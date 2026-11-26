"""
chat_cost_analyzer.py — Week 03, ITI Athar (GenAI & Agentic Systems)

Calls the OpenAI Chat Completions API and analyses tokens and cost for every request:
  1. Builds the messages from a system instruction + a prompt template.
  2. Estimates input tokens BEFORE the call (tiktoken) and projects the cost.
  3. Calls the API (optionally streaming the answer token by token).
  4. Reads the real token usage returned by the API (prompt, cached, completion, reasoning).
  5. Computes the actual cost from pricing.json, compares estimate vs actual,
     logs every run to reports/usage_log.csv and writes a Markdown cost report.

Author: Dr. Noha El-Shoafy (د. نهى الشعافي)

Usage examples
  python chat_cost_analyzer.py --template explain_concept --var topic="forward chaining" --stream
  python chat_cost_analyzer.py --prompt "What is an inference engine?"
  python chat_cost_analyzer.py --batch prompts/batch.jsonl
  python chat_cost_analyzer.py --batch prompts/batch.jsonl --mock     # offline, no API key needed
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
PRICING_FILE = BASE / "pricing.json"
TEMPLATES_FILE = BASE / "prompts" / "templates.json"
REPORT_DIR = BASE / "reports"
LOG_FILE = REPORT_DIR / "usage_log.csv"

# Chat-format overhead used by OpenAI's published counting recipe (per message + reply priming).
TOKENS_PER_MESSAGE = 3
TOKENS_REPLY_PRIMING = 3


# --------------------------------------------------------------------------- pricing
def load_pricing(path: Path = PRICING_FILE) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compute_cost(model: str, prompt_tokens: int, completion_tokens: int,
                 cached_tokens: int = 0, pricing: dict | None = None) -> dict:
    """Return a cost breakdown in USD. Prices in pricing.json are per 1M tokens."""
    pricing = pricing or load_pricing()
    if model not in pricing["models"]:
        raise KeyError(f"No price for model '{model}'. Add it to pricing.json.")
    p = pricing["models"][model]
    in_rate, cached_rate, out_rate = p["input"], p.get("cached_input", p["input"]), p["output"]

    long_ctx = p.get("long_context")
    is_long = bool(long_ctx) and prompt_tokens > long_ctx["threshold_tokens"]
    if is_long:
        in_rate *= long_ctx["input_multiplier"]
        cached_rate *= long_ctx["input_multiplier"]
        out_rate *= long_ctx["output_multiplier"]

    cached_tokens = min(cached_tokens, prompt_tokens)
    uncached = prompt_tokens - cached_tokens
    input_cost = uncached * in_rate / 1e6
    cached_cost = cached_tokens * cached_rate / 1e6
    output_cost = completion_tokens * out_rate / 1e6
    return {
        "input_cost": input_cost,
        "cached_input_cost": cached_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + cached_cost + output_cost,
        "long_context_rate": is_long,
    }


# --------------------------------------------------------------------------- tokens
class TokenCounter:
    """Counts tokens with tiktoken; falls back to a ~4 chars/token heuristic if the
    tokenizer file cannot be loaded (e.g. offline). The method used is always reported."""

    def __init__(self, model: str):
        self.method = "heuristic (~4 chars/token)"
        self._enc = None
        try:
            import tiktoken
            try:
                self._enc = tiktoken.encoding_for_model(model)
                self.method = f"tiktoken ({self._enc.name})"
            except KeyError:
                # Newer models may not be mapped yet; o200k_base is the encoding of recent OpenAI models.
                self._enc = tiktoken.get_encoding("o200k_base")
                self.method = "tiktoken (o200k_base, assumed)"
        except Exception:
            self._enc = None

    def count(self, text: str) -> int:
        if self._enc is not None:
            return len(self._enc.encode(text))
        return max(1, round(len(text) / 4))

    def count_messages(self, messages: list[dict]) -> int:
        total = TOKENS_REPLY_PRIMING
        for m in messages:
            total += TOKENS_PER_MESSAGE + self.count(m["role"]) + self.count(m["content"])
        return total


# --------------------------------------------------------------------------- prompts
def load_templates(path: Path = TEMPLATES_FILE) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_messages(user_text: str | None = None, template: str | None = None,
                   variables: dict | None = None, templates: dict | None = None) -> list[dict]:
    templates = templates or load_templates()
    values = {**templates.get("defaults", {}), **(variables or {})}
    system = templates["system"].format(**values)
    if template:
        if template not in templates["templates"]:
            raise KeyError(f"Unknown template '{template}'. Options: {list(templates['templates'])}")
        try:
            user_text = templates["templates"][template].format(**values)
        except KeyError as e:
            raise KeyError(f"Template '{template}' needs variable {e}. Pass it with --var name=value")
    if not user_text:
        raise ValueError("Give --prompt or --template.")
    return [{"role": "system", "content": system}, {"role": "user", "content": user_text}]


# --------------------------------------------------------------------------- API call
@dataclass
class RunResult:
    timestamp: str
    model: str
    label: str
    estimated_prompt_tokens: int
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    total_tokens: int
    estimated_input_cost: float
    input_cost: float
    cached_input_cost: float
    output_cost: float
    total_cost: float
    latency_s: float
    time_to_first_token_s: float | None
    streamed: bool
    mock: bool
    token_method: str
    answer: str = field(repr=False, default="")

    @property
    def estimate_error_pct(self) -> float:
        if not self.prompt_tokens:
            return 0.0
        return 100 * (self.estimated_prompt_tokens - self.prompt_tokens) / self.prompt_tokens


def _usage_fields(usage) -> tuple[int, int, int, int]:
    """Extract (prompt, cached, completion, reasoning) from an SDK usage object or dict."""
    def g(obj, name, default=0):
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(name, default) or default
        return getattr(obj, name, default) or default

    prompt = g(usage, "prompt_tokens")
    completion = g(usage, "completion_tokens")
    cached = g(g(usage, "prompt_tokens_details", None), "cached_tokens")
    reasoning = g(g(usage, "completion_tokens_details", None), "reasoning_tokens")
    return prompt, cached, completion, reasoning


def call_chat(client, model: str, messages: list[dict], stream: bool,
              max_completion_tokens: int | None, echo: bool = True):
    """Returns (answer_text, usage, latency_s, ttft_s)."""
    kwargs = {"model": model, "messages": messages}
    if max_completion_tokens:
        kwargs["max_completion_tokens"] = max_completion_tokens
    start = time.perf_counter()

    if not stream:
        resp = client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
        if echo:
            print(text)
        return text, resp.usage, time.perf_counter() - start, None

    # Streaming: ask the API to send a final chunk that carries the usage numbers.
    kwargs.update(stream=True, stream_options={"include_usage": True})
    parts, usage, ttft = [], None, None
    for chunk in client.chat.completions.create(**kwargs):
        if chunk.choices:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                if ttft is None:
                    ttft = time.perf_counter() - start
                parts.append(delta)
                if echo:
                    print(delta, end="", flush=True)
        if getattr(chunk, "usage", None):
            usage = chunk.usage
    if echo:
        print()
    return "".join(parts), usage, time.perf_counter() - start, ttft


class MockClient:
    """Offline stand-in for OpenAI(): returns a canned answer and realistic usage so the
    whole pipeline (estimate -> call -> cost -> report) can be tested without a key."""

    def __init__(self, counter: TokenCounter):
        self.counter = counter
        self.chat = self
        self.completions = self

    def create(self, model, messages, stream=False, **_):
        answer = ("[MOCK ANSWER] This is an offline placeholder response used to test "
                  "token counting and cost calculation without calling the API.")
        prompt = self.counter.count_messages(messages)
        usage = {"prompt_tokens": prompt, "completion_tokens": self.counter.count(answer),
                 "prompt_tokens_details": {"cached_tokens": 0},
                 "completion_tokens_details": {"reasoning_tokens": 0}}
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        if not stream:
            msg = type("M", (), {"content": answer})
            choice = type("C", (), {"message": msg})
            return type("R", (), {"choices": [choice], "usage": usage})
        return self._stream(answer, usage)

    @staticmethod
    def _stream(answer, usage):
        for word in answer.split(" "):
            delta = type("D", (), {"content": word + " "})
            yield type("K", (), {"choices": [type("C", (), {"delta": delta})], "usage": None})
        yield type("K", (), {"choices": [], "usage": usage})


def run_once(client, model: str, messages: list[dict], label: str, stream: bool,
             counter: TokenCounter, pricing: dict, mock: bool,
             max_completion_tokens: int | None = None, echo: bool = True) -> RunResult:
    est_prompt = counter.count_messages(messages)
    est_cost = compute_cost(model, est_prompt, 0, pricing=pricing)["input_cost"]
    if echo:
        print(f"\n── {label} ── estimated input: {est_prompt} tokens "
              f"(≈ ${est_cost:.6f}) via {counter.method}\n")

    text, usage, latency, ttft = call_chat(client, model, messages, stream,
                                           max_completion_tokens, echo)
    prompt, cached, completion, reasoning = _usage_fields(usage)
    if usage is None:  # should not happen, but never report zero cost silently
        prompt, completion = est_prompt, counter.count(text)
        print("⚠ API returned no usage; falling back to local token counts.", file=sys.stderr)
    cost = compute_cost(model, prompt, completion, cached, pricing)

    return RunResult(
        timestamp=datetime.now().isoformat(timespec="seconds"), model=model, label=label,
        estimated_prompt_tokens=est_prompt, prompt_tokens=prompt, cached_tokens=cached,
        completion_tokens=completion, reasoning_tokens=reasoning,
        total_tokens=prompt + completion, estimated_input_cost=est_cost,
        input_cost=cost["input_cost"], cached_input_cost=cost["cached_input_cost"],
        output_cost=cost["output_cost"], total_cost=cost["total_cost"],
        latency_s=round(latency, 3), time_to_first_token_s=round(ttft, 3) if ttft else None,
        streamed=stream, mock=mock, token_method=counter.method, answer=text,
    )


# --------------------------------------------------------------------------- reporting
def print_summary(r: RunResult) -> None:
    rows = [
        ("Prompt tokens (API)", f"{r.prompt_tokens}  (estimated {r.estimated_prompt_tokens}, "
                                f"error {r.estimate_error_pct:+.1f}%)"),
        ("  of which cached", r.cached_tokens),
        ("Completion tokens", f"{r.completion_tokens}  (reasoning {r.reasoning_tokens})"),
        ("Total tokens", r.total_tokens),
        ("Input cost", f"${r.input_cost + r.cached_input_cost:.6f}"),
        ("Output cost", f"${r.output_cost:.6f}"),
        ("TOTAL cost", f"${r.total_cost:.6f}"),
        ("Cost per 1,000 such calls", f"${r.total_cost * 1000:.4f}"),
        ("Latency", f"{r.latency_s}s" + (f"  (first token {r.time_to_first_token_s}s)"
                                         if r.time_to_first_token_s else "")),
    ]
    print("\n" + "─" * 64 + f"\nToken & cost analysis — {r.model}" + ("  [MOCK]" if r.mock else ""))
    for k, v in rows:
        print(f"  {k:<28} {v}")
    print("─" * 64)


def append_log(results: list[RunResult], path: Path = LOG_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [k for k in asdict(results[0]) if k != "answer"]
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        for r in results:
            w.writerow({k: v for k, v in asdict(r).items() if k != "answer"})


def projection_table(results: list[RunResult], pricing: dict) -> list[tuple[str, float]]:
    """Cost of the same token usage on every priced model (what-if comparison)."""
    out = []
    for m in pricing["models"]:
        total = sum(compute_cost(m, r.prompt_tokens, r.completion_tokens, r.cached_tokens,
                                 pricing)["total_cost"] for r in results)
        out.append((m, total))
    return sorted(out, key=lambda x: x[1])


def write_report(results: list[RunResult], pricing: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(results)
    tot = lambda a: sum(getattr(r, a) for r in results)
    mock = any(r.mock for r in results)
    lines = [
        f"# Token & Cost Report — {results[0].model}" + (" (MOCK RUN — not real API data)" if mock else ""),
        "",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M}  |  Requests: {n}  |  "
        f"Token estimator: {results[0].token_method}",
        f"Prices: {pricing.get('_note', '')}",
        "",
        "| # | Request | Est. prompt | Prompt | Cached | Completion | Reasoning | Cost (USD) | Latency (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(results, 1):
        lines.append(f"| {i} | {r.label} | {r.estimated_prompt_tokens} | {r.prompt_tokens} | "
                     f"{r.cached_tokens} | {r.completion_tokens} | {r.reasoning_tokens} | "
                     f"{r.total_cost:.6f} | {r.latency_s} |")
    total_cost = tot("total_cost")
    lines += [
        f"| | **Total** | {tot('estimated_prompt_tokens')} | {tot('prompt_tokens')} | "
        f"{tot('cached_tokens')} | {tot('completion_tokens')} | {tot('reasoning_tokens')} | "
        f"**{total_cost:.6f}** | {sum(r.latency_s for r in results):.2f} |",
        "",
        "## Summary",
        f"- Average cost per request: ${total_cost / n:.6f}",
        f"- Projected cost for 1,000 similar requests: ${total_cost / n * 1000:.4f}",
        f"- Output share of the bill: "
        f"{100 * tot('output_cost') / total_cost if total_cost else 0:.1f}%",
        f"- Mean prompt-estimate error vs API: "
        f"{sum(r.estimate_error_pct for r in results) / n:+.1f}%",
        "",
        "## Same usage on other models (what-if)",
        "| Model | Cost for these requests (USD) |",
        "|---|---:|",
    ]
    lines += [f"| {m} | {c:.6f} |" for m, c in projection_table(results, pricing)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------- CLI
def parse_vars(items: list[str]) -> dict:
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--var must look like name=value (got '{it}')")
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Chat Completion API call with token & cost analysis")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--prompt", help="free-text user prompt")
    src.add_argument("--template", help="template name from prompts/templates.json")
    src.add_argument("--batch", help="JSONL file: one {template, vars} or {prompt} per line")
    ap.add_argument("--var", action="append", help="template variable, name=value (repeatable)")
    ap.add_argument("--model", default=None, help="model id (default: $OPENAI_MODEL or gpt-6-luna)")
    ap.add_argument("--stream", action="store_true", help="stream the answer token by token")
    ap.add_argument("--max-tokens", type=int, default=None, help="max_completion_tokens cap")
    ap.add_argument("--budget", type=float, default=None, help="stop a batch above this USD total")
    ap.add_argument("--mock", action="store_true", help="offline mode, no API call")
    args = ap.parse_args(argv)

    try:
        from dotenv import load_dotenv
        load_dotenv(BASE / ".env")
    except ImportError:
        pass

    model = args.model or os.getenv("OPENAI_MODEL", "gpt-6-luna")
    budget = args.budget if args.budget is not None else (
        float(os.getenv("BUDGET_USD")) if os.getenv("BUDGET_USD") else None)
    pricing, templates = load_pricing(), load_templates()
    counter = TokenCounter(model)

    if args.mock:
        client = MockClient(counter)
    else:
        if not os.getenv("OPENAI_API_KEY"):
            print("OPENAI_API_KEY is missing. Copy .env.example to .env and add the key, "
                  "or run with --mock.", file=sys.stderr)
            return 1
        from openai import OpenAI
        client = OpenAI()

    if args.batch:
        jobs = []
        for line in Path(args.batch).read_text(encoding="utf-8").splitlines():
            if line.strip():
                j = json.loads(line)
                jobs.append((j.get("template"), j.get("prompt"), j.get("vars", {})))
    else:
        jobs = [(args.template, args.prompt, parse_vars(args.var))]

    results, spent = [], 0.0
    for tpl, prompt, variables in jobs:
        messages = build_messages(prompt, tpl, variables, templates)
        label = (tpl + (f": {next(iter(variables.values()))}" if variables else "")) if tpl else "free prompt"
        label = " ".join(label.split())[:60]  # one line, short, safe for tables
        if budget is not None:
            projected = spent + compute_cost(model, counter.count_messages(messages), 0,
                                             pricing=pricing)["total_cost"]
            if projected > budget:
                print(f"⛔ Budget ${budget} would be exceeded — stopping before '{label}'.")
                break
        try:
            r = run_once(client, model, messages, label, args.stream, counter, pricing,
                         args.mock, args.max_tokens)
        except Exception as e:  # keep the batch going and report the failure clearly
            print(f"✖ Request '{label}' failed: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        print_summary(r)
        results.append(r)
        spent += r.total_cost

    if not results:
        return 1
    append_log(results)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = write_report(results, pricing, REPORT_DIR / f"report_{stamp}{'_mock' if args.mock else ''}.md")
    print(f"\nRequests: {len(results)} | Total cost: ${spent:.6f}")
    print(f"Log: {LOG_FILE.relative_to(BASE)} | Report: {report.relative_to(BASE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
