# Week 03 — Chat Completion API with Automated Token & Cost Analysis

**ITI Athar — GenAI & Agentic Systems** · Course: *Gen-AI: Tools for the Modern Developer* (Mahara-Tech)
**Author:** Dr. Noha El-Shoafy — Faculty of Computers and AI, Matrouh University

A Python script that calls the OpenAI **Chat Completions API** and analyses the tokens and cost of every request automatically. The prompts come from **KnowBridge**, a teaching assistant for *CS461 Intelligent Systems*.

> **Scope note.** The Week 3 task is general: *"Python script calling Chat Completion API with automated token & cost analysis."* It names no specific domain. I chose to apply it to my own Athar project, **KnowBridge**, so the cost figures answer a real question: what would it cost to run a teaching assistant for CS461 students? The prompts live in `prompts/templates.json` and can be swapped for any topic without changing the code.

## What it does

| Step | What happens | Week 3 topic |
|---|---|---|
| 1 | Builds the messages from a **system instruction** plus a **prompt template** (`prompts/templates.json`) | Prompt templates, system instructions |
| 2 | Counts input tokens **before** the call with `tiktoken` and estimates the cost | Token economics |
| 3 | Calls `chat.completions.create`, optionally with **streaming** (`stream_options={"include_usage": true}`) | Direct API integration, streaming |
| 4 | Reads the real `usage` the API returns: prompt, **cached**, completion and **reasoning** tokens | Token economics |
| 5 | Works out the cost from `pricing.json` (Standard input, cached-input and output rates; long-context rates above 272K input tokens) and compares estimate with actual | Cost calculation |
| 6 | Logs every request to `reports/usage_log.csv` and writes a Markdown report, including a **what-if** table showing the same usage on other models | Cost analysis |
| + | `--budget` stops a batch before it goes over a set spend; `--mock` runs everything offline | Cost control |
| + | `--update-prices` reads OpenAI's official pricing page, shows what changed and saves only after you confirm | Token economics (prices change) |

## Results (real run, 2026-11-26)

Four CS461 requests on `gpt-6-luna` → **449 input + 1,073 output tokens = $0.000581 in total** (about $0.000145 per request).

| Request | Prompt | Completion (reasoning) | Cost (USD) |
|---|---:|---:|---:|
| explain_concept: forward chaining | 108 | 209 (64) | 0.000115 |
| explain_concept: certainty factors | 111 | 304 (117) | 0.000163 |
| make_mcq: frames | 106 | 422 (117) | 0.000222 |
| rule_trace: forward chaining | 124 | 138 (40) | 0.000081 |

Key findings: output is 92% of the bill; hidden reasoning tokens are 31% of the output; the local `tiktoken` estimate was within 1 token of the API count. Full report: [`reports/RESULTS_week03.md`](reports/RESULTS_week03.md).

## Setup

Tested on Windows with Python 3.14.0, `openai` 3.60.0 and `tiktoken` 0.14.0.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env            # then paste your key into .env
```

> If PowerShell blocks the activate script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

> 🔒 The API key goes in `.env` only. `.env` is listed in `.gitignore`, so it is never pushed to GitHub.

## Run

```bash
# One templated request, streamed
python chat_cost_analyzer.py --template explain_concept --var topic="forward chaining" --stream

# Free-text prompt
python chat_cost_analyzer.py --prompt "What is an inference engine?"

# Batch of 4 CS461 requests, with a spending cap
python chat_cost_analyzer.py --batch prompts/batch.jsonl --budget 0.05

# Offline test of the full pipeline (no key, no cost)
python chat_cost_analyzer.py --batch prompts/batch.jsonl --mock

# Refresh prices from OpenAI's official pricing page (shows changes, asks before saving)
python chat_cost_analyzer.py --update-prices

# Same, from a saved copy of the page (e.g. offline)
python chat_cost_analyzer.py --update-prices --prices-from-file pricing_page.md

# Unit tests (offline)
pytest -q
```

## Cost formula

```
cost = (prompt − cached) × input_rate + cached × cached_input_rate + completion × output_rate
```
Rates are in USD per 1M tokens. Reasoning tokens are already counted inside `completion_tokens`, so they are billed at the output rate. Requests above 272K input tokens use the page's long-context rates, which are also stored in `pricing.json`.

Default prices in `pricing.json` come from OpenAI's [official pricing page](https://developers.openai.com/api/docs/pricing), Standard tier, checked 2026-11-27. This is the source the course points to (Mahara-Tech VID05, *OpenAI APIs: Pricing Model*). Prices change, so re-check before relying on totals:

| Model | Input | Cached input | Output |
|---|---:|---:|---:|
| gpt-6-luna | $0.10 | $0.01 | $0.50 |
| gpt-6-sol | $2.00 | $0.20 | $10.00 |
| gpt-5.6-luna | $0.20 | $0.02 | $1.20 |

## Files

```
chat_cost_analyzer.py    main script
pricing.json             model prices (Standard + long-context rates)
price_updater.py         reads the official pricing page for --update-prices
prompts/templates.json   system instruction + 3 CS461 templates
prompts/batch.jsonl      sample batch
tests/                   14 offline tests (cost maths, templates, mock pipeline, price parsing)
reports/RESULTS_week03.md   results of the real run (tables + findings)
reports/SAMPLE_mock_report.md  report format from an offline mock run
reports/                 other generated logs and reports are git-ignored
```

## Design choices & limitations

- **Pre-call token estimate.** `tiktoken` 0.14.0 has no mapping for `gpt-6-luna`, so the script uses the `o200k_base` encoding and labels the estimate "assumed". In the real run this estimate was within **1 token of the API count on every request (+0.9% mean error)**. If no tokenizer can be loaded, the script falls back to about 4 characters per token and says so. Billing always uses the API's own `usage` figures, never the estimate.
- **Prices are data, not code.** Rates live in `pricing.json`, taken from OpenAI's official pricing page.
- **Price updates are confirmed, not automatic.** OpenAI publishes prices on a web page, not through an API, and the page lists each model several times (Standard, Batch, Flex, Fast, Ultrafast). Reading the wrong table would silently give wrong costs. So `--update-prices` reads **only** the *Standard pricing data* table from the Markdown copy of the page, stops if that table is missing or a value looks implausible, shows every change, and writes only after confirmation (keeping a `.bak` copy). Prices are not fetched on every run, so each report stays tied to known, dated prices.
- **Offline mode for testing and teaching.** `--mock` runs the whole pipeline without a key or cost, so the unit tests and classroom demos need no API access. `reports/SAMPLE_mock_report.md` shows the report format from such a run; the real results are in `reports/RESULTS_week03.md`.
