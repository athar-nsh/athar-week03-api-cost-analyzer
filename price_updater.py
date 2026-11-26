"""
price_updater.py — refresh pricing.json from OpenAI's official pricing page.

Design rule: never guess. The page lists the same model several times (Standard, Batch,
Flex, Fast, Ultrafast). We read ONLY the "Standard pricing data" section, show every change,
and write nothing unless the user confirms. Any parsing doubt stops the update and keeps
the existing prices.
"""
from __future__ import annotations

import json
import re
import shutil
import urllib.request
from datetime import date
from pathlib import Path

PRICING_URL = "https://developers.openai.com/api/docs/pricing"
PRICING_MD_URL = PRICING_URL + ".md"  # the docs serve a Markdown copy at <page>.md
STANDARD_HEADING = "### Standard pricing data"
LONG_CONTEXT_THRESHOLD = 272_000  # page: "Short context: ≤272K input tokens. Long context: >272K"


class PriceParseError(RuntimeError):
    """Raised when the page does not look the way the parser expects."""


def fetch_markdown(url: str = PRICING_MD_URL, timeout: int = 20) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "chat-cost-analyzer/1.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def _money(cell: str) -> float | None:
    cell = cell.strip()
    if cell in {"", "-"}:
        return None
    m = re.fullmatch(r"\$([0-9]+(?:\.[0-9]+)?)", cell)
    if not m:
        raise PriceParseError(f"Unexpected price cell: {cell!r}")
    return float(m.group(1))


def _clean_model(name: str) -> str:
    return re.sub(r"\s*\(.*?\)\s*$", "", name.strip())  # "gpt-5.5 (<272K context length)" -> "gpt-5.5"


def standard_section(md: str) -> str:
    start = md.find(STANDARD_HEADING)
    if start < 0:
        raise PriceParseError(f"'{STANDARD_HEADING}' not found — the page layout may have changed.")
    nxt = md.find("\n### ", start + len(STANDARD_HEADING))
    return md[start: nxt if nxt > 0 else len(md)]


def parse_standard_prices(md: str) -> dict[str, dict]:
    """Return {model: {input, cached_input, output, long_context?}} from the Standard tables only."""
    section = standard_section(md)
    prices: dict[str, dict] = {}
    header: list[str] | None = None
    for line in section.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            header = None if not line else header
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells[0] == "Model":
            header = [c.lower() for c in cells]
            continue
        if header is None or set(cells[0]) <= {"-", " "}:
            continue
        row = dict(zip(header, cells))
        short = {
            "input": _money(row.get("short context input", row.get("input", "-"))),
            "cached_input": _money(row.get("short context cached input", row.get("cached input", "-"))),
            "output": _money(row.get("short context output", row.get("output", "-"))),
        }
        if short["input"] is None or short["output"] is None:
            continue  # incomplete rows are skipped, never filled in
        entry = {k: v for k, v in short.items() if v is not None}
        if "long context input" in row:
            long_ = {
                "input": _money(row["long context input"]),
                "cached_input": _money(row.get("long context cached input", "-")),
                "output": _money(row["long context output"]),
            }
            if long_["input"] is not None and long_["output"] is not None:
                entry["long_context"] = {"threshold_tokens": LONG_CONTEXT_THRESHOLD,
                                         **{k: v for k, v in long_.items() if v is not None}}
        prices[_clean_model(cells[0])] = entry
    if not prices:
        raise PriceParseError("Standard section found but no price rows could be read.")
    return prices


def _sanity_check(model: str, p: dict) -> None:
    if not (0 < p["input"] <= p["output"]):
        raise PriceParseError(f"{model}: implausible prices {p} (expected 0 < input ≤ output).")


def diff_prices(old: dict, new: dict, models: list[str]) -> list[str]:
    changes = []
    for m in models:
        o, n = old.get(m, {}), new.get(m, {})
        for k in ("input", "cached_input", "output"):
            if o.get(k) != n.get(k):
                changes.append(f"{m:<14} {k:<17} {o.get(k)} -> {n.get(k)}")
        ol, nl = o.get("long_context", {}), n.get("long_context", {})
        for k in ("input", "cached_input", "output"):
            if ol.get(k) != nl.get(k):
                changes.append(f"{m:<14} long {k:<12} {ol.get(k)} -> {nl.get(k)}")
    return changes


def update_pricing_file(pricing_path: Path, md: str, assume_yes: bool = False,
                        source: str = PRICING_URL, ask=input) -> bool:
    """Update the models already listed in pricing.json. Returns True if the file was written."""
    current = json.loads(pricing_path.read_text(encoding="utf-8"))
    official = parse_standard_prices(md)
    tracked = list(current["models"])
    missing = [m for m in tracked if m not in official]
    if missing:
        print(f"⚠ Not found in the Standard table (kept unchanged): {', '.join(missing)}")
    updated = {m: official[m] for m in tracked if m in official}
    for m, p in updated.items():
        _sanity_check(m, p)

    changes = diff_prices(current["models"], updated, list(updated))
    if not changes:
        print("✓ Prices in pricing.json already match the official Standard table.")
        return False
    print("Changes found (Standard tier, USD per 1M tokens):")
    for c in changes:
        print("  " + c)
    if not assume_yes and ask("Save these prices to pricing.json? [y/N] ").strip().lower() != "y":
        print("No changes saved.")
        return False

    shutil.copyfile(pricing_path, pricing_path.with_suffix(".json.bak"))
    current["models"].update(updated)
    current["_note"] = (f"USD per 1M tokens, Standard tier, from OpenAI's official pricing page "
                        f"{source} (updated {date.today().isoformat()} with --update-prices). "
                        f"Prices change; re-check before relying on totals.")
    pricing_path.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
    print(f"✓ Saved. Previous file kept as {pricing_path.with_suffix('.json.bak').name}")
    return True
