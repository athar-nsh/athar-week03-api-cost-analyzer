"""Offline tests for price_updater — no network needed.  Run:  pytest -q"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chat_cost_analyzer as cca  # noqa: E402
import price_updater as pu  # noqa: E402

MD = (Path(__file__).parent / "fixtures" / "pricing_excerpt.md").read_text(encoding="utf-8")


def test_reads_standard_not_batch():
    p = pu.parse_standard_prices(MD)
    assert p["gpt-6-luna"]["input"] == 0.10      # Standard, not the Batch $0.05
    assert p["gpt-6-luna"]["output"] == 0.50
    assert p["gpt-6-luna"]["cached_input"] == 0.01


def test_long_context_and_second_table():
    p = pu.parse_standard_prices(MD)
    assert p["gpt-6-luna"]["long_context"] == {"threshold_tokens": 272_000, "input": 0.20,
                                               "cached_input": 0.02, "output": 0.75}
    assert p["gpt-5-mini"] == {"input": 0.25, "cached_input": 0.025, "output": 2.00}
    assert "cached_input" not in p["gpt-5.5-pro"]  # "-" is left out, never guessed


def test_missing_standard_section_stops():
    with pytest.raises(pu.PriceParseError):
        pu.parse_standard_prices(MD.replace("### Standard pricing data", "### Something else"))


def test_update_writes_only_after_confirmation(tmp_path):
    f = tmp_path / "pricing.json"
    f.write_text(json.dumps({"models": {"gpt-6-luna": {"input": 0.99, "cached_input": 0.01,
                                                       "output": 0.50}}}), encoding="utf-8")
    assert pu.update_pricing_file(f, MD, ask=lambda _: "n") is False
    assert json.loads(f.read_text())["models"]["gpt-6-luna"]["input"] == 0.99   # untouched
    assert pu.update_pricing_file(f, MD, ask=lambda _: "y") is True
    saved = json.loads(f.read_text())
    assert saved["models"]["gpt-6-luna"]["input"] == 0.10
    assert "official pricing page" in saved["_note"]
    assert (tmp_path / "pricing.json.bak").exists()


def test_explicit_long_context_rates_used_in_cost():
    pricing = {"models": {"m": {"input": 0.10, "cached_input": 0.01, "output": 0.50,
                                "long_context": {"threshold_tokens": 272_000, "input": 0.20,
                                                 "cached_input": 0.02, "output": 0.75}}}}
    c = cca.compute_cost("m", 300_000, 1_000, pricing=pricing)
    assert c["long_context_rate"]
    assert c["input_cost"] == pytest.approx(300_000 * 0.20 / 1e6)
    assert c["output_cost"] == pytest.approx(1_000 * 0.75 / 1e6)
