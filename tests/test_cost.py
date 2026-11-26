"""Offline tests — no API key or network needed.  Run:  pytest -q"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chat_cost_analyzer as cca  # noqa: E402

PRICING = {"models": {
    "m": {"input": 0.10, "cached_input": 0.01, "output": 0.50,
          "long_context": {"threshold_tokens": 1000, "input_multiplier": 2.0, "output_multiplier": 1.5}},
}}


def test_basic_cost():
    c = cca.compute_cost("m", 1_000, 500, pricing=PRICING)  # below the long-context threshold
    assert c["input_cost"] == pytest.approx(1_000 * 0.10 / 1e6)
    assert c["output_cost"] == pytest.approx(500 * 0.50 / 1e6)
    assert c["total_cost"] == pytest.approx(0.0001 + 0.00025)
    assert not c["long_context_rate"]


def test_cached_tokens_billed_at_cached_rate():
    c = cca.compute_cost("m", 800, 0, cached_tokens=600, pricing=PRICING)
    assert c["input_cost"] == pytest.approx(200 * 0.10 / 1e6)
    assert c["cached_input_cost"] == pytest.approx(600 * 0.01 / 1e6)


def test_long_context_multipliers():
    c = cca.compute_cost("m", 2_000, 100, pricing=PRICING)
    assert c["long_context_rate"]
    assert c["input_cost"] == pytest.approx(2_000 * 0.20 / 1e6)
    assert c["output_cost"] == pytest.approx(100 * 0.75 / 1e6)


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        cca.compute_cost("nope", 1, 1, pricing=PRICING)


def test_build_messages_from_template():
    msgs = cca.build_messages(template="explain_concept", variables={"topic": "frames"})
    assert msgs[0]["role"] == "system" and "CS461" in msgs[0]["content"]
    assert "frames" in msgs[1]["content"]


def test_missing_template_variable():
    with pytest.raises(KeyError):
        cca.build_messages(template="rule_trace", variables={"rules": "R1"})


def test_usage_fields_from_dict():
    u = {"prompt_tokens": 50, "completion_tokens": 20,
         "prompt_tokens_details": {"cached_tokens": 10},
         "completion_tokens_details": {"reasoning_tokens": 5}}
    assert cca._usage_fields(u) == (50, 10, 20, 5)


@pytest.mark.parametrize("stream", [False, True])
def test_mock_pipeline_end_to_end(stream, tmp_path):
    counter = cca.TokenCounter("gpt-6-luna")
    pricing = cca.load_pricing()
    msgs = cca.build_messages(prompt := "What is an inference engine?")
    r = cca.run_once(cca.MockClient(counter), "gpt-6-luna", msgs, "t", stream,
                     counter, pricing, mock=True, echo=False)
    assert r.prompt_tokens == r.estimated_prompt_tokens > 0
    assert r.completion_tokens > 0 and r.total_cost > 0
    assert r.answer.startswith("[MOCK ANSWER]")
    report = cca.write_report([r], pricing, tmp_path / "rep.md")
    assert "MOCK RUN" in report.read_text(encoding="utf-8")
