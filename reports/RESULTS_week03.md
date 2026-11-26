# Token & Cost Report — gpt-6-luna

Generated: 2026-11-26 23:48  |  Requests: 4  |  Token estimator: tiktoken (o200k_base, assumed)
Prices: USD per 1M tokens, OpenAI standard tier. Verify on https://developers.openai.com/api/docs/pricing before relying on totals. Checked 2026-11-26.

| # | Request | Est. prompt | Prompt | Cached | Completion | Reasoning | Cost (USD) | Latency (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | explain_concept: forward chaining | 109 | 108 | 0 | 209 | 64 | 0.000115 | 2.924 |
| 2 | explain_concept: certainty factors in expert systems | 112 | 111 | 0 | 304 | 117 | 0.000163 | 3.374 |
| 3 | make_mcq: knowledge representation with frames | 107 | 106 | 0 | 422 | 117 | 0.000222 | 3.828 |
| 4 | rule_trace: R1: IF A AND B THEN C R2: IF C THEN D R3: IF D A | 125 | 124 | 0 | 138 | 40 | 0.000081 | 1.415 |
| | **Total** | 453 | 449 | 0 | 1073 | 338 | **0.000581** | 11.54 |

## Summary
- Average cost per request: $0.000145
- Projected cost for 1,000 similar requests: $0.1454
- Output share of the bill: 92.3%
- Mean prompt-estimate error vs API: +0.9%

## Same usage on other models (what-if)
| Model | Cost for these requests (USD) |
|---|---:|
| gpt-6-luna | 0.000581 |
| gpt-5.6-luna | 0.001377 |
| gpt-6-sol | 0.011628 |

## Findings
1. **Output drives the bill.** 92% of the cost came from completion tokens. Output is priced 5× input, so capping answer length saves more than shortening prompts.
2. **Reasoning tokens are hidden but billed.** 338 of the 1,073 completion tokens (31%) were internal reasoning the student never sees, all charged at the output rate.
3. **Local estimation is reliable.** `tiktoken` (o200k_base) was within 1 token of the API count on every request (+0.9% mean error), so costs can be predicted before sending.
4. **Streaming improves perceived speed.** The first token arrived in about 1–2 s; full answers took 1.4–3.8 s.
5. **Model choice matters most.** The same four requests would cost about 2.4× more on gpt-5.6-luna and 20× more on gpt-6-sol.
6. **Scale estimate for CS461.** At about $0.000145 per question, 100 students × 20 questions each (2,000 requests) would cost about $0.29 on gpt-6-luna.

_Real run, 2026-11-26, gpt-6-luna, `python chat_cost_analyzer.py --batch prompts/batch.jsonl --stream --budget 0.05`._
