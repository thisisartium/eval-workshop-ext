# Decision HR Faithfulness Judge (parallel path)

Scores HR specialist responses with a **System One / decision model** via [LiteLLM](https://docs.litellm.ai/docs/decisions/) (`asystemone`) instead of always calling a binary LLM-as-judge. Same task as `experiments/hr_specialist.py`: run the specialist directly and evaluate groundedness against retrieved KB content.

Default model: OpenAI Decisions (`openai/gpt-6-luna`). Override with `DECISION_MODEL` (e.g. `typesafe/jev-1.14.0`). Public comparison: [Artificial Analysis Decision Index](https://artificialanalysis.ai/evaluations/artificial-analysis-decision-index).

## Why

- **Noul** returns a calibrated probability that the response is faithful to the KB (always)
- **Code** applies the pass threshold (`0.8`)
- **LLM critique** runs only when `score < 0.8` — clear passes skip the expensive judge call

Cat Cafe accepts continuous float scores; this path uses them as the evaluator score.

## Setup

Requires a provider key for the configured `DECISION_MODEL` (default needs `OPENAI_API_KEY`). See [decision_router.md](decision_router.md). Also needs the usual OpenAI/Anthropic key for fail-only critiques.

## Side-by-side experiment

```bash
uv run python scripts/register_dataset.py data/live_workshop_hr.jsonl --name live_hr

# LLM binary judge (critique every example)
uv run cat-experiments run experiments/hr_specialist.py \
  -c experiments/with_specialists.yaml \
  --dataset live_hr

# Decision probability + LLM critique on fails only
uv run cat-experiments run experiments/decision_hr_specialist.py \
  -c experiments/decision_hr_baseline.yaml \
  --dataset live_hr
```

Tuned agent variant:

```bash
uv run cat-experiments run experiments/decision_hr_specialist.py \
  -c experiments/decision_hr_tuned.yaml \
  --dataset live_hr
```

Compare in Cat Cafe under **Datasets → live_hr → Experiments**.

## Decision table

| `is_faithful` Noul | Label | Explanation |
| --- | --- | --- |
| `>= 0.8` | `pass` | Empty (no LLM call) |
| `< 0.8` | `fail` | LLM critique (same prompt as the baseline judge) |

Filter fails in Cat Cafe:

```
evaluations.hr_response_quality.score < 0.8
```

## Judge speed + cost benchmark

Generate HR specialist answers once, then compare judges on the same triples (pass rate, agreement with the LLM judge, wall-clock, tokens, USD, API calls):

- **LLM judge** — always-on critique
- **Decision (per call)** — one Noul per example; LLM critique only if `score < 0.8`
- **Decision (batch)** — one `asystemone` for all examples; LLM critique only on fails

Agent generation cost is printed separately and excluded from the judge table. Costs use [`pricing.py`](../src/helpdesk/pricing.py).

```bash
uv run python scripts/compare_hr_judge_bench.py \
  --dataset data/live_workshop_hr.jsonl
```

## Files

| Path | Role |
| --- | --- |
| `src/helpdesk/decision_hr_judge.py` | Noul + threshold + batch |
| `src/helpdesk/llm_faithfulness_critique.py` | Shared LLM critique prompt/call |
| `experiments/decision_hr_specialist.py` | Same HR task + hybrid `hr_response_quality` |
| `experiments/decision_hr_baseline.yaml` | Untuned agent + decision judge |
| `experiments/decision_hr_tuned.yaml` | Tuned agent + decision judge |
| `scripts/compare_hr_judge_bench.py` | LLM vs decision per-call vs batch speed/cost bench |
