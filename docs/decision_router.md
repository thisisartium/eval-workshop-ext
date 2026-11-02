# Decision Router (parallel path)

Escalate-only helpdesk routing using a **System One / decision model** via [LiteLLM](https://docs.litellm.ai/docs/decisions/) (`asystemone`) instead of the LLM concierge. Same surface as the routing baseline: classify department, then call `escalate_to_department`.

Default model: OpenAI Decisions (`openai/gpt-6-luna`). Override with `DECISION_MODEL` (e.g. `typesafe/jev-1.14.0`). For how decision models compare publicly, see the [Artificial Analysis Decision Index](https://artificialanalysis.ai/evaluations/artificial-analysis-decision-index).

## Why

- **Choice** over departments with calibrated probabilities
- **Noul** for “is this a real helpdesk request?” (handles appreciation / `None`)
- **Code** applies thresholds and escalates — the model judges, code decides

## Setup

1. For the default OpenAI model, set `OPENAI_API_KEY` in `.env`. Optional overrides:

```bash
# DECISION_MODEL=openai/gpt-6-luna   # default
# DECISION_MODEL=typesafe/jev-1.14.0 # needs TYPESAFE_API_KEY
```

2. Install deps (includes `litellm`):

```bash
uv sync
```

## Smoke test

```bash
uv run python scripts/decision_route_once.py "My laptop won't turn on"
uv run python scripts/decision_route_once.py "Thanks so much, you're the best!"
```

## Side-by-side experiment

Register the routing dataset (if needed), then run both paths against `live_routing`:

```bash
uv run python scripts/register_dataset.py data/live_workshop_routing.jsonl --name live_routing

# LLM baseline (agent + tools)
uv run cat-experiments run experiments/routing.py \
  -c experiments/baseline.yaml \
  --dataset live_routing

# Decision router
uv run cat-experiments run experiments/decision_routing.py \
  -c experiments/decision_baseline.yaml \
  --dataset live_routing
```

Compare in Cat Cafe under **Datasets → live_routing → Experiments** (`routing-baseline` vs `routing-decision-baseline`).

## Decision table

| `is_helpdesk_request` Noul | Action |
| --- | --- |
| `< 0.3` | Route `None` (not a ticket) |
| `> 0.7` | Escalate to Choice top department |
| Mid band | Abstain → `None` |

Department options: IT, HR, Facilities, Finance, Legal, Security. Criteria call out workshop confusions (employee expenses → HR vs vendor invoice → Finance, etc.).

## Files

| Path | Role |
| --- | --- |
| `src/helpdesk/decision_router.py` | Questions, thresholds, escalate |
| `experiments/decision_routing.py` | Experiment task + `department_match` |
| `experiments/decision_baseline.yaml` | Experiment config |
| `scripts/decision_route_once.py` | One-off smoke CLI |

For the parallel HR faithfulness judge (probability score + thresholded failure reason), see [decision_hr_judge.md](decision_hr_judge.md). Pitch: [why_decision_models.md](why_decision_models.md).
