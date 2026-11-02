# Jev Router (parallel path)

Escalate-only helpdesk routing using [TypeSafe Jev](https://typesafe.ai) instead of the LLM concierge. Same surface as the routing baseline: classify department, then call `escalate_to_department`.

## Why

- **Choice** over departments with calibrated probabilities
- **Noul** for “is this a real helpdesk request?” (handles appreciation / `None`)
- **Code** applies thresholds and escalates — Jev judges, code decides

## Setup

1. Create a TypeSafe API key and add it to `.env`:

```bash
TYPESAFE_API_KEY=...
```

2. Install deps (includes `typesafe-sdk`):

```bash
uv sync
```

## Smoke test

```bash
uv run python scripts/jev_route_once.py "My laptop won't turn on"
uv run python scripts/jev_route_once.py "Thanks so much, you're the best!"
```

## Side-by-side experiment

Register the routing dataset (if needed), then run both paths against `live_routing`:

```bash
uv run python scripts/register_dataset.py data/live_workshop_routing.jsonl --name live_routing

# LLM baseline (agent + tools)
uv run cat-experiments run experiments/routing.py \
  -c experiments/baseline.yaml \
  --dataset live_routing

# Jev router
uv run cat-experiments run experiments/jev_routing.py \
  -c experiments/jev_baseline.yaml \
  --dataset live_routing
```

Compare in Cat Cafe under **Datasets → live_routing → Experiments** (`routing-baseline` vs `routing-jev-baseline`).

## Decision table

| `is_helpdesk_request` Noul | Action |
| --- | --- |
| `< 0.3` | Route `None` (not a ticket) |
| `> 0.7` | Escalate to Choice top department |
| Mid band | Abstain → `None` |

Department options: IT, HR, Facilities, Finance, Legal, Security. Criteria call out workshop confusions (employee expenses → HR vs vendor invoice → Finance, etc.).

## Speed + cost benchmark

Compare the escalate-only LLM router vs Jev on the same dataset (accuracy, wall-clock, tokens, USD, API calls). The table has three rows:

- **Agent** — baseline escalate-only LLM router
- **Jev (per call)** — one TypeSafe request per example
- **Jev (batch)** — **one** TypeSafe `system_one` call for the whole dataset (parallel questions over `state.tickets`)

Costs use hardcoded rates in [`src/helpdesk/pricing.py`](../src/helpdesk/pricing.py) (update when list prices change — no live fetch). The summary prints speedup/cost delta/multiplier for agent vs batch and for per-call Jev vs batch.

```bash
uv run python scripts/compare_routing_bench.py \
  --dataset data/live_workshop_routing.jsonl
```

Uses `default_model()` for the agent (see `configs/models.yaml`). Override with `--model <id>` if needed.

## Files

| Path | Role |
| --- | --- |
| `src/helpdesk/jev_router.py` | Jev questions, thresholds, escalate |
| `src/helpdesk/pricing.py` | Hardcoded USD/MTok rates for cost estimates |
| `experiments/jev_routing.py` | Experiment task + `department_match` |
| `experiments/jev_baseline.yaml` | Experiment config |
| `scripts/jev_route_once.py` | One-off smoke CLI |
| `scripts/compare_routing_bench.py` | Agent vs Jev per-call vs Jev batch speed/cost bench |

For the parallel HR faithfulness judge (probability score + thresholded failure reason), see [jev_hr_judge.md](jev_hr_judge.md).
