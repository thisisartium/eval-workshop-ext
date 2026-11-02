#!/usr/bin/env python3
"""Compare LLM-as-judge vs decision-model hybrid (per-call and batch) on HR answers.

Generates specialist responses once, then times/costs only the judges.
Costs use ``helpdesk.pricing``.

Usage:
    uv run python scripts/compare_hr_judge_bench.py \
        --dataset data/live_workshop_hr.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from agent_platform.observability import (
    CostTracker,
    MultiProcessor,
    Span,
    ToolSpanData,
    TracingProcessor,
)
from agent_platform.runner import collect
from helpdesk.app import create_specialist_app, default_model
from helpdesk.decision_hr_judge import (
    FAITHFUL_PASS,
    judge_faithfulness,
    judge_faithfulness_batch,
)
from helpdesk.llm_faithfulness_critique import (
    faithfulness_critique_prompt,
    judge_with_llm,
)
from helpdesk.pricing import estimate_cost, format_rates_footer, resolve_rate


@dataclass
class CasePayload:
    """Shared inputs for all judges."""

    example_id: str
    question: str
    response: str
    retrieved_context: str


@dataclass
class JudgeTotals:
    """Accumulated judge-only metrics for one method."""

    name: str
    passes: int = 0
    total: int = 0
    agree_with_llm: int = 0
    seconds: float = 0.0
    api_calls: int = 0
    # TypeSafe usage
    ts_model: str = "decision"
    ts_input_tokens: int = 0
    ts_output_tokens: int = 0
    # LLM critique / judge usage
    llm_model: str = ""
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    llm_calls: int = 0
    labels: list[bool] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passes / self.total if self.total else 0.0

    @property
    def agree_rate(self) -> float:
        return self.agree_with_llm / self.total if self.total else 0.0

    @property
    def cost_usd(self) -> float:
        cost = 0.0
        if self.ts_input_tokens or self.ts_output_tokens:
            cost += estimate_cost(
                self.ts_model, self.ts_input_tokens, self.ts_output_tokens
            )
        if self.llm_model and (self.llm_input_tokens or self.llm_output_tokens):
            cost += estimate_cost(
                self.llm_model, self.llm_input_tokens, self.llm_output_tokens
            )
        return cost

    @property
    def in_tokens(self) -> int:
        return self.ts_input_tokens + self.llm_input_tokens

    @property
    def out_tokens(self) -> int:
        return self.ts_output_tokens + self.llm_output_tokens


class KbToolCollector(TracingProcessor):
    """Collects search_knowledge_base tool outputs for judge context."""

    def __init__(self) -> None:
        self.kb_chunks: list[str] = []

    def on_span_start(self, span: Span) -> None:
        return

    def on_span_end(self, span: Span) -> None:
        if not isinstance(span.span_data, ToolSpanData):
            return
        if span.span_data.name != "search_knowledge_base":
            return
        out = span.span_data.output
        if isinstance(out, str) and out.strip():
            self.kb_chunks.append(out)

    def clear(self) -> None:
        self.kb_chunks.clear()

    def joined(self) -> str:
        return "\n\n".join(self.kb_chunks)


def load_examples(path: Path) -> list[dict]:
    examples = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    return examples


def _print_compare(label: str, slow: JudgeTotals, fast: JudgeTotals) -> None:
    if fast.seconds > 0:
        print(f"Speedup ({label}): {slow.seconds / fast.seconds:.2f}x")
    print(f"Cost delta ({label}): ${slow.cost_usd - fast.cost_usd:.6f}")
    if fast.cost_usd > 0:
        print(
            f"Cost multiplier ({label}): "
            f"{slow.cost_usd / fast.cost_usd:.2f}x"
        )


def print_table(
    llm: JudgeTotals,
    decision: JudgeTotals,
    decision_batch: JudgeTotals,
) -> None:
    rows = [llm, decision, decision_batch]
    headers = (
        "Method",
        "Pass",
        "Agree LLM",
        "Total s",
        "Calls",
        "In tok",
        "Out tok",
        "Cost USD",
    )
    data = []
    for row in rows:
        data.append(
            [
                row.name,
                f"{row.passes}/{row.total} ({row.pass_rate:.0%})",
                f"{row.agree_with_llm}/{row.total} ({row.agree_rate:.0%})",
                f"{row.seconds:.2f}",
                str(row.api_calls),
                str(row.in_tokens),
                str(row.out_tokens),
                f"${row.cost_usd:.6f}",
            ]
        )

    widths = [len(h) for h in headers]
    for row in data:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(cols: list[str]) -> str:
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cols))

    print(fmt(list(headers)))
    print(fmt(["-" * w for w in widths]))
    for row in data:
        print(fmt(row))

    print()
    print(
        "Token note: In/Out tok and Cost USD combine TypeSafe + LLM critique "
        "tokens for hybrid methods."
    )
    print(
        f"  LLM judge:     ts=0 llm_calls={llm.llm_calls} "
        f"llm_tok={llm.llm_input_tokens}/{llm.llm_output_tokens}"
    )
    print(
        f"  Decision per-call:  ts_tok={decision.ts_input_tokens}/{decision.ts_output_tokens} "
        f"llm_calls={decision.llm_calls} "
        f"llm_tok={decision.llm_input_tokens}/{decision.llm_output_tokens}"
    )
    print(
        f"  Decision batch:     ts_tok={decision_batch.ts_input_tokens}/{decision_batch.ts_output_tokens} "
        f"llm_calls={decision_batch.llm_calls} "
        f"llm_tok={decision_batch.llm_input_tokens}/{decision_batch.llm_output_tokens}"
    )
    print()
    _print_compare("llm / decision-batch", llm, decision_batch)
    print()
    _print_compare("decision-per-call / decision-batch", decision, decision_batch)
    print()
    models = [
        m
        for m in (
            llm.llm_model,
            decision.ts_model,
            decision_batch.ts_model,
            decision.llm_model,
        )
        if m
    ]
    print(format_rates_footer(models))


async def generate_cases(
    examples: list[dict],
    model: str,
) -> tuple[list[CasePayload], float, int, int, float]:
    """Run HR specialist once per example; return cases + gen cost stats."""
    cost = CostTracker()
    kb = KbToolCollector()
    client = create_specialist_app(
        agent_config="configs/agents/hr_with_specialists.yaml",
        model=model,
        processor=MultiProcessor([cost, kb]),
    )

    cases: list[CasePayload] = []
    t0 = time.perf_counter()
    for ex in examples:
        ex_id = str(ex.get("id", f"ex_{len(cases)}"))
        question = ex.get("input", {}).get("request", "")
        kb.clear()
        text, _ = await collect(client.send(question))
        cases.append(
            CasePayload(
                example_id=ex_id,
                question=question,
                response=text,
                retrieved_context=kb.joined(),
            )
        )
        print(
            f"  [gen] {ex_id}: response_len={len(text)} "
            f"kb_len={len(kb.joined())}"
        )
    elapsed = time.perf_counter() - t0
    gen_cost = estimate_cost(model, cost.total_input_tokens, cost.total_output_tokens)
    return (
        cases,
        elapsed,
        cost.total_input_tokens,
        cost.total_output_tokens,
        gen_cost,
    )


async def run_llm_judge(
    cases: list[CasePayload],
    model: str,
) -> JudgeTotals:
    totals = JudgeTotals(name="LLM judge", llm_model=model)
    resolve_rate(model)

    for case in cases:
        prompt = faithfulness_critique_prompt(
            case.question, case.retrieved_context, case.response
        )
        t0 = time.perf_counter()
        result = await judge_with_llm(prompt, model)
        elapsed = time.perf_counter() - t0

        passed = bool(result["pass"])
        totals.seconds += elapsed
        totals.total += 1
        totals.api_calls += 1
        totals.llm_calls += 1
        totals.llm_input_tokens += int(result.get("input_tokens") or 0)
        totals.llm_output_tokens += int(result.get("output_tokens") or 0)
        totals.labels.append(passed)
        if passed:
            totals.passes += 1
        # Agreement with self is 100% by definition; filled after for others
        totals.agree_with_llm += 1
        print(
            f"  [llm]   {case.example_id}: "
            f"{'pass' if passed else 'fail'} {elapsed:.2f}s "
            f"tok={result.get('input_tokens')}/{result.get('output_tokens')}"
        )

    return totals


async def run_decision_per_call(
    cases: list[CasePayload],
    llm_model: str,
    llm_labels: list[bool],
) -> JudgeTotals:
    totals = JudgeTotals(name="Decision (per call)", llm_model=llm_model)
    resolve_rate(llm_model)

    for i, case in enumerate(cases):
        t0 = time.perf_counter()
        judgment = await judge_faithfulness(
            question=case.question,
            retrieved_context=case.retrieved_context,
            response=case.response,
        )
        totals.ts_input_tokens += judgment.input_tokens
        totals.ts_output_tokens += judgment.output_tokens
        totals.api_calls += 1
        if judgment.model:
            totals.ts_model = judgment.model
            resolve_rate(judgment.model)

        if not judgment.passed:
            prompt = faithfulness_critique_prompt(
                case.question, case.retrieved_context, case.response
            )
            critique = await judge_with_llm(prompt, llm_model)
            totals.llm_calls += 1
            totals.api_calls += 1
            totals.llm_input_tokens += int(critique.get("input_tokens") or 0)
            totals.llm_output_tokens += int(critique.get("output_tokens") or 0)

        elapsed = time.perf_counter() - t0
        totals.seconds += elapsed
        totals.total += 1
        totals.labels.append(judgment.passed)
        if judgment.passed:
            totals.passes += 1
        if judgment.passed == llm_labels[i]:
            totals.agree_with_llm += 1
        print(
            f"  [decision] {case.example_id}: "
            f"p={judgment.probability:.2f} "
            f"{'pass' if judgment.passed else 'fail'} "
            f"{elapsed:.2f}s tok={judgment.input_tokens}/{judgment.output_tokens}"
        )

    return totals


async def run_decision_batch(
    cases: list[CasePayload],
    llm_model: str,
    llm_labels: list[bool],
) -> JudgeTotals:
    totals = JudgeTotals(name="Decision (batch)", llm_model=llm_model)
    resolve_rate(llm_model)

    case_map = {
        c.example_id: {
            "question": c.question,
            "retrieved_context": c.retrieved_context,
            "response": c.response,
        }
        for c in cases
    }
    print(f"  Decision batch: 1 systemone call for {len(case_map)} examples")

    t0 = time.perf_counter()
    batch = await judge_faithfulness_batch(case_map)
    ts_elapsed = time.perf_counter() - t0

    totals.seconds += ts_elapsed
    totals.api_calls += batch.api_calls
    totals.ts_input_tokens = batch.input_tokens
    totals.ts_output_tokens = batch.output_tokens
    if batch.model:
        totals.ts_model = batch.model
        resolve_rate(batch.model)

    for i, case in enumerate(cases):
        judgment = batch.results[case.example_id]
        if not judgment.passed:
            t1 = time.perf_counter()
            prompt = faithfulness_critique_prompt(
                case.question, case.retrieved_context, case.response
            )
            critique = await judge_with_llm(prompt, llm_model)
            totals.seconds += time.perf_counter() - t1
            totals.llm_calls += 1
            totals.api_calls += 1
            totals.llm_input_tokens += int(critique.get("input_tokens") or 0)
            totals.llm_output_tokens += int(critique.get("output_tokens") or 0)

        totals.total += 1
        totals.labels.append(judgment.passed)
        if judgment.passed:
            totals.passes += 1
        if judgment.passed == llm_labels[i]:
            totals.agree_with_llm += 1
        print(
            f"  [batch] {case.example_id}: "
            f"p={judgment.probability:.2f} "
            f"{'pass' if judgment.passed else 'fail'}"
        )

    print(
        f"  [batch] ts_wall={ts_elapsed:.2f}s "
        f"tok={batch.input_tokens}/{batch.output_tokens} "
        f"calls={batch.api_calls} threshold={FAITHFUL_PASS}"
    )
    return totals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare LLM vs decision-model HR faithfulness judges (speed + cost)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/live_workshop_hr.jsonl"),
        help="JSONL dataset with input.request",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Agent + LLM judge model (default: helpdesk default_model())",
    )
    return parser.parse_args()


async def main() -> int:
    load_dotenv(override=True)
    args = parse_args()
    examples = load_examples(args.dataset)
    if not examples:
        print(f"No examples in {args.dataset}")
        return 1

    model = args.model or default_model()
    resolve_rate(model)

    print(f"Dataset: {args.dataset} ({len(examples)} examples)")
    print(f"Model (agent + LLM judge): {model}")
    print(f"Decision pass threshold: {FAITHFUL_PASS}")
    print()
    print("=== Generate HR specialist answers (shared) ===")
    cases, gen_s, gen_in, gen_out, gen_cost = await generate_cases(
        examples, model
    )
    print(
        f"  Agent gen: {gen_s:.2f}s tok={gen_in}/{gen_out} "
        f"cost=${gen_cost:.6f} (excluded from judge table)"
    )
    print()
    print("=== LLM judge (always-on) ===")
    llm = await run_llm_judge(cases, model)
    print()
    print("=== Decision (per call, LLM critique on fail) ===")
    decision = await run_decision_per_call(cases, model, llm.labels)
    print()
    print("=== Decision (batch, LLM critique on fail) ===")
    decision_batch = await run_decision_batch(cases, model, llm.labels)
    print()
    print_table(llm, decision, decision_batch)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
