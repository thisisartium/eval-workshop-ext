"""Jev-based HR response faithfulness judge.

Uses TypeSafe Jev for groundedness scoring:
- Noul for whether the agent response is faithful to retrieved KB content

Code owns the pass threshold. Explanations on fail come from a separate
LLM critique call in the experiment evaluator (not from Jev).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from dotenv import load_dotenv
from typesafe_sdk import AsyncTypeSafeClient, Noul

# Raised above Typesafe's 0.7 starting default so borderline scores
# (e.g. 0.75) fail and trigger an LLM critique.
FAITHFUL_PASS = 0.8

IS_FAITHFUL_INSTRUCTIONS = (
    "Is `response` faithful to `retrieved_context` for answering `question`?"
)

IS_FAITHFUL_CRITERIA = {
    "true": (
        "Every factual claim in `response` is supported by `retrieved_context`, "
        "or `response` correctly says the knowledge base has no relevant "
        "information. Generic courtesy phrasing (feel free to ask, contact HR) "
        "is not a factual claim."
    ),
    "false": (
        "`response` invents specific facts, numbers, dates, or policy details "
        "that are not present in `retrieved_context`."
    ),
}


@dataclass
class JevFaithfulnessResult:
    """Result of a Jev faithfulness judgment."""

    probability: float
    passed: bool = False
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0

    def metadata(self) -> dict:
        """Jev fields suitable for experiment / Cat Cafe metadata."""
        return {
            "probability": self.probability,
            "threshold": FAITHFUL_PASS,
            "passed": self.passed,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


@dataclass
class JevFaithfulnessBatchResult:
    """One TypeSafe call covering many faithfulness judgments."""

    results: dict[str, JevFaithfulnessResult]
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    api_calls: int = 1


def _normalize_case(
    question: str,
    retrieved_context: str,
    response: str,
) -> dict[str, str]:
    return {
        "question": question,
        "retrieved_context": retrieved_context.strip()
        or "(No content was retrieved from the knowledge base)",
        "response": response.strip() or "(No response provided)",
    }


def _case_noul(example_id: str) -> Noul:
    path = f"`cases.{example_id}`"
    return Noul(
        instructions=(
            f"Is {path}.response faithful to {path}.retrieved_context "
            f"for answering {path}.question?"
        ),
        criteria=IS_FAITHFUL_CRITERIA,
    )


async def judge_faithfulness(
    *,
    question: str,
    retrieved_context: str,
    response: str,
    client: AsyncTypeSafeClient | None = None,
) -> JevFaithfulnessResult:
    """Score HR response faithfulness with Jev.

    Args:
        question: User question.
        retrieved_context: KB snippets retrieved during the task.
        response: Agent response text.
        client: Optional shared TypeSafe client.

    Returns:
        JevFaithfulnessResult with probability, pass/fail, and usage.
    """
    load_dotenv(override=True)

    state = _normalize_case(question, retrieved_context, response)
    questions = {
        "is_faithful": Noul(
            instructions=IS_FAITHFUL_INSTRUCTIONS,
            criteria=IS_FAITHFUL_CRITERIA,
        ),
    }

    if client is not None:
        answer = await client.system_one(state=state, questions=questions)
    else:
        async with AsyncTypeSafeClient() as owned:
            answer = await owned.system_one(state=state, questions=questions)

    usage = answer.usage
    probability = float(answer.nouls["is_faithful"].noul)
    return JevFaithfulnessResult(
        probability=probability,
        passed=probability >= FAITHFUL_PASS,
        model=answer.model,
        input_tokens=int(usage.input_tokens or 0),
        output_tokens=int(usage.output_tokens or 0),
    )


async def judge_faithfulness_batch(
    cases: dict[str, dict[str, str]],
    *,
    client: AsyncTypeSafeClient | None = None,
) -> JevFaithfulnessBatchResult:
    """Score many faithfulness cases in one TypeSafe ``system_one`` call.

    Args:
        cases: Mapping of example id -> keys ``question``, ``retrieved_context``,
            ``response``.
        client: Optional shared TypeSafe client.

    Returns:
        JevFaithfulnessBatchResult with per-id results and aggregate usage.
    """
    if not cases:
        return JevFaithfulnessBatchResult(results={})

    load_dotenv(override=True)

    normalized = {
        example_id: _normalize_case(
            case.get("question", ""),
            case.get("retrieved_context", ""),
            case.get("response", ""),
        )
        for example_id, case in cases.items()
    }
    questions = {
        f"is_faithful_{example_id}": _case_noul(example_id)
        for example_id in normalized
    }
    state = {"cases": normalized}

    if client is not None:
        answer = await client.system_one(state=state, questions=questions)
    else:
        async with AsyncTypeSafeClient() as owned:
            answer = await owned.system_one(state=state, questions=questions)

    usage = answer.usage
    input_tokens = int(usage.input_tokens or 0)
    output_tokens = int(usage.output_tokens or 0)
    model = answer.model

    results: dict[str, JevFaithfulnessResult] = {}
    for example_id in normalized:
        probability = float(answer.nouls[f"is_faithful_{example_id}"].noul)
        results[example_id] = JevFaithfulnessResult(
            probability=probability,
            passed=probability >= FAITHFUL_PASS,
            model=model,
            input_tokens=0,
            output_tokens=0,
        )

    return JevFaithfulnessBatchResult(
        results=results,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        api_calls=1,
    )


def faithfulness_result_as_dict(result: JevFaithfulnessResult) -> dict:
    """Serialize a faithfulness result (tests / debugging)."""
    return asdict(result)
