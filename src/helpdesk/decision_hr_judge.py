"""Decision-model HR response faithfulness judge.

Uses a System One / decision model via LiteLLM ``asystemone`` for groundedness
scoring (default: TypeSafe Jev):
- Noul for whether the agent response is faithful to retrieved KB content

Code owns the pass threshold. Explanations on fail come from a separate
LLM critique call in the experiment evaluator (not from the decision model).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import litellm
from dotenv import load_dotenv

from helpdesk.decision_router import decision_model

# Raised above the common 0.7 starting default so borderline scores
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
class DecisionFaithfulnessResult:
    """Result of a decision-model faithfulness judgment."""

    probability: float
    passed: bool = False
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0

    def metadata(self) -> dict:
        """Fields suitable for experiment / Cat Cafe metadata."""
        return {
            "probability": self.probability,
            "threshold": FAITHFUL_PASS,
            "passed": self.passed,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


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


def _noul_question(instructions: str) -> dict[str, Any]:
    return {
        "type": "noul",
        "instructions": instructions,
        "criteria": IS_FAITHFUL_CRITERIA,
    }


def _usage_tokens(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    if isinstance(usage, dict):
        return int(usage.get("input_tokens") or 0), int(
            usage.get("output_tokens") or 0
        )
    return (
        int(getattr(usage, "input_tokens", 0) or 0),
        int(getattr(usage, "output_tokens", 0) or 0),
    )


def _answer_map(response: Any) -> dict[str, Any]:
    answers = getattr(response, "answers", None)
    if answers is None and isinstance(response, dict):
        answers = response.get("answers")
    if not isinstance(answers, dict):
        raise TypeError("LiteLLM systemone response missing answers map")
    return answers


def _get_noul(answer: Any) -> float:
    if isinstance(answer, dict):
        return float(answer["noul"])
    return float(answer.noul)


async def judge_faithfulness(
    *,
    question: str,
    retrieved_context: str,
    response: str,
) -> DecisionFaithfulnessResult:
    """Score HR response faithfulness with a decision model.

    Args:
        question: User question.
        retrieved_context: KB snippets retrieved during the task.
        response: Agent response text.

    Returns:
        DecisionFaithfulnessResult with probability, pass/fail, and usage.
    """
    load_dotenv(override=True)

    state = _normalize_case(question, retrieved_context, response)
    questions = {
        "is_faithful": _noul_question(IS_FAITHFUL_INSTRUCTIONS),
    }

    answer = await litellm.asystemone(
        model=decision_model(),
        state=state,
        questions=questions,
    )

    input_tokens, output_tokens = _usage_tokens(answer)
    probability = _get_noul(_answer_map(answer)["is_faithful"])
    model = getattr(answer, "model", None)
    if model is None and isinstance(answer, dict):
        model = answer.get("model")

    return DecisionFaithfulnessResult(
        probability=probability,
        passed=probability >= FAITHFUL_PASS,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def faithfulness_result_as_dict(result: DecisionFaithfulnessResult) -> dict:
    """Serialize a faithfulness result (tests / debugging)."""
    return asdict(result)
