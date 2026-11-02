"""Decision-model helpdesk router (escalate-only).

Uses a System One / decision model via LiteLLM ``asystemone`` for department
classification (default: TypeSafe Jev):
- Choice over IT / HR / Facilities / Finance / Legal / Security
- Companion Noul for whether the request belongs on the helpdesk at all

Code owns the decision thresholds and calls ``escalate_to_department``.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

import os

import litellm
from dotenv import load_dotenv

from agent_platform.observability import Span, ToolSpanData, TracingProcessor
from helpdesk.tools import escalate_to_department

# Default LiteLLM decision model (override with DECISION_MODEL).
_DEFAULT_DECISION_MODEL = "openai/gpt-6-luna"


def decision_model() -> str:
    """LiteLLM model id for System One calls."""
    return os.environ.get("DECISION_MODEL", _DEFAULT_DECISION_MODEL)


# Thresholds for the is_helpdesk_request Noul (starting defaults).
NOUL_YES = 0.7
NOUL_NO = 0.3

DEPARTMENT_CRITERIA: dict[str, str] = {
    "IT": (
        "Software, hardware, accounts, passwords, laptops, email, VPN, "
        "or other technology support for an employee's own devices/systems. "
        "Phishing or suspicious email that needs IT investigation belongs here "
        "unless it is clearly a physical security / badge access issue."
    ),
    "HR": (
        "Employee policies: time off, benefits, 401k, onboarding, payroll "
        "questions for the employee, expense reimbursement for the employee "
        "(travel, home office, conferences), workplace conduct, or people ops."
    ),
    "Facilities": (
        "Building and workplace environment: HVAC, projectors and conference "
        "room equipment, furniture, cleaning, office temperature, or space issues. "
        "Shared room AV gear is Facilities, not IT."
    ),
    "Finance": (
        "Company money moving outside employee reimbursement: vendor invoices, "
        "AP/AR, budgets, corporate purchases, or paying external suppliers. "
        "Employee expense reports go to HR, not Finance."
    ),
    "Legal": (
        "Contracts, compliance, intellectual property, legal review, or "
        "regulatory matters."
    ),
    "Security": (
        "Physical security and access control: badges, doors, building entry, "
        "theft, or on-site security incidents. Not general IT password resets."
    ),
}

IS_HELPDESK_INSTRUCTIONS = (
    "Does `request` ask the company helpdesk to take action or answer a "
    "work-related support question for an employee?"
)

IS_HELPDESK_CRITERIA = {
    "true": (
        "The message is a real helpdesk request: broken equipment, policy "
        "question, access issue, invoice, onboarding, benefits, etc."
    ),
    "false": (
        "The message is not a helpdesk ticket: thanks/appreciation only, "
        "small talk, or clearly unrelated to employee support."
    ),
}

DEPARTMENT_CHOICE_INSTRUCTIONS = (
    "Which department should handle `request`? Pick the single best match "
    "among the departments."
)


@dataclass
class DecisionRouteResult:
    """Result of a decision-model routing call plus escalation."""

    department: str
    response: str
    is_helpdesk_noul: float
    choice: str | None = None
    choice_confidence: float | None = None
    choice_probabilities: dict[str, float] = field(default_factory=dict)
    abstained: bool = False
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0

    def metadata(self) -> dict:
        """Fields suitable for experiment / Cat Cafe metadata."""
        return {
            "department": self.department,
            "abstained": self.abstained,
            "is_helpdesk_noul": self.is_helpdesk_noul,
            "choice": self.choice,
            "choice_confidence": self.choice_confidence,
            "choice_probabilities": self.choice_probabilities,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


def _decide_department(noul: float, choice: str | None) -> tuple[str, bool]:
    """Apply Noul thresholds; return (department, abstained)."""
    if noul <= NOUL_NO:
        return "None", True
    if noul < NOUL_YES:
        return "None", True
    return (choice or "None"), False


def _emit_tool_span(
    processor: TracingProcessor | None,
    *,
    department: str,
    request: str,
    result: str,
) -> None:
    if processor is None:
        return
    now = time.time()
    processor.on_span_end(
        Span(
            span_id=str(uuid.uuid4()),
            name="escalate_to_department",
            started_at=now,
            ended_at=now,
            data=ToolSpanData(
                name="escalate_to_department",
                arguments=json.dumps(
                    {"department": department, "request": request}
                ),
                result=result,
            ),
        )
    )


def _noul_question(instructions: str) -> dict[str, Any]:
    return {
        "type": "noul",
        "instructions": instructions,
        "criteria": IS_HELPDESK_CRITERIA,
    }


def _choice_question(instructions: str) -> dict[str, Any]:
    return {
        "type": "choice",
        "instructions": instructions,
        "criteria": DEPARTMENT_CRITERIA,
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


def _get_choice(answer: Any) -> tuple[str | None, float | None, dict[str, float]]:
    if isinstance(answer, dict):
        probs = answer.get("probabilities") or {}
        return (
            answer.get("choice"),
            answer.get("confidence"),
            {str(k): float(v) for k, v in probs.items()},
        )
    probs = getattr(answer, "probabilities", None) or {}
    return (
        getattr(answer, "choice", None),
        getattr(answer, "confidence", None),
        {str(k): float(v) for k, v in dict(probs).items()},
    )


async def route_with_decision(
    request: str,
    *,
    processor: TracingProcessor | None = None,
    escalate: bool = True,
) -> DecisionRouteResult:
    """Classify ``request`` with a decision model and escalate.

    Args:
        request: Employee helpdesk message.
        processor: Optional tracing processor (emits a tool span).
        escalate: If False, skip ``escalate_to_department`` (labels only).

    Returns:
        DecisionRouteResult with department, user-facing response, and probs.
    """
    load_dotenv(override=True)

    questions = {
        "is_helpdesk_request": _noul_question(IS_HELPDESK_INSTRUCTIONS),
        "department": _choice_question(DEPARTMENT_CHOICE_INSTRUCTIONS),
    }

    response = await litellm.asystemone(
        model=decision_model(),
        state={"request": request},
        questions=questions,
    )

    answers = _answer_map(response)
    noul_val = _get_noul(answers["is_helpdesk_request"])
    choice, confidence, probabilities = _get_choice(answers["department"])
    department, abstained = _decide_department(noul_val, choice)

    if escalate:
        escalate_text = await escalate_to_department(
            department=department, request=request
        )
        _emit_tool_span(
            processor,
            department=department,
            request=request,
            result=escalate_text,
        )
    else:
        escalate_text = ""

    input_tokens, output_tokens = _usage_tokens(response)
    model = getattr(response, "model", None)
    if model is None and isinstance(response, dict):
        model = response.get("model")

    return DecisionRouteResult(
        department=department,
        response=escalate_text,
        is_helpdesk_noul=noul_val,
        choice=choice,
        choice_confidence=confidence,
        choice_probabilities=probabilities,
        abstained=abstained,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def route_result_as_dict(result: DecisionRouteResult) -> dict:
    """Serialize a route result (tests / debugging)."""
    return asdict(result)
