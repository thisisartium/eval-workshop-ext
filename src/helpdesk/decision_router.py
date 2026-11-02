"""Jev-based helpdesk router (escalate-only).

Uses TypeSafe Jev for department classification:
- Choice over IT / HR / Facilities / Finance / Legal / Security
- Companion Noul for whether the request belongs on the helpdesk at all

Code owns the decision thresholds and calls ``escalate_to_department``.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field

from dotenv import load_dotenv
from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul

from agent_platform.observability import Span, ToolSpanData, TracingProcessor
from helpdesk.tools import escalate_to_department

# Thresholds for the is_helpdesk_request Noul (Typesafe starting defaults).
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
class JevRouteResult:
    """Result of a Jev routing decision plus escalation."""

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
        """Jev fields suitable for experiment / Cat Cafe metadata."""
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


@dataclass
class JevBatchResult:
    """One TypeSafe call covering many tickets."""

    results: dict[str, JevRouteResult]
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    api_calls: int = 1


def _decide_department(noul: float, choice: str) -> tuple[str, bool]:
    """Apply Noul thresholds; return (department, abstained)."""
    if noul < NOUL_NO:
        return "None", True
    if noul > NOUL_YES:
        return choice, False
    # Mid band: abstain rather than force a low-confidence route.
    return "None", True


def _emit_tool_span(
    processor: TracingProcessor | None,
    *,
    department: str,
    request: str,
    result: str,
) -> None:
    """Emit an escalate_to_department tool span for Cat Cafe / traces."""
    if processor is None:
        return
    trace_id = uuid.uuid4().hex[:16]
    span = Span(
        span_id=uuid.uuid4().hex[:8],
        trace_id=trace_id,
        parent_id=None,
        span_data=ToolSpanData(
            name="escalate_to_department",
            input=json.dumps({"department": department, "request": request}),
        ),
    )
    processor.on_span_start(span)
    assert isinstance(span.span_data, ToolSpanData)
    span.span_data.output = result
    span.ended_at = time.monotonic()
    processor.on_span_end(span)


def _ticket_path(example_id: str) -> str:
    """State path for a ticket in a batch request (backtick form for instructions)."""
    return f"tickets.{example_id}"


def _batch_questions_for_id(example_id: str) -> dict:
    """Noul + Choice for one ticket id, referencing ``tickets.<id>`` in state."""
    path = f"`{_ticket_path(example_id)}`"
    return {
        f"is_helpdesk_{example_id}": Noul(
            instructions=(
                f"Does {path} ask the company helpdesk to take action or "
                "answer a work-related support question for an employee?"
            ),
            criteria=IS_HELPDESK_CRITERIA,
        ),
        f"department_{example_id}": Choice(
            instructions=(
                f"Which department should handle {path}? Pick the single "
                "best match among the departments."
            ),
            criteria=DEPARTMENT_CRITERIA,
        ),
    }


async def route_with_jev(
    request: str,
    *,
    processor: TracingProcessor | None = None,
    client: AsyncTypeSafeClient | None = None,
    escalate: bool = True,
) -> JevRouteResult:
    """Classify ``request`` with Jev and escalate to the chosen department.

    Args:
        request: Employee helpdesk message.
        processor: Optional tracing processor (emits a tool span).
        client: Optional shared TypeSafe client (reused across bench calls).
        escalate: If False, skip ``escalate_to_department`` (bench / labels only).

    Returns:
        JevRouteResult with department, user-facing response, and Jev probs.
    """
    load_dotenv(override=True)

    questions = {
        "is_helpdesk_request": Noul(
            instructions=IS_HELPDESK_INSTRUCTIONS,
            criteria=IS_HELPDESK_CRITERIA,
        ),
        "department": Choice(
            instructions=DEPARTMENT_CHOICE_INSTRUCTIONS,
            criteria=DEPARTMENT_CRITERIA,
        ),
    }

    if client is not None:
        response = await client.system_one(
            state={"request": request},
            questions=questions,
        )
    else:
        async with AsyncTypeSafeClient() as owned:
            response = await owned.system_one(
                state={"request": request},
                questions=questions,
            )

    noul_answer = response.nouls["is_helpdesk_request"]
    choice_answer = response.choices["department"]
    department, abstained = _decide_department(
        noul_answer.noul, choice_answer.choice
    )

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

    usage = response.usage
    return JevRouteResult(
        department=department,
        response=escalate_text,
        is_helpdesk_noul=noul_answer.noul,
        choice=choice_answer.choice,
        choice_confidence=choice_answer.confidence,
        choice_probabilities=dict(choice_answer.probabilities),
        abstained=abstained,
        model=response.model,
        input_tokens=int(usage.input_tokens or 0),
        output_tokens=int(usage.output_tokens or 0),
    )


async def route_batch_with_jev(
    tickets: dict[str, str],
    *,
    client: AsyncTypeSafeClient | None = None,
) -> JevBatchResult:
    """Classify many tickets in a single TypeSafe ``system_one`` call.

    Args:
        tickets: Mapping of example id -> request text.
        client: Optional shared TypeSafe client.

    Returns:
        JevBatchResult with per-id route results and aggregate usage.
    """
    if not tickets:
        return JevBatchResult(results={})

    load_dotenv(override=True)

    questions: dict = {}
    for example_id in tickets:
        questions.update(_batch_questions_for_id(example_id))

    state = {"tickets": dict(tickets)}

    if client is not None:
        response = await client.system_one(state=state, questions=questions)
    else:
        async with AsyncTypeSafeClient() as owned:
            response = await owned.system_one(state=state, questions=questions)

    usage = response.usage
    input_tokens = int(usage.input_tokens or 0)
    output_tokens = int(usage.output_tokens or 0)
    model = response.model

    results: dict[str, JevRouteResult] = {}
    for example_id in tickets:
        noul_answer = response.nouls[f"is_helpdesk_{example_id}"]
        choice_answer = response.choices[f"department_{example_id}"]
        department, abstained = _decide_department(
            noul_answer.noul, choice_answer.choice
        )
        results[example_id] = JevRouteResult(
            department=department,
            response="",
            is_helpdesk_noul=noul_answer.noul,
            choice=choice_answer.choice,
            choice_confidence=choice_answer.confidence,
            choice_probabilities=dict(choice_answer.probabilities),
            abstained=abstained,
            model=model,
            input_tokens=0,
            output_tokens=0,
        )

    return JevBatchResult(
        results=results,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        api_calls=1,
    )


def route_result_as_dict(result: JevRouteResult) -> dict:
    """Serialize a route result (tests / debugging)."""
    return asdict(result)
