"""Jev-based helpdesk routing experiment.

Escalate-only department classification via TypeSafe Jev, comparable to
``experiments/routing.py`` with ``experiments/baseline.yaml``.

Run with:
    cat-experiments run experiments/jev_routing.py \\
        -c experiments/jev_baseline.yaml --dataset live_routing

Requires ``TYPESAFE_API_KEY`` in the environment.
"""

from __future__ import annotations

from dotenv import load_dotenv
from opentelemetry import trace

from agent_platform.tracing import OTelTracingProcessor
from cat.experiments.protocol import EvalInput, EvalOutput, TaskInput, TaskOutput
from cat.experiments.sdk import evaluator, task
from cat.experiments.sdk.tracing import extract_tool_calls
from helpdesk.jev_router import route_with_jev

load_dotenv()

name = "Helpdesk Routing (Jev)"
description = "Route helpdesk requests with TypeSafe Jev (Choice + Noul)"
params: dict = {}


@task
async def route_request(input_data: TaskInput) -> TaskOutput:
    """Classify a request with Jev and escalate to the chosen department."""
    tracer = trace.get_tracer("agent_platform")
    processor = OTelTracingProcessor(tracer, capture_content=True)

    request = input_data.input.get("request", "")
    result = await route_with_jev(request, processor=processor)

    return TaskOutput(
        output={
            "response": result.response,
            "department": result.department,
        },
        metadata=result.metadata(),
    )


@evaluator
async def department_match(eval_input: EvalInput) -> EvalOutput:
    """Evaluate if the Jev router routed to the correct department.

    Prefers tool-call extraction when present; falls back to department
    on the task output (Jev path always sets this).
    """
    expected_department = None
    if eval_input.expected_output:
        expected_department = eval_input.expected_output.get("department")

    tool_calls = extract_tool_calls(eval_input.task_spans or [])

    actual_department = None
    for tc in tool_calls:
        if tc.get("name") == "escalate_to_department":
            actual_department = tc.get("args", {}).get("department")
            break

    if not actual_department and isinstance(eval_input.actual_output, dict):
        actual_department = eval_input.actual_output.get("department")
    if not actual_department and isinstance(eval_input.task_metadata, dict):
        actual_department = eval_input.task_metadata.get("department")

    if expected_department and actual_department:
        match = expected_department == actual_department
        score = 1.0 if match else 0.0
    else:
        score = 0.0

    if score == 1.0:
        explanation = f"Correctly routed to {actual_department}."
    elif not expected_department:
        explanation = "No expected department in dataset."
    elif not actual_department:
        explanation = "Router did not route to any department."
    else:
        explanation = f"Expected {expected_department}, but got {actual_department}."

    meta = {
        "expected_department": expected_department,
        "actual_department": actual_department,
        "tool_calls": tool_calls,
    }
    if isinstance(eval_input.task_metadata, dict):
        for key in (
            "abstained",
            "is_helpdesk_noul",
            "choice",
            "choice_confidence",
            "choice_probabilities",
        ):
            if key in eval_input.task_metadata:
                meta[key] = eval_input.task_metadata[key]

    return EvalOutput(
        score=score,
        label="pass" if score == 1.0 else "fail",
        explanation=explanation,
        metadata=meta,
    )
