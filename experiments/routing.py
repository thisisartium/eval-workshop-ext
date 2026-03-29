"""Helpdesk routing experiment.

This experiment evaluates the helpdesk routing agent's ability to classify
requests into the correct department and optionally handle them via specialists.

Run with:
    # Baseline - simple routing prompt, no specialists
    cat-experiments run experiments/routing.py -c experiments/baseline.yaml \\
        --dataset live_routing

    # Tuned - improved routing prompt, no specialists
    cat-experiments run experiments/routing.py -c experiments/tuned.yaml \\
        --dataset live_routing

Configuration Files
===================

The experiment uses a layered config system:

1. configs/*.yaml - System configs (baseline, with_specialists, tuned)
2. configs/agents/*.yaml - Agent definitions (routing, hr per tier)
3. experiments/*.yaml - Experiment configs that reference system configs

Experiment Structure
====================

1. MODULE CONFIG: Default parameters (baseline = pure escalation)
2. TASK FUNCTION: Runs the agent via AgentClient (spans auto-captured)
3. EVALUATOR FUNCTION: Extracts tool calls from task_spans to compare routing

Configurable Parameters
=======================

- config: System config filename (e.g. "baseline.yaml", "tuned.yaml")
- prompt_file: Override routing prompt (relative to project root)
- model: Override model for all agents
"""

from __future__ import annotations

from dotenv import load_dotenv

from opentelemetry import trace

from agent_platform.runner import collect
from agent_platform.tracing import OTelTracingProcessor
from cat.experiments.protocol import EvalInput, EvalOutput, TaskInput, TaskOutput
from cat.experiments.sdk import evaluator, task
from cat.experiments.sdk.tracing import extract_tool_calls
from helpdesk.app import create_app, default_model

# Load environment variables for API keys
load_dotenv()


# =============================================================================
# MODULE-LEVEL CONFIG
# =============================================================================
# Default parameters for the experiment. Override via:
#   - YAML config file (-c experiments/baseline.yaml)
#   - CLI flags (--param model=claude-sonnet-4-6)

name = "Helpdesk Routing"
description = "Route helpdesk requests to appropriate departments"
params = {
    # System config file (relative to configs/)
    # None = use default system.yaml (tuned concierge + specialists)
    # "baseline.yaml" = pure escalation, no specialists
    # "with_specialists.yaml" = specialists with untuned prompts
    # "tuned.yaml" = specialists with fully tuned prompts
    "config": None,
    # Prompt override (relative to project root)
    "prompt_file": None,  # e.g., "prompts/routing/tuned.txt"
    # Model override — auto-detected from available API keys
    "model": default_model(),
}


# =============================================================================
# TASK FUNCTION
# =============================================================================


@task
async def route_request(input_data: TaskInput) -> TaskOutput:
    """Run the routing agent on a single helpdesk request.

    Creates a helpdesk app, sends the request, and returns the response.
    OTel spans are captured automatically for the evaluator to extract
    tool calls from.
    """
    p = input_data.params

    # Create tracing processor from the global TracerProvider
    # (set up by cat-experiments to capture spans for evaluation)
    tracer = trace.get_tracer("agent_platform")
    processor = OTelTracingProcessor(tracer, capture_content=True)

    # Create app with experiment config
    client = create_app(
        config=p.get("config"),
        model=p.get("model"),
        prompt_file=p.get("prompt_file"),
        processor=processor,
    )

    # Extract request from dataset example
    request = input_data.input.get("request", "")

    # Run agent and collect events
    text, _ = await collect(client.send(request))

    return TaskOutput(
        output={
            "response": text,
        },
        metadata={
            "config": p.get("config"),
            "model": p.get("model"),
            "prompt_file": p.get("prompt_file"),
        },
    )


# =============================================================================
# EVALUATOR FUNCTIONS
# =============================================================================


@evaluator
async def department_match(eval_input: EvalInput) -> EvalOutput:
    """Evaluate if the agent routed to the correct department.

    Extracts tool calls from task spans and compares the routed department
    against the expected department from the dataset. Looks for both
    ``escalate_to_department`` and ``call_specialist`` tool calls.
    """
    # Get expected department from dataset
    expected_department = None
    if eval_input.expected_output:
        expected_department = eval_input.expected_output.get("department")

    # Extract tool calls from task spans (captured automatically by executor)
    tool_calls = extract_tool_calls(eval_input.task_spans or [])

    # Get actual department from tool calls
    actual_department = None
    for tc in tool_calls:
        if tc.get("name") == "escalate_to_department":
            actual_department = tc.get("args", {}).get("department")
            break
        if tc.get("name") == "call_specialist":
            actual_department = tc.get("args", {}).get("specialist")
            break

    # Score: 1.0 if match, 0.0 otherwise
    if expected_department and actual_department:
        match = expected_department == actual_department
        score = 1.0 if match else 0.0
    else:
        score = 0.0

    # Build explanation
    if score == 1.0:
        explanation = f"Correctly routed to {actual_department}."
    elif not expected_department:
        explanation = "No expected department in dataset."
    elif not actual_department:
        explanation = "Agent did not route to any department."
    else:
        explanation = f"Expected {expected_department}, but got {actual_department}."

    return EvalOutput(
        score=score,
        label="pass" if score == 1.0 else "fail",
        explanation=explanation,
        metadata={
            "expected_department": expected_department,
            "actual_department": actual_department,
            "tool_calls": tool_calls,
        },
    )
