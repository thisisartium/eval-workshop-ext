"""HR Specialist experiment with pass/fail hallucination judge.

This experiment evaluates the HR specialist agent DIRECTLY (skipping the
concierge) on whether its responses are faithful to the knowledge base.

Run with:
    # Untuned HR specialist
    cat-experiments run experiments/hr_specialist.py -c experiments/with_specialists.yaml \\
        --dataset live_hr

    # Tuned HR specialist
    cat-experiments run experiments/hr_specialist.py -c experiments/hr_tuned.yaml \\
        --dataset live_hr

Evaluators
==========

hr_response_quality: Pass/fail hallucination judge that checks whether every
  factual claim in the response is supported by the retrieved KB content.
  Also passes when the specialist correctly reports that the KB has no
  relevant information.

Configurable Parameters
=======================

- agent_config: Path to HR specialist agent config (relative to helpdesk package)
- model: Model override for HR specialist (optional)
- prompt_file: Prompt file override (relative to helpdesk package, optional)
- judge_model: Model to use for LLM-as-judge evaluator (default: auto-detected)
"""

from __future__ import annotations

import json

from dotenv import load_dotenv
from opentelemetry import trace

from agent_platform.runner import collect
from agent_platform.tracing import OTelTracingProcessor
from cat.experiments.protocol import EvalInput, EvalOutput, TaskInput, TaskOutput
from cat.experiments.sdk import evaluator, task
from cat.experiments.sdk.tracing import extract_retrieval_context
from helpdesk.app import create_specialist_app, default_model


async def judge_with_llm(prompt: str, model: str) -> dict:
    """Run an LLM judge and parse the JSON verdict."""
    if model.startswith("claude"):
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic()
        response = await client.messages.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1024,
        )
        content = response.content[0].text if response.content else "{}"
    else:
        from openai import AsyncOpenAI

        client = AsyncOpenAI()
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"

    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        lines = [l for l in lines[1:] if l.strip() != "```"]
        stripped = "\n".join(lines)

    try:
        result = json.loads(stripped)
        return {
            "pass": bool(result.get("pass", False)),
            "critique": result.get("critique", ""),
        }
    except (json.JSONDecodeError, ValueError, TypeError):
        return {"pass": False, "critique": f"Failed to parse judge response: {content}"}

# Load environment variables for API keys
load_dotenv()


# =============================================================================
# MODULE-LEVEL CONFIG
# =============================================================================

name = "HR Specialist"
description = "Evaluate HR specialist directly with pass/fail quality judge"
params = {
    # HR specialist agent config (relative to helpdesk package dir)
    "agent_config": "configs/agents/hr_tuned.yaml",
    # Optional overrides for HR agent — auto-detected from available API keys
    "model": default_model(),
    "prompt_file": None,  # e.g., "prompts/specialists/hr/tuned.txt"
    # Model for LLM-as-judge evaluators — auto-detected from available API keys
    "judge_model": default_model(),
}


# =============================================================================
# TASK FUNCTION
# =============================================================================


@task
async def answer_hr_question(input_data: TaskInput) -> TaskOutput:
    """Run the HR specialist agent directly on an HR question.

    Bypasses the routing agent and tests the HR specialist in isolation.
    OTel spans are captured automatically for the evaluator.
    """
    p = input_data.params

    # Create tracing processor from the global TracerProvider
    # (set up by cat-experiments to capture spans for evaluation)
    tracer = trace.get_tracer("agent_platform")
    processor = OTelTracingProcessor(tracer, capture_content=True)

    client = create_specialist_app(
        agent_config=p.get("agent_config", "configs/agents/hr_tuned.yaml"),
        model=p.get("model"),
        prompt_file=p.get("prompt_file"),
        processor=processor,
    )

    request = input_data.input.get("request", "")

    text, events = await collect(client.send(request))

    return TaskOutput(
        output={
            "response": text,
        },
        metadata={
            "agent_config": p.get("agent_config"),
            "model": p.get("model"),
            "prompt_file": p.get("prompt_file"),
        },
    )


# =============================================================================
# EVALUATOR FUNCTIONS
# =============================================================================


@evaluator
async def hr_response_quality(eval_input: EvalInput) -> EvalOutput:
    """Pass/fail hallucination judge — checks if every claim is grounded in KB content."""
    # Extract question from input
    question = ""
    example_input = eval_input.example.get("input", {})
    if isinstance(example_input, dict):
        question = example_input.get("request", "")

    # Extract response from actual output
    response = ""
    if isinstance(eval_input.actual_output, dict):
        response = eval_input.actual_output.get("response", "")

    # Extract retrieved KB context from task spans
    contexts = extract_retrieval_context(eval_input.task_spans or [])
    retrieved_context = "\n\n".join(
        ctx["content"] for ctx in contexts if ctx.get("content")
    )

    judge_model = eval_input.params.get("judge_model") or default_model()

    prompt = f"""You are a hallucination judge. You are given a question, the content \
retrieved from a knowledge base, and an agent's response. Your job is to determine \
whether the agent's response is **faithful** to the retrieved KB content.

## Input

User Question:
{question}

Retrieved KB Content:
{retrieved_context if retrieved_context else "(No content was retrieved from the knowledge base)"}

Agent Response:
{response if response else "(No response provided)"}

## Evaluation Criteria

PASS if:
- Every factual claim in the response is supported by the retrieved KB content
- OR the agent stated that it could not find relevant information in the KB \
(this is a correct response when the KB has no relevant content)

FAIL if:
- The response contains specific facts, numbers, dates, or policy details that \
are NOT present in the retrieved KB content (hallucination)

## Instructions

Compare each factual claim in the agent's response against the retrieved KB content. \
If you find any claim that is not supported by the KB, the response fails.

Note: Generic phrasing like "feel free to ask" or "you can contact HR" is not a \
factual claim and should not be judged.

Respond with JSON:
{{
  "critique": "Your detailed reasoning, listing any unsupported claims if found...",
  "pass": true or false
}}"""

    judgment = await judge_with_llm(prompt, judge_model)

    passed = judgment["pass"]
    critique = judgment["critique"]

    return EvalOutput(
        score=1.0 if passed else 0.0,
        label="pass" if passed else "fail",
        explanation=critique,
        metadata={
            "question": question,
            "response": response[:500] if response else "",
            "retrieved_context": retrieved_context[:1000] if retrieved_context else "",
            "judge_model": judge_model,
        },
    )
