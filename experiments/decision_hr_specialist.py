"""HR Specialist experiment with TypeSafe Jev faithfulness judge.

Evaluates the HR specialist agent DIRECTLY (skipping the concierge) using
a Jev Noul probability for groundedness. LLM critique runs only when the
score is below the pass threshold (saves calls on clear passes).

Run with:
    # Untuned HR specialist
    cat-experiments run experiments/jev_hr_specialist.py \
        -c experiments/jev_hr_baseline.yaml --dataset live_hr

    # Tuned HR specialist
    cat-experiments run experiments/jev_hr_specialist.py \
        -c experiments/jev_hr_tuned.yaml --dataset live_hr

Requires ``TYPESAFE_API_KEY`` in the environment.

Evaluators
==========

hr_response_quality: Continuous faithfulness probability from Jev.
  Label is pass when score >= 0.8. Explanation (LLM critique) is set
  only on fail.
"""

from __future__ import annotations

from dotenv import load_dotenv
from opentelemetry import trace

from agent_platform.runner import collect
from agent_platform.tracing import OTelTracingProcessor
from cat.experiments.protocol import EvalInput, EvalOutput, TaskInput, TaskOutput
from cat.experiments.sdk import evaluator, task
from cat.experiments.sdk.tracing import extract_retrieval_context
from helpdesk.app import create_specialist_app, default_model
from helpdesk.jev_hr_judge import FAITHFUL_PASS, judge_faithfulness
from helpdesk.llm_faithfulness_critique import (
    faithfulness_critique_prompt,
    judge_with_llm,
)

load_dotenv()

name = "HR Specialist (Jev Judge)"
description = (
    "Evaluate HR specialist with Jev faithfulness probability; "
    "LLM critique only below threshold"
)
params = {
    "agent_config": "configs/agents/hr_tuned.yaml",
    "model": default_model(),
    "prompt_file": None,
    "judge_model": default_model(),
}


@task
async def answer_hr_question(input_data: TaskInput) -> TaskOutput:
    """Run the HR specialist agent directly on an HR question.

    Bypasses the routing agent and tests the HR specialist in isolation.
    OTel spans are captured automatically for the evaluator.
    """
    p = input_data.params

    tracer = trace.get_tracer("agent_platform")
    processor = OTelTracingProcessor(tracer, capture_content=True)

    client = create_specialist_app(
        agent_config=p.get("agent_config", "configs/agents/hr_tuned.yaml"),
        model=p.get("model"),
        prompt_file=p.get("prompt_file"),
        processor=processor,
    )

    request = input_data.input.get("request", "")

    text, _ = await collect(client.send(request))

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


@evaluator
async def hr_response_quality(eval_input: EvalInput) -> EvalOutput:
    """Faithfulness probability via Jev; LLM critique only below threshold."""
    question = ""
    example_input = eval_input.example.get("input", {})
    if isinstance(example_input, dict):
        question = example_input.get("request", "")

    response = ""
    if isinstance(eval_input.actual_output, dict):
        response = eval_input.actual_output.get("response", "")

    contexts = extract_retrieval_context(eval_input.task_spans or [])
    retrieved_context = "\n\n".join(
        ctx["content"] for ctx in contexts if ctx.get("content")
    )

    judgment = await judge_faithfulness(
        question=question,
        retrieved_context=retrieved_context,
        response=response,
    )

    explanation = ""
    critique_model = None
    if not judgment.passed:
        critique_model = eval_input.params.get("judge_model") or default_model()
        critique = await judge_with_llm(
            faithfulness_critique_prompt(question, retrieved_context, response),
            critique_model,
        )
        explanation = critique["critique"]

    metadata = {
        "question": question,
        "response": response[:500] if response else "",
        "retrieved_context": retrieved_context[:1000] if retrieved_context else "",
        "threshold": FAITHFUL_PASS,
        **judgment.metadata(),
    }
    if critique_model is not None:
        metadata["critique_model"] = critique_model

    return EvalOutput(
        score=judgment.probability,
        label="pass" if judgment.passed else "fail",
        explanation=explanation,
        metadata=metadata,
    )
