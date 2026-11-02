"""LLM faithfulness critique for HR specialist responses.

Shared by the always-on LLM judge experiment and the hybrid decision-model
path (which calls this only when the decision score is below threshold).
"""

from __future__ import annotations

import json


def _usage_from_openai(response) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    # Chat Completions uses prompt/completion; some SDKs expose input/output.
    input_tokens = getattr(usage, "prompt_tokens", None)
    if input_tokens is None:
        input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "completion_tokens", None)
    if output_tokens is None:
        output_tokens = getattr(usage, "output_tokens", 0) or 0
    return int(input_tokens), int(output_tokens)


def _usage_from_anthropic(response) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    return (
        int(getattr(usage, "input_tokens", 0) or 0),
        int(getattr(usage, "output_tokens", 0) or 0),
    )


async def judge_with_llm(prompt: str, model: str) -> dict:
    """Run an LLM judge and parse the JSON verdict.

    Returns:
        Dict with ``pass``, ``critique``, ``input_tokens``, ``output_tokens``.
    """
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
        input_tokens, output_tokens = _usage_from_anthropic(response)
    else:
        from openai import AsyncOpenAI

        client = AsyncOpenAI()
        # gpt-6-* chat models: no temperature=0; use max_completion_tokens.
        create_kwargs: dict = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
        }
        if model.startswith("gpt-6"):
            create_kwargs["max_completion_tokens"] = 1024
            create_kwargs["reasoning_effort"] = "low"
        else:
            create_kwargs["temperature"] = 0
            create_kwargs["max_tokens"] = 1024
        response = await client.chat.completions.create(**create_kwargs)
        content = response.choices[0].message.content or "{}"
        input_tokens, output_tokens = _usage_from_openai(response)

    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        lines = [line for line in lines[1:] if line.strip() != "```"]
        stripped = "\n".join(lines)

    try:
        result = json.loads(stripped)
        return {
            "pass": bool(result.get("pass", False)),
            "critique": result.get("critique", ""),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
    except (json.JSONDecodeError, ValueError, TypeError):
        return {
            "pass": False,
            "critique": f"Failed to parse judge response: {content}",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }


def faithfulness_critique_prompt(
    question: str,
    retrieved_context: str,
    response: str,
) -> str:
    """Build the hallucination-judge prompt used by HR evaluators."""
    kb = retrieved_context if retrieved_context else (
        "(No content was retrieved from the knowledge base)"
    )
    agent_response = response if response else "(No response provided)"

    return f"""You are a hallucination judge. You are given a question, the content \
retrieved from a knowledge base, and an agent's response. Your job is to determine \
whether the agent's response is **faithful** to the retrieved KB content.

## Input

User Question:
{question}

Retrieved KB Content:
{kb}

Agent Response:
{agent_response}

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
