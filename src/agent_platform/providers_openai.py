"""OpenAI provider implementation.

Wraps the OpenAI Responses API behind the LLMProvider protocol.
Conversation history is managed client-side via explicit message lists,
keeping behavior consistent across providers.
"""

import json
from typing import Any

from openai import AsyncOpenAI

from .providers import LLMResponse, ToolCallRequest


class OpenAIProvider:
    """LLM provider backed by the OpenAI Responses API.

    Conversation history is passed explicitly on each call (no server-side
    state). The runner maintains the message list; this provider just
    forwards it.

    Attributes:
        client: The AsyncOpenAI client instance.
    """

    def __init__(self, client: AsyncOpenAI) -> None:
        self.client = client

    async def call(
        self,
        model: str,
        instructions: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        """Send a request via the OpenAI Responses API.

        Passes the full conversation history as input each time.

        Args:
            model: OpenAI model ID.
            instructions: System prompt.
            messages: Full conversation history as input items.
            tools: Tool schemas in OpenAI function calling format.

        Returns:
            A normalized LLMResponse.
        """
        response = await self.client.responses.create(
            model=model,
            instructions=instructions,
            input=messages,
            tools=tools if tools else [],
        )

        # Extract tool calls
        raw_tool_calls = [item for item in response.output if item.type == "function_call"]

        tool_calls = [
            ToolCallRequest(
                id=tc.call_id,
                name=tc.name,
                arguments=tc.arguments,
            )
            for tc in raw_tool_calls
        ]

        # Extract usage
        usage = None
        if hasattr(response, "usage") and response.usage is not None:
            u = response.usage
            usage = {
                "input_tokens": u.input_tokens,
                "output_tokens": u.output_tokens,
            }
            if hasattr(u, "input_tokens_details") and u.input_tokens_details:
                if hasattr(u.input_tokens_details, "cached_tokens") and u.input_tokens_details.cached_tokens:
                    usage["cached_tokens"] = u.input_tokens_details.cached_tokens
            if hasattr(u, "output_tokens_details") and u.output_tokens_details:
                if hasattr(u.output_tokens_details, "reasoning_tokens") and u.output_tokens_details.reasoning_tokens:
                    usage["reasoning_tokens"] = u.output_tokens_details.reasoning_tokens

        text = None if tool_calls else response.output_text

        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            usage=usage,
            raw=response,
        )

    def format_tool_result(self, tool_call: ToolCallRequest, result: str) -> dict[str, Any]:
        """Format a tool result for the OpenAI Responses API.

        Args:
            tool_call: The original tool call request.
            result: The tool's string result.

        Returns:
            A function_call_output dict keyed by call_id.
        """
        return {
            "type": "function_call_output",
            "call_id": tool_call.id,
            "output": result,
        }

    def build_tool_results_message(
        self,
        tool_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """For OpenAI, tool results are passed directly as input items.

        Args:
            tool_results: List of function_call_output dicts.

        Returns:
            The tool results unchanged (OpenAI passes them as top-level input).
        """
        return tool_results

    def build_assistant_message(self, response: LLMResponse) -> dict[str, Any]:
        """Build an assistant message from the raw response output.

        For the Responses API, the assistant's output items (function_call
        entries) need to be included in subsequent input so the API can
        match tool results to their calls.

        Args:
            response: The normalized LLM response.

        Returns:
            A dict representing the assistant's output for the input array.
        """
        raw = response.raw
        # Return the raw output items directly — the Responses API expects
        # its own output format echoed back as input items.
        output_items = []
        for item in raw.output:
            if item.type == "function_call":
                output_items.append({
                    "type": "function_call",
                    "id": item.id,
                    "call_id": item.call_id,
                    "name": item.name,
                    "arguments": item.arguments,
                })
            elif item.type == "message":
                # Text output — include as assistant message
                text = ""
                for content in item.content:
                    if hasattr(content, "text"):
                        text += content.text
                if text:
                    output_items.append({
                        "role": "assistant",
                        "content": text,
                    })
        # Return as a list to be extended into messages (not appended)
        return output_items

    def convert_tool_schemas(self, openai_schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Pass through — schemas are already in OpenAI format.

        Args:
            openai_schemas: Tool schemas in OpenAI format.

        Returns:
            The schemas unchanged.
        """
        return openai_schemas
