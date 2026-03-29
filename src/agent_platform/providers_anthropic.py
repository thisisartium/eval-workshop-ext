"""Anthropic provider implementation.

Wraps the Anthropic Messages API behind the LLMProvider protocol.
Uses explicit message history for multi-turn conversation.
"""

import json
from typing import Any

from anthropic import AsyncAnthropic

from .providers import LLMResponse, ToolCallRequest


class AnthropicProvider:
    """LLM provider backed by the Anthropic Messages API.

    Manages multi-turn conversation via explicit message history —
    the caller is responsible for threading messages through calls.

    Attributes:
        client: The AsyncAnthropic client instance.
        max_tokens: Maximum tokens for each response.
    """

    def __init__(self, client: AsyncAnthropic, max_tokens: int = 4096) -> None:
        self.client = client
        self.max_tokens = max_tokens

    async def call(
        self,
        model: str,
        instructions: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        """Send a request via the Anthropic Messages API.

        Args:
            model: Anthropic model ID (e.g. "claude-sonnet-4-6").
            instructions: System prompt.
            messages: Conversation messages in Anthropic format.
            tools: Tool schemas in Anthropic format.

        Returns:
            A normalized LLMResponse.
        """
        kwargs: dict[str, Any] = {
            "model": model,
            "system": instructions,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools

        response = await self.client.messages.create(**kwargs)

        # Extract tool calls
        tool_calls = [
            ToolCallRequest(
                id=block.id,
                name=block.name,
                arguments=json.dumps(block.input),
            )
            for block in response.content
            if block.type == "tool_use"
        ]

        # Extract text
        text = None
        if not tool_calls:
            text_parts = [
                block.text for block in response.content if block.type == "text"
            ]
            text = "".join(text_parts) if text_parts else ""

        # Extract usage
        usage = None
        if response.usage:
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
            if hasattr(response.usage, "cache_creation_input_tokens") and response.usage.cache_creation_input_tokens:
                usage["cache_creation_input_tokens"] = response.usage.cache_creation_input_tokens
            if hasattr(response.usage, "cache_read_input_tokens") and response.usage.cache_read_input_tokens:
                usage["cache_read_input_tokens"] = response.usage.cache_read_input_tokens

        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            usage=usage,
            raw=response,
        )

    def format_tool_result(self, tool_call: ToolCallRequest, result: str) -> dict[str, Any]:
        """Format a tool result for the Anthropic Messages API.

        Args:
            tool_call: The original tool call request.
            result: The tool's string result.

        Returns:
            A tool_result content block.
        """
        return {
            "type": "tool_result",
            "tool_use_id": tool_call.id,
            "content": result,
        }

    def build_tool_results_message(
        self,
        tool_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Wrap tool results in a user message for Anthropic.

        Anthropic expects tool results as content blocks inside a user message.

        Args:
            tool_results: List of tool_result content blocks.

        Returns:
            A single-element list containing the user message.
        """
        return [{"role": "user", "content": tool_results}]

    def build_assistant_message(self, response: LLMResponse) -> list[dict[str, Any]]:
        """Build an assistant message from the raw Anthropic response.

        Preserves the full content blocks (text + tool_use) so that
        multi-turn history is correct.

        Args:
            response: The normalized LLM response.

        Returns:
            A single-element list with the assistant message dict.
        """
        raw = response.raw
        content: list[dict[str, Any]] = []
        for block in raw.content:
            if block.type == "text":
                content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
        return [{"role": "assistant", "content": content}]

    def convert_tool_schemas(self, openai_schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI function calling schemas to Anthropic tool format.

        Renames ``parameters`` to ``input_schema`` and drops OpenAI-specific
        fields like ``strict``.

        Args:
            openai_schemas: Tool schemas in OpenAI format.

        Returns:
            Tool schemas in Anthropic format.
        """
        anthropic_tools = []
        for schema in openai_schemas:
            anthropic_tools.append({
                "name": schema["name"],
                "description": schema.get("description", ""),
                "input_schema": schema["parameters"],
            })
        return anthropic_tools
