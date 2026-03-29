"""LLM provider protocol and normalized response types.

Defines the interface that provider implementations (OpenAI, Anthropic) must
satisfy, plus the normalized types the runner works with. The runner never
touches raw API responses directly — providers translate wire formats into
these common types.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCallRequest:
    """A single tool call extracted from the LLM response.

    Attributes:
        id: Provider-specific call ID (OpenAI call_id, Anthropic tool_use id).
        name: The tool/function name.
        arguments: JSON string of arguments.
    """

    id: str
    name: str
    arguments: str


@dataclass
class LLMResponse:
    """Normalized LLM response that the runner works with.

    Either text or tool_calls will be populated, not both.

    Attributes:
        text: The text response, or None if the LLM requested tool calls.
        tool_calls: List of tool call requests (empty if text response).
        usage: Token usage dict (input_tokens, output_tokens, etc.).
        raw: The original provider response for metadata extraction.
    """

    text: str | None = None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    usage: dict[str, Any] | None = None
    raw: Any = None


class LLMProvider(Protocol):
    """Protocol for LLM provider implementations.

    Providers translate between the runner's normalized types and the
    specific wire format of their API. The runner calls these methods
    without knowing which provider is active.
    """

    async def call(
        self,
        model: str,
        instructions: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        """Send a request to the LLM.

        Args:
            model: Model ID.
            instructions: System prompt.
            messages: Conversation messages in the provider's format.
            tools: Tool schemas in the provider's format.

        Returns:
            A normalized LLMResponse.
        """
        ...

    def format_tool_result(self, tool_call: ToolCallRequest, result: str) -> dict[str, Any]:
        """Format a tool result for the next API call.

        Args:
            tool_call: The original tool call request.
            result: The tool's string result.

        Returns:
            A dict in the provider's expected tool result format.
        """
        ...

    def build_tool_results_message(
        self,
        tool_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Wrap tool results into messages for the next LLM call.

        Args:
            tool_results: List of formatted tool result dicts.

        Returns:
            Messages to append to the conversation for the next call.
        """
        ...

    def build_assistant_message(self, response: LLMResponse) -> list[dict[str, Any]]:
        """Build assistant message(s) from the LLM response for conversation history.

        Returns one or more items to extend into the conversation messages list.
        The format is provider-specific (e.g. Anthropic uses a single assistant
        message with content blocks; OpenAI uses separate function_call items).

        Args:
            response: The normalized LLM response.

        Returns:
            A list of message/item dicts to extend into conversation history.
        """
        ...

    def convert_tool_schemas(self, openai_schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert tool schemas from the canonical (OpenAI) format to the provider's format.

        Args:
            openai_schemas: Tool schemas in OpenAI function calling format.

        Returns:
            Tool schemas in the provider's expected format.
        """
        ...
