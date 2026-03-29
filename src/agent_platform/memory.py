"""Context window management with pluggable overflow strategies.

ContextWindow handles token accounting — it counts tokens for the system
prompt, tool schemas, and message history, then delegates to an
OverflowStrategy when messages exceed the remaining budget.

The default strategy is SlidingWindow, which drops the oldest messages
first while always preserving the most recent message.
"""

import json
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import tiktoken


@runtime_checkable
class OverflowStrategy(Protocol):
    """Decides which messages to keep when history exceeds the token budget.

    Implementations receive the full message list, the remaining token
    budget (after instructions and tool schemas), and a token counting
    function. They return a (possibly shorter) list of messages that fits
    within the budget.
    """

    def apply(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        count_tokens: Callable[[str], int],
    ) -> list[dict[str, str]]:
        """Select messages that fit within the token budget.

        Args:
            messages: Full message history, oldest first.
            max_tokens: Token budget available for messages.
            count_tokens: Function to count tokens in a string.

        Returns:
            A list of messages that fits within max_tokens.
        """
        ...


# Per-message token overhead (role markers, formatting). Conservative
# estimate for the Responses API message format.
_MESSAGE_OVERHEAD_TOKENS = 4


class SlidingWindow:
    """Keep the most recent messages, dropping oldest first.

    Iterates backward from the newest message, accumulating token counts.
    Stops adding messages once the budget would be exceeded. The most
    recent message is always included regardless of budget.
    """

    def apply(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        count_tokens: Callable[[str], int],
    ) -> list[dict[str, str]]:
        """Keep messages from the end of the list that fit the budget.

        Args:
            messages: Full message history, oldest first.
            max_tokens: Token budget available for messages.
            count_tokens: Function to count tokens in a string.

        Returns:
            The most recent messages that fit within max_tokens.
            Always returns at least the last message if any exist.
        """
        if not messages:
            return []

        kept: list[dict[str, str]] = []
        tokens_used = 0

        for msg in reversed(messages):
            msg_tokens = (
                count_tokens(msg["role"])
                + count_tokens(msg["content"])
                + _MESSAGE_OVERHEAD_TOKENS
            )
            if kept and tokens_used + msg_tokens > max_tokens:
                break
            kept.append(msg)
            tokens_used += msg_tokens

        kept.reverse()
        return kept


class ContextWindow:
    """Builds a token-budgeted message list for the LLM.

    Counts tokens for the system prompt (instructions) and tool schemas
    as fixed overhead, then delegates to an OverflowStrategy to select
    which messages fit in the remaining budget.

    Attributes:
        max_tokens: Total token budget for the context window.
        strategy: The overflow strategy to use when messages exceed budget.
    """

    def __init__(
        self,
        max_tokens: int,
        model: str = "gpt-4o-mini",
        strategy: OverflowStrategy | None = None,
    ) -> None:
        self.max_tokens = max_tokens
        self.strategy = strategy or SlidingWindow()
        try:
            self._encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Non-OpenAI models (e.g. Claude) — use cl100k_base as a
            # reasonable approximation for context window budgeting.
            self._encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count tokens in a string using tiktoken.

        Args:
            text: The text to count tokens for.

        Returns:
            The number of tokens.
        """
        if not text:
            return 0
        return len(self._encoding.encode(text))

    def build(
        self,
        instructions: str,
        messages: list[dict[str, str]],
        tool_schemas: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, str]]:
        """Return messages that fit within the token budget.

        Counts instructions and tool schemas as fixed overhead (they are
        always present but passed separately to the API). The remaining
        budget is given to the overflow strategy to select messages.

        Args:
            instructions: System prompt text.
            messages: Full message history, oldest first.
            tool_schemas: Tool JSON schemas (counted for budget, not returned).

        Returns:
            A (possibly truncated) list of messages that fits the budget.
        """
        if not messages:
            return []

        overhead = self.count_tokens(instructions)
        if tool_schemas:
            overhead += self.count_tokens(json.dumps(tool_schemas))

        remaining = self.max_tokens - overhead
        if remaining <= 0:
            # Even without messages, instructions + tools exceed budget.
            # Return at least the most recent message so the agent can respond.
            return messages[-1:]

        return self.strategy.apply(messages, remaining, self.count_tokens)
