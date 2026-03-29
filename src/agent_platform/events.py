"""Typed event classes for SSE client streaming.

Events are frozen dataclasses yielded by the runner's async generator.
They inform SSE clients about agent lifecycle and progress — what tools
are being called, what text is being produced, when the agent starts and
finishes. Observability (token counting, cost, LLM call tracing) is
instrumented separately.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AgentStart:
    """Emitted when an agent begins a run."""

    agent_name: str
    run_id: str


@dataclass(frozen=True)
class AgentEnd:
    """Emitted when an agent finishes a run."""

    agent_name: str
    run_id: str


@dataclass(frozen=True)
class ToolCallStart:
    """Emitted before a tool is executed."""

    tool_name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolCallEnd:
    """Emitted after a tool finishes, with its result and wall-clock duration."""

    tool_name: str
    result: str
    duration: float


@dataclass(frozen=True)
class TextDelta:
    """Emitted for each chunk of streaming text output."""

    text: str


type Event = AgentStart | AgentEnd | ToolCallStart | ToolCallEnd | TextDelta
