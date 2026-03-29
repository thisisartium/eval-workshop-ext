"""Tracing subsystem for agent observability.

Follows the OpenAI Agents SDK pattern: the runner emits structured spans at
each stage of the ReAct loop, and pluggable TracingProcessor implementations
receive lifecycle callbacks. This is a separate concern from the client-facing
SSE event stream (events.py).

Key types:
    SpanData — typed payload per span kind (AgentSpanData, LLMSpanData, ToolSpanData)
    Span — wraps SpanData with timing, hierarchy (parent_id), and error tracking
    TracingProcessor — abstract interface for observability backends

Default processors:
    LoggingProcessor — logs span events via Python's logging module
    CostTracker — accumulates token counts from LLM spans
    MultiProcessor — fans out to multiple processors with error isolation
"""

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypedDict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SpanError — structured error type matching the OpenAI Agents SDK
# ---------------------------------------------------------------------------


class SpanError(TypedDict):
    """Structured error attached to a span.

    Attributes:
        message: A human-readable error description.
        data: Optional dictionary containing additional error context.
    """

    message: str
    data: dict[str, Any] | None


# ---------------------------------------------------------------------------
# Span data types — typed payloads per span kind
# ---------------------------------------------------------------------------


@dataclass
class AgentSpanData:
    """Payload for an agent lifecycle span.

    Attributes:
        name: The agent's name.
        tools: Names of tools available to this agent.
    """

    name: str
    tools: list[str] | None = None


@dataclass
class LLMSpanData:
    """Payload for an LLM call span.

    Follows the OpenAI Agents SDK GenerationSpanData pattern. Input and
    output hold the message sequences sent to and received from the model.
    Usage is a dict with token counts (populated after the call completes).

    Attributes:
        model: The model identifier used for this call.
        input: Sequence of input messages sent to the model.
        output: Sequence of output messages received from the model.
        usage: Token usage dict (e.g. {"input_tokens": N, "output_tokens": N}).
    """

    model: str
    input: Sequence[Mapping[str, Any]] | None = None
    output: Sequence[Mapping[str, Any]] | None = None
    system_instructions: str | None = None
    usage: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    response_id: str | None = None
    request_params: dict[str, float] | None = None


@dataclass
class ToolSpanData:
    """Payload for a tool execution span.

    Input (arguments JSON) is set on start; output (result) is set on end.

    Attributes:
        name: The tool's name.
        input: Arguments JSON string passed to the tool.
        output: Result string returned by the tool.
    """

    name: str
    input: str | None = None
    output: str | None = None


type SpanData = AgentSpanData | LLMSpanData | ToolSpanData

# ---------------------------------------------------------------------------
# Span — wraps SpanData with timing and hierarchy
# ---------------------------------------------------------------------------


@dataclass
class Span:
    """A single unit of traced work with timing, hierarchy, and payload.

    Spans are mutable: ended_at and error are set after creation,
    and SpanData fields may be populated as the operation progresses.

    Attributes:
        span_id: Unique identifier for this span.
        trace_id: Groups related spans (e.g. a full agent run).
        parent_id: ID of the parent span, or None for root spans.
        span_data: Typed payload (AgentSpanData, LLMSpanData, ToolSpanData).
        started_at: Monotonic timestamp when the span began.
        ended_at: Monotonic timestamp when the span finished (None if still open).
        error: Structured error if the span ended with a failure.
    """

    span_id: str
    trace_id: str
    parent_id: str | None
    span_data: SpanData
    started_at: float = field(default_factory=time.monotonic)
    ended_at: float | None = None
    error: SpanError | None = None


# ---------------------------------------------------------------------------
# TracingProcessor — the extension point
# ---------------------------------------------------------------------------


class TracingProcessor(ABC):
    """Abstract interface for observability backends.

    External systems (OTel exporters, logging, dashboards) implement this
    interface and register with the runner. The runner calls on_span_start
    when a span opens and on_span_end when it closes (with all data populated).
    """

    @abstractmethod
    def on_span_start(self, span: Span) -> None:
        """Called when a span begins."""

    @abstractmethod
    def on_span_end(self, span: Span) -> None:
        """Called when a span finishes, with all data populated."""

    def shutdown(self) -> None:
        """Called on application shutdown for cleanup."""


# ---------------------------------------------------------------------------
# MultiProcessor — fan-out with error isolation
# ---------------------------------------------------------------------------


class MultiProcessor(TracingProcessor):
    """Fans out span events to multiple processors.

    Catches and logs errors from individual processors so one broken
    processor doesn't disrupt others or the agent execution.
    """

    def __init__(self, processors: list[TracingProcessor]) -> None:
        self._processors = list(processors)

    def on_span_start(self, span: Span) -> None:
        """Fan out on_span_start to all processors."""
        for proc in self._processors:
            try:
                proc.on_span_start(span)
            except Exception:
                logger.exception(
                    "Processor %s failed on_span_start", type(proc).__name__
                )

    def on_span_end(self, span: Span) -> None:
        """Fan out on_span_end to all processors."""
        for proc in self._processors:
            try:
                proc.on_span_end(span)
            except Exception:
                logger.exception("Processor %s failed on_span_end", type(proc).__name__)

    def shutdown(self) -> None:
        """Shut down all processors, isolating errors."""
        for proc in self._processors:
            try:
                proc.shutdown()
            except Exception:
                logger.exception("Processor %s failed shutdown", type(proc).__name__)


# ---------------------------------------------------------------------------
# CostTracker — accumulates token counts from LLM spans
# ---------------------------------------------------------------------------


class CostTracker(TracingProcessor):
    """Accumulates token counts from LLM spans.

    Ignores non-LLM spans and LLM spans where token counts are None.
    Cost calculation is left to external consumers (pricing changes
    frequently and varies by agreement).
    """

    def __init__(self) -> None:
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0

    def on_span_start(self, span: Span) -> None:
        """No-op — token data is only available on span end."""

    def on_span_end(self, span: Span) -> None:
        """Accumulate tokens from LLM spans."""
        if not isinstance(span.span_data, LLMSpanData):
            return

        usage = span.span_data.usage
        if usage:
            self.total_input_tokens += usage.get("input_tokens", 0)
            self.total_output_tokens += usage.get("output_tokens", 0)

    def summary(self) -> dict:
        """Return a summary dict of accumulated totals.

        Returns:
            Dict with total_input_tokens and total_output_tokens.
        """
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
        }


# ---------------------------------------------------------------------------
# LoggingProcessor — structured trace output via Python logging
# ---------------------------------------------------------------------------


class LoggingProcessor(TracingProcessor):
    """Logs span lifecycle events via Python's logging module.

    Uses a named logger so output is controlled by standard logging
    configuration (level, handlers, format). Span starts are logged
    at DEBUG; span ends at INFO (or ERROR if the span has an error).
    """

    def __init__(self, logger_name: str = "agent_platform.tracing") -> None:
        self._logger = logging.getLogger(logger_name)

    def on_span_start(self, span: Span) -> None:
        """Log span start at DEBUG level."""
        kind, name = self._describe(span.span_data)
        self._logger.debug("%s START: %s", kind, name)

    def on_span_end(self, span: Span) -> None:
        """Log span end at INFO (or ERROR if span has an error)."""
        kind, name = self._describe(span.span_data)
        duration = ""
        if span.ended_at is not None:
            elapsed = span.ended_at - span.started_at
            duration = f" ({elapsed:.3f}s)"

        detail = self._detail(span.span_data)

        if span.error:
            self._logger.error(
                "%s END: %s%s%s ERROR: %s",
                kind,
                name,
                duration,
                detail,
                span.error["message"],
            )
        else:
            self._logger.info("%s END: %s%s%s", kind, name, duration, detail)

    def _describe(self, data: SpanData) -> tuple[str, str]:
        """Return (kind_label, name) for a SpanData."""
        match data:
            case AgentSpanData(name=name):
                return "agent", name
            case LLMSpanData(model=model):
                return "llm", model
            case ToolSpanData(name=name):
                return "tool", name

    def _detail(self, data: SpanData) -> str:
        """Return extra detail string for span end output."""
        if isinstance(data, LLMSpanData) and data.usage:
            parts = []
            if "input_tokens" in data.usage:
                parts.append(f"in={data.usage['input_tokens']}")
            if "output_tokens" in data.usage:
                parts.append(f"out={data.usage['output_tokens']}")
            if parts:
                return f" [{', '.join(parts)}]"
        return ""
