"""ReAct execution loop for agents.

Implements the core Reason + Act loop: call the LLM, inspect output for tool
calls, execute tools, feed results back, repeat until the LLM responds with
text or max_steps is exceeded.

Uses the LLMProvider protocol to support multiple backends (OpenAI, Anthropic).
The provider handles all API-specific wire format details; the runner works
only with normalized LLMResponse types.

Events are yielded at each stage via async generator for client-facing SSE
streaming. Observability (token counting, cost) is instrumented separately.
"""

import json
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from .events import (
    AgentEnd,
    AgentStart,
    Event,
    TextDelta,
    ToolCallEnd,
    ToolCallStart,
)
from .memory import ContextWindow
from .observability import (
    AgentSpanData,
    LLMSpanData,
    Span,
    SpanError,
    ToolSpanData,
    TracingProcessor,
)
from .providers import LLMProvider, LLMResponse, ToolCallRequest
from .registry import AgentRegistry
from .session import Session
from .tools import ToolContext, ToolRegistry


class MaxStepsExceeded(Exception):
    """Raised when the ReAct loop exceeds the configured maximum step count."""


class AgentRunner:
    """ReAct loop runner that executes agents with tools.

    Holds the LLM provider, tool registry, and configuration needed to
    run agent loops. The run() method is an async generator that yields
    typed events at each stage of execution.

    The ReAct loop follows a simple pattern:
        1. Call the LLM with the current conversation state
        2. If the LLM responds with text → done
        3. If the LLM requests tool calls → execute them, feed results back
        4. Repeat until text response or max_steps exceeded
    """

    def __init__(
        self,
        provider: LLMProvider,
        registry: ToolRegistry,
        max_steps: int = 10,
        agent_registry: AgentRegistry | None = None,
        context_window: ContextWindow | None = None,
        processor: TracingProcessor | None = None,
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.max_steps = max_steps
        self.agent_registry = agent_registry
        self.context_window = context_window
        self.processor = processor

    async def run(
        self,
        name: str,
        model: str,
        instructions: str,
        message: str,
        tool_names: list[str] | None = None,
        session: Session | None = None,
        _trace_id: str | None = None,
        _parent_span_id: str | None = None,
    ) -> AsyncGenerator[Event, None]:
        """Run a ReAct agent loop, yielding typed events at each stage.

        Calls the LLM, executes tool calls, feeds results back, and repeats
        until the LLM returns a text response or max_steps is exceeded.

        When a session is provided, the runner builds the initial input from
        the session's message history (truncated by the context window if
        configured). The user message and assistant response are appended
        to the session so multi-turn conversations persist.

        Args:
            name: Human-readable name for the agent (used in events).
            model: Model ID to use for LLM calls.
            instructions: System prompt passed to the provider.
            message: The user's input message for this run.
            tool_names: Names of tools registered in the registry to expose.
                If None or empty, the LLM is called without tools.
            session: Optional session for multi-turn conversation. When
                provided, history is included in the first LLM call and
                new messages are appended after the run.
            _trace_id: Internal. Trace ID inherited from parent agent during
                delegation. When None, a new trace is created.
            _parent_span_id: Internal. Parent span ID for delegation hierarchy.

        Yields:
            Typed Event instances in order: AgentStart, then per-step
            ToolCallStart/End events, then TextDelta and AgentEnd on
            completion.

        Raises:
            MaxStepsExceeded: If the loop runs more than max_steps iterations
                without the LLM producing a text response.
        """
        run_id = uuid.uuid4().hex[:8]
        trace_id = _trace_id or run_id

        # --- Agent span: wraps the entire run ---
        agent_span = self._start_span(
            AgentSpanData(name=name, tools=tool_names),
            trace_id=trace_id,
            parent_id=_parent_span_id,
        )

        yield AgentStart(agent_name=name, run_id=run_id)

        # Get canonical (OpenAI-format) schemas for tracing, then convert for provider
        canonical_schemas = self.registry.get_schemas(tool_names) if tool_names else []
        provider_tools = self.provider.convert_tool_schemas(canonical_schemas) if canonical_schemas else []

        # Build initial messages from session history or bare message
        messages = self._build_initial_messages(message, instructions, canonical_schemas, session)

        for _step in range(self.max_steps):
            # --- LLM span: wraps each LLM call ---
            llm_span = self._start_span(
                LLMSpanData(
                    model=model,
                    input=list(messages),
                    system_instructions=instructions,
                    tools=canonical_schemas if canonical_schemas else None,
                ),
                trace_id=trace_id,
                parent_id=agent_span.span_id,
            )

            response = await self.provider.call(
                model, instructions, messages, provider_tools
            )

            # Populate span output and metadata
            llm_span.span_data.output = self._build_llm_output(response)
            if response.usage:
                llm_span.span_data.usage = response.usage

            raw = response.raw
            if hasattr(raw, "id") and raw.id:
                llm_span.span_data.response_id = raw.id

            request_params: dict[str, float] = {}
            if hasattr(raw, "temperature") and raw.temperature is not None:
                request_params["temperature"] = raw.temperature
            if hasattr(raw, "top_p") and raw.top_p is not None:
                request_params["top_p"] = raw.top_p
            if request_params:
                llm_span.span_data.request_params = request_params

            self._end_span(llm_span)

            if not response.tool_calls:
                # Terminal condition: LLM responded with text instead of tool calls
                final_text = response.text or ""
                if session is not None:
                    session.add_message("assistant", final_text)
                yield TextDelta(text=final_text)
                yield AgentEnd(agent_name=name, run_id=run_id)
                self._end_span(agent_span)
                return

            # Append assistant output to conversation history
            messages.extend(self.provider.build_assistant_message(response))

            # Execute tools and prepare results for the next LLM call
            tool_results: list[dict[str, Any]] = []
            for tc in response.tool_calls:
                yield ToolCallStart(
                    tool_name=tc.name,
                    arguments=json.loads(tc.arguments),
                )
                end_event = await self._execute_tool_call(
                    tc, trace_id=trace_id, parent_span_id=agent_span.span_id
                )
                yield end_event
                tool_results.append(
                    self.provider.format_tool_result(tc, end_event.result)
                )

            # Append tool results to messages for next iteration
            result_messages = self.provider.build_tool_results_message(tool_results)
            messages.extend(result_messages)

        self._end_span(agent_span, error="Max steps exceeded")
        raise MaxStepsExceeded(
            f"Agent '{name}' exceeded maximum of {self.max_steps} steps"
        )

    def _build_initial_messages(
        self,
        message: str,
        instructions: str,
        tools: list[dict[str, Any]],
        session: Session | None,
    ) -> list[dict[str, Any]]:
        """Build the initial message list for the first LLM call.

        Without a session, returns a single user message. With a session,
        appends the new user message to history, optionally truncates via
        ContextWindow, and returns the full message list.

        Args:
            message: The user's new message.
            instructions: System prompt (for ContextWindow token accounting).
            tools: Tool schemas (for ContextWindow token accounting).
            session: Optional session with conversation history.

        Returns:
            A list of message dicts.
        """
        if session is None:
            return [{"role": "user", "content": message}]

        session.add_message("user", message)
        history = session.messages

        if self.context_window is not None:
            history = self.context_window.build(instructions, history, tools)

        return [{"role": m["role"], "content": m["content"]} for m in history]

    def _build_llm_output(self, response: LLMResponse) -> list[dict[str, Any]]:
        """Build structured output messages from an LLM response for tracing.

        Args:
            response: The normalized LLM response.

        Returns:
            A list of message dicts capturing the LLM's output.
        """
        if response.tool_calls:
            tc_list: list[dict[str, Any]] = []
            for tc in response.tool_calls:
                tc_dict: dict[str, Any] = {
                    "name": tc.name,
                    "arguments": json.loads(tc.arguments) if tc.arguments else {},
                }
                tc_dict["id"] = tc.id
                tc_list.append(tc_dict)
            return [{"role": "assistant", "tool_calls": tc_list}]
        return [{"role": "assistant", "content": response.text or ""}]

    def _start_span(
        self,
        span_data: AgentSpanData | LLMSpanData | ToolSpanData,
        trace_id: str,
        parent_id: str | None = None,
    ) -> Span:
        """Create and start a new span, notifying the processor.

        Args:
            span_data: Typed payload for this span.
            trace_id: Trace ID grouping related spans.
            parent_id: Parent span ID, or None for root spans.

        Returns:
            The started Span instance.
        """
        span = Span(
            span_id=uuid.uuid4().hex[:8],
            trace_id=trace_id,
            parent_id=parent_id,
            span_data=span_data,
            started_at=time.monotonic(),
        )
        if self.processor is not None:
            self.processor.on_span_start(span)
        return span

    def _end_span(self, span: Span, error: str | None = None) -> None:
        """End a span and notify the processor.

        Args:
            span: The span to close.
            error: Optional error message if the span ended with a failure.
        """
        span.ended_at = time.monotonic()
        if error is not None:
            span.error = SpanError(message=error, data=None)
        if self.processor is not None:
            self.processor.on_span_end(span)

    def _make_context(
        self,
        trace_id: str | None = None,
        parent_span_id: str | None = None,
    ) -> ToolContext:
        """Build a ToolContext whose run_agent() delegates back into this runner.

        The context captures this runner instance via closure, allowing
        tools to delegate to other agents without importing AgentRunner.

        Args:
            trace_id: Trace ID to propagate to child agent runs.
            parent_span_id: Parent span ID for the child agent's hierarchy.

        Returns:
            A ToolContext bound to this runner's agent registry.
        """

        async def _run_agent(agent_name: str, message: str) -> str:
            if self.agent_registry is None:
                return "Error: no AgentRegistry configured"
            agent = self.agent_registry.get(agent_name)
            if agent is None:
                return f"Error: agent '{agent_name}' not found"
            text, _events = await collect(
                self.run(
                    agent.display_name or agent.name,
                    agent.model,
                    agent.system_prompt,
                    message,
                    tool_names=agent.tool_names,
                    _trace_id=trace_id,
                    _parent_span_id=parent_span_id,
                )
            )
            return text

        return ToolContext(_run_agent=_run_agent)

    async def _execute_tool_call(
        self,
        tool_call: ToolCallRequest,
        trace_id: str | None = None,
        parent_span_id: str | None = None,
    ) -> ToolCallEnd:
        """Execute a single tool call via the registry.

        Runs the tool, measures wall-clock duration, and returns a
        ToolCallEnd event. Emits a tool span around the execution and
        threads trace context through to any delegated agent runs.

        Args:
            tool_call: A normalized ToolCallRequest.
            trace_id: Trace ID for span hierarchy.
            parent_span_id: Parent span ID for this tool span.

        Returns:
            A ToolCallEnd event with the tool's result and duration.
        """
        # --- Tool span: wraps the tool execution ---
        tool_span = (
            self._start_span(
                ToolSpanData(name=tool_call.name, input=tool_call.arguments),
                trace_id=trace_id or "",
                parent_id=parent_span_id,
            )
            if self.processor is not None
            else None
        )

        start_time = time.monotonic()
        context = self._make_context(
            trace_id=trace_id, parent_span_id=tool_span.span_id if tool_span else None
        )
        result = await self.registry.execute(tool_call.name, tool_call.arguments, context=context)
        duration = time.monotonic() - start_time

        if tool_span is not None:
            tool_span.span_data.output = result
            self._end_span(tool_span)

        return ToolCallEnd(tool_name=tool_call.name, result=result, duration=duration)


async def collect(
    events: AsyncGenerator[Event, None],
) -> tuple[str, list[Event]]:
    """Drain an event generator, returning the final text and all events.

    Convenience function for tests and scripts that want to run an agent
    and get results without writing an async for loop.

    Args:
        events: An async generator of Event objects from AgentRunner.run().

    Returns:
        A (final_text, events) tuple where final_text is the concatenation
        of all TextDelta.text values and events is the full list.
    """
    collected: list[Event] = []
    text_parts: list[str] = []
    async for event in events:
        collected.append(event)
        if isinstance(event, TextDelta):
            text_parts.append(event.text)
    return "".join(text_parts), collected
