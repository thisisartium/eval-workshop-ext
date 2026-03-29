"""Platform client interface.

AgentClient is the canonical way to interact with the agent platform.
It wraps the runner, agent registry, and session store behind a clean
API: send a message, get an event stream back.

This is the platform's primary port. In-process consumers (CLI, server)
use it directly. A future HTTP client will mirror the same interface
for remote consumers.

Use the ``create`` class method to build a client from high-level
inputs (agents and tools) without needing to understand the internal
platform components::

    client = AgentClient.create(
        agents=[router, hr_specialist],
        tools=[route_message],
        default_agent_name="router",
    )
"""

from collections.abc import AsyncGenerator, Callable
from pathlib import Path

from .agent import Agent
from .events import Event
from .llm import create_provider
from .memory import ContextWindow
from .observability import TracingProcessor
from .providers import LLMProvider
from .registry import AgentRegistry
from .runner import AgentRunner
from .session import Session, SessionStore
from .tools import ToolRegistry


class AgentClient:
    """Primary interface to the agent platform.

    Encapsulates agent resolution, session lifecycle, and runner
    execution behind a simple send/receive API. Consumers provide
    a message and optionally an agent name and session ID; the client
    handles everything else.

    Attributes:
        runner: The configured AgentRunner.
        agent_registry: Registry of available agents.
        session_store: Persistent session storage.
        default_agent_name: Agent to use when no name is specified.
    """

    def __init__(
        self,
        runner: AgentRunner,
        agent_registry: AgentRegistry,
        session_store: SessionStore,
        default_agent_name: str,
    ) -> None:
        self.runner = runner
        self.agent_registry = agent_registry
        self.session_store = session_store
        self.default_agent_name = default_agent_name

    @classmethod
    def create(
        cls,
        agents: list[Agent],
        tools: list[Callable] | None = None,
        default_agent_name: str | None = None,
        session_dir: Path | None = None,
        max_tokens: int = 8000,
        max_steps: int = 10,
        processor: TracingProcessor | None = None,
        provider: LLMProvider | None = None,
    ) -> "AgentClient":
        """Create a fully wired client from agents and tools.

        Factory method that assembles all platform internals — LLM provider,
        tool registry, agent registry, context window, session store, and
        runner — so consumers don't need to understand the individual
        components.

        The LLM provider is inferred from the first agent's model name
        (models starting with "claude" use Anthropic, others use OpenAI)
        unless an explicit provider is passed.

        Args:
            agents: List of Agent configs to register. At least one
                is required.
            tools: List of ``@tool``-decorated functions to register.
                If None, no tools are available.
            default_agent_name: Name of the agent to use when none is
                specified in ``send()``. Defaults to the first agent's
                name.
            session_dir: Directory for session JSONL files. Defaults to
                ``data/sessions`` relative to the working directory.
            max_tokens: Token budget for context window. Defaults to
                8000.
            max_steps: Maximum ReAct loop iterations. Defaults to 10.
            processor: Optional TracingProcessor for observability.
                If provided, the runner emits spans to it during
                execution.
            provider: Optional LLMProvider instance. If None, one is
                created automatically based on the first agent's model.

        Returns:
            A fully configured AgentClient ready to use.
        """
        if provider is None:
            provider = create_provider(model=agents[0].model)

        tool_registry = ToolRegistry()
        for tool_func in tools or []:
            tool_registry.register(tool_func)

        agent_registry = AgentRegistry()
        for agent in agents:
            agent_registry.register(agent)

        context_window = ContextWindow(max_tokens=max_tokens)
        session_store = SessionStore(session_dir or Path("data/sessions"))

        runner = AgentRunner(
            provider=provider,
            registry=tool_registry,
            agent_registry=agent_registry,
            context_window=context_window,
            max_steps=max_steps,
            processor=processor,
        )

        return cls(
            runner=runner,
            agent_registry=agent_registry,
            session_store=session_store,
            default_agent_name=default_agent_name or agents[0].name,
        )

    async def send(
        self,
        message: str,
        agent_name: str | None = None,
        session_id: str | None = None,
    ) -> AsyncGenerator[Event, None]:
        """Send a message to an agent, yielding events.

        Resolves the agent from the registry, optionally loads or creates
        a session, runs the agent via the runner, and yields events as
        they are produced. If a session ID is provided, the session is
        persisted after the run completes.

        Args:
            message: The user's input message.
            agent_name: Name of the agent to run. Defaults to the
                client's default agent.
            session_id: Optional session ID for multi-turn persistence.
                If provided, the session is loaded (or created) and
                saved after the run.

        Yields:
            Typed Event instances from the runner.

        Raises:
            ValueError: If the specified agent name is not found in
                the registry.
        """
        name = agent_name or self.default_agent_name
        agent = self.agent_registry.get(name)
        if agent is None:
            raise ValueError(f"Agent '{name}' not found in registry")

        session: Session | None = None
        if session_id is not None:
            session = self.session_store.get_or_create(session_id)

        async for event in self.runner.run(
            name=agent.display_name or agent.name,
            model=agent.model,
            instructions=agent.system_prompt,
            message=message,
            tool_names=agent.tool_names,
            session=session,
        ):
            yield event

        if session_id is not None and session is not None:
            self.session_store.save(session)

    def list_sessions(self) -> list[str]:
        """List all available session IDs.

        Returns:
            A list of session ID strings.
        """
        return self.session_store.list_sessions()

    def get_session(self, session_id: str) -> Session | None:
        """Get a session by ID.

        Args:
            session_id: The session ID to look up.

        Returns:
            The session if found, otherwise None.
        """
        return self.session_store.get(session_id)
