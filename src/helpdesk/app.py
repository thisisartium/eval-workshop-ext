"""Helpdesk application wiring.

Defines the helpdesk domain's agents and tools, then uses the platform's
AgentClient.create() factory to assemble everything. This is the only
place that knows about both the platform and the helpdesk domain.

Agents are loaded from YAML config files with separate prompt files,
following the platform's config-driven agent lifecycle pattern. The
knowledge base vectorstore is initialized here and injected into
the search tool via a closure factory.

System configurations control which agents and tools are active:

- ``baseline.yaml`` — concierge with pure escalation, no specialists
- ``with_specialists.yaml`` — concierge + specialists, untuned prompts
- ``tuned.yaml`` — concierge + specialists, fully tuned prompts
- ``system.yaml`` — production default (points to tuned)

Config files, prompts, and KB documents live at the project root
(not inside the package) for easy access during workshops.
"""

import os
from dataclasses import replace
from pathlib import Path

import yaml
from dotenv import load_dotenv

from agent_platform.agent import Agent
from agent_platform.client import AgentClient
from agent_platform.observability import TracingProcessor
from helpdesk.kb.vectorstore import VectorStore
from helpdesk.tools import (
    escalate_to_department,
    make_call_specialist,
    make_search_knowledge_base,
)

CONFIGS_DIR = Path("configs")
KB_DIR = Path("kb")
MODELS_CONFIG = CONFIGS_DIR / "models.yaml"

# Fallback defaults if models.yaml is missing
_FALLBACK_OPENAI_MODEL = "gpt-4o-mini"
_FALLBACK_ANTHROPIC_MODEL = "claude-sonnet-4-6"


def _load_model_defaults() -> tuple[str, str]:
    """Load default model IDs from configs/models.yaml.

    Returns:
        Tuple of (anthropic_model, openai_model).
    """
    if MODELS_CONFIG.exists():
        with open(MODELS_CONFIG) as f:
            config = yaml.safe_load(f) or {}
        return (
            config.get("anthropic", _FALLBACK_ANTHROPIC_MODEL),
            config.get("openai", _FALLBACK_OPENAI_MODEL),
        )
    return _FALLBACK_ANTHROPIC_MODEL, _FALLBACK_OPENAI_MODEL


def default_model() -> str:
    """Return the default model based on available API keys.

    Reads model IDs from ``configs/models.yaml`` and selects one based
    on which API keys are set in the environment:

    - OPENAI_API_KEY only → openai model from config
    - ANTHROPIC_API_KEY set (or both) → anthropic model from config

    Returns:
        A model ID string.
    """
    load_dotenv(override=True)
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))

    anthropic_model, openai_model = _load_model_defaults()

    if has_openai and not has_anthropic:
        return openai_model
    return anthropic_model


def create_app(
    config: str | None = None,
    model: str | None = None,
    prompt_file: str | None = None,
    session_dir: Path | None = None,
    processor: TracingProcessor | None = None,
) -> AgentClient:
    """Create a fully wired helpdesk application.

    Loads a system configuration that defines which agents are active,
    initializes tools based on the configuration, and delegates all
    platform wiring to AgentClient.create().

    All paths are resolved relative to the current working directory
    (project root).

    Args:
        config: System config filename, either relative to the configs
            directory (e.g. ``"baseline.yaml"``) or a full path
            (e.g. ``"configs/baseline.yaml"``). Defaults to
            ``"system.yaml"``.
        model: Override the model for all agents. If None, uses model
            from each agent's YAML config.
        prompt_file: Override the routing agent's prompt file, relative
            to the project root (e.g. ``"prompts/routing/tuned.txt"``).
            If None, uses the prompt from the routing agent's YAML config.
        session_dir: Directory for session JSONL files. Defaults to
            ``data/sessions`` relative to the working directory.
        processor: Optional TracingProcessor for observability.
            Use ``configure_tracing()`` from ``agent_platform.tracing``
            to create one with OTLP export.

    Returns:
        An AgentClient ready to send messages and receive events.
    """
    # Pick a model that matches an available API key
    model = model or default_model()

    # All paths resolve from project root (cwd)
    base_dir = Path(".")

    # Load system config — accept both "system.yaml" and "configs/system.yaml"
    config_name = config or "system.yaml"
    config_path = Path(config_name)
    if not config_path.exists():
        config_path = CONFIGS_DIR / config_name
    with open(config_path) as f:
        system_config = yaml.safe_load(f)

    # Load routing agent
    routing_path = base_dir / system_config["routing"]
    router = Agent.from_yaml(routing_path, base_dir=base_dir)

    # Apply overrides
    if prompt_file:
        prompt_path = base_dir / prompt_file
        router = replace(router, system_prompt=prompt_path.read_text().strip())
    router = replace(router, model=model)

    # Build tool list — always include escalate_to_department
    tools = [escalate_to_department]
    agents = [router]

    # Load specialists if configured
    specialists_config = system_config.get("specialists", {})
    if specialists_config:
        # Initialize knowledge base for specialist tools
        vectorstore = VectorStore.from_directory(KB_DIR / "hr")
        search_kb = make_search_knowledge_base(vectorstore)
        tools.append(search_kb)

        specialist_names = []
        for name, agent_path in specialists_config.items():
            specialist = Agent.from_yaml(base_dir / agent_path, base_dir=base_dir)
            specialist = replace(specialist, model=model)
            agents.append(specialist)
            specialist_names.append(name)

        # Create call_specialist tool with dynamic specialist list
        call_specialist = make_call_specialist(specialist_names)
        tools.append(call_specialist)

    return AgentClient.create(
        agents=agents,
        tools=tools,
        default_agent_name="router",
        session_dir=session_dir,
        processor=processor,
    )


def create_specialist_app(
    agent_config: str = "configs/agents/hr_tuned.yaml",
    model: str | None = None,
    prompt_file: str | None = None,
    session_dir: Path | None = None,
    processor: TracingProcessor | None = None,
) -> AgentClient:
    """Create an app that runs a single specialist agent directly.

    Used by experiments that need to evaluate a specialist in isolation
    (bypassing the routing agent).

    Args:
        agent_config: Agent config path relative to the project root
            (e.g. ``"configs/agents/hr_tuned.yaml"``).
        model: Override model. If None, uses the agent config value.
        prompt_file: Override prompt file, relative to the project root.
            If None, uses the agent config value.
        session_dir: Directory for session JSONL files.
        processor: Optional TracingProcessor for observability.

    Returns:
        An AgentClient with the specialist as the default agent.
    """
    model = model or default_model()

    base_dir = Path(".")

    agent = Agent.from_yaml(base_dir / agent_config, base_dir=base_dir)

    if prompt_file:
        prompt_path = base_dir / prompt_file
        agent = replace(agent, system_prompt=prompt_path.read_text().strip())
    agent = replace(agent, model=model)

    # Build tools for the specialist
    vectorstore = VectorStore.from_directory(KB_DIR / "hr")
    search_kb = make_search_knowledge_base(vectorstore)
    tools = [search_kb, escalate_to_department]

    return AgentClient.create(
        agents=[agent],
        tools=tools,
        default_agent_name=agent.name,
        session_dir=session_dir,
        processor=processor,
    )
