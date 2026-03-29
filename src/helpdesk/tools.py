"""Helpdesk domain tools.

Tools specific to the helpdesk application, built on the agent_platform
tool system. These are registered with the platform's ToolRegistry and
called by agents during the ReAct loop.

There are three tools, each with a distinct role:

``escalate_to_department``
    Forwards a request to a department's human team. Used by the router
    in Phase 1 (routing-only eval) and by specialists when they cannot
    resolve a request from the knowledge base.

``make_call_specialist``
    Closure factory that creates a ``call_specialist`` tool listing
    available specialist agents in its description. The router uses
    this in Phase 2 to delegate requests to specialist agents.

``make_search_knowledge_base``
    Closure factory that creates a ``search_knowledge_base`` tool bound
    to a VectorStore instance. Specialists use this to look up policy
    information.
"""

from collections.abc import Callable

from agent_platform.tools import ToolContext, tool
from helpdesk.kb.vectorstore import VectorStore


@tool
async def escalate_to_department(department: str, request: str) -> str:
    """Escalate a helpdesk request to a department's human team."""
    if department == "None":
        return "This request is not appropriate for the company helpdesk."
    return (
        f"I've forwarded your request to the {department} team. "
        "They will follow up with you shortly."
    )


def make_call_specialist(specialist_names: list[str]) -> Callable:
    """Create a call_specialist tool listing available specialists.

    The tool description dynamically includes the specialist names so
    the LLM knows which specialists it can delegate to.

    Args:
        specialist_names: Names of registered specialist agents
            (e.g. ``["HR"]``).

    Returns:
        A @tool-decorated async function ready for registration.
    """
    specialists_list = ", ".join(specialist_names)

    @tool
    async def call_specialist(
        specialist: str, request: str, context: ToolContext
    ) -> str:
        """Call a specialist agent to handle a request."""
        return await context.run_agent(specialist, request)

    # Override the schema description with the dynamic specialist list
    call_specialist._tool_schema = {
        **call_specialist._tool_schema,
        "description": (
            f"Call a specialist agent to handle a request. "
            f"Available specialists: {specialists_list}"
        ),
    }
    return call_specialist


def make_search_knowledge_base(vectorstore: VectorStore) -> Callable:
    """Create a search_knowledge_base tool bound to the given vectorstore.

    Uses a closure so the tool can access the vectorstore without
    needing ToolContext extension or module-level globals.

    Args:
        vectorstore: A VectorStore instance to search against.

    Returns:
        A @tool-decorated async function ready for registration.
    """

    @tool
    async def search_knowledge_base(query: str) -> str:
        """Search the HR knowledge base for policy information relevant to the query."""
        results = vectorstore.search(query, n_results=3)
        if not results:
            return "No relevant information found in the knowledge base."
        return "\n\n---\n\n".join(results)

    return search_knowledge_base
