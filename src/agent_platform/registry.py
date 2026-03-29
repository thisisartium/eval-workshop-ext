"""Agent registry.

AgentRegistry manages the set of agents available in the system.
Agents are registered by name and looked up at runtime by the runner
and ToolContext for delegation.
"""

from .agent import Agent


class AgentRegistry:
    """Registry that manages named agents.

    Stores resolved Agent configs by name. Used by the runner to look up
    agents for execution and by ToolContext for delegation.
    """

    def __init__(self) -> None:
        self._agents: dict[str, Agent] = {}

    def register(self, agent: Agent) -> None:
        """Register an agent by its name.

        Args:
            agent: A resolved Agent configuration.
        """
        self._agents[agent.name] = agent

    def get(self, name: str) -> Agent | None:
        """Look up an agent by name.

        Args:
            name: The agent's registered name.

        Returns:
            The Agent, or None if not found.
        """
        return self._agents.get(name)
