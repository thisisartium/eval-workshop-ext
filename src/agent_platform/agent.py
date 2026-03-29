"""Agent configuration: dataclass and YAML loading.

Defines the Agent dataclass that holds everything needed to run an agent:
name, model, system prompt, and tool names. The from_yaml() class method
loads agent config from a YAML file and reads the system prompt from a
separate text file, keeping prompts version-controllable independently.
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Agent:
    """Immutable agent configuration.

    Holds the four pieces of information the runner needs to execute an agent:
    a human-readable name, an LLM model identifier, the system prompt text,
    and the list of tool names the agent is allowed to use.

    Attributes:
        name: Human-readable agent name (e.g. "router", "hr_specialist").
        model: OpenAI model ID (e.g. "gpt-4o-mini").
        system_prompt: Full system prompt text loaded from a prompt file.
        tool_names: Names of tools this agent can call. Empty list means
            the agent has no tools and will respond with text only.
    """

    name: str
    model: str
    system_prompt: str
    tool_names: list[str] = field(default_factory=list)
    display_name: str | None = None

    @classmethod
    def from_yaml(cls, yaml_path: Path, base_dir: Path | None = None) -> "Agent":
        """Load an agent configuration from a YAML file.

        The YAML file should contain:
            name: agent name
            model: OpenAI model ID
            prompt_file: path to the system prompt text file
            tools: list of tool names (optional, defaults to [])

        The prompt_file path is resolved relative to base_dir. If base_dir
        is not provided, it defaults to the YAML file's parent directory.

        Args:
            yaml_path: Path to the YAML config file.
            base_dir: Base directory for resolving prompt_file paths.
                Defaults to the YAML file's parent directory.

        Returns:
            A fully populated Agent instance.

        Raises:
            FileNotFoundError: If the YAML file or prompt file doesn't exist.
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Agent config not found: {yaml_path}")

        with open(yaml_path) as f:
            config = yaml.safe_load(f)

        if base_dir is None:
            base_dir = yaml_path.parent

        prompt_path = Path(base_dir) / config["prompt_file"]
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

        system_prompt = prompt_path.read_text().strip()

        return cls(
            name=config["name"],
            model=config.get("model", ""),
            system_prompt=system_prompt,
            tool_names=config.get("tools", []),
            display_name=config.get("display_name"),
        )
