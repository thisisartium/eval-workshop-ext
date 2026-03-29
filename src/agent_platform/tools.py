"""Tool system: decorator and registry.

Provides the @tool decorator for defining agent tools from async functions
and ToolRegistry for managing and executing them.

The @tool decorator inspects type hints to generate JSON schemas compatible
with the OpenAI Responses API's strict function calling mode. Schema generation
is powered by Pydantic (via create_model + to_strict_json_schema), so all
standard Python types map correctly without hand-rolled conversion.
"""

import functools
import inspect
import json
import typing
from collections.abc import Awaitable, Callable as AbcCallable
from dataclasses import dataclass
from typing import Any, Callable

import pydantic

from openai.lib._pydantic import to_strict_json_schema
from openai.types.responses import FunctionToolParam


@dataclass
class ToolContext:
    """Platform context injected into tools that declare it.

    Tools that need to delegate to other agents declare a parameter typed
    ToolContext. The runner injects this at call time; it is hidden from
    the LLM's JSON schema.

    Attributes:
        _run_agent: Async callable that runs a named agent and returns
            its final text response. Supplied by AgentRunner at call time.
    """

    _run_agent: AbcCallable[[str, str], Awaitable[str]]

    async def run_agent(self, agent_name: str, message: str) -> str:
        """Delegate a message to a named agent and return its text reply.

        Args:
            agent_name: The registered name of the agent to run.
            message: The message to send to the agent.

        Returns:
            The agent's final text response.
        """
        return await self._run_agent(agent_name, message)


def tool(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator that marks an async function as an agent tool.

    Extracts a JSON schema from the function's type hints and docstring,
    compatible with the OpenAI Responses API's strict function calling mode.

    The decorated function gains two attributes:
        _is_tool: True — used by ToolRegistry.register() for validation.
        _tool_schema: FunctionToolParam dict for the OpenAI API.

    Args:
        func: An async function to expose as an agent tool.

    Returns:
        The wrapped function with tool metadata attached.
    """
    hints = typing.get_type_hints(func)
    sig = inspect.signature(func)

    # Build Pydantic field definitions from type hints.
    # Each field is a (type, ...) tuple — the ellipsis means required.
    # ToolContext-typed params are skipped (injected by runner, hidden from LLM).
    fields: dict[str, Any] = {}
    context_param: str | None = None
    for name in sig.parameters:
        if name not in hints:
            continue
        if hints[name] is ToolContext:
            context_param = name
            continue
        fields[name] = (hints[name], ...)

    model = pydantic.create_model(func.__name__, **fields)
    parameters = to_strict_json_schema(model)

    description = (inspect.getdoc(func) or func.__name__).split("\n")[0].strip()

    schema: FunctionToolParam = {
        "type": "function",
        "name": func.__name__,
        "description": description,
        "parameters": parameters,
        "strict": True,
    }

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        return await func(*args, **kwargs)

    wrapper._is_tool = True  # type: ignore[attr-defined]
    wrapper._tool_schema = schema  # type: ignore[attr-defined]
    wrapper._context_param = context_param  # type: ignore[attr-defined]
    return wrapper


class ToolRegistry:
    """Registry for agent tools with schema lookup and execution.

    Tools are registered explicitly via register(), looked up by name,
    and executed with automatic argument parsing. The registry provides
    schemas in the format expected by the OpenAI Responses API.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Callable[..., Any]] = {}

    def register(self, func: Callable[..., Any]) -> None:
        """Register a @tool-decorated function.

        Args:
            func: A function decorated with @tool.

        Raises:
            ValueError: If the function is not decorated with @tool.
        """
        if not getattr(func, "_is_tool", False):
            raise ValueError(f"{func.__name__} is not decorated with @tool")
        self._tools[func._tool_schema["name"]] = func  # type: ignore[attr-defined]

    def get(self, name: str) -> Callable[..., Any] | None:
        """Look up a tool by name.

        Args:
            name: The tool's registered name (matches the function name).

        Returns:
            The tool function, or None if not found.
        """
        return self._tools.get(name)

    def get_schemas(self, tool_names: list[str]) -> list[FunctionToolParam]:
        """Get OpenAI-compatible schemas for the given tool names.

        Args:
            tool_names: Names of tools to include.

        Returns:
            List of FunctionToolParam dicts for the OpenAI Responses API.

        Raises:
            KeyError: If a tool name is not registered.
        """
        schemas = []
        for name in tool_names:
            func = self._tools.get(name)
            if func is None:
                raise KeyError(f"Tool '{name}' not registered")
            schemas.append(func._tool_schema)  # type: ignore[attr-defined]
        return schemas

    async def execute(
        self,
        name: str,
        arguments_json: str,
        context: ToolContext | None = None,
    ) -> str:
        """Execute a tool by name with JSON arguments from the LLM.

        Parses the JSON arguments string and calls the tool. If the tool
        declares a ToolContext parameter, it is injected automatically.
        Exceptions are caught and returned as error strings so the LLM
        can see what went wrong.

        Args:
            name: The registered tool name.
            arguments_json: JSON string of arguments from the LLM.
            context: Optional ToolContext to inject into tools that
                declare a ToolContext-typed parameter.

        Returns:
            The tool's string result, or an error message.
        """
        func = self._tools.get(name)
        if func is None:
            return f"Error: tool '{name}' not found"

        try:
            arguments = json.loads(arguments_json)
        except json.JSONDecodeError as e:
            return f"Error: invalid arguments JSON: {e}"

        try:
            # Inject ToolContext if the tool declares it
            context_param = getattr(func, "_context_param", None)
            if context_param and context is not None:
                arguments[context_param] = context

            result = await func(**arguments)
            if not isinstance(result, str):
                raise TypeError(
                    f"Tool '{name}' must return str, got {type(result).__name__}"
                )
            return result
        except Exception as e:
            return f"Error executing tool '{name}': {e}"
