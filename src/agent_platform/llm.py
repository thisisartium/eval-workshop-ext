"""LLM provider factory.

Centralizes provider creation with environment-based configuration.
Supports OpenAI and Anthropic — provider is inferred from the model name
or specified explicitly.
"""

import os
from typing import Any

from dotenv import load_dotenv

from .providers import LLMProvider


class MissingAPIKeyError(RuntimeError):
    """Raised when the required API key for the selected model is not set."""


def create_provider(provider: str | None = None, model: str | None = None) -> LLMProvider:
    """Create an LLM provider instance.

    Provider can be specified explicitly or inferred from the model name.
    If neither is given, defaults to OpenAI.

    Args:
        provider: Explicit provider name ("openai" or "anthropic").
        model: Model name used to infer provider if not specified.
            Models starting with "claude" use Anthropic.

    Returns:
        An LLMProvider instance ready to use.

    Raises:
        MissingAPIKeyError: If the required API key for the inferred
            provider is not set in the environment.
    """
    load_dotenv(override=True)

    if provider is None and model is not None:
        provider = "anthropic" if model.startswith("claude") else "openai"

    if provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise MissingAPIKeyError(
                f"Model '{model or 'claude-*'}' requires ANTHROPIC_API_KEY. "
                "Set it in your environment or .env file.\n"
                "Get a key at: https://console.anthropic.com/settings/keys"
            )
        from anthropic import AsyncAnthropic

        from .providers_anthropic import AnthropicProvider

        return AnthropicProvider(AsyncAnthropic())
    else:
        if not os.environ.get("OPENAI_API_KEY"):
            raise MissingAPIKeyError(
                f"Model '{model or 'gpt-*'}' requires OPENAI_API_KEY. "
                "Set it in your environment or .env file.\n"
                "Get a key at: https://platform.openai.com/api-keys"
            )
        from openai import AsyncOpenAI

        from .providers_openai import OpenAIProvider

        return OpenAIProvider(AsyncOpenAI())


# Backwards compatibility
def create_client() -> Any:
    """Create an AsyncOpenAI client with API key loaded from environment.

    .. deprecated::
        Use ``create_provider()`` instead.
    """
    from openai import AsyncOpenAI

    load_dotenv(override=True)
    return AsyncOpenAI()
