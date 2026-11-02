"""Hardcoded token rates for routing cost estimates.

``ModelRate.input_per_mtok`` / ``output_per_mtok`` are always **USD per
1 million tokens (MTok)**. Provider list prices sometimes use other
units (e.g. TypeSafe quotes **USD per billion tokens / BTok**); comments
on each entry record the published unit and the converted MTok value.

Rates dated 2026-10-30. Update when list prices change — no live fetches.
"""

from __future__ import annotations

from dataclasses import dataclass

# As of 2026-10-30.
PRICE_DATE = "2026-10-30"


@dataclass(frozen=True)
class ModelRate:
    """USD per 1M tokens (MTok) for input and output."""

    input_per_mtok: float  # USD / MTok
    output_per_mtok: float  # USD / MTok
    label: str


# Exact model ids used by this workshop.
# Stored values are USD/MTok. Comments show the provider's published unit.
RATES: dict[str, ModelRate] = {
    # TypeSafe Jev (example decision model) — published $42 / BTok input,
    # $0 output (docs.typesafe.ai/models). $42/BTok = $0.042/MTok. Input only.
    "jev": ModelRate(0.042, 0.0, "TypeSafe Jev"),
    "jev-latest": ModelRate(0.042, 0.0, "TypeSafe Jev"),
    "jev-preview": ModelRate(0.042, 0.0, "TypeSafe Jev"),
    "jev-1.14.0": ModelRate(0.042, 0.0, "TypeSafe Jev 1.14"),
    "jev-1.13.0": ModelRate(0.042, 0.0, "TypeSafe Jev 1.13"),
    "typesafe/jev-1.14.0": ModelRate(0.042, 0.0, "TypeSafe Jev 1.14"),
    "typesafe/jev-latest": ModelRate(0.042, 0.0, "TypeSafe Jev"),
    # OpenAI — published USD / MTok (developers.openai.com pricing).
    "gpt-6-luna": ModelRate(0.10, 0.50, "OpenAI gpt-6-luna"),  # $0.10 / $0.50 per MTok
    "gpt-4o-mini": ModelRate(0.15, 0.60, "OpenAI gpt-4o-mini"),  # $0.15 / $0.60 per MTok
    "gpt-4o": ModelRate(2.50, 10.00, "OpenAI gpt-4o"),  # $2.50 / $10.00 per MTok
    # Anthropic — published USD / MTok (docs.anthropic.com pricing).
    "claude-sonnet-4-6": ModelRate(3.00, 15.00, "Anthropic Claude Sonnet 4.6"),  # $3 / $15 per MTok
}


def resolve_rate(model: str) -> ModelRate:
    """Look up a model rate; supports jev-* / typesafe/jev-* fallback.

    Args:
        model: Provider model id from the API response or config.

    Returns:
        ModelRate for cost estimation (USD/MTok fields).

    Raises:
        KeyError: If the model is unknown.
    """
    if model in RATES:
        return RATES[model]
    # LiteLLM ids like typesafe/jev-1.14.0
    bare = model.split("/", 1)[-1] if "/" in model else model
    if bare in RATES:
        return RATES[bare]
    # Versioned Jev ids like jev-1.14.0-rc1
    if bare.startswith("jev-1.14") or bare.startswith("jev-1.13"):
        return RATES["jev"]
    if bare.startswith("jev"):
        return RATES["jev"]
    known = ", ".join(sorted(RATES))
    raise KeyError(
        f"No hardcoded rate for model {model!r}. "
        f"Known keys: {known}. Update helpdesk.pricing.RATES."
    )


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Estimate USD cost for a token count under hardcoded rates.

    Args:
        model: Model id.
        input_tokens: Prompt / input tokens.
        output_tokens: Completion / output tokens.

    Returns:
        Estimated cost in USD.
    """
    rate = resolve_rate(model)
    return (
        input_tokens * rate.input_per_mtok
        + output_tokens * rate.output_per_mtok
    ) / 1_000_000


def format_rates_footer(models: list[str]) -> str:
    """Human-readable rate lines for the models used in a bench run."""
    lines = [f"Price constants as of {PRICE_DATE} (USD per MTok):"]
    seen: set[str] = set()
    for model in models:
        rate = resolve_rate(model)
        key = rate.label
        if key in seen:
            continue
        seen.add(key)
        extra = ""
        if rate.label.startswith("TypeSafe"):
            # Same rate, provider's published unit.
            extra = " (= $42/BTok in, $0 out)"
        lines.append(
            f"  {rate.label} ({model}): "
            f"in ${rate.input_per_mtok:g}/MTok / "
            f"out ${rate.output_per_mtok:g}/MTok{extra}"
        )
    return "\n".join(lines)
