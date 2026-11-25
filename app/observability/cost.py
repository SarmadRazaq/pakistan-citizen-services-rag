"""Cost of one request, from token counts and the configured per-model list prices.

- `list_price_usd`: tokens x `Settings.model_prices` (USD per 1M input/output tokens). Models with
  no configured price count as 0, so the figure is only ever a lower bound of "what it would cost".
- `cost_usd`: what was actually paid. The NVIDIA free tier costs 0 unless `NVIDIA_BILLED=true`;
  the local Ollama fallback always costs 0.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from app.config import Settings

PER_MILLION = 1_000_000


@dataclass(frozen=True, slots=True)
class ModelUsage:
    model: str
    endpoint: str  # "nvidia" | "ollama"
    input_tokens: int
    output_tokens: int


def request_cost(settings: Settings, calls: Iterable[ModelUsage]) -> tuple[float, float]:
    """Return `(cost_usd, list_price_usd)` summed over every model call of one request."""
    cost = list_price = 0.0
    for call in calls:
        price = settings.model_prices.get(call.model)
        if price is None:
            continue
        priced = (call.input_tokens * price[0] + call.output_tokens * price[1]) / PER_MILLION
        list_price += priced
        if call.endpoint == "nvidia" and settings.nvidia_billed:
            cost += priced
    return cost, list_price
