from __future__ import annotations

import pytest

from app.config import Settings
from app.observability.cost import ModelUsage, request_cost

PRICES = {"big": (1.0, 2.0), "small": (0.5, 1.0)}


def make_settings(**kw: object) -> Settings:
    return Settings(_env_file=None, app_env="test", **kw)  # type: ignore[call-arg]


def test_no_prices_means_zero() -> None:
    assert request_cost(make_settings(), [ModelUsage("big", "hosted", 1000, 1000)]) == (0.0, 0.0)


def test_unbilled_hosted_calls_have_list_price_but_no_cost() -> None:
    cost, listed = request_cost(
        make_settings(model_prices=PRICES, llm_billed=False),
        [ModelUsage("big", "hosted", 1_000_000, 500_000)],
    )
    assert cost == 0.0 and listed == pytest.approx(2.0)


def test_hosted_calls_are_billed_by_default_and_summed() -> None:
    calls = [ModelUsage("small", "hosted", 2_000_000, 0), ModelUsage("big", "hosted", 0, 1_000_000)]
    cost, listed = request_cost(make_settings(model_prices=PRICES), calls)
    assert cost == pytest.approx(3.0) and listed == pytest.approx(3.0)


def test_local_fallback_is_free_and_unknown_model_ignored() -> None:
    calls = [ModelUsage("big", "ollama", 1_000_000, 0), ModelUsage("mystery", "hosted", 5, 5)]
    cost, listed = request_cost(make_settings(model_prices=PRICES, llm_billed=True), calls)
    assert cost == 0.0 and listed == pytest.approx(1.0)


def test_model_prices_from_env_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_PRICES", '{"big": [1.5, 3]}')
    assert make_settings().model_prices == {"big": (1.5, 3.0)}
