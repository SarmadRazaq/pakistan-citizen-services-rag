from __future__ import annotations

from typing import Any

import pytest

from app.config import Settings

GOOD: dict[str, Any] = {
    "jwt_secret": "x" * 48,
    "nvidia_api_key": "nvapi-test",
    "cors_origins": ["https://app.example.org"],
    "log_question_text": False,
}


def prod(**overrides: Any) -> Settings:
    return Settings(_env_file=None, app_env="prod", **{**GOOD, **overrides})  # type: ignore[call-arg]


def test_valid_prod_config_passes() -> None:
    assert prod().app_env == "prod"


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"jwt_secret": "change-me-in-.env"}, "JWT_SECRET is still the default"),
        ({"jwt_secret": "short"}, "JWT_SECRET must be at least 32"),
        ({"nvidia_api_key": None}, "LLM_API_KEY"),
        ({"cors_origins": []}, "CORS_ORIGINS must not be empty"),
        ({"cors_origins": ["*"]}, "CORS_ORIGINS must not contain"),
        ({"log_question_text": True}, "LOG_QUESTION_TEXT"),
    ],
)
def test_each_rule_fails_alone(overrides: dict[str, Any], fragment: str) -> None:
    with pytest.raises(ValueError) as exc:
        prod(**overrides)
    message = str(exc.value)
    assert fragment in message
    assert message.count(";") == 0  # only one problem listed


def test_all_problems_are_listed_without_secret_values() -> None:
    with pytest.raises(ValueError) as exc:
        prod(jwt_secret="tiny-secret", nvidia_api_key=None, cors_origins=["*"], log_question_text=True)
    message = str(exc.value)
    for fragment in ("JWT_SECRET", "LLM_API_KEY", "CORS_ORIGINS", "LOG_QUESTION_TEXT"):
        assert fragment in message
    assert "tiny-secret" not in message


@pytest.mark.parametrize("env", ["dev", "test"])
def test_non_prod_defaults_pass(env: str) -> None:
    assert Settings(_env_file=None, app_env=env).app_env == env  # type: ignore[call-arg]
