from __future__ import annotations

import pytest

from app.config import Settings
from app.core.logging import REDACTED, redact_text, redaction_processor


def test_defaults_match_blueprint(settings: Settings) -> None:
    assert settings.public_owner_id == "u_public"
    assert settings.response_cache == "eval_only"
    assert settings.jurisdictions == ["federal", "punjab", "sindh", "kp"]
    assert settings.max_upload_bytes == 50 * 1024 * 1024


def test_secrets_never_appear_in_repr() -> None:
    s = Settings(_env_file=None, jwt_secret="super-secret-value")  # type: ignore[call-arg]
    assert "super-secret-value" not in repr(s)
    assert s.jwt_secret.get_secret_value() == "super-secret-value"


def test_csv_lists_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GENERATION_MODELS", "model-a, model-b")
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.generation_models == ["model-a", "model-b"]


@pytest.mark.parametrize(
    "secret",
    [
        "sk-abcdefghijklmnopqrstuvwxyz",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1X2EifQ.c2lnbmF0dXJlLXBhcnQ",
        "35202-1234567-1",  # CNIC-shaped number
        "0x" + "a" * 64,
        "sk-" + "x" * 40,
    ],
)
def test_secret_patterns_are_redacted(secret: str) -> None:
    assert secret not in redact_text(f"value={secret} end")


def test_sensitive_keys_are_redacted() -> None:
    event = {"event": "login", "password": "hunter2", "headers": {"Authorization": "Bearer x"}}
    out = redaction_processor(None, "info", event)
    assert out["password"] == REDACTED
    assert out["headers"]["Authorization"] == REDACTED
    assert out["event"] == "login"
