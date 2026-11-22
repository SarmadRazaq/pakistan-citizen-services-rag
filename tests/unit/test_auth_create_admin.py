"""scripts/create_admin.py: the only way to grant the admin role (§5.8.1)."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from types import ModuleType

import pytest

from app.config import Settings
from app.core.auth_service import AuthService
from app.core.tenancy import Role
from app.db.session import Database

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "create_admin.py"
PASSWORD = "a long admin password"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("create_admin", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        app_env="test",
        jwt_secret="test-jwt-secret-that-is-long-enough-for-hs256-0123456789",
        database_url=f"sqlite+aiosqlite:///{(tmp_path / 'app.sqlite3').as_posix()}",
    )


async def _role(settings: Settings, email: str) -> Role | None:
    db = Database(settings.database_url)
    try:
        user = await AuthService(settings, db).find_user(email)
        return user.role if user else None
    finally:
        await db.dispose()


async def test_creates_then_promotes_then_refuses_last_demote(settings: Settings) -> None:
    script = _load()
    prompts: list[str] = []

    def read(prompt: str) -> str:
        prompts.append(prompt)
        return PASSWORD

    assert (
        await script.run(
            "Admin@Example.org", display_name="Ops", demote=False, settings=settings, read_password=read
        )
        == 0
    )
    assert len(prompts) == 2  # entered twice, through getpass
    assert await _role(settings, "admin@example.org") is Role.ADMIN

    prompts.clear()
    assert (
        await script.run(
            "admin@example.org", display_name=None, demote=False, settings=settings, read_password=read
        )
        == 0
    )
    assert prompts == []  # promoting an existing account never asks for a password

    assert (
        await script.run("admin@example.org", display_name=None, demote=True, settings=settings) == 1
    )  # last admin


def test_password_is_never_a_cli_argument() -> None:
    script = _load()
    source = inspect.getsource(script)
    assert "getpass" in source
    assert "--password" not in source
    with pytest.raises(SystemExit):
        script.main(["admin@example.org", "--password", "hunter2hunter2"])


def test_prompt_rejects_short_or_mismatched_passwords() -> None:
    script = _load()
    with pytest.raises(SystemExit, match="at least"):
        script.prompt_password(lambda _: "short")
    answers = iter([PASSWORD, PASSWORD + "x"])
    with pytest.raises(SystemExit, match="do not match"):
        script.prompt_password(lambda _: next(answers))
