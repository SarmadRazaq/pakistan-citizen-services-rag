from __future__ import annotations

from collections.abc import AsyncIterator, Callable

import httpx
import pytest
from fastapi import FastAPI

from app.api.deps import get_tenant
from app.config import Settings
from app.core.health import HealthService, Probe
from app.core.tenancy import Role, Scope, TenantContext
from app.main import create_app

_PROVIDER_ENV = (
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "LLM_BILLED",
    "LLM_TOKEN_PARAM",
    "OPENAI_API_KEY",
    "EMBEDDING_PROVIDER",
    "RERANKER_PROVIDER",
)


@pytest.fixture(autouse=True)
def _clean_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer's exported OPENAI_API_KEY / LLM_* variables must not leak into tests."""
    for name in _PROVIDER_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def settings() -> Settings:
    # Built explicitly so a developer's .env never leaks into tests.
    return Settings(_env_file=None, app_env="test", log_level="WARNING")  # type: ignore[call-arg]


def ok_probe(name: str, *, required: bool = True) -> Probe:
    async def check() -> tuple[bool, str | None]:
        return True, None

    return Probe(name, check, required)


def down_probe(name: str, *, required: bool = True) -> Probe:
    async def check() -> tuple[bool, str | None]:
        return False, "connection refused"

    return Probe(name, check, required)


@pytest.fixture
def make_app(settings: Settings) -> Callable[..., FastAPI]:
    def factory(probes: list[Probe] | None = None, tenant: TenantContext | None = None) -> FastAPI:
        health = HealthService(probes if probes is not None else [ok_probe("qdrant")], timeout_s=1.0)
        app = create_app(settings, health_service=health)
        if tenant is not None:
            app.dependency_overrides[get_tenant] = lambda: tenant
        return app

    return factory


@pytest.fixture
async def client(make_app: Callable[..., FastAPI]) -> AsyncIterator[httpx.AsyncClient]:
    app = make_app()
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
def user_a() -> TenantContext:
    return TenantContext.for_user("u_a", Role.USER, Scope.ALL)


@pytest.fixture
def admin() -> TenantContext:
    return TenantContext.for_user("u_admin", Role.ADMIN, Scope.ALL)
