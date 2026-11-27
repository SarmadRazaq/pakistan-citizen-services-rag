"""GET /api/v1/metrics/summary through the real app."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable

import httpx
import pytest
from fastapi import FastAPI

from app.core.tenancy import TenantContext, system_context
from app.db.models import RequestLog
from app.db.session import Database
from app.observability.metrics import MetricsService
from app.schemas import MetricsSummary


@pytest.fixture
async def db() -> AsyncIterator[Database]:
    database = Database("sqlite+aiosqlite:///:memory:")
    await database.create_all()
    async with database.session(system_context("test seed")) as s:
        for i, user in enumerate(["u_a", "u_b"]):
            s.add(
                RequestLog(
                    request_id=f"r{i}",
                    user_id=user,
                    route="ask",
                    status=200,
                    latency_ms=100 * (i + 1),
                    llm_model=f"m{i}",
                    input_tokens=10,
                    output_tokens=5,
                )
            )
    yield database
    await database.dispose()


async def _get(
    make_app: Callable[..., FastAPI], db: Database, tenant: TenantContext | None, url: str
) -> httpx.Response:
    app = make_app(tenant=tenant)
    app.state.metrics = MetricsService(db)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        return await c.get(url)


async def test_user_sees_own_metrics(
    make_app: Callable[..., FastAPI], db: Database, user_a: TenantContext
) -> None:
    r = await _get(make_app, db, user_a, "/api/v1/metrics/summary?window=1h")
    assert r.status_code == 200, r.text
    body = MetricsSummary.model_validate(r.json())
    assert body.scope == "own" and body.requests == 1 and body.provider_mix is None


async def test_admin_sees_global_provider_mix(
    make_app: Callable[..., FastAPI], db: Database, admin: TenantContext
) -> None:
    r = await _get(make_app, db, admin, "/api/v1/metrics/summary")
    body = MetricsSummary.model_validate(r.json())
    assert body.scope == "global" and body.requests == 2 and body.window == "24h"
    assert body.provider_mix == {"m0": 1, "m1": 1}


async def test_requires_auth(make_app: Callable[..., FastAPI], db: Database) -> None:
    r = await _get(make_app, db, None, "/api/v1/metrics/summary")
    assert r.status_code == 401


async def test_bad_window_is_422(
    make_app: Callable[..., FastAPI], db: Database, user_a: TenantContext
) -> None:
    r = await _get(make_app, db, user_a, "/api/v1/metrics/summary?window=30d")
    assert r.status_code == 422
