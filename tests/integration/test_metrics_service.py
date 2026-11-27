"""MetricsService over an in-memory SQLite database (B3)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from itertools import count

import pytest

from app.core.tenancy import Role, Scope, TenantContext, system_context
from app.db.models import RequestLog
from app.db.session import Database
from app.observability.metrics import MetricsService, percentile

NOW = datetime(2026, 1, 10, 12, 0, tzinfo=UTC)
_ids = count()

USER_A = TenantContext.for_user("u_a", Role.USER, Scope.ALL)
ADMIN = TenantContext.for_user("u_admin", Role.ADMIN, Scope.ALL)


def row(
    user: str = "u_a",
    *,
    age: timedelta = timedelta(minutes=5),
    latency: int = 100,
    status: int = 200,
    route: str = "ask",
    model: str | None = "m1",
    degraded: bool = False,
    tin: int = 100,
    tout: int = 50,
    cost: float = 0.0,
    price: float = 0.0,
) -> RequestLog:
    return RequestLog(
        request_id=f"r{next(_ids)}",
        user_id=user,
        route=route,
        status=status,
        latency_ms=latency,
        llm_model=model,
        degraded=degraded,
        input_tokens=tin,
        output_tokens=tout,
        cost_usd=cost,
        list_price_usd=price,
        created_at=NOW - age,
    )


@pytest.fixture
async def db() -> AsyncIterator[Database]:
    database = Database("sqlite+aiosqlite:///:memory:")
    await database.create_all()
    yield database
    await database.dispose()


async def seed(db: Database, *rows: RequestLog) -> MetricsService:
    async with db.session(system_context("test seed")) as s:
        s.add_all(rows)
    return MetricsService(db, clock=lambda: NOW)


def test_percentile_interpolates() -> None:
    assert percentile([], 50) is None
    assert percentile([7], 95) == 7
    assert percentile([10, 20, 30, 40], 50) == 25
    assert percentile(list(range(1, 101)), 95) == pytest.approx(95.05)


async def test_empty_window(db: Database) -> None:
    out = await (await seed(db)).summary(USER_A, "24h")
    assert out.requests == 0 and out.p50_latency_ms is None and out.p95_latency_ms is None
    assert out.avg_input_tokens is None and out.error_rate == 0.0 and out.provider_mix is None


async def test_percentiles_and_averages(db: Database) -> None:
    svc = await seed(db, *[row(latency=ms, tin=100, tout=40) for ms in (100, 200, 300, 400)])
    out = await svc.summary(USER_A, "1h")
    assert out.requests == 4 and out.scope == "own"
    assert out.p50_latency_ms == 250 and out.p95_latency_ms == pytest.approx(385)
    assert out.avg_input_tokens == 100 and out.avg_output_tokens == 40


async def test_window_filtering(db: Database) -> None:
    svc = await seed(
        db,
        row(age=timedelta(minutes=30)),
        row(age=timedelta(hours=3)),
        row(age=timedelta(days=3)),
        row(age=timedelta(days=30)),
    )
    assert (await svc.summary(USER_A, "1h")).requests == 1
    assert (await svc.summary(USER_A, "24h")).requests == 2
    assert (await svc.summary(USER_A, "7d")).requests == 3


async def test_own_vs_global_scope(db: Database) -> None:
    svc = await seed(db, row("u_a", model="m1"), row("u_b", model="m2"), row("u_b", model="m2"))
    own = await svc.summary(USER_A, "24h")
    assert own.requests == 1 and own.scope == "own" and own.provider_mix is None
    glob = await svc.summary(ADMIN, "24h")
    assert glob.requests == 3 and glob.scope == "global"
    assert glob.provider_mix == {"m1": 1, "m2": 2}


async def test_error_and_fallback_rates(db: Database) -> None:
    svc = await seed(
        db,
        row(),
        row(degraded=True),
        row(status=503, model=None),
        row(status=429, model=None),
    )
    out = await svc.summary(USER_A, "24h")
    assert out.error_rate == 0.25  # only the 503 is a server error
    assert out.fallback_rate == 0.5  # 1 degraded of 2 completed asks


async def test_cost_per_question(db: Database) -> None:
    svc = await seed(
        db,
        row(cost=0.02, price=0.04),
        row(cost=0.0, price=0.02),
        row(route="search", model=None, tin=0, tout=0),
        row(status=503, model=None, tin=0, tout=0),
    )
    out = await svc.summary(USER_A, "24h")
    assert out.cost_per_question_usd == pytest.approx(0.01)
    assert out.list_price_per_question_usd == pytest.approx(0.03)
    assert out.avg_input_tokens == 100  # search and failed rows do not dilute token averages
