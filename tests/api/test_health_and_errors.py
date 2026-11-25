from __future__ import annotations

from collections.abc import Callable

import httpx
from fastapi import FastAPI

from app.core.errors import PROBLEM_JSON, ErrorCode
from app.core.tenancy import TenantContext
from tests.conftest import down_probe, ok_probe


def _client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
    )


async def test_live(client: httpx.AsyncClient) -> None:
    r = await client.get("/health/live")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


async def test_ready_ok_when_required_up(make_app: Callable[..., FastAPI]) -> None:
    app = make_app([ok_probe("qdrant"), ok_probe("llm_api"), down_probe("ollama", required=False)])
    async with _client(app) as c:
        r = await c.get("/health/ready")
    assert r.status_code == 200
    assert r.json()["status"] == "degraded"


async def test_ready_503_when_required_down(make_app: Callable[..., FastAPI]) -> None:
    app = make_app([ok_probe("qdrant"), down_probe("llm_api")])
    async with _client(app) as c:
        r = await c.get("/health/ready")
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "unavailable"
    assert {c["name"]: c["ok"] for c in body["components"]} == {"qdrant": True, "llm_api": False}


async def test_missing_nvidia_key_makes_readiness_fail_without_network() -> None:
    import httpx as _httpx

    from app.config import Settings
    from app.core.health import HealthService, default_probes

    settings = Settings(_env_file=None, nvidia_api_key=None)  # type: ignore[call-arg]
    async with _httpx.AsyncClient() as client:
        probes = {p.name: p for p in default_probes(settings, client)}
        ok, detail = await probes["llm_api"].check()
    assert not ok
    assert detail == "LLM_API_KEY is not set"
    assert HealthService  # imported for type completeness


async def test_request_id_is_echoed_or_created(client: httpx.AsyncClient) -> None:
    r = await client.get("/health/live", headers={"X-Request-ID": "abc-123"})
    assert r.headers["x-request-id"] == "abc-123"
    r = await client.get("/health/live", headers={"X-Request-ID": "bad id with spaces"})
    assert r.headers["x-request-id"] != "bad id with spaces"
    assert len(r.headers["x-request-id"]) == 32


async def test_protected_route_without_auth_is_401_problem(client: httpx.AsyncClient) -> None:
    r = await client.get("/api/v1/documents")
    assert r.status_code == 401
    assert r.headers["content-type"].startswith(PROBLEM_JSON)
    body = r.json()
    assert body["code"] == ErrorCode.UNAUTHORIZED.slug
    assert body["request_id"] == r.headers["x-request-id"]
    assert set(body) == {"type", "title", "status", "code", "detail", "request_id", "extra", "errors"}


async def test_unknown_route_is_404_problem(client: httpx.AsyncClient) -> None:
    r = await client.get("/api/v1/nope")
    assert r.status_code == 404
    assert r.json()["code"] == "not_found"


async def test_validation_error_is_422_problem_with_field_errors(
    make_app: Callable[..., FastAPI], user_a: TenantContext
) -> None:
    async with _client(make_app(tenant=user_a)) as c:
        r = await c.post("/api/v1/ask", json={"question": "   "})
    assert r.status_code == 422
    body = r.json()
    assert body["code"] == "validation_failed"
    assert body["errors"] and body["errors"][0]["loc"][-1] == "question"


async def test_stub_routes_return_501_problem(
    make_app: Callable[..., FastAPI], user_a: TenantContext
) -> None:
    async with _client(make_app(tenant=user_a)) as c:
        r = await c.post("/api/v1/ask", json={"question": "Passport ki fee kitni hai?"})
    assert r.status_code == 501
    assert r.json()["code"] == "not_implemented"


async def test_unhandled_exception_is_generic_500(make_app: Callable[..., FastAPI]) -> None:
    app = make_app()

    @app.get("/boom")
    async def boom() -> None:
        raise RuntimeError("secret internal detail")

    async with _client(app) as c:
        r = await c.get("/boom")
    assert r.status_code == 500
    assert "secret internal detail" not in r.text
    assert r.json()["code"] == "internal_error"
