from __future__ import annotations

import httpx
from fastapi import FastAPI, Response

from app.config import Settings
from app.core.middleware import SecurityHeadersMiddleware

BASE = {
    "x-content-type-options": "nosniff",
    "x-frame-options": "DENY",
    "referrer-policy": "no-referrer",
    "cross-origin-opener-policy": "same-origin",
}


async def test_security_headers_on_success_and_error(client: httpx.AsyncClient) -> None:
    for path in ("/health/live", "/no-such-route"):
        r = await client.get(path)
        for name, value in BASE.items():
            assert r.headers[name] == value
        assert "strict-transport-security" not in r.headers


async def test_hsts_only_in_prod() -> None:
    from app.main import create_app

    settings = Settings(  # type: ignore[call-arg]
        _env_file=None,
        app_env="prod",
        jwt_secret="x" * 48,
        llm_api_key="sk-test",
        log_level="WARNING",
    )
    app = create_app(settings)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/health/live")
    assert r.headers["strict-transport-security"] == "max-age=63072000; includeSubDomains"
    assert r.headers["x-frame-options"] == "DENY"


async def test_route_set_header_is_not_overridden() -> None:
    app = FastAPI()
    app.add_middleware(SecurityHeadersMiddleware)

    @app.get("/x")
    def x() -> Response:
        return Response("ok", headers={"X-Frame-Options": "SAMEORIGIN"})

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/x")
    assert r.headers.get_list("x-frame-options") == ["SAMEORIGIN"]
    assert r.headers["x-content-type-options"] == "nosniff"
