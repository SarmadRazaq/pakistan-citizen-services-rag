"""Request id, timing and access-log middleware (pure ASGI, so it also wraps error responses)."""

from __future__ import annotations

import re
import time
import uuid

import structlog
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.context import set_request_id

log = structlog.get_logger("access")

REQUEST_ID_HEADER = "x-request-id"
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def new_request_id() -> str:
    return uuid.uuid4().hex


class RequestContextMiddleware:
    """Echo a well-formed client X-Request-ID (or create one), time the request, log one access line."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = dict(scope.get("headers") or []).get(REQUEST_ID_HEADER.encode())
        candidate = incoming.decode("latin-1") if incoming else ""
        request_id = candidate if _VALID_REQUEST_ID.match(candidate) else new_request_id()
        set_request_id(request_id)
        started = time.perf_counter()
        status_code = 500

        async def send_with_id(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
                headers = list(message.get("headers", []))
                headers.append((REQUEST_ID_HEADER.encode(), request_id.encode()))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            log.info(
                "request",
                method=scope.get("method"),
                route=scope.get("path"),
                status=status_code,
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
            )
            set_request_id(None)
