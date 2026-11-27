"""Per-route rate limits (blueprint §5.8.6) on slowapi's limiter, applied by one ASGI middleware.

| Route                                   | Limit               | Keyed by                              |
|-----------------------------------------|---------------------|---------------------------------------|
| POST /ask, POST /ask/stream             | `RATE_LIMIT_ASK`    | `user_id` from the JWT, else client IP |
| POST /documents, PUT /documents/{id}    | `RATE_LIMIT_UPLOAD` | `user_id` from the JWT, else client IP |
| POST /auth/login                        | `RATE_LIMIT_LOGIN`  | client IP **and** the submitted email  |
| POST /auth/signup                       | `RATE_LIMIT_LOGIN`  | client IP                              |

A limited request gets **429** `rate_limited` as problem+json with `Retry-After`.

The middleware is appended as the *innermost* user middleware, so the request-id and CORS
middleware still wrap a 429 (it carries `X-Request-ID` and is readable by the browser). Routes are
matched by method and path, so the routers owned by other work packages need no decorators.

The JWT is only signature-checked here to pick the key (an invalid token falls back to the IP and is
rejected later by `get_tenant`). Emails are hashed before they become limiter keys. Storage is
in-process memory, which is correct for the single-worker deployment (§12.8).
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from dataclasses import dataclass
from urllib.parse import parse_qs

from fastapi import FastAPI
from limits import RateLimitItem, parse
from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.middleware import Middleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.config import Settings
from app.core.errors import AppError, ErrorCode, problem_response
from app.core.security import decode_access_token

LOGIN_BODY_MAX_BYTES = 16 * 1024  # larger login bodies are not parsed for the email key


@dataclass(frozen=True, slots=True)
class RateLimitRule:
    name: str
    methods: frozenset[str]
    path: re.Pattern[str]
    limit: RateLimitItem
    keys: tuple[str, ...]  # "user_or_ip", "ip", "email"
    needs_body: bool = False


def _client_ip(scope: Scope) -> str:
    client = scope.get("client")
    return str(client[0]) if client else "unknown"


def _header(scope: Scope, name: bytes) -> str:
    for key, value in scope.get("headers") or []:
        if key.lower() == name:
            return bytes(value).decode("latin-1")
    return ""


def _email_from_form(scope: Scope, body: bytes | None) -> str | None:
    if not body or "application/x-www-form-urlencoded" not in _header(scope, b"content-type"):
        return None
    try:
        values = parse_qs(body.decode("utf-8"), max_num_fields=20).get("username")
    except (UnicodeDecodeError, ValueError):
        return None
    email = values[0].strip().lower() if values else ""
    return hashlib.sha256(email.encode("utf-8")).hexdigest() if email else None


def build_rules(settings: Settings) -> list[RateLimitRule]:
    prefix = re.escape(settings.api_prefix.rstrip("/"))
    ask, upload, login = (
        parse(settings.rate_limit_ask),
        parse(settings.rate_limit_upload),
        parse(settings.rate_limit_login),
    )
    post, put = frozenset({"POST"}), frozenset({"PUT"})
    return [
        RateLimitRule("ask", post, re.compile(rf"^{prefix}/ask(/stream)?/?$"), ask, ("user_or_ip",)),
        RateLimitRule("upload", post, re.compile(rf"^{prefix}/documents/?$"), upload, ("user_or_ip",)),
        RateLimitRule("upload", put, re.compile(rf"^{prefix}/documents/[^/]+/?$"), upload, ("user_or_ip",)),
        RateLimitRule(
            "login", post, re.compile(rf"^{prefix}/auth/login/?$"), login, ("ip", "email"), needs_body=True
        ),
        RateLimitRule("signup", post, re.compile(rf"^{prefix}/auth/signup/?$"), login, ("ip",)),
    ]


class RateLimitMiddleware:
    def __init__(self, app: ASGIApp, *, settings: Settings, limiter: Limiter) -> None:
        self.app = app
        self._settings = settings
        self._strategy = limiter.limiter
        self._rules = build_rules(settings)

    def _user_or_ip(self, scope: Scope) -> str:
        scheme, _, token = _header(scope, b"authorization").partition(" ")
        if scheme.lower() == "bearer" and token.strip():
            try:
                return "user:" + decode_access_token(self._settings, token.strip()).sub
            except AppError:
                pass
        return "ip:" + _client_ip(scope)

    def _keys(self, rule: RateLimitRule, scope: Scope, body: bytes | None) -> list[str]:
        keys: list[str] = []
        for kind in rule.keys:
            if kind == "user_or_ip":
                keys.append(self._user_or_ip(scope))
            elif kind == "ip":
                keys.append("ip:" + _client_ip(scope))
            elif kind == "email" and (email := _email_from_form(scope, body)):
                keys.append("email:" + email)
        return keys

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        method, path = scope.get("method", ""), scope.get("path", "")
        rule = next((r for r in self._rules if method in r.methods and r.path.match(path)), None)
        if rule is None:
            await self.app(scope, receive, send)
            return

        body: bytes | None = None
        if rule.needs_body:
            body, receive = await _buffer_body(receive)

        limited = False
        retry_after = 0.0
        for key in self._keys(rule, scope, body):  # every key is hit: IP and email both count
            if not self._strategy.hit(rule.limit, rule.name, key):
                limited = True
                stats = self._strategy.get_window_stats(rule.limit, rule.name, key)
                retry_after = max(retry_after, stats.reset_time - time.time())
        if limited:
            seconds = max(1, math.ceil(retry_after))
            response = problem_response(
                ErrorCode.RATE_LIMITED,
                f"Too many requests. Try again in {seconds} seconds.",
                extra={"retry_after": seconds},
                headers={"Retry-After": str(seconds)},
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


async def _buffer_body(receive: Receive) -> tuple[bytes | None, Receive]:
    """Read the request body once and hand back a `receive` that replays it to the app."""
    chunks: list[bytes] = []
    size = 0
    more = True
    first: list[Message] = []
    while more:
        message = await receive()
        if message["type"] != "http.request":
            first.append(message)  # e.g. a disconnect: replay it as is
            break
        chunk = message.get("body", b"")
        chunks.append(chunk)
        size += len(chunk)
        more = bool(message.get("more_body", False))
    body = b"".join(chunks)
    replay: list[Message] = [{"type": "http.request", "body": body, "more_body": False}, *first]

    async def replay_receive() -> Message:
        if replay:
            return replay.pop(0)
        return await receive()

    return (body if size <= LOGIN_BODY_MAX_BYTES else None), replay_receive


def install_rate_limits(app: FastAPI, settings: Settings) -> Limiter:
    """Add the rate-limit middleware as the innermost user middleware. Call before the app starts.

    Also stores the slowapi `Limiter` on `app.state.limiter` (slowapi's convention), so a route can
    still use `@app.state.limiter.limit(...)` if it ever needs a custom limit.
    """
    if app.middleware_stack is not None:
        raise RuntimeError("install_rate_limits must be called before the app handles its first request")
    limiter = Limiter(key_func=get_remote_address, storage_uri="memory://", strategy="moving-window")
    app.state.limiter = limiter
    app.user_middleware.append(Middleware(RateLimitMiddleware, settings=settings, limiter=limiter))
    return limiter
