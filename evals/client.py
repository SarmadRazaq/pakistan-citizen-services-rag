"""HTTP client for the running service: `/auth/login`, `/search`, `/ask`, with an on-disk response cache.

The cache (`evals/.cache/api/<endpoint>/<sha256>.json`) is keyed by the endpoint, the JSON request body
and the X-Eval-Config header, so the same request under the same config is answered from disk. It
stores the raw response body, the client-measured latency and the X-Eval-Config-Applied header, so an
`--offline` re-run reproduces every number (latency included) without the service.

- Login is lazy (only when a request actually goes to the network) and repeated once on a 401, since
  the token lives 30 minutes and a full run can take longer.
- 429 is retried, honouring Retry-After (the /ask rate limit is 30/minute); 5xx is retried with backoff.
- Only 200 responses are cached; an error is reported for that item and retried on the next run.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx

from evals.eval_config import APPLIED_HEADER, EVAL_HEADER
from evals.llm import retry_after

Endpoint = Literal["search", "ask"]
Sleep = Callable[[float], Awaitable[None]]


class ServiceError(RuntimeError):
    pass


@dataclass
class ApiResult:
    endpoint: Endpoint
    body: dict[str, Any] | None
    status: int | None
    latency_ms: float | None
    cached: bool
    applied: str | None = None  # X-Eval-Config-Applied echoed by the service, if any
    error: str | None = None


def request_key(endpoint: str, body: dict[str, Any], header: str | None) -> str:
    raw = json.dumps({"endpoint": endpoint, "body": body, "eval_config": header}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


class ServiceClient:
    def __init__(
        self,
        base_url: str,
        *,
        cache_dir: Path,
        api_prefix: str = "/api/v1",
        email: str | None = None,
        password: str | None = None,
        use_cache: bool = True,
        offline: bool = False,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_s: float = 180.0,
        max_attempts: int = 5,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.prefix = "/" + api_prefix.strip("/") if api_prefix.strip("/") else ""
        self.cache_dir = cache_dir
        self.email = email
        self.password = password
        self.use_cache = use_cache
        self.offline = offline
        self.max_attempts = max_attempts
        self._sleep = sleep
        self._transport = transport
        self._timeout = timeout_s
        self._client: httpx.AsyncClient | None = None
        self._token: str | None = None
        self._login_error: str | None = None
        self._login_lock = asyncio.Lock()
        self.network_calls = 0

    async def __aenter__(self) -> ServiceClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url, timeout=self._timeout, transport=self._transport
            )
        return self._client

    async def login(self, *, force: bool = False) -> str:
        async with self._login_lock:
            if self._token and not force:
                return self._token
            if self._login_error:  # sticky: never hammer /auth/login (5/minute) after a failure
                raise ServiceError(self._login_error)
            try:
                self._token = await self._login()
            except ServiceError as exc:
                self._login_error = str(exc)
                raise
            return self._token

    async def _login(self) -> str:
        if not self.email or not self.password:
            raise ServiceError("no credentials: pass --email/--password or set EVAL_EMAIL / EVAL_PASSWORD")
        r = await self._http().post(
            f"{self.prefix}/auth/login",
            data={"username": self.email, "password": self.password},
        )
        if r.status_code != 200:
            raise ServiceError(f"login failed: HTTP {r.status_code}")
        token = r.json().get("access_token")
        if not token:
            raise ServiceError("login response has no access_token")
        return str(token)

    # ------------------------------------------------------------ cache

    def _path(self, endpoint: str, key: str) -> Path:
        return self.cache_dir / endpoint / f"{key}.json"

    def _read(self, endpoint: Endpoint, key: str) -> ApiResult | None:
        path = self._path(endpoint, key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return ApiResult(
            endpoint=endpoint,
            body=data.get("body"),
            status=data.get("status", 200),
            latency_ms=data.get("latency_ms"),
            cached=True,
            applied=data.get("applied"),
        )

    def _write(self, endpoint: Endpoint, key: str, request: dict[str, Any], result: ApiResult) -> None:
        path = self._path(endpoint, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "endpoint": endpoint,
            "request": request,
            "status": result.status,
            "latency_ms": result.latency_ms,
            "applied": result.applied,
            "body": result.body,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)

    # ------------------------------------------------------------ requests

    async def call(self, endpoint: Endpoint, body: dict[str, Any], *, eval_header: str | None) -> ApiResult:
        key = request_key(endpoint, body, eval_header)
        if self.use_cache or self.offline:
            hit = self._read(endpoint, key)
            if hit is not None:
                return hit
        if self.offline:
            return ApiResult(endpoint, None, None, None, cached=False, error="cache_miss (offline)")
        try:
            result = await self._post(endpoint, body, eval_header)
        except (ServiceError, httpx.HTTPError) as exc:
            return ApiResult(endpoint, None, None, None, cached=False, error=str(exc) or type(exc).__name__)
        if result.status == 200 and result.body is not None:
            self._write(endpoint, key, body, result)
        return result

    async def _post(self, endpoint: Endpoint, body: dict[str, Any], eval_header: str | None) -> ApiResult:
        token = await self.login()
        relogged = False
        last = ""
        for attempt in range(1, self.max_attempts + 1):
            headers = {"Authorization": f"Bearer {token}"}
            if eval_header:
                headers[EVAL_HEADER] = eval_header
            started = time.perf_counter()
            self.network_calls += 1
            try:
                r = await self._http().post(f"{self.prefix}/{endpoint}", json=body, headers=headers)
            except httpx.TransportError as exc:
                last = type(exc).__name__
                await self._sleep(min(30.0, 2.0**attempt))
                continue
            latency = round((time.perf_counter() - started) * 1000, 1)
            if r.status_code == 401 and not relogged:
                token = await self.login(force=True)
                relogged = True
                continue
            if r.status_code == 429 or r.status_code >= 500:
                last = f"HTTP {r.status_code}"
                await self._sleep(retry_after(r, attempt))
                continue
            if r.status_code != 200:
                return ApiResult(
                    endpoint, None, r.status_code, latency, cached=False, error=f"HTTP {r.status_code}"
                )
            return ApiResult(
                endpoint,
                r.json(),
                200,
                latency,
                cached=False,
                applied=r.headers.get(APPLIED_HEADER),
            )
        return ApiResult(endpoint, None, None, None, cached=False, error=f"gave up ({last})")
