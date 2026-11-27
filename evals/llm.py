"""A minimal OpenAI-compatible `/chat/completions` client for the eval: judge and closed-book.

- Calls the API directly with httpx (no service code), key and base URL from `Settings`.
- **Disk cache** keyed by sha256(model, prompt): one JSON file per call. Re-runs are free and the
  published numbers re-run offline. `use_cache=False` (`--no-cache`) skips reads but still writes.
- **Offline mode** never touches the network: a cache miss raises `CacheMissError`.
- **Client-side rate limit** (`LLM_MAX_RPM` requests/minute).
- Retries 429/5xx with backoff, honouring `Retry-After`; JSON mode falls back to plain prompting when
  a model rejects `response_format`.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from app.config import Settings

Message = dict[str, str]
Sleep = Callable[[float], Awaitable[None]]


class LLMError(RuntimeError):
    pass


class CacheMissError(LLMError):
    """Offline mode and the call is not in the cache."""


class MissingKeyError(LLMError):
    """No LLM key (LLM_API_KEY / OPENAI_API_KEY) in the environment / .env."""


class RateLimiter:
    """Spaces calls at least 60/rpm seconds apart (a simple, deterministic client-side limit)."""

    def __init__(
        self, rpm: int, *, clock: Callable[[], float] = time.monotonic, sleep: Sleep = asyncio.sleep
    ):
        self.interval = 60.0 / max(1, rpm)
        self._clock = clock
        self._sleep = sleep
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = self._clock()
            if self._next > now:
                await self._sleep(self._next - now)
                now = self._next
            self._next = now + self.interval


def cache_key(model: str, messages: Sequence[Message]) -> str:
    prompt = json.dumps(list(messages), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(f"{model}\n{prompt}".encode()).hexdigest()


@dataclass
class ChatResult:
    content: str
    model: str
    cached: bool
    latency_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


class ChatClient:
    def __init__(
        self,
        settings: Settings,
        *,
        cache_dir: Path,
        use_cache: bool = True,
        offline: bool = False,
        transport: httpx.AsyncBaseTransport | None = None,
        max_rpm: int | None = None,
        limiter: RateLimiter | None = None,
        timeout_s: float = 120.0,
        max_attempts: int = 4,
        sleep: Sleep = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings = settings
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.offline = offline
        self.max_attempts = max_attempts
        self._sleep = sleep
        rpm = max_rpm or settings.llm_max_rpm
        # Share one limiter between clients that use the same key (judge + closed-book).
        self.limiter = limiter or RateLimiter(rpm, clock=clock, sleep=sleep)
        self._transport = transport
        self._timeout = timeout_s
        self._client: httpx.AsyncClient | None = None
        self.network_calls = 0

    async def __aenter__(self) -> ChatClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            key = self.settings.llm_key
            if not key:
                raise MissingKeyError("LLM_API_KEY (or OPENAI_API_KEY) is not set (add it to .env)")
            self._client = httpx.AsyncClient(
                base_url=self.settings.llm_base_url.rstrip("/"),
                headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
                timeout=self._timeout,
                transport=self._transport,
            )
        return self._client

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _read(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return data

    def _write(self, key: str, data: dict[str, Any]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = self._path(key).with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self._path(key))

    async def complete(
        self,
        model: str,
        messages: Sequence[Message],
        *,
        max_tokens: int = 1024,
        json_mode: bool = True,
    ) -> ChatResult:
        key = cache_key(model, messages)
        if self.use_cache or self.offline:
            hit = self._read(key)
            if hit is not None:
                return ChatResult(
                    content=str(hit.get("content", "")),
                    model=model,
                    cached=True,
                    latency_ms=hit.get("latency_ms"),
                    input_tokens=hit.get("input_tokens"),
                    output_tokens=hit.get("output_tokens"),
                )
        if self.offline:
            raise CacheMissError(f"offline and no cached response for {model} ({key[:12]})")
        result = await self._call(model, messages, max_tokens=max_tokens, json_mode=json_mode)
        self._write(
            key,
            {
                "model": model,
                "messages": list(messages),
                "content": result.content,
                "latency_ms": result.latency_ms,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )
        return result

    async def _call(
        self, model: str, messages: Sequence[Message], *, max_tokens: int, json_mode: bool
    ) -> ChatResult:
        client = self._http()
        body: dict[str, Any] = {"model": model, "messages": list(messages)}
        if self.settings.llm_token_param == "max_completion_tokens":  # noqa: S105
            body["max_completion_tokens"] = max_tokens  # reasoning models: no `temperature`
        else:
            body["temperature"] = 0
            body["max_tokens"] = max_tokens
        if json_mode and self.settings.llm_json_mode:
            body["response_format"] = {"type": "json_object"}
        last = ""
        for attempt in range(1, self.max_attempts + 1):
            await self.limiter.acquire()
            started = time.perf_counter()
            self.network_calls += 1
            try:
                r = await client.post("/chat/completions", json=body)
            except httpx.TransportError as exc:
                last = f"{type(exc).__name__}"
                await self._sleep(min(30.0, 2.0**attempt))
                continue
            latency = round((time.perf_counter() - started) * 1000, 1)
            if r.status_code in (400, 422) and "response_format" in body:
                body.pop("response_format")  # the model rejects JSON mode: plain prompting instead
                continue
            if r.status_code == 429 or r.status_code >= 500:
                last = f"HTTP {r.status_code}"
                await self._sleep(retry_after(r, attempt))
                continue
            if r.status_code != 200:
                raise LLMError(f"{model}: HTTP {r.status_code}")
            data = r.json()
            message = (data.get("choices") or [{}])[0].get("message") or {}
            content = message.get("content") or message.get("reasoning_content") or ""
            usage = data.get("usage") or {}
            return ChatResult(
                content=str(content),
                model=model,
                cached=False,
                latency_ms=latency,
                input_tokens=usage.get("prompt_tokens"),
                output_tokens=usage.get("completion_tokens"),
            )
        raise LLMError(f"{model}: gave up after {self.max_attempts} attempts ({last})")


def retry_after(response: httpx.Response, attempt: int) -> float:
    header = response.headers.get("retry-after")
    if header:
        try:
            return min(60.0, max(0.0, float(header)))
        except ValueError:
            pass
    return min(30.0, 2.0**attempt)
