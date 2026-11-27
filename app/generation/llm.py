"""`LLMRouter`: the hosted (OpenAI) model chain, optional Ollama fallback (blueprint §3.7, §12.2, E14/E15).

Transport choice: the router calls the OpenAI-compatible `POST /chat/completions` directly with
`httpx.AsyncClient` instead of going through LangChain's `ChatOpenAI`. Raw httpx gives exact control
over what reliability depends on: per-attempt timeouts, which statuses are retried, honouring
`Retry-After`, the client-side rate limit and retrying without `response_format` when a model
rejects JSON mode. It is also fully testable offline with `httpx.MockTransport`.

Per call, for each model of the chain in order:
- skip it while its circuit breaker is open (3 consecutive failures, half-open after 60 s);
- hosted calls first take a token from the shared bucket (`LLM_MAX_RPM`);
- 429 / 5xx are retried at most twice with exponential backoff + jitter, honouring `Retry-After`;
- a 400/422 while JSON mode was requested is retried once without `response_format`; when that
  succeeds the model is remembered as non-JSON-mode;
- a timeout, transport error or any other 4xx moves on to the next model (a 401 skips every
  hosted model, since the key itself is bad).
Hosted models fall back across `GENERATION_MODELS`. Only when `LOCAL_FALLBACK_ENABLED=true` (off by
default) does the local Ollama model then answer with `degraded=True` (`generate` and `helper` only).
When the whole chain is exhausted the router raises `AppError(LLM_UNAVAILABLE)` with `Retry-After`.

Logging: model, endpoint, status, latency and attempt number only. The API key, the prompts and
the answer text are never logged.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import random
import time
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import httpx
import structlog

from app.config import Settings
from app.core.errors import AppError, ErrorCode
from app.core.resilience import (
    CircuitBreaker,
    Clock,
    Sleep,
    TokenBucket,
    backoff_delay,
    parse_retry_after,
)
from app.protocols import LLMMessage, LLMResult

log = structlog.get_logger(__name__)

Purpose = Literal["generate", "helper", "judge"]
Endpoint = Literal["hosted", "ollama"]

# Module-level defaults (not in Settings; see the work-package report).
MAX_RETRIES = 2  # 429/5xx retries on the same model
BREAKER_FAILURES = 3
BREAKER_RESET_S = 60.0
CONNECT_TIMEOUT_S = 5.0  # §12.2
CHAIN_BUDGET_FACTOR = 2.0  # hosted chain capped at 2 x the per-attempt timeout (90 s for generation)
LLM_BURST = 5  # token-bucket burst on top of LLM_MAX_RPM
DEFAULT_RETRY_AFTER_S = 30


@dataclass(frozen=True, slots=True)
class _Target:
    endpoint: Endpoint
    model: str

    @property
    def key(self) -> str:
        return f"{self.endpoint}:{self.model}"


class _ModelFailedError(Exception):
    """One model of the chain gave up (after its own retries)."""

    def __init__(
        self, reason: str, attempts: int, *, retry_after: float | None = None, bad_key: bool = False
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.attempts = attempts
        self.retry_after = retry_after
        self.bad_key = bad_key


@dataclass(frozen=True, slots=True)
class _Completion:
    text: str
    input_tokens: int
    output_tokens: int
    attempts: int


def _dedupe(models: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(m for m in models if m))


def _parse_completion(response: httpx.Response) -> tuple[str, int, int] | None:
    try:
        data = response.json()
        message = data["choices"][0].get("message") or {}
    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
        return None
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        return None
    usage = data.get("usage") or {}
    try:
        return content, int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
    except (TypeError, ValueError):
        return content, 0, 0


class LLMRouter:
    """Implements `LLMClient`.

    Chains per purpose: `generate` = GENERATION_MODELS; `helper` = HELPER_MODEL + GENERATION_MODELS;
    `judge` = JUDGE_MODEL. `generate` and `helper` then fall back to Ollama (`degraded=True`) only when
    `LOCAL_FALLBACK_ENABLED`. With no
    LLM_API_KEY the hosted models are skipped. `local_only=True` builds a router that uses only
    Ollama (for `PRIVATE_CONTEXT_POLICY=local_only` answers over private chunks).

    Disk response cache (eval only): enabled only when the router is constructed with `cache=True`
    **and** `RESPONSE_CACHE=eval_only`. The serving app must never construct it with `cache=True`:
    the key has no tenant component, so a cached answer built from one user's documents could be
    served to another (§5.8.5, E37). The key includes the chain, so an `X-Eval-Config`
    `generation_model` override (a one-model chain) never shares an entry with the default chain
    or with another model. Degraded (Ollama) answers are never cached, so a bad network
    day cannot leak fallback answers into published eval numbers.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        local_only: bool = False,
        cache: bool = False,
        client: httpx.AsyncClient | None = None,
        clock: Clock = time.monotonic,
        sleep: Sleep = asyncio.sleep,
        rng: random.Random | None = None,
    ) -> None:
        self._settings = settings
        self._local_only = local_only
        self._api_key = settings.llm_key
        self._client = client or httpx.AsyncClient()
        self._owns_client = client is None
        self._clock = clock
        self._sleep = sleep
        self._rng = rng or random.Random()  # noqa: S311 (backoff jitter, not cryptography)
        self._bucket = TokenBucket(settings.llm_max_rpm, LLM_BURST, clock=clock, sleep=sleep)
        self._breakers: dict[str, CircuitBreaker] = {}
        self._no_json_mode: set[str] = set()
        self._cache_dir: Path | None = (
            settings.llm_cache_dir if cache and settings.response_cache == "eval_only" else None
        )

    # ------------------------------------------------------------ public API

    @property
    def cache_enabled(self) -> bool:
        return self._cache_dir is not None

    @property
    def has_hosted(self) -> bool:
        return bool(self._api_key) and not self._local_only

    def chain(self, purpose: Purpose, model: str | None = None) -> list[_Target]:
        """`model` (eval override, `generate` only): the hosted part of the chain is just that model."""
        local = _Target("ollama", self._settings.local_fallback_model)
        if self._local_only:
            return [local]
        hosted: list[str] = []
        if self._api_key:
            s = self._settings
            names = {
                "generate": list(s.generation_models),
                "helper": [s.helper_model, *s.generation_models],
                "judge": [s.judge_model],
            }[purpose]
            hosted = _dedupe(names)
            if model and purpose == "generate":
                hosted = [model]
        targets = [_Target("hosted", m) for m in hosted]
        if purpose != "judge" and self._settings.local_fallback_enabled:
            targets.append(local)
        return targets

    def breaker_state(self, endpoint: Endpoint, model: str) -> str:
        return self._breaker(_Target(endpoint, model)).state

    def json_mode_disabled(self, model: str, endpoint: Endpoint = "hosted") -> bool:
        return _Target(endpoint, model).key in self._no_json_mode

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        purpose: Purpose,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        model: str | None = None,
    ) -> LLMResult:
        chain = self.chain(purpose, model)
        cache_path = self._cache_path(chain, messages, max_tokens, temperature)
        if cache_path is not None:
            cached = self._cache_read(cache_path)
            if cached is not None:
                log.info("llm.cache_hit", model=cached.model, endpoint=cached.endpoint, purpose=purpose)
                return cached

        timeout = self._settings.llm_helper_timeout_s if purpose == "helper" else self._settings.llm_timeout_s
        deadline = self._clock() + CHAIN_BUDGET_FACTOR * timeout
        started = time.perf_counter()
        attempts = 0
        retry_hint: float | None = None
        bad_key = False

        for target in chain:
            hosted = target.endpoint == "hosted"
            if hosted and bad_key:
                continue
            remaining = deadline - self._clock()
            if hosted and remaining <= 0:
                log.warning("llm.skip", reason="chain_budget", model=target.model, endpoint=target.endpoint)
                continue
            breaker = self._breaker(target)
            if not breaker.allow():
                log.info("llm.skip", reason="circuit_open", model=target.model, endpoint=target.endpoint)
                wait = breaker.seconds_until_retry()
                retry_hint = wait if retry_hint is None else min(retry_hint, wait)
                continue
            try:
                done = await self._call_model(
                    target,
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    attempt_timeout_s=min(timeout, remaining) if hosted else timeout,
                    deadline=deadline if hosted else None,
                )
            except _ModelFailedError as failure:
                attempts += failure.attempts
                breaker.record_failure()
                bad_key = bad_key or failure.bad_key
                if failure.retry_after is not None:
                    retry_hint = max(retry_hint or 0.0, failure.retry_after)
                log.warning(
                    "llm.model_failed",
                    model=target.model,
                    endpoint=target.endpoint,
                    reason=failure.reason,
                    attempts=failure.attempts,
                    breaker=breaker.state,
                )
                continue
            attempts += done.attempts
            breaker.record_success()
            result = LLMResult(
                text=done.text,
                endpoint=target.endpoint,
                model=target.model,
                attempts=attempts,
                input_tokens=done.input_tokens,
                output_tokens=done.output_tokens,
                latency_ms=round((time.perf_counter() - started) * 1000),
                degraded=target.endpoint == "ollama",
            )
            log.info(
                "llm.complete",
                model=result.model,
                endpoint=result.endpoint,
                purpose=purpose,
                attempts=result.attempts,
                latency_ms=result.latency_ms,
                degraded=result.degraded,
            )
            if cache_path is not None and not result.degraded:
                self._cache_write(cache_path, result)
            return result

        retry_after = max(1, math.ceil(retry_hint if retry_hint else DEFAULT_RETRY_AFTER_S))
        log.error("llm.unavailable", purpose=purpose, attempts=attempts, models=len(chain))
        raise AppError(
            ErrorCode.LLM_UNAVAILABLE,
            "The answer model is unavailable right now. Try again shortly.",
            extra={"attempts": attempts},
            headers={"Retry-After": str(retry_after)},
        )

    async def available_models(self) -> list[str]:
        """Model ids visible to the key (GET /models). The local-only router lists Ollama's models."""
        if self._local_only:
            url, headers = f"{self._base(_Target('ollama', ''))}/models", self._headers(hosted=False)
        else:
            if not self._api_key:
                return []
            url, headers = f"{self._base(_Target('hosted', ''))}/models", self._headers(hosted=True)
        try:
            response = await self._client.get(
                url,
                headers=headers,
                timeout=httpx.Timeout(self._settings.llm_helper_timeout_s, connect=CONNECT_TIMEOUT_S),
            )
        except httpx.HTTPError as exc:
            log.warning("llm.models_failed", reason=type(exc).__name__)
            raise AppError(ErrorCode.LLM_UNAVAILABLE, "Could not list the available models.") from exc
        if response.status_code != 200:
            log.warning("llm.models_failed", status=response.status_code)
            raise AppError(ErrorCode.LLM_UNAVAILABLE, "Could not list the available models.")
        try:
            data = response.json().get("data", [])
            return sorted(str(item["id"]) for item in data if isinstance(item, dict) and "id" in item)
        except (ValueError, AttributeError) as exc:
            raise AppError(ErrorCode.LLM_UNAVAILABLE, "Could not list the available models.") from exc

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # ------------------------------------------------------------ one model

    def _breaker(self, target: _Target) -> CircuitBreaker:
        breaker = self._breakers.get(target.key)
        if breaker is None:
            breaker = CircuitBreaker(BREAKER_FAILURES, BREAKER_RESET_S, clock=self._clock)
            self._breakers[target.key] = breaker
        return breaker

    def _base(self, target: _Target) -> str:
        base = self._settings.llm_base_url if target.endpoint == "hosted" else self._settings.ollama_base_url
        return base.rstrip("/")

    def _headers(self, *, hosted: bool) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if hosted and self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def _call_model(
        self,
        target: _Target,
        messages: Sequence[LLMMessage],
        *,
        max_tokens: int,
        temperature: float,
        attempt_timeout_s: float,
        deadline: float | None,
    ) -> _Completion:
        hosted = target.endpoint == "hosted"
        url = f"{self._base(target)}/chat/completions"
        headers = self._headers(hosted=hosted)
        json_mode = self._settings.llm_json_mode and target.key not in self._no_json_mode
        json_fallback_used = False
        retries = 0
        attempts = 0
        payload_messages = [{"role": m.role, "content": m.content} for m in messages]

        while True:
            body: dict[str, Any] = {
                "model": target.model,
                "messages": payload_messages,
                "stream": False,
            }
            if hosted and self._settings.llm_token_param == "max_completion_tokens":  # noqa: S105
                # OpenAI reasoning models (gpt-5*, o*): no `max_tokens`, no non-default `temperature`.
                body["max_completion_tokens"] = max_tokens
            else:
                body["max_tokens"] = max_tokens
                body["temperature"] = temperature
            if json_mode:
                body["response_format"] = {"type": "json_object"}
            if hosted:
                await self._bucket.acquire()
            attempts += 1
            sent = time.perf_counter()
            try:
                async with asyncio.timeout(attempt_timeout_s):
                    response = await self._client.post(
                        url,
                        json=body,
                        headers=headers,
                        timeout=httpx.Timeout(
                            attempt_timeout_s, connect=min(CONNECT_TIMEOUT_S, attempt_timeout_s)
                        ),
                    )
            except (TimeoutError, httpx.TimeoutException) as exc:
                self._log_attempt(target, "timeout", sent, attempts)
                raise _ModelFailedError("timeout", attempts) from exc
            except httpx.HTTPError as exc:
                self._log_attempt(target, type(exc).__name__, sent, attempts)
                raise _ModelFailedError("transport_error", attempts) from exc

            status = response.status_code
            self._log_attempt(target, status, sent, attempts)
            if status == 200:
                parsed = _parse_completion(response)
                if parsed is None:
                    raise _ModelFailedError("bad_response", attempts)
                if json_fallback_used:
                    self._no_json_mode.add(target.key)
                    log.info("llm.json_mode_disabled", model=target.model, endpoint=target.endpoint)
                text, input_tokens, output_tokens = parsed
                return _Completion(text, input_tokens, output_tokens, attempts)

            if status in (400, 422) and json_mode and not json_fallback_used:
                json_mode, json_fallback_used = False, True  # maybe it rejected response_format
                continue

            if status == 429 or status >= 500:
                retry_after = parse_retry_after(response.headers.get("Retry-After"))
                if retries < MAX_RETRIES:
                    delay = backoff_delay(retries, retry_after_s=retry_after, rng=self._rng)
                    if deadline is not None and self._clock() + delay >= deadline:
                        raise _ModelFailedError(f"http_{status}", attempts, retry_after=retry_after)
                    retries += 1
                    await self._sleep(delay)
                    continue
                raise _ModelFailedError(f"http_{status}", attempts, retry_after=retry_after)

            raise _ModelFailedError(f"http_{status}", attempts, bad_key=hosted and status == 401)

    @staticmethod
    def _log_attempt(target: _Target, status: int | str, sent: float, attempt: int) -> None:
        log.info(
            "llm.attempt",
            model=target.model,
            endpoint=target.endpoint,
            status=status,
            latency_ms=round((time.perf_counter() - sent) * 1000),
            attempt=attempt,
        )

    # ------------------------------------------------------------ eval disk cache

    def _cache_path(
        self, chain: Sequence[_Target], messages: Sequence[LLMMessage], max_tokens: int, temperature: float
    ) -> Path | None:
        if self._cache_dir is None:
            return None
        material = {
            "chain": [t.key for t in chain],
            "messages": [[m.role, m.content] for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "json_mode": self._settings.llm_json_mode,
        }
        digest = hashlib.sha256(
            json.dumps(material, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        return self._cache_dir / digest[:2] / f"{digest}.json"

    @staticmethod
    def _cache_read(path: Path) -> LLMResult | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return LLMResult(**data)
        except FileNotFoundError:
            return None
        except (OSError, ValueError, TypeError):
            log.warning("llm.cache_unreadable", file=path.name)
            return None

    @staticmethod
    def _cache_write(path: Path, result: LLMResult) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
            tmp.write_text(json.dumps(asdict(result), ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
        except OSError:
            log.warning("llm.cache_write_failed", file=path.name)


def build_llm_clients(
    settings: Settings, *, eval_cache: bool = False, client: httpx.AsyncClient | None = None
) -> tuple[LLMRouter, LLMRouter]:
    """Return (hosted, local_only).

    `hosted` is the full hosted chain (→ Ollama when LOCAL_FALLBACK_ENABLED). `local_only` uses only
    Ollama; the ask pipeline uses it when `PRIVATE_CONTEXT_POLICY=local_only` and private chunks are in
    the context. The serving app passes `eval_cache=False` (the default); only the eval runner may pass True.
    """
    hosted = LLMRouter(settings, cache=eval_cache, client=client)
    local_only = LLMRouter(settings, local_only=True, cache=eval_cache, client=client)
    return hosted, local_only
