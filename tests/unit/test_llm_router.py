"""LLMRouter against httpx.MockTransport: no network, no real sleeps (blueprint §3.7, §12.2, E14/E15)."""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from structlog.testing import capture_logs

from app.config import Settings
from app.core.errors import AppError, ErrorCode
from app.core.resilience import CircuitBreaker, TokenBucket, backoff_delay, parse_retry_after
from app.generation.llm import LLMRouter, build_llm_clients
from app.protocols import LLMClient, LLMMessage

KEY = "sk-test-key-0123456789abcdefghij"
LOCAL = "qwen2.5:3b-instruct"
MESSAGES = [LLMMessage("system", "Answer in JSON."), LLMMessage("user", "secret question about passports")]

Step = httpx.Response | Exception | Callable[[httpx.Request, dict[str, Any]], Any]


def ok(
    text: str = '{"answer": "fine"}', prompt_tokens: int = 12, completion_tokens: int = 3
) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        },
    )


def status(code: int, headers: dict[str, str] | None = None, body: str = "{}") -> httpx.Response:
    return httpx.Response(code, headers=headers, content=body.encode())


def timeout_step(request: httpx.Request, body: dict[str, Any]) -> httpx.Response:
    raise httpx.ReadTimeout("read timed out", request=request)


class Script:
    """Per-model scripted replies. The last step of a model repeats forever."""

    def __init__(self, plan: dict[str, list[Step]]) -> None:
        self.plan = plan
        self.calls: list[tuple[str, str, dict[str, Any], httpx.Headers]] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        model = body["model"]
        self.calls.append((request.url.host, model, body, request.headers))
        steps = self.plan.get(model) or [status(404)]
        step = steps.pop(0) if len(steps) > 1 else steps[0]
        if isinstance(step, Exception):
            raise step
        if callable(step) and not isinstance(step, httpx.Response):
            result = step(request, body)
            if asyncio.iscoroutine(result):
                result = await result
            assert isinstance(result, httpx.Response)
            return result
        return step

    def models(self) -> list[str]:
        return [model for _, model, _, _ in self.calls]


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = dict(
        _env_file=None,
        app_env="test",
        llm_api_key=KEY,
        generation_models=["gen-a", "gen-b"],
        helper_model="helper-h",
        judge_model="judge-j",
        local_fallback_model=LOCAL,
        local_fallback_enabled=True,  # these tests exercise the full chain, Ollama included
        llm_max_rpm=6000,
        llm_cache_dir=tmp_path / "llm-cache",
    )
    values.update(overrides)
    return Settings(**values)


@pytest.fixture
def sleeps() -> list[float]:
    return []


@pytest.fixture
def make_router(tmp_path: Path, sleeps: list[float]) -> Callable[..., tuple[LLMRouter, Script]]:
    def factory(
        plan: dict[str, list[Step]],
        *,
        clock: Callable[[], float] | None = None,
        cache: bool = False,
        local_only: bool = False,
        **overrides: Any,
    ) -> tuple[LLMRouter, Script]:
        async def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)

        script = Script(plan)
        kwargs: dict[str, Any] = {"sleep": fake_sleep, "rng": random.Random(7)}  # noqa: S311
        if clock is not None:
            kwargs["clock"] = clock
        router = LLMRouter(
            make_settings(tmp_path, **overrides),
            client=httpx.AsyncClient(transport=httpx.MockTransport(script)),
            cache=cache,
            local_only=local_only,
            **kwargs,
        )
        return router, script

    return factory


# ---------------------------------------------------------------- happy path and parsing


async def test_success_parses_text_and_usage(make_router: Callable[..., tuple[LLMRouter, Script]]) -> None:
    router, script = make_router({"gen-a": [ok("hello", 120, 30)]})
    assert isinstance(router, LLMClient)

    result = await router.complete(MESSAGES, purpose="generate", max_tokens=200)

    assert (result.text, result.endpoint, result.model) == ("hello", "hosted", "gen-a")
    assert (result.attempts, result.input_tokens, result.output_tokens) == (1, 120, 30)
    assert result.degraded is False
    host, _, body, headers = script.calls[0]
    assert host == "api.openai.com"
    assert headers["authorization"] == f"Bearer {KEY}"
    assert body["response_format"] == {"type": "json_object"}
    assert body["max_tokens"] == 200
    assert body["messages"][1] == {"role": "user", "content": "secret question about passports"}


async def test_json_mode_off_sends_no_response_format(
    make_router: Callable[..., tuple[LLMRouter, Script]],
) -> None:
    router, script = make_router({"gen-a": [ok()]}, llm_json_mode=False)
    await router.complete(MESSAGES, purpose="generate")
    assert "response_format" not in script.calls[0][2]


# ---------------------------------------------------------------- retries and failover


async def test_429_is_retried_honouring_retry_after(
    make_router: Callable[..., tuple[LLMRouter, Script]], sleeps: list[float]
) -> None:
    router, script = make_router({"gen-a": [status(429, {"Retry-After": "2"}), ok("after wait")]})

    result = await router.complete(MESSAGES, purpose="generate")

    assert result.text == "after wait"
    assert result.model == "gen-a"
    assert result.attempts == 2
    assert sleeps == [2.0]
    assert script.models() == ["gen-a", "gen-a"]


async def test_three_500s_fail_over_to_next_model(
    make_router: Callable[..., tuple[LLMRouter, Script]], sleeps: list[float]
) -> None:
    router, script = make_router({"gen-a": [status(500)], "gen-b": [ok("from b")]})

    result = await router.complete(MESSAGES, purpose="generate")

    assert (result.model, result.text, result.attempts) == ("gen-b", "from b", 4)
    assert script.models() == ["gen-a", "gen-a", "gen-a", "gen-b"]
    assert len(sleeps) == 2  # two retries with backoff, then failover
    assert all(0 < s <= 8 for s in sleeps)


async def test_other_4xx_is_not_retried(make_router: Callable[..., tuple[LLMRouter, Script]]) -> None:
    router, script = make_router({"gen-a": [status(404)], "gen-b": [ok()]})
    result = await router.complete(MESSAGES, purpose="generate")
    assert result.model == "gen-b"
    assert script.models() == ["gen-a", "gen-b"]


async def test_json_mode_rejection_retries_without_response_format(
    make_router: Callable[..., tuple[LLMRouter, Script]],
) -> None:
    def rejects_json_mode(request: httpx.Request, body: dict[str, Any]) -> httpx.Response:
        if "response_format" in body:
            return status(400, body='{"error": "response_format is not supported by this model"}')
        return ok("plain prompting")

    router, script = make_router({"gen-a": [rejects_json_mode]})

    result = await router.complete(MESSAGES, purpose="generate")

    assert (result.text, result.model, result.attempts) == ("plain prompting", "gen-a", 2)
    assert "response_format" in script.calls[0][2]
    assert "response_format" not in script.calls[1][2]
    assert router.json_mode_disabled("gen-a")

    await router.complete(MESSAGES, purpose="generate")  # remembered: no rejected first try
    assert len(script.calls) == 3
    assert "response_format" not in script.calls[2][2]


async def test_transport_timeout_fails_over(make_router: Callable[..., tuple[LLMRouter, Script]]) -> None:
    router, script = make_router({"gen-a": [timeout_step], "gen-b": [ok("b answered")]})
    result = await router.complete(MESSAGES, purpose="generate")
    assert (result.model, result.attempts) == ("gen-b", 2)
    assert script.models() == ["gen-a", "gen-b"]


async def test_per_attempt_timeout_cuts_a_hanging_model(
    make_router: Callable[..., tuple[LLMRouter, Script]],
) -> None:
    async def hangs(request: httpx.Request, body: dict[str, Any]) -> httpx.Response:
        await asyncio.sleep(5)
        return ok()

    router, _ = make_router({"gen-a": [hangs], "gen-b": [ok("fast")]}, llm_timeout_s=0.05)
    result = await asyncio.wait_for(router.complete(MESSAGES, purpose="generate"), timeout=2)
    assert result.model == "gen-b"


async def test_bad_key_skips_remaining_hosted_models(
    make_router: Callable[..., tuple[LLMRouter, Script]],
) -> None:
    router, script = make_router({"gen-a": [status(401)], "gen-b": [ok()], LOCAL: [ok("local")]})
    result = await router.complete(MESSAGES, purpose="generate")
    assert result.endpoint == "ollama"
    assert script.models() == ["gen-a", LOCAL]


# ---------------------------------------------------------------- circuit breaker


async def test_circuit_breaker_opens_after_three_failures_and_half_opens(
    make_router: Callable[..., tuple[LLMRouter, Script]],
) -> None:
    clock = FakeClock()
    router, script = make_router({"gen-a": [status(404)], "gen-b": [ok()]}, clock=clock)

    for _ in range(3):
        assert (await router.complete(MESSAGES, purpose="generate")).model == "gen-b"
    assert router.breaker_state("hosted", "gen-a") == "open"

    script.calls.clear()
    await router.complete(MESSAGES, purpose="generate")
    assert script.models() == ["gen-b"]  # open breaker: gen-a is not called at all

    clock.now += 61
    assert router.breaker_state("hosted", "gen-a") == "half_open"
    script.calls.clear()
    script.plan["gen-a"] = [ok("recovered")]
    result = await router.complete(MESSAGES, purpose="generate")
    assert result.model == "gen-a"  # the half-open trial succeeded
    assert router.breaker_state("hosted", "gen-a") == "closed"


def test_breaker_failed_trial_reopens() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker(3, 60, clock=clock)
    for _ in range(3):
        assert breaker.allow()
        breaker.record_failure()
    assert not breaker.allow()
    clock.now += 60
    assert breaker.allow()  # the one trial
    assert not breaker.allow()  # no second concurrent trial
    breaker.record_failure()
    assert breaker.state == "open"
    assert breaker.seconds_until_retry() == pytest.approx(60)


# ---------------------------------------------------------------- local fallback and exhaustion


async def test_all_hosted_down_falls_back_to_ollama_degraded(
    make_router: Callable[..., tuple[LLMRouter, Script]],
) -> None:
    router, script = make_router(
        {"gen-a": [status(503)], "gen-b": [status(502)], LOCAL: [ok("local answer", 50, 5)]}
    )

    result = await router.complete(MESSAGES, purpose="generate")

    assert (result.endpoint, result.model, result.text) == ("ollama", LOCAL, "local answer")
    assert result.degraded is True
    assert result.attempts == 7
    host, _, _, headers = script.calls[-1]
    assert host == "127.0.0.1"
    assert "authorization" not in headers


async def test_everything_down_raises_503_llm_unavailable(
    make_router: Callable[..., tuple[LLMRouter, Script]],
) -> None:
    router, _ = make_router(
        {"gen-a": [status(500)], "gen-b": [status(429, {"Retry-After": "7"})], LOCAL: [timeout_step]}
    )

    with pytest.raises(AppError) as info:
        await router.complete(MESSAGES, purpose="generate")

    assert info.value.code is ErrorCode.LLM_UNAVAILABLE
    assert info.value.code.status == 503
    assert info.value.code.slug == "llm_unavailable"
    assert info.value.headers["Retry-After"] == "7"


async def test_missing_key_goes_straight_to_ollama(
    make_router: Callable[..., tuple[LLMRouter, Script]],
) -> None:
    router, script = make_router({"gen-a": [ok()], LOCAL: [ok("local")]}, llm_api_key=None)
    result = await router.complete(MESSAGES, purpose="generate")
    assert (result.endpoint, result.degraded) == ("ollama", True)
    assert script.models() == [LOCAL]
    assert router.has_hosted is False


async def test_missing_key_and_ollama_down_is_503(
    make_router: Callable[..., tuple[LLMRouter, Script]],
) -> None:
    router, _ = make_router({LOCAL: [status(500)]}, llm_api_key=None)
    with pytest.raises(AppError) as info:
        await router.complete(MESSAGES, purpose="helper")
    assert info.value.code is ErrorCode.LLM_UNAVAILABLE


async def test_purpose_chains(make_router: Callable[..., tuple[LLMRouter, Script]]) -> None:
    router, _ = make_router({}, generation_models=["gen-a", "helper-h", "gen-b"])
    keys = {p: [t.key for t in router.chain(p)] for p in ("generate", "helper", "judge")}  # type: ignore[arg-type]
    assert keys["generate"] == ["hosted:gen-a", "hosted:helper-h", "hosted:gen-b", f"ollama:{LOCAL}"]
    assert keys["helper"] == ["hosted:helper-h", "hosted:gen-a", "hosted:gen-b", f"ollama:{LOCAL}"]
    assert keys["judge"] == ["hosted:judge-j"]  # never judged by the local fallback


async def test_judge_down_has_no_local_fallback(
    make_router: Callable[..., tuple[LLMRouter, Script]],
) -> None:
    router, script = make_router({"judge-j": [status(404)], LOCAL: [ok()]})
    with pytest.raises(AppError):
        await router.complete(MESSAGES, purpose="judge")
    assert script.models() == ["judge-j"]


async def test_helper_uses_helper_timeout(make_router: Callable[..., tuple[LLMRouter, Script]]) -> None:
    seen: list[Any] = []

    def capture(request: httpx.Request, body: dict[str, Any]) -> httpx.Response:
        seen.append(request.extensions.get("timeout"))
        return ok()

    router, script = make_router({"helper-h": [capture]}, llm_helper_timeout_s=7.0)
    await router.complete(MESSAGES, purpose="helper")
    assert script.models() == ["helper-h"]
    assert seen[0]["read"] == pytest.approx(7.0)


async def test_local_only_router_never_calls_hosted(
    make_router: Callable[..., tuple[LLMRouter, Script]],
) -> None:
    router, script = make_router({"gen-a": [ok()], LOCAL: [ok("private stays local")]}, local_only=True)
    result = await router.complete(MESSAGES, purpose="generate")
    assert (result.endpoint, result.degraded) == ("ollama", True)
    assert script.models() == [LOCAL]


# ---------------------------------------------------------------- eval disk cache


async def test_cache_hit_avoids_second_call(
    make_router: Callable[..., tuple[LLMRouter, Script]], tmp_path: Path
) -> None:
    router, script = make_router({"gen-a": [ok("cached answer", 10, 2)]}, cache=True)
    assert router.cache_enabled

    first = await router.complete(MESSAGES, purpose="generate", max_tokens=300)
    second = await router.complete(MESSAGES, purpose="generate", max_tokens=300)

    assert len(script.calls) == 1
    assert second == first
    assert list((tmp_path / "llm-cache").rglob("*.json"))

    await router.complete(MESSAGES, purpose="generate", max_tokens=301)  # different params: miss
    assert len(script.calls) == 2


async def test_cache_disabled_by_default(make_router: Callable[..., tuple[LLMRouter, Script]]) -> None:
    router, script = make_router({"gen-a": [ok()]})
    assert not router.cache_enabled
    await router.complete(MESSAGES, purpose="generate")
    await router.complete(MESSAGES, purpose="generate")
    assert len(script.calls) == 2


async def test_cache_needs_eval_only_mode(make_router: Callable[..., tuple[LLMRouter, Script]]) -> None:
    router, script = make_router({"gen-a": [ok()]}, cache=True, response_cache="off")
    assert not router.cache_enabled
    await router.complete(MESSAGES, purpose="generate")
    await router.complete(MESSAGES, purpose="generate")
    assert len(script.calls) == 2


async def test_degraded_answers_are_not_cached(
    make_router: Callable[..., tuple[LLMRouter, Script]],
) -> None:
    router, script = make_router({LOCAL: [ok("local")]}, cache=True, llm_api_key=None)
    await router.complete(MESSAGES, purpose="generate")
    await router.complete(MESSAGES, purpose="generate")
    assert len(script.calls) == 2


# ---------------------------------------------------------------- models, logging, wiring


async def test_available_models_lists_hosted_ids(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        assert request.headers["authorization"] == f"Bearer {KEY}"
        return httpx.Response(200, json={"data": [{"id": "z/model"}, {"id": "a/model"}]})

    router = LLMRouter(
        make_settings(tmp_path), client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    assert await router.available_models() == ["a/model", "z/model"]

    no_key = LLMRouter(make_settings(tmp_path, llm_api_key=None))
    assert await no_key.available_models() == []
    await no_key.aclose()


async def test_available_models_failure_is_llm_unavailable(tmp_path: Path) -> None:
    router = LLMRouter(
        make_settings(tmp_path),
        client=httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(401))),
    )
    with pytest.raises(AppError) as info:
        await router.available_models()
    assert info.value.code is ErrorCode.LLM_UNAVAILABLE


async def test_logs_never_contain_key_prompt_or_answer(
    make_router: Callable[..., tuple[LLMRouter, Script]],
) -> None:
    answer = "the confidential answer text"
    router, _ = make_router({"gen-a": [status(500)], "gen-b": [status(404)], LOCAL: [ok(answer)]})
    with capture_logs() as logs:
        await router.complete(MESSAGES, purpose="generate")
    dumped = json.dumps(logs, default=str)
    assert logs  # the router does log attempts
    for secret in (KEY, "secret question about passports", "Answer in JSON.", answer):
        assert secret not in dumped
    assert {"model", "endpoint", "status", "latency_ms", "attempt"} <= set(
        next(e for e in logs if e["event"] == "llm.attempt")
    )


def test_build_llm_clients(tmp_path: Path) -> None:
    hosted, local_only = build_llm_clients(make_settings(tmp_path))
    assert [t.endpoint for t in hosted.chain("generate")] == ["hosted", "hosted", "ollama"]
    assert [t.key for t in local_only.chain("generate")] == [f"ollama:{LOCAL}"]
    assert not hosted.cache_enabled and not local_only.cache_enabled
    cached, _ = build_llm_clients(make_settings(tmp_path), eval_cache=True)
    assert cached.cache_enabled


# ---------------------------------------------------------------- resilience helpers


def test_parse_retry_after() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    assert parse_retry_after("3") == 3.0
    assert parse_retry_after(None) is None
    assert parse_retry_after("soon") is None
    assert parse_retry_after(format_datetime(now + timedelta(seconds=9), usegmt=True), now=now) == 9.0


def test_backoff_is_exponential_capped_and_honours_retry_after() -> None:
    rng = random.Random(1)  # noqa: S311
    delays = [backoff_delay(i, rng=rng) for i in range(6)]
    assert delays[0] < 0.76 and delays[1] < 1.51
    assert max(delays) <= 8.0
    assert backoff_delay(0, retry_after_s=4.0) == 4.0
    assert backoff_delay(0, retry_after_s=600.0) == 30.0


async def test_token_bucket_spaces_requests() -> None:
    clock = FakeClock()
    waits: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        waits.append(seconds)

    bucket = TokenBucket(60, burst=2, clock=clock, sleep=fake_sleep)  # 1 token per second
    assert await bucket.acquire() == 0
    assert await bucket.acquire() == 0
    assert await bucket.acquire() == pytest.approx(1.0)
    assert await bucket.acquire() == pytest.approx(2.0)  # reservations queue up
    clock.now += 10
    assert await bucket.acquire() == 0  # refilled (but never above the burst)
    assert waits == [pytest.approx(1.0), pytest.approx(2.0)]
