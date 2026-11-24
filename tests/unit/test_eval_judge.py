"""LLM judge and the eval LLM client: parsing, disk cache, offline, rate limit (no network)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from evals.golden import GoldenItem
from evals.judge import Judge, JudgeInput, build_messages, parse_verdict
from evals.llm import CacheMissError, ChatClient, MissingKeyError, RateLimiter, cache_key

from app.config import Settings

JUDGE_MODEL = "openai/gpt-oss-120b"


def _settings(**over: Any) -> Settings:
    return Settings(_env_file=None, nvidia_api_key="nvapi-test", **over)  # type: ignore[call-arg]


def _item() -> GoldenItem:
    return GoldenItem.model_validate(
        {
            "id": "q_t_adv",
            "question": "Meri date of birth galat hai, court case chal raha hai, kya karun?",
            "language": "roman_ur",
            "answer_language": "roman_ur",
            "type": "advice_boundary",
            "answerable": True,
            "gold_sources": [{"source_id": "ID-05", "pages": [14]}],
            "agencies": ["NADRA"],
            "jurisdictions": ["federal"],
            "expected_notices": ["not_advice", "referral"],
            "source_ids": ["ID-05"],
            "reference_answer": "State the general rule only, with the not-advice notice.",
            "split": "test",
            "review_status": "approved",
            "author": "s",
            "reviewed_by": "s",
        }
    )


class Recorder:
    """A MockTransport handler that returns scripted chat completions and records requests."""

    def __init__(self, contents: list[str] | None = None, statuses: list[int] | None = None) -> None:
        self.contents = contents or ['{"correctness": 2, "citation_support": "yes"}']
        self.statuses = statuses or []
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(
            {"body": body, "auth": request.headers.get("authorization"), "url": str(request.url)}
        )
        if self.statuses:
            status = self.statuses.pop(0)
            if status != 200:
                return httpx.Response(status, headers={"retry-after": "0"}, json={"error": "x"})
        content = self.contents[min(len(self.requests), len(self.contents)) - 1]
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": content}}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 10},
            },
        )


async def _no_sleep(_: float) -> None:
    return None


def _client(tmp_path: Path, rec: Recorder, **over: Any) -> ChatClient:
    kwargs: dict[str, Any] = {
        "cache_dir": tmp_path / "judge",
        "transport": httpx.MockTransport(rec),
        "sleep": _no_sleep,
    }
    kwargs.update(over)
    settings = kwargs.pop("settings", _settings())
    return ChatClient(settings, **kwargs)


# ---------------------------------------------------------------- parsing


@pytest.mark.parametrize(
    "text",
    [
        '{"correctness": 2, "citation_support": "yes", "refusal_appropriate": true, '
        '"disclaimer_present": true, "personal_directive": false, "rationale": "ok"}',
        '```json\n{"correctness": "2", "citation_support": "y", "refusal_appropriate": "true", '
        '"disclaimer_present": "yes", "personal_directive": "no", "rationale": "ok"}\n```',
        '<think>{"correctness": 0} draft</think>Here is my verdict: {"correctness": 2.0, '
        '"citation_support": true, "refusal_appropriate": 1, "disclaimer_present": true, '
        '"personal_directive": false, "rationale": "ok"} Thanks.',
    ],
)
def test_parse_verdict_tolerant(text: str) -> None:
    v = parse_verdict(text)
    assert v.parsed
    assert v.correctness == 2
    assert v.citation_support == "yes"
    assert v.refusal_appropriate is True
    assert v.advice_ok is True


def test_parse_verdict_degrades() -> None:
    v = parse_verdict("correctness: 1 (partly right), citations unclear")
    assert not v.parsed and v.correctness == 1
    assert parse_verdict("no idea").correctness is None
    assert parse_verdict('{"correctness": 7, "citation_support": null}').correctness == 2  # clamped
    assert parse_verdict('{"correctness": "partial", "citation_support": "n/a"}').citation_support == "na"
    directive = parse_verdict('{"correctness": 2, "disclaimer_present": true, "personal_directive": true}')
    assert directive.advice_ok is False


def test_prompt_carries_rubric_and_context() -> None:
    inp = JudgeInput(
        item=_item(),
        answer="Aam qaida yeh hai ... [S1]",
        refused=False,
        notices=["not_advice"],
        citations=[{"marker": "S1", "file_name": "ID-05__x.pdf", "page": 14, "quote": "date of birth"}],
    )
    msgs = build_messages(inp)
    assert msgs[0]["role"] == "system" and "JSON" in msgs[0]["content"]
    user = msgs[1]["content"]
    for needle in (
        "Reference answer",
        "State the general rule",
        "[S1] ID-05__x.pdf, p.14",
        "not_advice, referral",
        "personal_directive",
        "Roman Urdu",
    ):
        assert needle in user


# ---------------------------------------------------------------- client: cache, offline, limits


async def test_judge_calls_nvidia_then_uses_disk_cache(tmp_path: Path) -> None:
    rec = Recorder(['{"correctness": 1, "citation_support": "no", "rationale": "half"}'])
    inp = JudgeInput(item=_item(), answer="x", refused=False)
    async with _client(tmp_path, rec) as client:
        judge = Judge(client, JUDGE_MODEL)
        first = await judge.judge(inp)
        second = await judge.judge(inp)
    assert first.correctness == 1 and not first.cached
    assert second.correctness == 1 and second.cached
    assert len(rec.requests) == 1  # the second verdict came from evals/.cache/judge
    req = rec.requests[0]
    assert req["auth"] == "Bearer nvapi-test"
    assert req["url"].endswith("/v1/chat/completions")
    assert req["body"]["model"] == JUDGE_MODEL and req["body"]["temperature"] == 0
    assert req["body"]["response_format"] == {"type": "json_object"}
    key = cache_key(JUDGE_MODEL, build_messages(inp))
    cached = json.loads((tmp_path / "judge" / f"{key}.json").read_text(encoding="utf-8"))
    assert cached["model"] == JUDGE_MODEL and "half" in cached["content"]


async def test_no_cache_bypasses_reads(tmp_path: Path) -> None:
    rec = Recorder()
    msgs = [{"role": "user", "content": "hi"}]
    async with _client(tmp_path, rec) as client:
        await client.complete(JUDGE_MODEL, msgs)
    async with _client(tmp_path, rec, use_cache=False) as client:
        result = await client.complete(JUDGE_MODEL, msgs)
    assert not result.cached and len(rec.requests) == 2


async def test_offline_reads_cache_and_never_calls(tmp_path: Path) -> None:
    rec = Recorder()
    msgs = [{"role": "user", "content": "hi"}]
    async with _client(tmp_path, rec) as client:
        await client.complete(JUDGE_MODEL, msgs)
    offline = _client(tmp_path, rec, offline=True, settings=Settings(_env_file=None))  # type: ignore[call-arg]
    assert (await offline.complete(JUDGE_MODEL, msgs)).cached
    with pytest.raises(CacheMissError):
        await offline.complete(JUDGE_MODEL, [{"role": "user", "content": "new"}])
    assert len(rec.requests) == 1


async def test_missing_key_is_reported(tmp_path: Path) -> None:
    client = _client(tmp_path, Recorder(), settings=Settings(_env_file=None))  # type: ignore[call-arg]
    with pytest.raises(MissingKeyError):
        await client.complete(JUDGE_MODEL, [{"role": "user", "content": "hi"}])
    verdict, error = await Judge(client, JUDGE_MODEL).try_judge(
        JudgeInput(item=_item(), answer="x", refused=False)
    )
    assert verdict is None and error and "NVIDIA_API_KEY" in error


async def test_retries_429_and_drops_json_mode_when_rejected(tmp_path: Path) -> None:
    rec = Recorder(statuses=[429, 400, 200])
    async with _client(tmp_path, rec) as client:
        result = await client.complete(JUDGE_MODEL, [{"role": "user", "content": "hi"}])
    assert result.content and len(rec.requests) == 3
    assert "response_format" in rec.requests[1]["body"]
    assert "response_format" not in rec.requests[2]["body"]  # the model rejected JSON mode


async def test_rate_limiter_spaces_calls() -> None:
    now = [0.0]
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    limiter = RateLimiter(30, clock=lambda: now[0], sleep=fake_sleep)
    for _ in range(3):
        await limiter.acquire()
    assert sleeps == [pytest.approx(2.0), pytest.approx(2.0)]  # 60 / 30 rpm


def test_client_caps_rpm_at_35(tmp_path: Path) -> None:
    client = _client(tmp_path, Recorder(), settings=_settings(llm_max_rpm=1000))
    assert client.limiter.interval == pytest.approx(60 / 35)
