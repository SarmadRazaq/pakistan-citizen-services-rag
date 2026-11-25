"""X-Eval-Config on /ask and /search: ignored unless enabled and admin, validated, echoed when applied."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest
from fastapi import FastAPI

from app.config import Settings
from app.core.tenancy import TenantContext
from tests.test_ask_support import (
    PASSPORT_TEXT,
    World,
    add_doc,
    answer_reply,
    close_worlds,  # noqa: F401 (autouse fixture)
    helper_reply,
    make_world,
)

QUOTE = "Urgent passport fee for 36 pages is Rs 9,000"
MODEL = "qwen/qwen3-235b-a22b"
HEADER = "X-Eval-Config"
APPLIED = "X-Eval-Config-Applied"
ASK = {"question": "What is the urgent passport fee?"}
SEARCH = {"query": "urgent passport fee"}


def _header(**overrides: object) -> dict[str, str]:
    return {HEADER: json.dumps({"config": "T1", **overrides}, separators=(",", ":"))}


@pytest.fixture
async def world() -> World:
    w = await make_world(
        hosted=[helper_reply(), answer_reply("The urgent fee is Rs 9,000 [S1].", [("S1", QUOTE)])]
    )
    await add_doc(w, "d_pass", PASSPORT_TEXT, agency="DGIP")
    return w


@pytest.fixture
def build(make_app: Callable[..., FastAPI], world: World, settings: Settings) -> Callable[..., FastAPI]:
    def factory(tenant: TenantContext, *, enabled: bool = True) -> FastAPI:
        app = make_app(tenant=tenant)
        app.state.ask = world.service
        app.state.settings = settings.model_copy(
            update={
                "eval_overrides_enabled": enabled,
                "generation_models": ["deepseek-ai/deepseek-v3.1", MODEL],
            }
        )
        return app

    return factory


async def _call(app: FastAPI, path: str, body: dict[str, object], headers: dict[str, str]) -> httpx.Response:
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        return await c.post(f"/api/v1/{path}", json=body, headers=headers)


async def test_default_setting_is_off(settings: Settings) -> None:
    assert settings.eval_overrides_enabled is False


@pytest.mark.parametrize(("path", "body"), [("ask", ASK), ("search", SEARCH)])
async def test_ignored_when_disabled_even_for_admin(
    build: Callable[..., FastAPI], admin: TenantContext, world: World, path: str, body: dict[str, object]
) -> None:
    r = await _call(build(admin, enabled=False), path, body, _header(retrieval_mode="dense"))
    assert r.status_code == 200, r.text
    assert APPLIED not in r.headers
    assert world.retriever.queries[-1].overrides is None


async def test_bad_header_is_ignored_when_disabled(
    build: Callable[..., FastAPI], admin: TenantContext
) -> None:
    r = await _call(build(admin, enabled=False), "search", SEARCH, {HEADER: "not json"})
    assert r.status_code == 200 and APPLIED not in r.headers


@pytest.mark.parametrize(("path", "body"), [("ask", ASK), ("search", SEARCH)])
async def test_ignored_for_non_admin(
    build: Callable[..., FastAPI], user_a: TenantContext, world: World, path: str, body: dict[str, object]
) -> None:
    r = await _call(build(user_a), path, body, _header(retrieval_mode="dense", rerank=False))
    assert r.status_code == 200, r.text
    assert APPLIED not in r.headers
    assert world.retriever.queries[-1].overrides is None


async def test_non_admin_bad_header_is_ignored_too(
    build: Callable[..., FastAPI], user_a: TenantContext
) -> None:
    r = await _call(build(user_a), "search", SEARCH, {HEADER: "not json"})
    assert r.status_code == 200


@pytest.mark.parametrize(("path", "body"), [("ask", ASK), ("search", SEARCH)])
async def test_honoured_for_admin_and_echoed(
    build: Callable[..., FastAPI], admin: TenantContext, world: World, path: str, body: dict[str, object]
) -> None:
    r = await _call(build(admin), path, body, _header(retrieval_mode="dense", rerank=False))
    assert r.status_code == 200, r.text
    assert r.headers[APPLIED] == '{"config":"T1","rerank":false,"retrieval_mode":"dense"}'
    seen = world.retriever.queries[-1].overrides
    assert seen is not None and seen.retrieval_mode == "dense" and seen.rerank is False


async def test_no_header_means_no_echo_for_admin(build: Callable[..., FastAPI], admin: TenantContext) -> None:
    r = await _call(build(admin), "search", SEARCH, {})
    assert r.status_code == 200 and APPLIED not in r.headers


async def test_generation_model_reaches_the_generate_call(
    build: Callable[..., FastAPI], admin: TenantContext, world: World
) -> None:
    r = await _call(build(admin), "ask", ASK, _header(generation_model=MODEL))
    assert r.status_code == 200, r.text
    assert r.headers[APPLIED].count(MODEL) == 1
    assert world.hosted.models[-1] == MODEL  # the generate call
    assert world.hosted.models[0] is None  # the helper call keeps its own chain


@pytest.mark.parametrize(
    "value",
    [
        "not json",
        '{"retrieval_mode":"graph"}',
        '{"unknown":1}',
        '{"rerank":"yes"}',
        '{"generation_model":"not/allowed"}',
    ],
)
@pytest.mark.parametrize(("path", "body"), [("ask", ASK), ("search", SEARCH)])
async def test_bad_header_is_422_problem(
    build: Callable[..., FastAPI], admin: TenantContext, path: str, body: dict[str, object], value: str
) -> None:
    r = await _call(build(admin), path, body, {HEADER: value})
    assert r.status_code == 422
    assert r.headers["content-type"].startswith("application/problem+json")
    assert r.json()["code"] == "validation_failed"
    assert APPLIED not in r.headers
