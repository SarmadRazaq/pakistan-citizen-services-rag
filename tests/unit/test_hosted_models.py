"""The OpenAI embedder against httpx.MockTransport (request shape, batching, retries, validation)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from app.config import Settings
from app.core.errors import AppError, ErrorCode
from app.protocols import Embedder, Embedding
from app.retrieval.hosted import LexicalSparseEncoder, OpenAIEmbedder

KEY = "sk-test-openai-0123456789abcdefghijkl"
DIM = 4
EMBED_URL = "https://api.openai.com/v1/embeddings"
MODEL = "text-embedding-3-large"

Handler = Callable[[httpx.Request, dict[str, Any]], httpx.Response]


def make_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = dict(
        _env_file=None, app_env="test", openai_api_key=KEY, embedding_dimension=DIM, embed_batch_size=2
    )
    values.update(overrides)
    return Settings(**values)


class Recorder:
    def __init__(self, handler: Handler | list[httpx.Response]) -> None:
        self.handler = handler
        self.calls: list[tuple[httpx.Request, dict[str, Any]]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        self.calls.append((request, body))
        if isinstance(self.handler, list):
            steps = self.handler
            return steps.pop(0) if len(steps) > 1 else steps[0]
        return self.handler(request, body)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self))


def vec_for(text: str) -> list[float]:
    return [float(len(text)), 1.0, 0.0, 2.0]


def embed_handler(*, shuffle: bool = False) -> Handler:
    def handle(request: httpx.Request, body: dict[str, Any]) -> httpx.Response:
        data = [
            {"index": i, "embedding": vec_for(t), "object": "embedding"} for i, t in enumerate(body["input"])
        ]
        if shuffle:
            data.reverse()
        return httpx.Response(200, json={"data": data, "model": body["model"]})

    return handle


@pytest.fixture
def sleeps() -> list[float]:
    return []


@pytest.fixture
def fake_sleep(sleeps: list[float]) -> Callable[[float], Any]:
    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    return sleep


# ---------------------------------------------------------------- embedder


async def test_embedder_attributes_and_lifecycle_make_no_network_call(fake_sleep: Any) -> None:
    rec = Recorder(embed_handler())
    emb = OpenAIEmbedder(make_settings(), client=rec.client(), sleep=fake_sleep)
    assert isinstance(emb, Embedder)
    assert emb.model_name == MODEL
    assert emb.dimension == DIM
    assert emb.loaded is True
    await emb.warmup()
    emb.unload()
    await emb.aclose()
    assert rec.calls == []


async def test_embed_documents_request_shape(fake_sleep: Any) -> None:
    rec = Recorder(embed_handler())
    emb = OpenAIEmbedder(make_settings(embed_batch_size=32), client=rec.client(), sleep=fake_sleep)
    await emb.embed_documents(["alpha", "beta"])
    ((request, body),) = rec.calls
    assert request.method == "POST"
    assert str(request.url) == EMBED_URL
    assert request.headers["authorization"] == f"Bearer {KEY}"
    assert body == {
        "model": MODEL,
        "input": ["alpha", "beta"],
        "encoding_format": "float",
        "dimensions": DIM,
    }


async def test_embed_documents_returns_embeddings_with_sparse(fake_sleep: Any) -> None:
    rec = Recorder(embed_handler())
    sparse = LexicalSparseEncoder()
    emb = OpenAIEmbedder(make_settings(), client=rec.client(), sparse=sparse, sleep=fake_sleep)
    texts = ["passport fee", "visa office"]
    out = await emb.embed_documents(texts)
    assert all(isinstance(e, Embedding) for e in out)
    assert [e.dense for e in out] == [tuple(vec_for(t)) for t in texts]
    assert [e.sparse for e in out] == [sparse.encode_document(t) for t in texts]


async def test_embed_documents_batches_by_setting(fake_sleep: Any) -> None:
    rec = Recorder(embed_handler())
    emb = OpenAIEmbedder(make_settings(embed_batch_size=2), client=rec.client(), sleep=fake_sleep)
    texts = ["a", "bb", "ccc", "dddd", "eeeee"]
    out = await emb.embed_documents(texts)
    assert [len(b["input"]) for _, b in rec.calls] == [2, 2, 1]
    assert [e.dense[0] for e in out] == [float(len(t)) for t in texts]


async def test_embed_documents_orders_by_response_index(fake_sleep: Any) -> None:
    rec = Recorder(embed_handler(shuffle=True))
    emb = OpenAIEmbedder(make_settings(embed_batch_size=32), client=rec.client(), sleep=fake_sleep)
    out = await emb.embed_documents(["a", "bb", "ccc"])
    assert [e.dense[0] for e in out] == [1.0, 2.0, 3.0]


async def test_embed_documents_empty_makes_no_call(fake_sleep: Any) -> None:
    rec = Recorder(embed_handler())
    emb = OpenAIEmbedder(make_settings(), client=rec.client(), sleep=fake_sleep)
    assert await emb.embed_documents([]) == []
    assert rec.calls == []


async def test_embed_query_uses_query_sparse(fake_sleep: Any) -> None:
    rec = Recorder(embed_handler())
    sparse = LexicalSparseEncoder()
    emb = OpenAIEmbedder(make_settings(), client=rec.client(), sparse=sparse, sleep=fake_sleep)
    e = await emb.embed_query("passport fee passport")
    ((request, body),) = rec.calls
    assert str(request.url) == EMBED_URL
    assert body["input"] == ["passport fee passport"]
    assert e.dense == tuple(vec_for("passport fee passport"))
    assert e.sparse == sparse.encode_query("passport fee passport")


async def test_embed_query_is_cached(fake_sleep: Any) -> None:
    rec = Recorder(embed_handler())
    emb = OpenAIEmbedder(make_settings(), client=rec.client(), sleep=fake_sleep)
    first = await emb.embed_query("same question")
    second = await emb.embed_query("same question")
    assert first == second
    assert len(rec.calls) == 1
    await emb.embed_query("another question")
    assert len(rec.calls) == 2


async def test_embedder_rejects_wrong_dimension(fake_sleep: Any) -> None:
    def handle(request: httpx.Request, body: dict[str, Any]) -> httpx.Response:
        data = [{"index": i, "embedding": [0.1, 0.2]} for i, _ in enumerate(body["input"])]
        return httpx.Response(200, json={"data": data})

    emb = OpenAIEmbedder(make_settings(), client=Recorder(handle).client(), sleep=fake_sleep)
    with pytest.raises(AppError) as exc:
        await emb.embed_query("x")
    assert exc.value.code is ErrorCode.DEPENDENCY_UNAVAILABLE


@pytest.mark.parametrize("code", [429, 500, 502, 503])
async def test_embedder_retries_twice_then_dependency_unavailable(
    code: int, fake_sleep: Any, sleeps: list[float]
) -> None:
    rec = Recorder([httpx.Response(code)])
    emb = OpenAIEmbedder(make_settings(), client=rec.client(), sleep=fake_sleep)
    with pytest.raises(AppError) as exc:
        await emb.embed_query("x")
    assert exc.value.code is ErrorCode.DEPENDENCY_UNAVAILABLE
    assert len(rec.calls) == 3
    assert len(sleeps) == 2


async def test_embedder_recovers_after_transient_failure(fake_sleep: Any, sleeps: list[float]) -> None:
    good = httpx.Response(200, json={"data": [{"index": 0, "embedding": vec_for("x")}]})
    rec = Recorder([httpx.Response(429), httpx.Response(503), good])
    emb = OpenAIEmbedder(make_settings(), client=rec.client(), sleep=fake_sleep)
    e = await emb.embed_query("x")
    assert e.dense == tuple(vec_for("x"))
    assert len(rec.calls) == 3
    assert len(sleeps) == 2


@pytest.mark.parametrize("code", [401, 403])
async def test_embedder_auth_failure_is_not_retried(code: int, fake_sleep: Any, sleeps: list[float]) -> None:
    rec = Recorder([httpx.Response(code)])
    emb = OpenAIEmbedder(make_settings(), client=rec.client(), sleep=fake_sleep)
    with pytest.raises(AppError) as exc:
        await emb.embed_documents(["x"])
    assert exc.value.code is ErrorCode.DEPENDENCY_UNAVAILABLE
    assert len(rec.calls) == 1
    assert sleeps == []


@pytest.mark.parametrize("code", [400, 422])
async def test_embedder_rejected_request_is_not_retried(
    code: int, fake_sleep: Any, sleeps: list[float]
) -> None:
    rec = Recorder([httpx.Response(code)])
    emb = OpenAIEmbedder(make_settings(), client=rec.client(), sleep=fake_sleep)
    with pytest.raises(AppError) as exc:
        await emb.embed_documents(["x"])
    assert exc.value.code is ErrorCode.DEPENDENCY_UNAVAILABLE
    assert len(rec.calls) == 1
    assert sleeps == []


async def test_embedder_missing_key_makes_no_call(fake_sleep: Any) -> None:
    rec = Recorder(embed_handler())
    emb = OpenAIEmbedder(make_settings(openai_api_key=None), client=rec.client(), sleep=fake_sleep)
    with pytest.raises(AppError) as exc:
        await emb.embed_query("x")
    assert exc.value.code is ErrorCode.DEPENDENCY_UNAVAILABLE
    assert rec.calls == []
