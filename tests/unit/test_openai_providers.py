"""OpenAI / neutral-LLM setup: env names, OpenAI embedder, factory, validation, chat body,
health probes, cost and the check_models script. httpx.MockTransport only; no network."""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from evals.llm import ChatClient

from app.config import Settings
from app.core.errors import AppError
from app.core.health import default_probes
from app.generation.llm import LLMRouter
from app.observability.cost import ModelUsage, request_cost
from app.protocols import LLMMessage
from app.retrieval.factory import build_retrieval
from app.retrieval.hosted import LexicalSparseEncoder, OpenAIEmbedder
from app.retrieval.reranker import NoReranker

OPENAI_KEY = "sk-test-openai-0123456789abcdefghijkl"
OTHER_BASE = "https://llm.example.test/v1"
OPENAI_BASE = "https://api.openai.com/v1"
DIM = 4
ENV_NAMES = (
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "OPENAI_API_KEY",
    "LLM_BILLED",
    "LLM_TOKEN_PARAM",
    "EMBEDDING_PROVIDER",
    "RERANKER_PROVIDER",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def make_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = dict(_env_file=None, app_env="test", embedding_dimension=DIM)
    values.update(overrides)
    return Settings(**values)


def openai_settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = dict(
        llm_base_url=OPENAI_BASE,
        llm_api_key=OPENAI_KEY,
        embedding_provider="openai",
        reranker_provider="none",
    )
    base.update(overrides)
    return make_settings(**base)


# ---------------------------------------------------------------- neutral LLM env names


def test_llm_env_names_are_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_BASE_URL", OTHER_BASE)
    monkeypatch.setenv("LLM_API_KEY", OPENAI_KEY)
    s = make_settings()
    assert s.llm_base_url == OTHER_BASE
    assert s.llm_api_key is not None and s.llm_api_key.get_secret_value() == OPENAI_KEY
    assert not s.llm_is_openai and s.llm_key == OPENAI_KEY


def test_llm_defaults_to_the_openai_api() -> None:
    s = make_settings()
    assert s.llm_base_url == OPENAI_BASE and s.llm_is_openai


def test_field_names_still_work_as_init_arguments() -> None:
    s = make_settings(llm_api_key=OPENAI_KEY, llm_base_url=OTHER_BASE)
    assert s.llm_key == OPENAI_KEY and s.llm_base_url == OTHER_BASE


def test_openai_key_alone_serves_llm_and_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    s = make_settings()
    assert s.llm_key == OPENAI_KEY and s.openai_key == OPENAI_KEY


def test_llm_api_key_wins_over_openai_key_for_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "sk-chat-only")
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    s = make_settings()
    assert s.llm_key == "sk-chat-only" and s.openai_key == OPENAI_KEY


def test_openai_key_is_never_reused_for_another_llm_host() -> None:
    s = make_settings(llm_base_url=OTHER_BASE, openai_api_key=OPENAI_KEY)
    assert s.llm_key == ""  # the OpenAI key is not sent to a non-OpenAI chat endpoint


def test_llm_key_is_not_reused_for_embeddings_on_another_llm_host() -> None:
    s = make_settings(llm_base_url=OTHER_BASE, llm_api_key="sk-other-host")
    assert s.openai_key == ""


def test_openai_defaults() -> None:
    s = make_settings()
    assert s.openai_base_url == OPENAI_BASE
    assert s.openai_embedding_model == "text-embedding-3-large"
    assert s.openai_api_key is None
    assert s.llm_token_param == "max_tokens"


# ---------------------------------------------------------------- OpenAI embedder


def vec_for(text: str) -> list[float]:
    return [float(len(text)), 1.0, 0.0, 2.0]


class Recorder:
    def __init__(
        self, handler: Callable[[httpx.Request, dict[str, Any]], httpx.Response] | None = None
    ) -> None:
        self.handler = handler or self.ok
        self.calls: list[tuple[httpx.Request, dict[str, Any]]] = []

    @staticmethod
    def ok(request: httpx.Request, body: dict[str, Any]) -> httpx.Response:
        data = [{"index": i, "embedding": vec_for(t)} for i, t in enumerate(body["input"])]
        return httpx.Response(200, json={"data": data})

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.calls.append((request, body))
        return self.handler(request, body)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self))


async def test_openai_embedder_request_shape() -> None:
    rec = Recorder()
    emb = OpenAIEmbedder(openai_settings(), client=rec.client())
    out = await emb.embed_documents(["passport fee", "cnic"])
    request, body = rec.calls[0]
    assert str(request.url) == f"{OPENAI_BASE}/embeddings"
    assert request.headers["authorization"] == f"Bearer {OPENAI_KEY}"
    assert body == {
        "model": "text-embedding-3-large",
        "input": ["passport fee", "cnic"],
        "encoding_format": "float",
        "dimensions": DIM,
    }
    assert "input_type" not in body and "truncate" not in body
    assert [e.dense for e in out] == [tuple(vec_for("passport fee")), tuple(vec_for("cnic"))]
    assert out[0].sparse.indices  # lexical sparse half


async def test_openai_embedder_sparse_half_is_the_lexical_encoder() -> None:
    text = "passport fee Rs 4,500"
    openai = await OpenAIEmbedder(openai_settings(), client=Recorder().client()).embed_query(text)
    assert openai.sparse == LexicalSparseEncoder().encode_query(text)


async def test_openai_embedder_batches_and_orders() -> None:
    def reversed_rows(request: httpx.Request, body: dict[str, Any]) -> httpx.Response:
        data = [{"index": i, "embedding": vec_for(t)} for i, t in enumerate(body["input"])]
        return httpx.Response(200, json={"data": data[::-1]})

    rec = Recorder(reversed_rows)
    emb = OpenAIEmbedder(openai_settings(embed_batch_size=2), client=rec.client())
    texts = ["a1", "bb22", "ccc333", "d", "ee"]
    out = await emb.embed_documents(texts)
    assert [len(b["input"]) for _, b in rec.calls] == [2, 2, 1]
    assert [e.dense[0] for e in out] == [float(len(t)) for t in texts]


async def test_openai_embedder_batch_never_exceeds_openai_input_limit() -> None:
    emb = OpenAIEmbedder(openai_settings(embed_batch_size=5000), client=Recorder().client())
    assert emb._batch_size == 2048


async def test_openai_embedder_key_falls_back_to_the_llm_key_on_the_openai_base() -> None:
    rec = Recorder()
    await OpenAIEmbedder(openai_settings(), client=rec.client()).embed_query("x")
    assert rec.calls[0][0].headers["authorization"] == f"Bearer {OPENAI_KEY}"


async def test_openai_embedder_prefers_its_own_key() -> None:
    rec = Recorder()
    s = openai_settings(openai_api_key="sk-embeddings-only")
    await OpenAIEmbedder(s, client=rec.client()).embed_query("x")
    assert rec.calls[0][0].headers["authorization"] == "Bearer sk-embeddings-only"


async def test_openai_embedder_missing_key_is_a_dependency_error_without_network() -> None:
    rec = Recorder()
    s = make_settings(embedding_provider="openai")  # no OPENAI_API_KEY, no LLM key
    with pytest.raises(AppError) as exc:
        await OpenAIEmbedder(s, client=rec.client()).embed_query("x")
    assert "OPENAI_API_KEY" in str(exc.value) and rec.calls == []


@pytest.mark.parametrize(
    "bad",
    [
        lambda r, b: httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 2.0]}]}),
        lambda r, b: httpx.Response(200, json={"data": []}),
        lambda r, b: httpx.Response(200, json={"nope": 1}),
    ],
    ids=["wrong_dimension", "wrong_count", "no_data"],
)
async def test_openai_embedder_validates_count_and_dimension(bad: Any) -> None:
    emb = OpenAIEmbedder(openai_settings(), client=Recorder(bad).client())
    with pytest.raises(AppError):
        await emb.embed_query("passport")


async def test_openai_embedder_retries_429_then_succeeds() -> None:
    replies = [httpx.Response(429, headers={"Retry-After": "1"})]

    def handler(request: httpx.Request, body: dict[str, Any]) -> httpx.Response:
        return replies.pop(0) if replies else Recorder.ok(request, body)

    async def no_sleep(_: float) -> None:
        return None

    rec = Recorder(handler)
    emb = OpenAIEmbedder(openai_settings(), client=rec.client(), sleep=no_sleep)
    await emb.embed_query("x")
    assert len(rec.calls) == 2


# ---------------------------------------------------------------- factory and reranker none


@pytest.mark.filterwarnings("ignore:Payload indexes have no effect:UserWarning")
async def test_factory_builds_openai_embedder_and_no_reranker() -> None:
    bundle = build_retrieval(openai_settings(), qdrant_location=":memory:")
    assert isinstance(bundle.embedder, OpenAIEmbedder)
    assert isinstance(bundle.reranker, NoReranker)
    assert bundle.store.dense_dim == DIM
    await bundle.startup(warmup=True)
    await bundle.shutdown()


async def test_no_reranker_refuses_to_be_called() -> None:
    with pytest.raises(RuntimeError):
        await NoReranker().rerank("q", ["p"])


def test_invalid_provider_values_rejected() -> None:
    with pytest.raises(ValueError):
        make_settings(embedding_provider="cohere")
    with pytest.raises(ValueError):
        make_settings(reranker_provider="openai")
    with pytest.raises(ValueError):
        make_settings(embedding_provider="bogus")
    with pytest.raises(ValueError):
        make_settings(llm_token_param="max_output_tokens")


# ---------------------------------------------------------------- production validation


GOOD_PROD: dict[str, Any] = {
    "jwt_secret": "x" * 48,
    "cors_origins": ["https://app.example.org"],
    "log_question_text": False,
}
LOCAL_RETRIEVAL: dict[str, Any] = {"embedding_provider": "local", "reranker_provider": "none"}


def prod(**overrides: Any) -> Settings:
    return make_settings(app_env="prod", **{**GOOD_PROD, **overrides})


def test_prod_one_openai_key_serves_chat_and_embeddings() -> None:
    s = prod(llm_api_key=OPENAI_KEY)
    assert s.llm_key == OPENAI_KEY and s.openai_key == OPENAI_KEY


def test_prod_openai_key_alone_is_enough_on_openai_base() -> None:
    s = prod(openai_api_key=OPENAI_KEY)
    assert s.llm_key == OPENAI_KEY


def test_prod_llm_key_always_required() -> None:
    with pytest.raises(ValueError, match="LLM_API_KEY"):
        prod(**LOCAL_RETRIEVAL)


def test_prod_openai_embeddings_need_an_openai_key() -> None:
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        prod(llm_base_url=OTHER_BASE, llm_api_key="sk-other-host")  # key is not an OpenAI key here


def test_prod_local_embeddings_need_no_openai_key() -> None:
    s = prod(llm_base_url=OTHER_BASE, llm_api_key="sk-other-host", **LOCAL_RETRIEVAL)
    assert s.openai_key == ""


def test_prod_error_does_not_echo_keys() -> None:
    with pytest.raises(ValueError) as exc:
        prod(llm_base_url=OTHER_BASE, llm_api_key=OPENAI_KEY)
    assert "OPENAI_API_KEY" in str(exc.value)
    assert OPENAI_KEY not in str(exc.value)


# ---------------------------------------------------------------- chat compatibility


MESSAGES = [LLMMessage("system", "Answer in JSON."), LLMMessage("user", "passport fee?")]


def chat_ok() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": '{"a": 1}'}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        },
    )


async def router_request(settings: Settings) -> tuple[dict[str, Any], httpx.Request]:
    seen: list[tuple[dict[str, Any], httpx.Request]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((json.loads(request.content), request))
        return chat_ok()

    router = LLMRouter(settings, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    await router.complete(MESSAGES, purpose="generate", max_tokens=321, temperature=0.2)
    return seen[0]


async def test_router_default_body_uses_max_tokens_and_temperature() -> None:
    body, request = await router_request(
        openai_settings(generation_models=["gpt-4.1-mini"], llm_max_rpm=6000)
    )
    assert str(request.url) == f"{OPENAI_BASE}/chat/completions"
    assert request.headers["authorization"] == f"Bearer {OPENAI_KEY}"
    assert body["max_tokens"] == 321 and body["temperature"] == 0.2
    assert body["response_format"] == {"type": "json_object"}
    assert set(body) == {"model", "messages", "stream", "max_tokens", "temperature", "response_format"}


async def test_router_reasoning_body_uses_max_completion_tokens_and_no_temperature() -> None:
    body, _ = await router_request(
        openai_settings(
            generation_models=["gpt-5-mini"], llm_token_param="max_completion_tokens", llm_max_rpm=6000
        )
    )
    assert body["max_completion_tokens"] == 321
    assert "max_tokens" not in body and "temperature" not in body


async def test_router_uses_openai_key_when_only_openai_key_is_set() -> None:
    s = make_settings(
        openai_api_key=OPENAI_KEY,
        generation_models=["gpt-4.1-mini"],
        llm_max_rpm=6000,
    )
    _, request = await router_request(s)
    assert request.headers["authorization"] == f"Bearer {OPENAI_KEY}"


async def eval_body(settings: Settings, tmp_path: Path) -> dict[str, Any]:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return chat_ok()

    client = ChatClient(
        settings,
        cache_dir=tmp_path,
        use_cache=False,
        transport=httpx.MockTransport(handler),
        max_rpm=6000,
    )
    async with client:
        await client.complete("gpt-4o", [{"role": "user", "content": "json please"}], max_tokens=77)
    return seen[0]


async def test_eval_client_default_body(tmp_path: Path) -> None:
    body = await eval_body(openai_settings(), tmp_path)
    assert body["max_tokens"] == 77 and body["temperature"] == 0
    assert "max_completion_tokens" not in body


async def test_eval_client_reasoning_body(tmp_path: Path) -> None:
    body = await eval_body(openai_settings(llm_token_param="max_completion_tokens"), tmp_path)
    assert body["max_completion_tokens"] == 77
    assert "max_tokens" not in body and "temperature" not in body


async def test_eval_client_honours_llm_max_rpm_above_35(tmp_path: Path) -> None:
    client = ChatClient(openai_settings(llm_max_rpm=3000), cache_dir=tmp_path)
    assert client.limiter.interval == pytest.approx(60 / 3000)
    await client.aclose()


# ---------------------------------------------------------------- cost


PRICES = {"gpt-4.1-mini": (0.4, 1.6)}


def test_hosted_calls_are_billed_by_default() -> None:
    cost, listed = request_cost(
        make_settings(model_prices=PRICES), [ModelUsage("gpt-4.1-mini", "hosted", 1_000_000, 1_000_000)]
    )
    assert cost == pytest.approx(2.0) == listed


def test_llm_billed_false_makes_hosted_calls_free_but_keeps_list_price() -> None:
    usage = [ModelUsage("gpt-4.1-mini", "hosted", 1_000_000, 0)]
    free, listed = request_cost(make_settings(model_prices=PRICES, llm_billed=False), usage)
    paid, _ = request_cost(make_settings(model_prices=PRICES, llm_billed=True), usage)
    assert free == 0.0 and listed == pytest.approx(0.4) and paid == pytest.approx(0.4)


def test_llm_billed_env_name(monkeypatch: pytest.MonkeyPatch) -> None:
    assert make_settings().hosted_billed is True
    monkeypatch.setenv("LLM_BILLED", "false")
    assert make_settings().hosted_billed is False


def test_local_fallback_is_free_even_on_openai() -> None:
    cost, _ = request_cost(
        openai_settings(model_prices=PRICES), [ModelUsage("gpt-4.1-mini", "ollama", 1_000_000, 0)]
    )
    assert cost == 0.0


# ---------------------------------------------------------------- health probes


def probe_client(seen: list[httpx.Request]) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"data": [{"id": "m"}]})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_probes_for_the_openai_setup() -> None:
    seen: list[httpx.Request] = []
    async with probe_client(seen) as client:
        probes = {p.name: p for p in default_probes(openai_settings(), client)}
        assert set(probes) == {"qdrant", "llm_api", "openai_embeddings", "ollama"}
        for name in ("llm_api", "openai_embeddings"):
            ok, _ = await probes[name].check()
            assert ok
    assert {str(r.url) for r in seen} == {f"{OPENAI_BASE}/models"}
    assert all(r.headers["authorization"] == f"Bearer {OPENAI_KEY}" for r in seen)


async def test_probe_without_key_fails_without_network() -> None:
    seen: list[httpx.Request] = []
    async with probe_client(seen) as client:
        s = make_settings()
        probes = {p.name: p for p in default_probes(s, client)}
        ok, detail = await probes["llm_api"].check()
    assert not ok and detail == "LLM_API_KEY is not set" and seen == []


async def test_local_embeddings_need_no_openai_probe() -> None:
    s = openai_settings(embedding_provider="local")
    async with httpx.AsyncClient() as client:
        names = [p.name for p in default_probes(s, client)]
    assert names == ["qdrant", "llm_api", "ollama"]


async def test_llm_probe_uses_the_configured_base_url() -> None:
    seen: list[httpx.Request] = []
    async with probe_client(seen) as client:
        s = make_settings(llm_base_url=OTHER_BASE, llm_api_key="sk-other-host", embedding_provider="local")
        probes = {p.name: p for p in default_probes(s, client)}
        ok, _ = await probes["llm_api"].check()
    assert ok and [str(r.url) for r in seen] == [f"{OTHER_BASE}/models"]


# ---------------------------------------------------------------- scripts/check_models.py


def load_check_models() -> Any:
    path = Path(__file__).resolve().parents[2] / "scripts" / "check_models.py"
    spec = importlib.util.spec_from_file_location("check_models_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_check(settings: Settings) -> tuple[dict[str, Any], dict[str, str], list[httpx.Request]]:
    module = load_check_models()
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body = json.loads(request.content)
        if "input" in body:
            return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.0] * DIM}]})
        return httpx.Response(200, json={"rankings": []})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        out, skipped = module.check_retrieval(client, settings)
    return out, skipped, seen


def test_check_models_probes_only_openai_embeddings_when_there_is_no_reranker() -> None:
    out, skipped, seen = run_check(openai_settings())
    assert list(out) == ["text-embedding-3-large"] and out["text-embedding-3-large"]["reachable"] is True
    assert [str(r.url) for r in seen] == [f"{OPENAI_BASE}/embeddings"]
    assert json.loads(seen[0].content)["dimensions"] == DIM
    assert seen[0].headers["authorization"] == f"Bearer {OPENAI_KEY}"
    assert set(skipped) == {"reranker"} and "none" in skipped["reranker"]


def test_check_models_flags_wrong_embedding_dimension() -> None:
    module = load_check_models()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.0] * (DIM + 1)}]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        out, _ = module.check_retrieval(client, openai_settings())
    assert out["text-embedding-3-large"]["reachable"] is False


def test_check_models_defaults_probe_openai_embeddings_and_report_the_local_reranker() -> None:
    out, skipped, seen = run_check(make_settings(llm_api_key=OPENAI_KEY))
    assert {e["role"] for e in out.values()} == {"embedding", "reranker"}
    reranker = out["BAAI/bge-reranker-v2-m3"]
    assert isinstance(reranker["package_installed"], bool) and isinstance(reranker["weights_cached"], bool)
    assert [str(r.url) for r in seen] == [f"{OPENAI_BASE}/embeddings"]  # the reranker is not called
    assert skipped == {}


def test_check_models_local_embeddings_are_skipped() -> None:
    out, skipped, seen = run_check(make_settings(embedding_provider="local", reranker_provider="none"))
    assert out == {} and seen == [] and set(skipped) == {"embedding", "reranker"}


def test_check_models_embeddings_without_a_key_are_skipped() -> None:
    out, skipped, seen = run_check(make_settings(reranker_provider="none"))
    assert out == {} and seen == []
    assert "OPENAI_API_KEY" in skipped["embedding"]


def test_check_models_local_reranker_reports_missing_package(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_check_models()
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda name: None)
    skipped: dict[str, str] = {}
    info = module.check_local_reranker(make_settings(), skipped)
    assert info is not None and info["package_installed"] is False and skipped == {}


def test_check_models_probe_body_for_reasoning_models() -> None:
    module = load_check_models()
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return chat_ok()

    s = openai_settings(llm_token_param="max_completion_tokens")
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = module._probe(client, OPENAI_BASE, "gpt-5-mini", s, OPENAI_KEY)
    assert result["ok"] is True
    body = seen[0]
    assert "max_completion_tokens" in body and "temperature" not in body and "max_tokens" not in body
