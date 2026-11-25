"""OpenAI / neutral-LLM setup: env aliases, OpenAI embedder, factory, validation, chat body,
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
from app.guardrails.input_guards import build_injection_classifier
from app.observability.cost import ModelUsage, request_cost
from app.protocols import LLMMessage
from app.retrieval.factory import build_retrieval
from app.retrieval.hosted import NvidiaEmbedder, OpenAIEmbedder
from app.retrieval.reranker import NoReranker

OPENAI_KEY = "sk-test-openai-0123456789abcdefghijkl"
NVIDIA_KEY = "nvapi-test-key-0123456789abcdefghij"
OPENAI_BASE = "https://api.openai.com/v1"
DIM = 4
ENV_NAMES = (
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "NVIDIA_BASE_URL",
    "NVIDIA_API_KEY",
    "NVIDIA_RETRIEVAL_API_KEY",
    "OPENAI_API_KEY",
    "LLM_BILLED",
    "NVIDIA_BILLED",
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
        nvidia_base_url=OPENAI_BASE,
        nvidia_api_key=OPENAI_KEY,
        embedding_provider="openai",
        reranker_provider="none",
        injection_provider="none",
    )
    base.update(overrides)
    return make_settings(**base)


# ---------------------------------------------------------------- neutral LLM env names


def test_llm_env_names_are_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_BASE_URL", OPENAI_BASE)
    monkeypatch.setenv("LLM_API_KEY", OPENAI_KEY)
    s = make_settings()
    assert s.nvidia_base_url == OPENAI_BASE
    assert s.nvidia_api_key is not None and s.nvidia_api_key.get_secret_value() == OPENAI_KEY
    assert s.llm_is_openai and s.llm_key == OPENAI_KEY


def test_old_nvidia_env_names_still_work(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NVIDIA_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("NVIDIA_API_KEY", NVIDIA_KEY)
    s = make_settings()
    assert s.nvidia_base_url == "https://example.test/v1"
    assert s.llm_key == NVIDIA_KEY


def test_llm_names_win_over_old_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", OPENAI_KEY)
    monkeypatch.setenv("NVIDIA_API_KEY", NVIDIA_KEY)
    s = make_settings()
    assert s.llm_key == OPENAI_KEY
    assert s.nvidia_key == NVIDIA_KEY  # the NVIDIA retrieval key still reads NVIDIA_API_KEY


def test_field_names_still_work_as_init_arguments() -> None:
    s = make_settings(nvidia_api_key=NVIDIA_KEY, nvidia_base_url="https://integrate.api.nvidia.com/v1")
    assert s.llm_key == NVIDIA_KEY and s.nvidia_key == NVIDIA_KEY and s.llm_is_nvidia


def test_openai_key_alone_serves_llm_and_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    s = make_settings(nvidia_base_url=OPENAI_BASE)
    assert s.llm_key == OPENAI_KEY and s.openai_key == OPENAI_KEY


def test_openai_key_is_never_reused_for_nvidia() -> None:
    s = openai_settings(embedding_provider="nvidia", reranker_provider="nvidia", injection_provider="nvidia")
    assert s.nvidia_key == ""
    assert build_injection_classifier(s) is None  # no NVIDIA key: the OpenAI key is not sent to NVIDIA


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


async def test_openai_embedder_uses_same_sparse_as_nvidia() -> None:
    text = "passport fee Rs 4,500"
    openai = await OpenAIEmbedder(openai_settings(), client=Recorder().client()).embed_query(text)
    nv = NvidiaEmbedder(make_settings(nvidia_api_key=NVIDIA_KEY), client=Recorder().client())
    assert openai.sparse == (await nv.embed_query(text)).sparse


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


async def test_openai_embedder_key_falls_back_to_openai_llm_key() -> None:
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
    s = make_settings(embedding_provider="openai")  # LLM base is NVIDIA, no OpenAI key
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


def test_invalid_new_provider_values_rejected() -> None:
    with pytest.raises(ValueError):
        make_settings(embedding_provider="cohere")
    with pytest.raises(ValueError):
        make_settings(reranker_provider="openai")
    with pytest.raises(ValueError):
        make_settings(llm_token_param="max_output_tokens")


# ---------------------------------------------------------------- production validation


GOOD_PROD: dict[str, Any] = {
    "jwt_secret": "x" * 48,
    "cors_origins": ["https://app.example.org"],
    "log_question_text": False,
}
NO_NVIDIA: dict[str, Any] = {"reranker_provider": "none", "injection_provider": "none"}


def prod(**overrides: Any) -> Settings:
    return make_settings(app_env="prod", **{**GOOD_PROD, **overrides})


def test_prod_openai_setup_needs_no_nvidia_key() -> None:
    s = prod(nvidia_base_url=OPENAI_BASE, nvidia_api_key=OPENAI_KEY, embedding_provider="openai", **NO_NVIDIA)
    assert s.nvidia_key == ""


def test_prod_openai_key_alone_is_enough_on_openai_base() -> None:
    s = prod(nvidia_base_url=OPENAI_BASE, openai_api_key=OPENAI_KEY, embedding_provider="openai", **NO_NVIDIA)
    assert s.llm_key == OPENAI_KEY


def test_prod_llm_key_always_required() -> None:
    with pytest.raises(ValueError, match="LLM_API_KEY"):
        prod(embedding_provider="openai", openai_api_key=OPENAI_KEY, **NO_NVIDIA)


def test_prod_openai_embeddings_need_an_openai_key() -> None:
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        prod(nvidia_api_key=NVIDIA_KEY, embedding_provider="openai", **NO_NVIDIA)  # NVIDIA LLM


def test_prod_nvidia_retrieval_needs_an_nvidia_key_when_llm_is_openai() -> None:
    common: dict[str, Any] = dict(
        nvidia_base_url=OPENAI_BASE,
        nvidia_api_key=OPENAI_KEY,
        embedding_provider="openai",
        reranker_provider="nvidia",
        injection_provider="none",
    )
    with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
        prod(**common)
    ok = prod(nvidia_retrieval_api_key=NVIDIA_KEY, **common)
    assert ok.nvidia_key == NVIDIA_KEY


def test_prod_error_does_not_echo_keys() -> None:
    with pytest.raises(ValueError) as exc:
        prod(nvidia_base_url=OPENAI_BASE, nvidia_api_key=OPENAI_KEY, embedding_provider="nvidia")
    assert "NVIDIA_API_KEY" in str(exc.value)
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
        nvidia_base_url=OPENAI_BASE,
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


def test_openai_base_url_counts_as_real_cost() -> None:
    cost, listed = request_cost(
        openai_settings(model_prices=PRICES), [ModelUsage("gpt-4.1-mini", "nvidia", 1_000_000, 1_000_000)]
    )
    assert cost == pytest.approx(2.0) == listed


def test_nvidia_base_url_stays_free_by_default() -> None:
    cost, listed = request_cost(
        make_settings(model_prices=PRICES), [ModelUsage("gpt-4.1-mini", "nvidia", 1_000_000, 0)]
    )
    assert cost == 0.0 and listed == pytest.approx(0.4)


def test_llm_billed_overrides_the_base_url_guess() -> None:
    usage = [ModelUsage("gpt-4.1-mini", "nvidia", 1_000_000, 0)]
    free, _ = request_cost(openai_settings(model_prices=PRICES, llm_billed=False), usage)
    paid, _ = request_cost(make_settings(model_prices=PRICES, llm_billed=True), usage)
    assert free == 0.0 and paid == pytest.approx(0.4)


def test_old_nvidia_billed_env_name_still_works(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NVIDIA_BILLED", "true")
    assert make_settings().hosted_billed is True
    monkeypatch.setenv("LLM_BILLED", "false")
    assert make_settings(nvidia_base_url=OPENAI_BASE).hosted_billed is False


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
        s = make_settings(nvidia_base_url=OPENAI_BASE)
        probes = {p.name: p for p in default_probes(s, client)}
        ok, detail = await probes["llm_api"].check()
    assert not ok and detail == "LLM_API_KEY is not set" and seen == []


async def test_nvidia_defaults_keep_the_single_llm_probe() -> None:
    async with httpx.AsyncClient() as client:
        names = [p.name for p in default_probes(make_settings(nvidia_api_key=NVIDIA_KEY), client)]
    assert names == ["qdrant", "llm_api", "ollama"]


async def test_nvidia_retrieval_probe_added_when_llm_is_openai() -> None:
    s = openai_settings(reranker_provider="nvidia", nvidia_retrieval_api_key=NVIDIA_KEY)
    async with httpx.AsyncClient() as client:
        names = [p.name for p in default_probes(s, client)]
    assert "nvidia_retrieval" in names


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
        out, skipped = module.check_retrieval(client, settings, [])
    return out, skipped, seen


def test_check_models_probes_only_openai_embeddings_in_the_openai_setup() -> None:
    out, skipped, seen = run_check(openai_settings())
    assert list(out) == ["text-embedding-3-large"] and out["text-embedding-3-large"]["reachable"] is True
    assert [str(r.url) for r in seen] == [f"{OPENAI_BASE}/embeddings"]
    assert json.loads(seen[0].content)["dimensions"] == DIM
    assert seen[0].headers["authorization"] == f"Bearer {OPENAI_KEY}"
    assert set(skipped) == {"reranker", "injection"}
    assert "none" in skipped["reranker"] and "none" in skipped["injection"]


def test_check_models_flags_wrong_embedding_dimension() -> None:
    module = load_check_models()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.0] * (DIM + 1)}]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        out, _ = module.check_retrieval(client, openai_settings(), [])
    assert out["text-embedding-3-large"]["reachable"] is False


def test_check_models_nvidia_defaults_probe_embedder_and_reranker() -> None:
    out, skipped, seen = run_check(make_settings(nvidia_api_key=NVIDIA_KEY))
    assert {e["role"] for e in out.values()} == {"embedding", "reranker"}
    assert all(r.headers["authorization"] == f"Bearer {NVIDIA_KEY}" for r in seen)
    assert set(skipped) == {"injection"}


def test_check_models_local_providers_are_skipped() -> None:
    out, skipped, seen = run_check(
        make_settings(embedding_provider="local", reranker_provider="local", injection_provider="none")
    )
    assert out == {} and seen == [] and set(skipped) == {"embedding", "reranker", "injection"}


def test_check_models_never_sends_openai_key_to_nvidia() -> None:
    s = openai_settings(embedding_provider="nvidia", reranker_provider="nvidia")
    out, skipped, seen = run_check(s)
    assert out == {} and seen == []
    assert "no NVIDIA key" in skipped["embedding"] and "no NVIDIA key" in skipped["reranker"]


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
