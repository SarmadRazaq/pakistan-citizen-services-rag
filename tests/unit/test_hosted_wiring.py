"""Settings defaults, retrieval factory and LLM chain wiring for the hosted (OpenAI) setup."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.generation.llm import LLMRouter
from app.retrieval.embedder import BGEM3Embedder
from app.retrieval.factory import build_retrieval
from app.retrieval.hosted import OpenAIEmbedder
from app.retrieval.reranker import BGEReranker, NoReranker

KEY = "sk-test-openai-0123456789abcdefghijkl"


def make_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = dict(_env_file=None, app_env="test", llm_api_key=KEY, openai_api_key=KEY)
    values.update(overrides)
    return Settings(**values)


# ---------------------------------------------------------------- settings


def test_hosted_setting_defaults() -> None:
    s = make_settings(llm_api_key=None)
    assert s.llm_base_url == "https://api.openai.com/v1"
    assert s.generation_models == ["gpt-4.1-mini", "gpt-4o-mini"]
    assert s.helper_model == "gpt-4.1-mini"
    assert s.judge_model == "gpt-4o"
    assert s.llm_max_rpm == 300
    assert s.hosted_billed is True
    assert s.embedding_provider == "openai"
    assert s.openai_embedding_model == "text-embedding-3-large"
    assert s.embedding_dimension == 1024
    assert s.reranker_provider == "local"
    assert s.reranker_model == "BAAI/bge-reranker-v2-m3"
    assert s.local_fallback_enabled is False
    assert s.hosted_retrieval_timeout_s == 20.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("embedding_provider", "bogus"),
        ("embedding_provider", "hosted"),
        ("reranker_provider", "bogus"),
        ("reranker_provider", "hosted"),
    ],
)
def test_invalid_provider_rejected(field: str, value: str) -> None:
    with pytest.raises(ValidationError):
        make_settings(**{field: value})


def test_removed_injection_provider_setting_is_ignored() -> None:
    assert not hasattr(make_settings(), "injection_provider")


def test_local_only_policy_requires_local_fallback() -> None:
    with pytest.raises(ValidationError):
        make_settings(private_context_policy="local_only", local_fallback_enabled=False)
    ok = make_settings(private_context_policy="local_only", local_fallback_enabled=True)
    assert ok.private_context_policy == "local_only"


def test_hosted_with_notice_without_fallback_is_valid() -> None:
    s = make_settings(private_context_policy="hosted_with_notice", local_fallback_enabled=False)
    assert s.local_fallback_enabled is False


# ---------------------------------------------------------------- factory


@pytest.mark.filterwarnings("ignore:Payload indexes have no effect:UserWarning")
async def test_factory_defaults_build_openai_embedder_and_local_reranker() -> None:
    settings = make_settings(embedding_dimension=768)
    bundle = build_retrieval(settings, qdrant_location=":memory:")
    assert isinstance(bundle.embedder, OpenAIEmbedder)
    assert isinstance(bundle.reranker, BGEReranker)
    assert bundle.store.dense_dim == 768 == bundle.embedder.dimension
    await bundle.startup()
    await bundle.shutdown()


@pytest.mark.filterwarnings("ignore:Payload indexes have no effect:UserWarning")
async def test_factory_local_providers_build_bge_models() -> None:
    settings = make_settings(embedding_provider="local", reranker_provider="local")
    bundle = build_retrieval(settings, qdrant_location=":memory:")
    assert isinstance(bundle.embedder, BGEM3Embedder)
    assert isinstance(bundle.reranker, BGEReranker)
    assert not bundle.embedder.loaded
    assert not bundle.reranker.loaded
    await bundle.shutdown()


@pytest.mark.filterwarnings("ignore:Payload indexes have no effect:UserWarning")
async def test_factory_mixed_providers() -> None:
    settings = make_settings(embedding_provider="local", reranker_provider="none")
    bundle = build_retrieval(settings, qdrant_location=":memory:")
    assert isinstance(bundle.embedder, BGEM3Embedder)
    assert isinstance(bundle.reranker, NoReranker)
    await bundle.shutdown()


# ---------------------------------------------------------------- LLM chain


def router(tmp_path: Path, **overrides: Any) -> LLMRouter:
    return LLMRouter(make_settings(llm_cache_dir=tmp_path / "c", **overrides))


@pytest.mark.parametrize("purpose", ["generate", "helper"])
def test_chain_has_no_ollama_by_default(tmp_path: Path, purpose: str) -> None:
    chain = router(tmp_path).chain(purpose)  # type: ignore[arg-type]
    assert chain
    assert all(t.endpoint == "hosted" for t in chain)


@pytest.mark.parametrize("purpose", ["generate", "helper"])
def test_chain_has_ollama_last_when_enabled(tmp_path: Path, purpose: str) -> None:
    chain = router(tmp_path, local_fallback_enabled=True).chain(purpose)  # type: ignore[arg-type]
    assert chain[-1].endpoint == "ollama"
    assert all(t.endpoint == "hosted" for t in chain[:-1])


def test_judge_chain_never_has_ollama(tmp_path: Path) -> None:
    for enabled in (False, True):
        chain = router(tmp_path, local_fallback_enabled=enabled).chain("judge")
        assert chain
        assert all(t.endpoint == "hosted" for t in chain)
