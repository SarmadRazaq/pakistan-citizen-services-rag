"""Settings, factory, injection-classifier builder and LLM chain wiring for hosted models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.generation.llm import LLMRouter
from app.guardrails.input_guards import build_injection_classifier
from app.protocols import InjectionClassifier
from app.retrieval.embedder import BGEM3Embedder
from app.retrieval.factory import build_retrieval
from app.retrieval.hosted import NvidiaEmbedder, NvidiaJailbreakClassifier, NvidiaReranker
from app.retrieval.reranker import BGEReranker

KEY = "nvapi-test-key-0123456789abcdefghij"
JAILBREAK_URL = "https://ai.api.nvidia.com/v1/security/nvidia/nemoguard-jailbreak-detect"


def make_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = dict(_env_file=None, app_env="test", nvidia_api_key=KEY)
    values.update(overrides)
    return Settings(**values)


# ---------------------------------------------------------------- settings


def test_hosted_setting_defaults() -> None:
    s = make_settings(nvidia_api_key=None)
    assert s.embedding_provider == "nvidia"
    assert s.hosted_embedding_model == "baai/bge-m3"
    assert s.embedding_dimension == 1024
    assert s.reranker_provider == "nvidia"
    assert s.hosted_reranker_model == "nvidia/llama-3.2-nv-rerankqa-1b-v2"
    assert s.hosted_reranker_url is None
    assert s.nvidia_retrieval_base_url == "https://ai.api.nvidia.com/v1/retrieval"
    assert s.injection_provider == "nvidia"
    assert s.jailbreak_detect_url == JAILBREAK_URL
    assert s.local_fallback_enabled is False
    assert s.hosted_retrieval_timeout_s == 20.0


@pytest.mark.parametrize("field", ["embedding_provider", "reranker_provider", "injection_provider"])
def test_invalid_provider_rejected(field: str) -> None:
    with pytest.raises(ValidationError):
        make_settings(**{field: "bogus"})


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
async def test_factory_defaults_build_hosted_models() -> None:
    settings = make_settings(embedding_dimension=768)
    bundle = build_retrieval(settings, qdrant_location=":memory:")
    assert isinstance(bundle.embedder, NvidiaEmbedder)
    assert isinstance(bundle.reranker, NvidiaReranker)
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
    settings = make_settings(embedding_provider="nvidia", reranker_provider="local")
    bundle = build_retrieval(settings, qdrant_location=":memory:")
    assert isinstance(bundle.embedder, NvidiaEmbedder)
    assert isinstance(bundle.reranker, BGEReranker)
    await bundle.shutdown()


# ---------------------------------------------------------------- injection classifier builder


def test_builder_nvidia_with_key() -> None:
    clf = build_injection_classifier(make_settings(injection_provider="nvidia"))
    assert isinstance(clf, NvidiaJailbreakClassifier)
    assert isinstance(clf, InjectionClassifier)


def test_builder_none_provider() -> None:
    assert build_injection_classifier(make_settings(injection_provider="none")) is None


def test_builder_nvidia_without_key() -> None:
    settings = make_settings(injection_provider="nvidia", nvidia_api_key=None)
    assert build_injection_classifier(settings) is None


# ---------------------------------------------------------------- LLM chain


def router(tmp_path: Path, **overrides: Any) -> LLMRouter:
    return LLMRouter(make_settings(llm_cache_dir=tmp_path / "c", **overrides))


@pytest.mark.parametrize("purpose", ["generate", "helper"])
def test_chain_has_no_ollama_by_default(tmp_path: Path, purpose: str) -> None:
    chain = router(tmp_path).chain(purpose)  # type: ignore[arg-type]
    assert chain
    assert all(t.endpoint == "nvidia" for t in chain)


@pytest.mark.parametrize("purpose", ["generate", "helper"])
def test_chain_has_ollama_last_when_enabled(tmp_path: Path, purpose: str) -> None:
    chain = router(tmp_path, local_fallback_enabled=True).chain(purpose)  # type: ignore[arg-type]
    assert chain[-1].endpoint == "ollama"
    assert all(t.endpoint == "nvidia" for t in chain[:-1])


def test_judge_chain_never_has_ollama(tmp_path: Path) -> None:
    for enabled in (False, True):
        chain = router(tmp_path, local_fallback_enabled=enabled).chain("judge")
        assert chain
        assert all(t.endpoint == "nvidia" for t in chain)
