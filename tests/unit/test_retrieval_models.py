"""BGE-M3 embedder, reranker, GPU worker, device selection and factory, with fake models (no torch)."""

from __future__ import annotations

import asyncio
import math
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from app.config import Settings
from app.core import device as device_mod
from app.protocols import Embedder, Reranker, SparseVector
from app.retrieval.embedder import BGEM3Embedder, to_sparse
from app.retrieval.factory import build_retrieval
from app.retrieval.gpu import GpuWorker
from app.retrieval.reranker import BGEReranker


class FakeM3:
    """Mimics BGEM3FlagModel.encode: unnormalised dense rows and str-keyed lexical weights."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.kwargs: list[dict[str, Any]] = []

    def encode(self, sentences: list[str], **kwargs: Any) -> Mapping[str, Any]:
        self.calls.append(list(sentences))
        self.kwargs.append(kwargs)
        dense = [[float(len(s)), 1.0, 0.0, 2.0] for s in sentences]
        lexical = [{"42": 0.5, "7": 0.25, "3": 0.0, 7: 0.25} for _ in sentences]
        return {"dense_vecs": dense, "lexical_weights": lexical, "colbert_vecs": None}


class FakeCrossEncoder:
    def __init__(self) -> None:
        self.pairs: list[list[list[str]]] = []

    def compute_score(self, sentence_pairs: list[list[str]], **kwargs: Any) -> Any:
        self.pairs.append(sentence_pairs)
        assert kwargs.get("normalize") is True
        if len(sentence_pairs) == 1:
            return 0.75  # FlagReranker returns a bare float for one pair
        return [1.2, -0.1, *([0.5] * (len(sentence_pairs) - 2))]


def _embedder(
    settings: Settings, model: FakeM3, loads: list[int] | None = None, **kwargs: Any
) -> BGEM3Embedder:
    def factory() -> FakeM3:
        if loads is not None:
            loads.append(1)
        return model

    return BGEM3Embedder(settings, GpuWorker(1), model_factory=factory, dimension=4, **kwargs)


# ---------------------------------------------------------------- embedder


def test_to_sparse_sorts_ints_and_drops_zeros() -> None:
    sv = to_sparse({"42": 0.5, "7": 0.25, "3": 0.0, 7: 0.25, "9": -1.0})
    assert sv == SparseVector((7, 42), (0.5, 0.5))


async def test_embedder_is_lazy_and_loads_once(settings: Settings) -> None:
    loads: list[int] = []
    emb = _embedder(settings, FakeM3(), loads)
    assert isinstance(emb, Embedder)
    assert not emb.loaded and loads == []
    await asyncio.gather(emb.embed_query("a"), emb.embed_query("b"), emb.embed_documents(["c"]))
    assert loads == [1]
    assert emb.loaded


async def test_embedder_output_shape_and_normalisation(settings: Settings) -> None:
    model = FakeM3()
    emb = _embedder(settings, model)
    (e,) = await emb.embed_documents(["hello"])
    assert len(e.dense) == 4
    assert math.isclose(math.sqrt(sum(v * v for v in e.dense)), 1.0, rel_tol=1e-9)
    assert e.sparse.indices == (7, 42)
    kw = model.kwargs[0]
    assert kw["return_dense"] and kw["return_sparse"] and not kw["return_colbert_vecs"]


async def test_embedder_batches(settings: Settings) -> None:
    model = FakeM3()
    emb = _embedder(settings.model_copy(update={"embed_batch_size": 2}), model)
    out = await emb.embed_documents(["a", "bb", "ccc", "dddd", "eeeee"])
    assert len(out) == 5
    assert [len(c) for c in model.calls] == [2, 2, 1]


async def test_query_cache_is_lru(settings: Settings) -> None:
    model = FakeM3()
    emb = _embedder(settings, model, query_cache_size=2)
    await emb.embed_query("q1")
    await emb.embed_query("q1")
    assert len(model.calls) == 1
    await emb.embed_query("q2")
    await emb.embed_query("q3")  # evicts q1
    await emb.embed_query("q1")
    assert len(model.calls) == 4


async def test_chunk_cache_is_per_owner_namespace(settings: Settings, tmp_path: Path) -> None:
    model = FakeM3()
    emb = _embedder(settings, model, cache_dir=tmp_path)
    first, fresh = await emb.embed_chunks(
        "u_public", [("h1", "text one"), ("h2", "text two"), ("h1", "text one")]
    )
    assert fresh == 2 and len(first) == 3 and first[0] == first[2]
    again, fresh = await emb.embed_chunks("u_public", [("h1", "text one")])
    assert fresh == 0 and again[0] == first[0]
    # Another tenant uploading the same text gets no cache hit (§5.8.5 side channel).
    _, fresh = await emb.embed_chunks("u_a", [("h1", "text one")])
    assert fresh == 1
    # The disk copy survives a restart.
    model2 = FakeM3()
    restarted = _embedder(settings, model2, cache_dir=tmp_path)
    cached, fresh = await restarted.embed_chunks("u_public", [("h2", "text two")])
    assert fresh == 0 and model2.calls == []
    assert cached[0].sparse == first[1].sparse
    assert cached[0].dense == pytest.approx(first[1].dense, abs=1e-6)
    restarted.cache.drop_namespace(restarted.model_name, "u_public")
    _, fresh = await restarted.embed_chunks("u_public", [("h2", "text two")])
    assert fresh == 1


async def test_embedder_rejects_wrong_dimension(settings: Settings) -> None:
    emb = BGEM3Embedder(settings, GpuWorker(1), model_factory=FakeM3)  # expects 1024-d
    with pytest.raises(RuntimeError):
        await emb.embed_query("x")


# ---------------------------------------------------------------- reranker


async def test_reranker_scores(settings: Settings) -> None:
    model = FakeCrossEncoder()
    loads: list[int] = []

    def factory() -> FakeCrossEncoder:
        loads.append(1)
        return model

    rr = BGEReranker(settings, GpuWorker(1), model_factory=factory)
    assert isinstance(rr, Reranker)
    assert await rr.rerank("q", []) == [] and loads == []
    assert await rr.rerank("q", ["p"]) == [0.75]
    scores = await rr.rerank("q", ["a", "b", "c"])
    assert scores == [1.0, 0.0, 0.5]  # clamped into [0, 1]
    assert model.pairs[-1] == [["q", "a"], ["q", "b"], ["q", "c"]]
    assert loads == [1]


# ---------------------------------------------------------------- GPU worker


async def test_gpu_worker_serialises_calls() -> None:
    worker = GpuWorker(1)
    state = {"now": 0, "max": 0}
    lock = threading.Lock()

    def job() -> int:
        with lock:
            state["now"] += 1
            state["max"] = max(state["max"], state["now"])
        time.sleep(0.01)
        with lock:
            state["now"] -= 1
        return threading.get_ident()

    threads = await asyncio.gather(*(worker.run(job) for _ in range(4)))
    assert state["max"] == 1
    assert threading.get_ident() not in threads  # ran off the event-loop thread
    worker.shutdown()
    with pytest.raises(RuntimeError):
        await worker.run(job)


def test_gpu_worker_validates() -> None:
    with pytest.raises(ValueError):
        GpuWorker(0)


# ---------------------------------------------------------------- device


def _fake_torch(available: bool) -> Any:
    cuda = SimpleNamespace(
        is_available=lambda: available,
        mem_get_info=lambda i: (3 * 1024**3, 6 * 1024**3),
        get_device_name=lambda i: "GeForce RTX 3050",
        max_memory_allocated=lambda: 512 * 1024**2,
        empty_cache=lambda: None,
    )
    return SimpleNamespace(cuda=cuda)


def test_device_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    assert device_mod.resolve_device("cpu") == "cpu"
    monkeypatch.setattr(device_mod, "_import_torch", lambda: None)
    assert device_mod.resolve_device("auto") == "cpu"
    with pytest.raises(RuntimeError):
        device_mod.resolve_device("cuda")
    assert device_mod.device_info("auto").torch_available is False
    assert device_mod.max_memory_allocated_mb() is None

    monkeypatch.setattr(device_mod, "_import_torch", lambda: _fake_torch(True))
    assert device_mod.resolve_device("auto") == "cuda:0"
    info = device_mod.device_info("auto")
    assert info.is_cuda and info.vram_total_mb == 6144 and info.vram_free_mb == 3072
    assert info.name == "GeForce RTX 3050"
    assert device_mod.max_memory_allocated_mb() == 512

    monkeypatch.setattr(device_mod, "_import_torch", lambda: _fake_torch(False))
    assert device_mod.resolve_device("auto") == "cpu"


# ---------------------------------------------------------------- factory


@pytest.mark.filterwarnings("ignore:Payload indexes have no effect:UserWarning")
async def test_factory_builds_without_loading_models(settings: Settings) -> None:
    local = settings.model_copy(update={"embedding_provider": "local", "reranker_provider": "local"})
    bundle = build_retrieval(local, qdrant_location=":memory:")
    assert not bundle.embedder.loaded and not bundle.reranker.loaded
    assert bundle.embedder.dimension == 1024
    assert bundle.store.dense_dim == 1024
    await bundle.startup()  # ensure_collection only, no warm-up
    (probe,) = bundle.probes()
    assert await probe.check() == (True, None)
    await bundle.shutdown()
