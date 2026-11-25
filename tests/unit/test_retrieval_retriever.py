"""HybridRetriever with the shared fakes: variant fusion, dedupe, rerank gate fields, context selection."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest

from app.config import Settings
from app.core.eval_overrides import EvalOverrides
from app.core.ids import point_id
from app.core.tenancy import Role, Scope, TenantContext
from app.protocols import (
    ChunkPoint,
    Embedding,
    RetrievalQuery,
    Retriever,
    ScoredPoint,
    SearchFilters,
)
from app.retrieval.retriever import HybridRetriever, dedupe, fuse_rrf
from tests.fakes import FakeEmbedder, FakeReranker, InMemoryVectorStore, make_payload

PUBLIC = "u_public"


async def _add(
    store: InMemoryVectorStore, ctx: TenantContext, owner: str, doc_id: str, texts: list[str]
) -> None:
    emb = FakeEmbedder()
    embeddings = await emb.embed_documents(texts)
    points = []
    for i, (text, e) in enumerate(zip(texts, embeddings, strict=True)):
        p = make_payload(doc_id, text, chunk_index=i, chunk_count=len(texts))
        points.append(ChunkPoint(point_id(doc_id, p.chunk_hash), e, p))
    await store.upsert(ctx, owner, points)


class RecordingReranker(FakeReranker):
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def rerank(self, query: str, passages: Sequence[str]) -> list[float]:
        self.queries.append(query)
        return await super().rerank(query, passages)


class FixedReranker:
    model_name = "fixed"

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores

    async def rerank(self, query: str, passages: Sequence[str]) -> list[float]:
        return [self.scores.get(p, 0.0) for p in passages]


class ConcurrencyStore(InMemoryVectorStore):
    """Counts how many hybrid_search calls overlap, to prove variants run in parallel."""

    in_flight: int = 0
    max_in_flight: int = 0

    async def hybrid_search(  # type: ignore[override]
        self, ctx: TenantContext, query: Embedding, **kwargs: object
    ) -> list[ScoredPoint]:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(0.01)
        try:
            return await super().hybrid_search(ctx, query, **kwargs)  # type: ignore[arg-type]
        finally:
            self.in_flight -= 1


@pytest.fixture
def store() -> InMemoryVectorStore:
    return InMemoryVectorStore()


@pytest.fixture
async def corpus(
    store: InMemoryVectorStore, admin: TenantContext, user_a: TenantContext
) -> InMemoryVectorStore:
    await _add(
        store,
        admin,
        PUBLIC,
        "d_passport",
        [
            "passport fee schedule normal processing 3000 rupees",
            "urgent passport fee is 5000 rupees",
            "machine readable passport application documents list",
        ],
    )
    await _add(
        store, admin, PUBLIC, "d_cnic", ["cnic renewal fee at nadra centre", "smart card delivery time"]
    )
    await _add(store, admin, PUBLIC, "d_urdu", ["پاسپورٹ فیس کی تفصیل"])
    await _add(store, user_a, "u_a", "d_mine", ["my notes about passport renewal"])
    return store


def _retriever(
    settings: Settings, store: InMemoryVectorStore, reranker: object | None = None
) -> HybridRetriever:
    return HybridRetriever(settings, FakeEmbedder(), store, reranker or FakeReranker())  # type: ignore[arg-type]


def test_implements_protocol(settings: Settings, store: InMemoryVectorStore) -> None:
    assert isinstance(_retriever(settings, store), Retriever)


async def test_variant_fusion_finds_hits_from_every_variant(
    settings: Settings, corpus: InMemoryVectorStore, user_a: TenantContext
) -> None:
    retriever = _retriever(settings, corpus)
    english_only = await retriever.retrieve(user_a, RetrievalQuery(variants=("passport fee",)))
    fused = await retriever.retrieve(
        user_a, RetrievalQuery(variants=("passport fee", "پاسپورٹ فیس", "passport fee "))
    )
    docs = {c.point.payload.doc_id for c in fused.candidates}
    assert "d_urdu" in docs  # reached through the Urdu-script variant
    # Duplicate variants (after stripping) are searched once: 2 searches, not 3.
    assert corpus.calls.count("hybrid_search") == 1 + 2
    assert len(english_only.candidates) >= 1
    assert all(c.point.rrf_score is not None and c.point.rrf_score > 0 for c in fused.candidates)


async def test_variants_are_searched_in_parallel(settings: Settings, user_a: TenantContext) -> None:
    store = ConcurrencyStore()
    await _add(store, user_a, "u_a", "d1", ["alpha beta", "gamma delta"])
    retriever = _retriever(settings, store)
    await retriever.retrieve(user_a, RetrievalQuery(variants=("alpha", "gamma", "beta")))
    assert store.max_in_flight == 3


def test_fuse_rrf_rewards_points_found_by_several_variants() -> None:
    def sp(pid: str, sim: float) -> ScoredPoint:
        return ScoredPoint(pid, PUBLIC, make_payload("d", pid), similarity=sim, sparse_score=sim)

    v1 = [sp("a", 0.9), sp("b", 0.8), sp("c", 0.7)]
    v2 = [sp("c", 0.95), sp("d", 0.6)]
    fused = fuse_rrf([v1, v2], k=60)
    assert [p.point_id for p in fused] == ["c", "a", "b", "d"]
    c = fused[0]
    assert c.rrf_score == pytest.approx(1 / 63 + 1 / 61)
    assert c.similarity == pytest.approx(0.95)  # best over the variants


async def test_identical_chunk_hash_prefers_private_copy(
    settings: Settings, store: InMemoryVectorStore, admin: TenantContext, user_a: TenantContext
) -> None:
    text = "birth certificate is issued by the union council"
    await _add(store, admin, PUBLIC, "d_pub", [text])
    await _add(store, user_a, "u_a", "d_priv", [text])
    result = await _retriever(settings, store).retrieve(
        user_a, RetrievalQuery(variants=("birth certificate",))
    )
    matching = [c for c in result.candidates if c.point.payload.text == text]
    assert len(matching) == 1
    assert matching[0].point.owner_id == "u_a"
    assert matching[0].point.payload.doc_id == "d_priv"


async def test_near_duplicates_are_dropped(
    settings: Settings, store: InMemoryVectorStore, admin: TenantContext
) -> None:
    base = "the fee for a new domicile certificate in punjab is 200 rupees and it takes seven days"
    await _add(store, admin, PUBLIC, "d1", [base])
    await _add(store, admin, PUBLIC, "d2", [base + "."])
    await _add(store, admin, PUBLIC, "d3", ["a completely different chunk about domicile appeals"])
    ctx = TenantContext.for_user("u_x", Role.USER)
    result = await _retriever(settings, store).retrieve(
        ctx, RetrievalQuery(variants=("domicile fee punjab",))
    )
    texts = [c.point.payload.text for c in result.candidates]
    assert sum(1 for t in texts if t.startswith("the fee for a new domicile")) == 1
    assert "a completely different chunk about domicile appeals" in texts


def test_dedupe_keeps_order_and_threshold() -> None:
    def sp(pid: str, owner: str, text: str, chunk_hash: str) -> ScoredPoint:
        return ScoredPoint(pid, owner, make_payload("d", text, chunk_hash=chunk_hash), similarity=0.5)

    points = [
        sp("1", PUBLIC, "same text", "h1"),
        sp("2", "u_a", "other text entirely", "h2"),
        sp("3", "u_a", "same text", "h1"),
    ]
    out = dedupe(points, public_owner_id=PUBLIC, near_dup_ratio=95.0)
    assert [p.point_id for p in out] == ["3", "2"]  # the private copy takes the public one's place


async def test_rerank_uses_original_question_and_sets_gate_fields(
    settings: Settings, corpus: InMemoryVectorStore, user_a: TenantContext
) -> None:
    reranker = RecordingReranker()
    retriever = _retriever(settings, corpus, reranker)
    result = await retriever.retrieve(
        user_a, RetrievalQuery(variants=("urgent passport fee", "پاسپورٹ فیس"), candidates_k=5, context_k=2)
    )
    assert reranker.queries == ["urgent passport fee"]
    assert 1 <= len(result.candidates) <= 5
    assert [c.rank for c in result.candidates] == list(range(1, len(result.candidates) + 1))
    scores = [c.rerank_score or 0.0 for c in result.candidates]
    assert scores == sorted(scores, reverse=True)
    assert result.max_rerank == scores[0]
    assert result.candidates[0].point.payload.text == "urgent passport fee is 5000 rupees"
    assert set(result.latency_ms) >= {"embed", "search", "rerank"}
    assert all(isinstance(v, int) for v in result.latency_ms.values())


async def test_context_selection_respects_tau_keep_and_context_k(
    settings: Settings, store: InMemoryVectorStore, admin: TenantContext
) -> None:
    texts = ["chunk one", "chunk two", "chunk three", "chunk four"]
    await _add(store, admin, PUBLIC, "d", texts)
    ctx = TenantContext.for_user("u_x", Role.USER)
    reranker = FixedReranker({"chunk one": 0.9, "chunk two": 0.1, "chunk three": 0.5, "chunk four": 0.3})
    s = settings.model_copy(update={"tau_keep": 0.25})
    result = await HybridRetriever(s, FakeEmbedder(), store, reranker).retrieve(
        ctx, RetrievalQuery(variants=("chunk",), context_k=2)
    )
    assert [c.point.payload.text for c in result.context] == ["chunk one", "chunk three"]
    assert [c.rank for c in result.context] == [1, 2]
    used = {c.point.payload.text for c in result.candidates if c.used_in_context}
    assert used == {"chunk one", "chunk three"}
    assert len(result.candidates) == 4
    assert result.max_rerank == pytest.approx(0.9)


async def test_context_has_at_least_one_chunk_below_tau_keep(
    settings: Settings, store: InMemoryVectorStore, admin: TenantContext
) -> None:
    await _add(store, admin, PUBLIC, "d", ["low one", "low two"])
    ctx = TenantContext.for_user("u_x", Role.USER)
    reranker = FixedReranker({"low one": 0.05, "low two": 0.1})
    result = await HybridRetriever(settings, FakeEmbedder(), store, reranker).retrieve(
        ctx, RetrievalQuery(variants=("low",))
    )
    assert len(result.context) == 1
    assert result.context[0].point.payload.text == "low two"
    assert result.max_rerank == pytest.approx(0.1)  # below tau_refuse: the ask pipeline refuses


async def test_tenant_isolation_through_the_retriever(
    settings: Settings, corpus: InMemoryVectorStore, user_a: TenantContext
) -> None:
    user_b = TenantContext.for_user("u_b", Role.USER)
    await _add(corpus, user_b, "u_b", "d_b", ["passport fee secret of b"])
    result = await _retriever(settings, corpus).retrieve(user_a, RetrievalQuery(variants=("passport fee",)))
    assert all(c.point.owner_id in {"u_a", PUBLIC} for c in result.candidates)
    mine = await _retriever(settings, corpus).retrieve(
        user_a.with_scope(Scope.MINE), RetrievalQuery(variants=("passport",))
    )
    assert {c.point.payload.doc_id for c in mine.candidates} == {"d_mine"}


async def test_filters_are_passed_through(
    settings: Settings, corpus: InMemoryVectorStore, user_a: TenantContext
) -> None:
    result = await _retriever(settings, corpus).retrieve(
        user_a, RetrievalQuery(variants=("fee",), filters=SearchFilters(doc_ids=("d_cnic",)))
    )
    assert {c.point.payload.doc_id for c in result.candidates} == {"d_cnic"}


async def test_empty_inputs(settings: Settings, store: InMemoryVectorStore, user_a: TenantContext) -> None:
    retriever = _retriever(settings, store)
    empty = await retriever.retrieve(user_a, RetrievalQuery(variants=("  ",)))
    assert empty.candidates == () and empty.context == () and empty.max_rerank is None
    nothing = await retriever.retrieve(user_a, RetrievalQuery(variants=("no documents yet",)))
    assert nothing.candidates == () and nothing.max_rerank is None


# ---------------------------------------------------------------- eval overrides (X-Eval-Config)


class _TracingStore(InMemoryVectorStore):
    """Records the `mode` of every search."""

    def __init__(self) -> None:
        super().__init__()
        self.modes: list[str] = []

    async def hybrid_search(self, ctx, query, **kwargs):  # type: ignore[no-untyped-def,override]
        self.modes.append(kwargs.get("mode", "hybrid"))
        return await super().hybrid_search(ctx, query, **kwargs)


@pytest.fixture
async def traced(admin: TenantContext) -> _TracingStore:
    store = _TracingStore()
    await _add(
        store,
        admin,
        PUBLIC,
        "d1",
        ["urgent passport fee is Rs 9,000", "cnic renewal at nadra centre", "driving licence rules"],
    )
    return store


def _ovr(**kw: object) -> RetrievalQuery:
    return RetrievalQuery(
        variants=("urgent passport fee",),
        overrides=EvalOverrides(**kw),  # type: ignore[arg-type]
        context_k=2,
    )


async def test_default_query_is_hybrid_and_reranked(
    settings: Settings, traced: _TracingStore, user_a: TenantContext
) -> None:
    reranker = RecordingReranker()
    res = await _retriever(settings, traced, reranker).retrieve(
        user_a, RetrievalQuery(variants=("urgent passport fee",))
    )
    assert traced.modes == ["hybrid"] and reranker.queries
    assert all(c.rerank_score is not None for c in res.candidates)


@pytest.mark.parametrize("mode", ["dense", "sparse"])
async def test_single_leg_modes_are_forwarded_to_the_store(
    settings: Settings, traced: _TracingStore, user_a: TenantContext, mode: str
) -> None:
    res = await _retriever(settings, traced).retrieve(user_a, _ovr(retrieval_mode=mode))
    assert traced.modes == [mode] and res.candidates


async def test_fake_store_dense_skips_sparse_and_sparse_skips_dense(
    traced: _TracingStore, user_a: TenantContext
) -> None:
    emb = await FakeEmbedder().embed_query("urgent passport fee")
    kw = {"filters": SearchFilters(), "prefetch_k": 10, "limit": 10}
    dense = await traced.hybrid_search(user_a, emb, mode="dense", **kw)
    sparse = await traced.hybrid_search(user_a, emb, mode="sparse", **kw)
    assert [p.similarity for p in dense] == sorted((p.similarity for p in dense), reverse=True)
    assert [p.sparse_score for p in sparse] == sorted((p.sparse_score or 0 for p in sparse), reverse=True)
    assert all(p.rrf_score is None for p in dense + sparse)


async def test_rerank_false_skips_reranker_and_gates_on_dense_similarity(
    settings: Settings, traced: _TracingStore, user_a: TenantContext
) -> None:
    reranker = RecordingReranker()
    res = await _retriever(settings, traced, reranker).retrieve(user_a, _ovr(rerank=False))
    assert reranker.queries == []
    assert res.candidates and all(c.rerank_score is None for c in res.candidates)
    assert res.max_rerank == max(c.point.similarity for c in res.candidates)
    assert 1 <= len(res.context) <= 2
    assert [c.rank for c in res.candidates] == list(range(1, len(res.candidates) + 1))


async def test_rerank_false_keeps_at_least_one_context_chunk(
    settings: Settings, traced: _TracingStore, user_a: TenantContext
) -> None:
    strict = settings.model_copy(update={"tau_keep": 0.999})
    res = await _retriever(strict, traced).retrieve(user_a, _ovr(rerank=False))
    assert len(res.context) == 1  # the relevance gate decides on max_rerank
