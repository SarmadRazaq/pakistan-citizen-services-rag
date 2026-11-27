"""TenantScopedVectorStore against qdrant-client's local in-memory mode (no server, no network).

Local mode supports named dense + sparse vectors, aliases, Prefetch + FusionQuery(RRF), filters,
scroll, count and delete-by-filter. Payload indexes are accepted but have no effect there (a warning),
so index behaviour itself is only checked against a real server.
"""

from __future__ import annotations

import math
from collections.abc import AsyncIterator
from datetime import date

import httpx
import pytest
from fastapi import FastAPI
from structlog.testing import capture_logs

from app.config import Settings
from app.core.errors import AppError, ErrorCode, MissingTenantFilterError, install_error_handlers
from app.core.ids import point_id
from app.core.tenancy import Role, Scope, TenantContext, Visibility, system_context
from app.protocols import ChunkPoint, Embedding, SearchFilters, SparseVector
from app.retrieval.vector_store import (
    TenantScopedVectorStore,
    payload_from_qdrant,
    payload_to_qdrant,
)
from tests.fakes import FakeEmbedder, make_payload

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect:UserWarning")

PUBLIC = "u_public"
EMB = FakeEmbedder()


@pytest.fixture
async def store(settings: Settings) -> AsyncIterator[TenantScopedVectorStore]:
    s = TenantScopedVectorStore.from_settings(settings, dense_dim=EMB.dimension, location=":memory:")
    await s.ensure_collection()
    yield s
    await s.close()


@pytest.fixture
def user_b() -> TenantContext:
    return TenantContext.for_user("u_b", Role.USER, Scope.ALL)


async def _points(doc_id: str, texts: list[str], **payload: object) -> list[ChunkPoint]:
    embeddings = await EMB.embed_documents(texts)
    out = []
    for i, (text, emb) in enumerate(zip(texts, embeddings, strict=True)):
        p = make_payload(doc_id, text, chunk_index=i, chunk_count=len(texts), **payload)
        out.append(ChunkPoint(point_id(doc_id, p.chunk_hash), emb, p))
    return out


async def _search(
    store: TenantScopedVectorStore, ctx: TenantContext, text: str, filters: SearchFilters | None = None
) -> list[str]:
    q = await EMB.embed_query(text)
    hits = await store.hybrid_search(ctx, q, filters=filters or SearchFilters(), prefetch_k=30, limit=20)
    return [h.payload.doc_id for h in hits]


@pytest.fixture
async def seeded(
    store: TenantScopedVectorStore, user_a: TenantContext, user_b: TenantContext, admin: TenantContext
) -> TenantScopedVectorStore:
    await store.upsert(
        admin,
        PUBLIC,
        await _points("d_pub", ["passport fee schedule federal", "cnic renewal fee nadra"], agency="DGIP"),
    )
    await store.upsert(user_a, "u_a", await _points("d_a", ["passport fee my private note"], agency="NADRA"))
    await store.upsert(
        user_b, "u_b", await _points("d_b", ["passport fee secret of user b"], jurisdiction="sindh")
    )
    return store


# ---------------------------------------------------------------- collection lifecycle


async def test_ensure_collection_is_idempotent_and_ready(settings: Settings) -> None:
    s = TenantScopedVectorStore.from_settings(settings, dense_dim=8, location=":memory:")
    assert await s.ready() is False
    await s.ensure_collection()
    await s.ensure_collection()
    assert await s.ready() is True
    assert s.collection == "chunks__bge-m3__v1"
    assert s.alias == settings.qdrant_collection_alias
    await s.close()


# ---------------------------------------------------------------- tenant isolation


async def test_user_never_sees_another_users_private_points(
    seeded: TenantScopedVectorStore, user_a: TenantContext, user_b: TenantContext
) -> None:
    a_docs = await _search(seeded, user_a, "passport fee")
    assert "d_b" not in a_docs
    assert {"d_a", "d_pub"} <= set(a_docs)
    b_docs = await _search(seeded, user_b, "passport fee")
    assert "d_a" not in b_docs
    assert {"d_b", "d_pub"} <= set(b_docs)
    # Even an explicit doc_ids filter on B's document returns nothing for A.
    assert await _search(seeded, user_a, "passport", SearchFilters(doc_ids=("d_b",))) == []
    assert await seeded.point_ids(user_a, "d_b") == set()
    assert await seeded.count(user_a, "d_b") == 0
    assert await seeded.count(user_a) == 3  # 2 public + 1 own


async def test_scopes_narrow_the_readable_set(seeded: TenantScopedVectorStore, user_a: TenantContext) -> None:
    mine = await _search(seeded, user_a.with_scope(Scope.MINE), "passport fee")
    public = await _search(seeded, user_a.with_scope(Scope.PUBLIC), "passport fee")
    assert set(mine) == {"d_a"}
    assert set(public) == {"d_pub"}


async def test_system_context_reads_everything(seeded: TenantScopedVectorStore) -> None:
    ctx = system_context("test")
    assert set(await _search(seeded, ctx, "passport fee")) == {"d_pub", "d_a", "d_b"}
    assert await seeded.count(ctx) == 4


async def test_non_admin_cannot_write_public_or_other_owner(
    store: TenantScopedVectorStore, user_a: TenantContext
) -> None:
    points = await _points("d_x", ["some text"])
    for owner in (PUBLIC, "u_b"):
        with pytest.raises(AppError) as exc:
            await store.upsert(user_a, owner, points)
        assert exc.value.code is ErrorCode.ADMIN_REQUIRED
    assert await store.count(system_context("test")) == 0


async def test_non_admin_cannot_delete_or_patch_public(
    seeded: TenantScopedVectorStore, user_a: TenantContext
) -> None:
    public_ids = await seeded.point_ids(user_a, "d_pub")
    assert len(public_ids) == 2
    remaining = await seeded.delete_document(user_a, "d_pub")
    assert remaining == 2  # refused: nothing was deleted, and the caller can see that
    await seeded.delete_points(user_a, sorted(public_ids))
    await seeded.set_payload(user_a, sorted(public_ids), {"agency": "HACKED"})
    assert await seeded.point_ids(user_a, "d_pub") == public_ids
    q = await EMB.embed_query("passport fee schedule")
    hits = await seeded.hybrid_search(user_a, q, filters=SearchFilters(), prefetch_k=10, limit=10)
    assert all(h.payload.agency != "HACKED" for h in hits)
    # Nor can A touch B's private points by id.
    b_ids = await seeded.point_ids(system_context("test"), "d_b")
    await seeded.delete_points(user_a, sorted(b_ids))
    assert await seeded.count(system_context("test"), "d_b") == 1


async def test_admin_can_delete_public(seeded: TenantScopedVectorStore, admin: TenantContext) -> None:
    assert await seeded.delete_document(admin, "d_pub") == 0
    assert await seeded.count(admin, "d_pub") == 0


async def test_missing_context_raises(store: TenantScopedVectorStore) -> None:
    q = await EMB.embed_query("x")
    calls = [
        store.upsert(None, "u_a", []),  # type: ignore[arg-type]
        store.hybrid_search(None, q, filters=SearchFilters(), prefetch_k=5, limit=5),  # type: ignore[arg-type]
        store.point_ids(None, "d"),  # type: ignore[arg-type]
        store.set_payload(None, ["x"], {"agency": "A"}),  # type: ignore[arg-type]
        store.delete_points(None, ["x"]),  # type: ignore[arg-type]
        store.delete_document(None, "d"),  # type: ignore[arg-type]
        store.count(None),  # type: ignore[arg-type]
    ]
    for call in calls:
        with pytest.raises(MissingTenantFilterError):
            await call


async def test_missing_tenant_filter_raises(store: TenantScopedVectorStore) -> None:
    """E43: no TenantContext -> MissingTenantFilterError (HTTP 500), logged as a security event."""
    assert ErrorCode.TENANT_FILTER_MISSING.status == 500
    with capture_logs() as logs, pytest.raises(MissingTenantFilterError) as err:
        await store.count(None)  # type: ignore[arg-type]
    assert err.value.code is ErrorCode.TENANT_FILTER_MISSING
    assert any(
        e["event"] == "security.tenant_filter_missing" and e["log_level"] == "error" and e["operation"]
        for e in logs
    )

    app = FastAPI()
    install_error_handlers(app)

    @app.get("/boom")
    async def boom() -> None:
        await store.count(None)  # type: ignore[arg-type]

    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/boom")
    assert r.status_code == 500
    assert "An internal error occurred." in r.text
    assert "count" not in r.text  # the operation name is logged, never returned


# ---------------------------------------------------------------- search behaviour


async def test_filters(seeded: TenantScopedVectorStore, admin: TenantContext) -> None:
    ctx = system_context("test")
    assert set(await _search(seeded, ctx, "passport", SearchFilters(agencies=("NADRA",)))) == {"d_a"}
    assert set(await _search(seeded, ctx, "passport", SearchFilters(jurisdictions=("sindh",)))) == {"d_b"}
    assert set(await _search(seeded, ctx, "fee", SearchFilters(doc_ids=("d_pub", "d_a")))) == {"d_pub", "d_a"}
    assert await _search(seeded, ctx, "fee", SearchFilters(doc_ids=())) == []
    # Filters combine with the tenant condition: admin (scope all) cannot reach u_b through agency.
    assert await _search(seeded, admin, "passport", SearchFilters(jurisdictions=("sindh",))) == []


def _emb(dense: list[float], sparse: dict[int, float]) -> Embedding:
    items = sorted(sparse.items())
    return Embedding(tuple(dense), SparseVector(tuple(i for i, _ in items), tuple(v for _, v in items)))


async def test_rrf_ordering_and_similarity_for_sparse_only_hits(settings: Settings) -> None:
    store = TenantScopedVectorStore.from_settings(settings, dense_dim=4, location=":memory:")
    await store.ensure_collection()
    admin = TenantContext.for_user("u_admin", Role.ADMIN)
    specs = {
        "both": _emb([0.9, 0.1, 0.0, 0.0], {1: 0.5}),  # rank 2 dense, rank 2 sparse
        "dense": _emb([1.0, 0.0, 0.0, 0.0], {7: 1.0}),  # rank 1 dense only
        "sparse": _emb([0.2, 0.0, 0.98, 0.0], {1: 2.0}),  # rank 1 sparse only
        "none": _emb([0.0, 1.0, 0.0, 0.0], {9: 1.0}),  # in neither leg
    }
    points = [
        ChunkPoint(point_id("d_r", name), emb, make_payload("d_r", name, chunk_hash=name))
        for name, emb in specs.items()
    ]
    await store.upsert(admin, PUBLIC, points)
    query = _emb([1.0, 0.0, 0.0, 0.0], {1: 1.0})
    hits = await store.hybrid_search(admin, query, filters=SearchFilters(), prefetch_k=2, limit=10)
    names = [h.payload.text for h in hits]
    assert names[0] == "both"
    assert set(names) == {"both", "dense", "sparse"}
    rrf = [h.rrf_score or 0.0 for h in hits]
    assert rrf == sorted(rrf, reverse=True)
    by_name = {h.payload.text: h for h in hits}
    sparse_only = by_name["sparse"]
    expected = 0.2 / math.sqrt(0.2**2 + 0.98**2)
    assert sparse_only.similarity == pytest.approx(expected, abs=1e-4)
    assert sparse_only.sparse_score == pytest.approx(2.0)
    assert by_name["dense"].similarity == pytest.approx(1.0, abs=1e-5)
    assert by_name["dense"].sparse_score == pytest.approx(0.0)
    assert by_name["both"].sparse_score == pytest.approx(0.5)
    assert all(h.owner_id == PUBLIC for h in hits)
    await store.close()


async def test_dense_only_query_when_sparse_is_empty(
    seeded: TenantScopedVectorStore, user_a: TenantContext
) -> None:
    q = await EMB.embed_query("passport fee")
    hits = await seeded.hybrid_search(
        user_a, Embedding(q.dense, SparseVector((), ())), filters=SearchFilters(), prefetch_k=10, limit=10
    )
    assert {h.payload.doc_id for h in hits} == {"d_a", "d_pub"}


# ---------------------------------------------------------------- writes and payloads


async def test_delete_document_returns_zero(seeded: TenantScopedVectorStore, user_a: TenantContext) -> None:
    assert await seeded.count(user_a, "d_a") == 1
    assert await seeded.delete_document(user_a, "d_a") == 0
    assert await seeded.point_ids(user_a, "d_a") == set()
    assert await seeded.delete_document(user_a, "d_missing") == 0


async def test_reupsert_is_idempotent(store: TenantScopedVectorStore, user_a: TenantContext) -> None:
    points = await _points("d_i", ["one chunk", "two chunk", "three chunk"])
    await store.upsert(user_a, "u_a", points)
    ids_first = await store.point_ids(user_a, "d_i")
    again = await _points("d_i", ["one chunk", "two chunk", "three chunk"])
    assert [p.point_id for p in again] == [p.point_id for p in points]  # deterministic ids
    await store.upsert(user_a, "u_a", again)
    assert await store.point_ids(user_a, "d_i") == ids_first
    assert await store.count(user_a, "d_i") == 3


async def test_set_payload_and_reserved_keys(seeded: TenantScopedVectorStore, user_a: TenantContext) -> None:
    ids = sorted(await seeded.point_ids(user_a, "d_a"))
    await seeded.set_payload(user_a, ids, {"topic": "passport", "effective_date": date(2025, 7, 1)})
    q = await EMB.embed_query("passport fee my private note")
    hits = await seeded.hybrid_search(
        user_a, q, filters=SearchFilters(doc_ids=("d_a",)), prefetch_k=5, limit=5
    )
    assert hits[0].payload.topic == "passport"
    assert hits[0].payload.effective_date == date(2025, 7, 1)
    for bad in ({"user_id": "u_b"}, {"visibility": "public"}, {"not_a_field": 1}):
        with pytest.raises(ValueError):
            await seeded.set_payload(user_a, ids, bad)


async def test_payload_round_trip(store: TenantScopedVectorStore, admin: TenantContext) -> None:
    payload = make_payload(
        "d_rt",
        "fee table row",
        effective_date=date(2025, 7, 1),
        section_path=("Fees", "Urgent"),
        agency="DGIP",
        is_table=True,
        page_start=2,
        page_end=3,
    )
    raw = payload_to_qdrant(payload, owner_id=PUBLIC, visibility=Visibility.PUBLIC, point_id="p1")
    assert raw["effective_date"] == "2025-07-01"
    assert raw["section_path"] == ["Fees", "Urgent"]
    assert raw["user_id"] == PUBLIC and raw["visibility"] == "public" and raw["chunk_id"] == "p1"
    assert payload_from_qdrant(raw) == payload
    (emb,) = await EMB.embed_documents(["fee table row"])
    await store.upsert(admin, PUBLIC, [ChunkPoint(point_id("d_rt", payload.chunk_hash), emb, payload)])
    hits = await store.hybrid_search(admin, emb, filters=SearchFilters(), prefetch_k=5, limit=5)
    assert hits[0].payload == payload
    assert hits[0].similarity == pytest.approx(1.0, abs=1e-5)


async def test_wrong_dimension_is_rejected(store: TenantScopedVectorStore, user_a: TenantContext) -> None:
    bad = ChunkPoint("p", _emb([1.0, 0.0], {}), make_payload("d", "t"))
    with pytest.raises(ValueError):
        await store.upsert(user_a, "u_a", [bad])


async def test_connection_errors_map_to_503() -> None:
    async def refused() -> None:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(AppError) as exc:
        await TenantScopedVectorStore._guard("count", refused())
    assert exc.value.code is ErrorCode.DEPENDENCY_UNAVAILABLE
    assert exc.value.code.status == 503
