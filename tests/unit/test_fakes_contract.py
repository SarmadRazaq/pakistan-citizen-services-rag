"""The shared fakes must satisfy the protocols and the tenant rules, or every package's tests lie."""

from __future__ import annotations

from app.core.tenancy import PUBLIC_OWNER_ID, Role, Scope, TenantContext, system_context
from app.protocols import ChunkPoint, Embedder, LLMClient, Reranker, SearchFilters, VectorStore
from tests.fakes import FakeEmbedder, FakeLLM, FakeReranker, InMemoryVectorStore, make_payload


def test_fakes_implement_protocols() -> None:
    assert isinstance(FakeEmbedder(), Embedder)
    assert isinstance(InMemoryVectorStore(), VectorStore)
    assert isinstance(FakeReranker(), Reranker)
    assert isinstance(FakeLLM(), LLMClient)


async def test_store_enforces_tenants() -> None:
    emb, store = FakeEmbedder(), InMemoryVectorStore()
    seed = system_context("test")
    for owner, doc, text in [
        (PUBLIC_OWNER_ID, "d_pub", "passport urgent fee schedule"),
        ("u_a", "d_a", "passport fee note from helpdesk A"),
        ("u_b", "d_b", "passport secret canary for B"),
    ]:
        (vec,) = await emb.embed_documents([text])
        await store.upsert(seed, owner, [ChunkPoint(f"p_{doc}", vec, make_payload(doc, text))])

    a = TenantContext.for_user("u_a", Role.USER, Scope.ALL)
    q = await emb.embed_query("passport fee")
    hits = await store.hybrid_search(a, q, filters=SearchFilters(), prefetch_k=10, limit=10)
    assert {h.payload.doc_id for h in hits} == {"d_pub", "d_a"}
    assert await store.delete_document(a, "d_pub") == 1  # user cannot delete public points
    assert await store.count(a.with_scope(Scope.MINE)) == 1
