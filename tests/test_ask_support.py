"""Shared helpers for the ask-pipeline tests (no tests here).

`StoreRetriever` is a small `Retriever` over the shared fakes (FakeEmbedder + InMemoryVectorStore +
FakeReranker): variants → hybrid search → union → rerank → context. It records every query, so tests
can assert on variants and filters.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pytest

from app.config import Settings
from app.core.ids import point_id
from app.core.tenancy import PUBLIC_OWNER_ID, Role, Scope, TenantContext, system_context
from app.db.models import Document
from app.db.session import Database
from app.generation.service import RagAskService
from app.guardrails.advice import Referral
from app.protocols import (
    Candidate,
    ChunkPoint,
    RetrievalQuery,
    RetrievalResult,
    ScoredPoint,
)
from tests.fakes import FakeEmbedder, FakeLLM, FakeReranker, InMemoryVectorStore, make_payload

TODAY = date(2026, 11, 22)

PASSPORT_TEXT = (
    "Machine Readable Passport fee schedule. Urgent passport fee for 36 pages is Rs 9,000 "
    "and normal passport fee is Rs 4,500. Applications are accepted at all regional passport offices."
)
CNIC_TEXT = (
    "NADRA CNIC renewal: a citizen can renew the computerised national identity card at any NADRA "
    "registration centre. The executive renewal fee is Rs 2,500."
)
PRIVATE_TEXT = (
    "Helpdesk note: our office accepts urgent passport fee payments only by bank challan before 11 am."
)


@dataclass
class StoreRetriever:
    store: InMemoryVectorStore
    embedder: FakeEmbedder = field(default_factory=FakeEmbedder)
    reranker: FakeReranker = field(default_factory=FakeReranker)
    tau_keep: float = 0.2
    queries: list[RetrievalQuery] = field(default_factory=list)

    async def retrieve(self, ctx: TenantContext, query: RetrievalQuery) -> RetrievalResult:
        self.queries.append(query)
        merged: dict[str, ScoredPoint] = {}
        for variant in query.variants:
            emb = await self.embedder.embed_query(variant)
            hits = await self.store.hybrid_search(
                ctx, emb, filters=query.filters, prefetch_k=30, limit=query.candidates_k
            )
            for hit in hits:
                best = merged.get(hit.point_id)
                if best is None or (hit.rrf_score or 0) > (best.rrf_score or 0):
                    merged[hit.point_id] = hit
        points = sorted(merged.values(), key=lambda p: -(p.rrf_score or 0))[: query.candidates_k]
        if not points:
            return RetrievalResult(candidates=(), context=(), max_rerank=None, latency_ms={"search": 1})
        per_variant = [
            await self.reranker.rerank(v, [p.payload.text for p in points]) for v in query.variants
        ]
        scores = [max(col) for col in zip(*per_variant, strict=True)]
        ranked = sorted(zip(points, scores, strict=True), key=lambda ps: -ps[1])
        keep = [p.point_id for p, s in ranked if s >= self.tau_keep][: query.context_k] or [
            ranked[0][0].point_id
        ]
        candidates = tuple(
            Candidate(point=p, rank=i, rerank_score=s, used_in_context=p.point_id in keep)
            for i, (p, s) in enumerate(ranked, start=1)
        )
        context = tuple(c for c in candidates if c.used_in_context)
        return RetrievalResult(candidates, context, max(scores), latency_ms={"search": 1, "rerank": 1})


@dataclass
class World:
    settings: Settings
    db: Database
    store: InMemoryVectorStore
    retriever: StoreRetriever
    hosted: FakeLLM
    local: FakeLLM
    service: RagAskService


_OPEN_DATABASES: list[Database] = []


@pytest.fixture(autouse=True)
async def close_worlds() -> AsyncIterator[None]:
    """Dispose every in-memory database a test created (import this fixture into the test module)."""
    yield
    while _OPEN_DATABASES:
        await _OPEN_DATABASES.pop().dispose()


def make_settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, app_env="test", log_level="WARNING", **overrides)  # type: ignore[call-arg]


async def add_doc(
    world: World,
    doc_id: str,
    text: str,
    *,
    owner: str = PUBLIC_OWNER_ID,
    status: str = "READY",
    **payload: Any,
) -> str:
    seed = system_context("test seed")
    async with world.db.session(seed) as s:
        s.add(
            Document(
                id=doc_id,
                user_id=owner,
                visibility="public" if owner == PUBLIC_OWNER_ID else "private",
                file_name=f"{doc_id}.pdf",
                file_type="pdf",
                size_bytes=len(text),
                file_sha256=doc_id.ljust(64, "0"),
                status=status,
            )
        )
    (vec,) = await world.retriever.embedder.embed_documents([text])
    pl = make_payload(doc_id, text, **payload)
    pid = point_id(doc_id, pl.chunk_hash)
    await world.store.upsert(seed, owner, [ChunkPoint(pid, vec, pl)])
    return pid


async def make_world(
    *,
    hosted: Sequence[Any] = (),
    local: Sequence[Any] = (),
    referrals: Sequence[Referral] = (),
    **settings_overrides: Any,
) -> World:
    settings = make_settings(**settings_overrides)
    db = Database("sqlite+aiosqlite:///:memory:")
    await db.create_all()
    _OPEN_DATABASES.append(db)
    store = InMemoryVectorStore()
    retriever = StoreRetriever(store)
    hosted_llm = FakeLLM(replies=list(hosted))
    local_llm = FakeLLM(replies=list(local), endpoint="ollama", model="fake-local")
    service = RagAskService(
        settings, db, retriever, hosted_llm, local_llm, referrals=referrals, clock=lambda: TODAY
    )
    return World(settings, db, store, retriever, hosted_llm, local_llm, service)


def user(user_id: str = "u_a", *, request_id: str | None = None) -> TenantContext:
    return TenantContext.for_user(user_id, Role.USER, Scope.ALL, request_id=request_id)


def helper_reply(**overrides: Any) -> str:
    body: dict[str, Any] = {
        "english_query": None,
        "urdu_query": None,
        "sub_queries": [],
        "agencies": [],
        "jurisdictions": [],
        "advice_request": False,
        "off_topic": False,
        "injection_suspect": False,
        "emergency": False,
    }
    body.update(overrides)
    return "Here is the analysis:\n" + json.dumps(body)


def answer_reply(
    answer: str,
    citations: Sequence[tuple[str, str]] = (),
    *,
    answerable: bool = True,
    unanswered_parts: Sequence[str] = (),
) -> str:
    return json.dumps(
        {
            "answerable": answerable,
            "answer": answer,
            "citations": [{"source": s, "quote": q} for s, q in citations],
            "unanswered_parts": list(unanswered_parts),
        },
        ensure_ascii=False,
    )


def generate_calls(llm: FakeLLM) -> list[list[Any]]:
    """The messages of every *answer* call (the helper call is excluded)."""
    return [m for m in llm.seen if "SOURCES:" in m[-1].content or any("SOURCES:" in x.content for x in m)]
