"""Shared test doubles for the protocols in app/protocols.py (frozen contract, used by every work package).

- FakeEmbedder: deterministic hash vectors, so the same text always gets the same embedding.
- InMemoryVectorStore: implements VectorStore with the same tenant rules as TenantScopedVectorStore.
- FakeReranker: token-overlap scores in [0, 1].
- FakeLLM: scripted replies, failures and timeouts, for router and pipeline tests.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from app.core.errors import AppError, ErrorCode, MissingTenantFilterError
from app.core.tenancy import TenantContext
from app.protocols import (
    ChunkPayload,
    ChunkPoint,
    Embedding,
    LLMMessage,
    LLMResult,
    ScoredPoint,
    SearchFilters,
    SparseVector,
)

_TOKEN = re.compile(r"[\w؀-ۿ]+", re.UNICODE)


def tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


class FakeEmbedder:
    model_name = "fake-hash-embedder"
    dimension = 64

    def _dense(self, text: str) -> tuple[float, ...]:
        vec = [0.0] * self.dimension
        for tok in tokens(text):
            h = int(hashlib.md5(tok.encode(), usedforsecurity=False).hexdigest(), 16)
            vec[h % self.dimension] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return tuple(v / norm for v in vec)

    def _sparse(self, text: str) -> SparseVector:
        weights: dict[int, float] = {}
        for tok in tokens(text):
            idx = int(hashlib.md5(tok.encode(), usedforsecurity=False).hexdigest(), 16) % 100_000
            weights[idx] = weights.get(idx, 0.0) + 1.0
        items = sorted(weights.items())
        return SparseVector(tuple(i for i, _ in items), tuple(w for _, w in items))

    async def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        return [Embedding(self._dense(t), self._sparse(t)) for t in texts]

    async def embed_query(self, text: str) -> Embedding:
        return Embedding(self._dense(text), self._sparse(text))


def _cos(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _sparse_dot(a: SparseVector, b: SparseVector) -> float:
    bw = dict(zip(b.indices, b.values, strict=True))
    return sum(w * bw.get(i, 0.0) for i, w in zip(a.indices, a.values, strict=True))


@dataclass
class _Stored:
    owner_id: str
    point: ChunkPoint


@dataclass
class InMemoryVectorStore:
    """Same tenant rules as the real store: reads by readable owners, writes by writable owner."""

    points: dict[str, _Stored] = field(default_factory=dict)
    calls: list[str] = field(default_factory=list)

    @staticmethod
    def _check(ctx: TenantContext | None, op: str) -> TenantContext:
        if ctx is None:
            raise MissingTenantFilterError(op)
        return ctx

    def _readable(self, ctx: TenantContext) -> list[_Stored]:
        return [s for s in self.points.values() if ctx.can_read(s.owner_id)]

    async def upsert(self, ctx: TenantContext, owner_id: str, points: Sequence[ChunkPoint]) -> None:
        self._check(ctx, "upsert")
        if not ctx.can_write(owner_id):
            raise AppError(ErrorCode.ADMIN_REQUIRED)
        self.calls.append("upsert")
        for p in points:
            self.points[p.point_id] = _Stored(owner_id, p)

    async def hybrid_search(
        self,
        ctx: TenantContext,
        query: Embedding,
        *,
        filters: SearchFilters,
        prefetch_k: int,
        limit: int,
        mode: Literal["hybrid", "dense", "sparse"] = "hybrid",
    ) -> list[ScoredPoint]:
        self._check(ctx, "hybrid_search")
        self.calls.append("hybrid_search")
        pool = [
            s
            for s in self._readable(ctx)
            if (filters.doc_ids is None or s.point.payload.doc_id in filters.doc_ids)
            and (filters.agencies is None or s.point.payload.agency in filters.agencies)
            and (filters.jurisdictions is None or s.point.payload.jurisdiction in filters.jurisdictions)
        ]
        dense = sorted(pool, key=lambda s: -_cos(query.dense, s.point.embedding.dense))[:prefetch_k]
        sparse = sorted(pool, key=lambda s: -_sparse_dot(query.sparse, s.point.embedding.sparse))[:prefetch_k]
        rrf: dict[str, float | None] = {}
        if mode == "hybrid":
            fused: dict[str, float] = {}
            for ranked in (dense, sparse):
                for rank, s in enumerate(ranked, start=1):
                    fused[s.point.point_id] = fused.get(s.point.point_id, 0.0) + 1.0 / (60 + rank)
            rrf = dict(sorted(fused.items(), key=lambda kv: -kv[1])[:limit])
        else:  # a single leg: its own order, no fused score
            rrf = {s.point.point_id: None for s in (dense if mode == "dense" else sparse)[:limit]}
        out = []
        for pid, score in rrf.items():
            s = self.points[pid]
            out.append(
                ScoredPoint(
                    point_id=pid,
                    owner_id=s.owner_id,
                    payload=s.point.payload,
                    similarity=_cos(query.dense, s.point.embedding.dense),
                    sparse_score=_sparse_dot(query.sparse, s.point.embedding.sparse),
                    rrf_score=score,
                )
            )
        return out

    async def point_ids(self, ctx: TenantContext, doc_id: str) -> set[str]:
        self._check(ctx, "point_ids")
        return {s.point.point_id for s in self._readable(ctx) if s.point.payload.doc_id == doc_id}

    async def set_payload(
        self, ctx: TenantContext, point_ids: Sequence[str], payload: dict[str, Any]
    ) -> None:
        self._check(ctx, "set_payload")
        for pid in point_ids:
            s = self.points.get(pid)
            if s and ctx.can_write(s.owner_id):
                s.point = replace(s.point, payload=replace(s.point.payload, **payload))

    async def delete_points(self, ctx: TenantContext, point_ids: Sequence[str]) -> None:
        self._check(ctx, "delete_points")
        for pid in point_ids:
            s = self.points.get(pid)
            if s and ctx.can_write(s.owner_id):
                del self.points[pid]

    async def delete_document(self, ctx: TenantContext, doc_id: str) -> int:
        self._check(ctx, "delete_document")
        doomed = [
            pid
            for pid, s in self.points.items()
            if s.point.payload.doc_id == doc_id and ctx.can_write(s.owner_id)
        ]
        for pid in doomed:
            del self.points[pid]
        return sum(1 for s in self.points.values() if s.point.payload.doc_id == doc_id)

    async def count(self, ctx: TenantContext, doc_id: str | None = None) -> int:
        self._check(ctx, "count")
        return sum(1 for s in self._readable(ctx) if doc_id is None or s.point.payload.doc_id == doc_id)

    async def ready(self) -> bool:
        return True


class FakeReranker:
    model_name = "fake-overlap-reranker"

    async def rerank(self, query: str, passages: Sequence[str]) -> list[float]:
        q = set(tokens(query))
        scores = []
        for p in passages:
            pt = set(tokens(p))
            scores.append(len(q & pt) / len(q) if q else 0.0)
        return scores


Reply = str | Exception | Callable[[Sequence[LLMMessage]], str]


@dataclass
class FakeLLM:
    """Scripted LLM. `replies` are consumed in order; an Exception is raised; "TIMEOUT" sleeps forever."""

    replies: list[Reply] = field(default_factory=list)
    default: Reply = '{"answerable": false, "answer": "", "citations": []}'
    seen: list[list[LLMMessage]] = field(default_factory=list)
    models: list[str | None] = field(default_factory=list)  # the `model` override of each call
    endpoint: Literal["hosted", "ollama"] = "hosted"
    model: str = "fake-model"

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        purpose: Literal["generate", "helper", "judge"],
        max_tokens: int = 1024,
        temperature: float = 0.0,
        model: str | None = None,
    ) -> LLMResult:
        self.seen.append(list(messages))
        self.models.append(model)
        reply = self.replies.pop(0) if self.replies else self.default
        if isinstance(reply, Exception):
            raise reply
        text = reply(messages) if callable(reply) else reply
        if text == "TIMEOUT":
            await asyncio.sleep(3600)
        return LLMResult(
            text=text, endpoint=self.endpoint, model=self.model, input_tokens=100, output_tokens=20
        )

    async def available_models(self) -> list[str]:
        return [self.model]


def make_payload(doc_id: str, text: str, **overrides: Any) -> ChunkPayload:
    base: dict[str, Any] = dict(
        doc_id=doc_id,
        doc_version=1,
        file_name=f"{doc_id}.pdf",
        file_type="pdf",
        chunk_hash=hashlib.sha256(text.encode()).hexdigest(),
        chunk_index=0,
        chunk_count=1,
        text=text,
        token_count=len(tokens(text)),
    )
    base.update(overrides)
    return ChunkPayload(**base)
