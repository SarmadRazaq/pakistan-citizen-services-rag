"""HybridRetriever: variants → hybrid search → cross-variant RRF → dedupe → rerank (blueprint §3.6, §4).

1. Embed each query variant (original, English, Urdu script) and run `hybrid_search` for all of them
   in parallel.
2. Fuse the per-variant lists with RRF (k=60) by `point_id`. `rrf_score` on the returned points is this
   cross-variant score; `similarity` and `sparse_score` are the best over the variants.
3. Dedupe: identical `chunk_hash` collapses to one (the private copy wins, §5.8.4), then near-duplicate
   text (rapidfuzz ratio ≥ 95, a cheap stand-in for cosine > 0.95) is dropped below its better twin.
4. Rerank the top `candidates_k` against the original question (`variants[0]`).
5. Context = the top `context_k` with `rerank_score ≥ tau_keep`, at least one; `max_rerank` drives the
   relevance gate in the ask pipeline.

Eval overrides (`RetrievalQuery.overrides`, from the `X-Eval-Config` header; absent in normal serving):
- `retrieval_mode` `dense` / `sparse` searches a single leg (the other leg is never queried);
- `rerank=false` skips the reranker. Candidates keep the fused order, `rerank_score` is None, and the
  gate signal changes: `max_rerank` becomes the best **dense cosine similarity** among the candidates
  and the context keeps the top `context_k` with `similarity ≥ tau_keep` (at least one). `tau_keep` and
  `tau_refuse` were tuned for reranker scores, so a no-rerank run refuses at different rates than the
  default; that difference is part of what the comparison measures.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, replace

import structlog
from rapidfuzz import fuzz

from app.config import Settings
from app.core.eval_overrides import EvalOverrides
from app.core.tenancy import TenantContext
from app.protocols import (
    Candidate,
    Embedder,
    Reranker,
    RetrievalQuery,
    RetrievalResult,
    ScoredPoint,
    VectorStore,
)

log = structlog.get_logger(__name__)

RRF_K = 60


@dataclass
class _Fused:
    point: ScoredPoint
    score: float
    first_seen: int  # tie-breaker: earlier variant / rank first


def fuse_rrf(result_lists: list[list[ScoredPoint]], k: int = RRF_K) -> list[ScoredPoint]:
    """Reciprocal Rank Fusion over variant result lists, by point_id. Highest fused score first."""
    fused: dict[str, _Fused] = {}
    order = 0
    for results in result_lists:
        for rank, point in enumerate(results, start=1):
            contribution = 1.0 / (k + rank)
            entry = fused.get(point.point_id)
            if entry is None:
                fused[point.point_id] = _Fused(point, contribution, order)
            else:
                entry.score += contribution
                best = entry.point
                entry.point = replace(
                    best,
                    similarity=max(best.similarity, point.similarity),
                    sparse_score=_max_opt(best.sparse_score, point.sparse_score),
                )
            order += 1
    ranked = sorted(fused.values(), key=lambda e: (-e.score, e.first_seen))
    return [replace(e.point, rrf_score=e.score) for e in ranked]


def _max_opt(a: float | None, b: float | None) -> float | None:
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


def dedupe(points: list[ScoredPoint], *, public_owner_id: str, near_dup_ratio: float) -> list[ScoredPoint]:
    """Collapse identical chunk_hash (private beats public, §5.8.4), then drop near-duplicate text.

    The surviving copy of an identical-hash group takes the best position the group had.
    """
    groups: dict[str, int] = {}  # chunk_hash → index in `kept`
    kept: list[ScoredPoint] = []
    for point in points:
        idx = groups.get(point.payload.chunk_hash)
        if idx is None:
            groups[point.payload.chunk_hash] = len(kept)
            kept.append(point)
            continue
        current = kept[idx]
        if current.owner_id == public_owner_id and point.owner_id != public_owner_id:
            kept[idx] = replace(
                point,
                similarity=max(point.similarity, current.similarity),
                rrf_score=_max_opt(point.rrf_score, current.rrf_score),
            )
    out: list[ScoredPoint] = []
    for point in kept:
        if any(fuzz.ratio(point.payload.text, better.payload.text) >= near_dup_ratio for better in out):
            continue
        out.append(point)
    return out


def _ms(start: float) -> int:
    return round((time.perf_counter() - start) * 1000)


class HybridRetriever:
    def __init__(
        self, settings: Settings, embedder: Embedder, store: VectorStore, reranker: Reranker
    ) -> None:
        self._settings = settings
        self._embedder = embedder
        self._store = store
        self._reranker = reranker

    @staticmethod
    def _variants(query: RetrievalQuery) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for variant in query.variants:
            text = variant.strip()
            if text and text not in seen:
                seen.add(text)
                out.append(text)
        return out

    async def retrieve(self, ctx: TenantContext, query: RetrievalQuery) -> RetrievalResult:
        variants = self._variants(query)
        if not variants:
            return RetrievalResult(candidates=(), context=(), max_rerank=None, latency_ms={})
        latency: dict[str, int] = {}
        started = time.perf_counter()

        t = time.perf_counter()
        embeddings = await asyncio.gather(*(self._embedder.embed_query(v) for v in variants))
        latency["embed"] = _ms(t)

        overrides = query.overrides or EvalOverrides()
        candidates_k = max(1, query.candidates_k)
        prefetch_k = max(self._settings.prefetch_k, candidates_k)
        t = time.perf_counter()
        per_variant = await asyncio.gather(
            *(
                self._store.hybrid_search(
                    ctx,
                    emb,
                    filters=query.filters,
                    prefetch_k=prefetch_k,
                    limit=candidates_k,
                    **({} if overrides.retrieval_mode == "hybrid" else {"mode": overrides.retrieval_mode}),
                )
                for emb in embeddings
            )
        )
        latency["search"] = _ms(t)

        fused = fuse_rrf([list(r) for r in per_variant])
        # §5.8.2 layer 4: never let a point the context cannot read go further, whatever the store did.
        safe = [p for p in fused if ctx.can_read(p.owner_id)]
        if len(safe) != len(fused):
            log.error("security.cross_tenant_point_dropped", dropped=len(fused) - len(safe))
        unique = dedupe(
            safe,
            public_owner_id=ctx.public_owner_id,
            near_dup_ratio=self._settings.dedupe_cosine * 100.0,
        )
        top = unique[:candidates_k]

        # `gate_scores` is what the relevance gate and the context cut compare with tau_keep / tau_refuse:
        # the reranker scores, or (rerank=false, eval only) the dense cosine similarity.
        t = time.perf_counter()
        if overrides.rerank:
            scores = await self._reranker.rerank(variants[0], [p.payload.text for p in top]) if top else []
            if len(scores) != len(top):
                raise RuntimeError("reranker returned a different number of scores than passages")
            gate_scores = scores
            order = sorted(range(len(top)), key=lambda i: (-scores[i], i))
        else:
            scores = []
            gate_scores = [p.similarity for p in top]
            order = list(range(len(top)))  # keep the fused (or single-leg) order
        latency["rerank"] = _ms(t)

        context_k = max(1, query.context_k)
        context_idx: set[int] = set()
        for i in order:
            if len(context_idx) >= context_k:
                break
            if gate_scores[i] >= self._settings.tau_keep:
                context_idx.add(i)
        if not context_idx and order:
            context_idx.add(order[0])  # at least one chunk; the relevance gate decides on max_rerank

        candidates = tuple(
            Candidate(
                point=top[i],
                rank=rank,
                rerank_score=scores[i] if overrides.rerank else None,
                used_in_context=i in context_idx,
            )
            for rank, i in enumerate(order, start=1)
        )
        context = tuple(c for c in candidates if c.used_in_context)
        latency["total"] = _ms(started)
        return RetrievalResult(
            candidates=candidates,
            context=context,
            max_rerank=max(gate_scores) if gate_scores else None,
            latency_ms=latency,
        )
