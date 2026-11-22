"""Wiring for the retrieval stack: one GPU worker, BGE-M3, the Qdrant store, the reranker, the retriever.

    bundle = build_retrieval(settings)
    await bundle.startup(warmup=True)   # in the FastAPI lifespan
    ...
    await bundle.shutdown()

Building the bundle never loads model weights or opens a network connection; `startup()` creates the
collection and alias if missing, and `warmup=True` loads both models with one tiny forward pass each.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import structlog

from app.config import Settings
from app.core.device import empty_cuda_cache, log_device
from app.core.health import Probe
from app.retrieval.embedder import BGEM3Embedder
from app.retrieval.gpu import GpuWorker
from app.retrieval.reranker import BGEReranker
from app.retrieval.retriever import HybridRetriever
from app.retrieval.vector_store import TenantScopedVectorStore

log = structlog.get_logger(__name__)


@dataclass
class RetrievalBundle:
    embedder: BGEM3Embedder
    store: TenantScopedVectorStore
    reranker: BGEReranker
    retriever: HybridRetriever
    worker: GpuWorker
    settings: Settings

    async def startup(self, *, warmup: bool = False) -> None:
        await self.store.ensure_collection()
        if warmup:
            log_device(self.settings.device)
            await self.embedder.warmup()
            await self.reranker.warmup()
            log.info("retrieval.warmed_up")

    async def shutdown(self) -> None:
        await self.store.close()
        self.embedder.unload()
        self.reranker.unload()
        self.worker.shutdown()
        empty_cuda_cache()

    def probes(self) -> list[Probe]:
        """Readiness probe for /health/ready: the alias resolves to a collection."""

        async def qdrant() -> tuple[bool, str | None]:
            ok = await self.store.ready()
            return ok, None if ok else f"collection alias {self.store.alias} not reachable"

        return [Probe("qdrant", qdrant)]


def build_retrieval(
    settings: Settings,
    *,
    qdrant_location: str | None = None,
    embedding_cache_dir: Path | None = None,
) -> RetrievalBundle:
    worker = GpuWorker(settings.gpu_max_concurrent)
    embedder = BGEM3Embedder(settings, worker, cache_dir=embedding_cache_dir)
    store = TenantScopedVectorStore.from_settings(
        settings, dense_dim=embedder.dimension, location=qdrant_location
    )
    reranker = BGEReranker(settings, worker)
    retriever = HybridRetriever(settings, embedder, store, reranker)
    return RetrievalBundle(
        embedder=embedder,
        store=store,
        reranker=reranker,
        retriever=retriever,
        worker=worker,
        settings=settings,
    )


async def startup(bundle: RetrievalBundle, *, warmup: bool = False) -> None:
    await bundle.startup(warmup=warmup)


async def shutdown(bundle: RetrievalBundle) -> None:
    await bundle.shutdown()
