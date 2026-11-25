"""Wiring for the retrieval stack: embedder, Qdrant store, reranker and retriever.

The embedder and reranker are hosted on the NVIDIA API catalog by default (`EMBEDDING_PROVIDER` /
`RERANKER_PROVIDER` = nvidia) and run locally (BGE-M3, bge-reranker-v2-m3 on the shared GPU worker)
when set to `local`; the two providers are chosen independently. `EMBEDDING_PROVIDER=openai` uses OpenAI
embeddings; `RERANKER_PROVIDER=none` builds no reranker (the retriever gates on dense similarity).

    bundle = build_retrieval(settings)
    await bundle.startup(warmup=True)   # in the FastAPI lifespan
    ...
    await bundle.shutdown()

Building the bundle never loads model weights or opens a network connection; `startup()` creates the
collection and alias if missing, and `warmup=True` loads local models with one tiny forward pass each
(hosted models need no warm-up). `shutdown()` closes the hosted HTTP clients.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from app.config import Settings
from app.core.device import empty_cuda_cache, log_device
from app.core.health import Probe
from app.protocols import Embedder, Reranker
from app.retrieval.embedder import BGEM3Embedder
from app.retrieval.gpu import GpuWorker
from app.retrieval.hosted import NvidiaEmbedder, NvidiaReranker, OpenAIEmbedder
from app.retrieval.reranker import BGEReranker, NoReranker
from app.retrieval.retriever import HybridRetriever
from app.retrieval.vector_store import TenantScopedVectorStore

log = structlog.get_logger(__name__)


@dataclass
class RetrievalBundle:
    embedder: Embedder
    store: TenantScopedVectorStore
    reranker: Reranker
    retriever: HybridRetriever
    worker: GpuWorker
    settings: Settings

    async def startup(self, *, warmup: bool = False) -> None:
        await self.store.ensure_collection()
        if warmup:
            log_device(self.settings.device)
            await self.embedder.warmup()  # type: ignore[attr-defined]
            await self.reranker.warmup()  # type: ignore[attr-defined]
            log.info("retrieval.warmed_up")

    async def shutdown(self) -> None:
        await self.store.close()
        models: tuple[Any, ...] = (self.embedder, self.reranker)
        for model in models:
            model.unload()
            aclose = getattr(model, "aclose", None)
            if aclose is not None:
                await aclose()
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
    embedder: Embedder
    if settings.embedding_provider == "nvidia":
        embedder = NvidiaEmbedder(settings)
    elif settings.embedding_provider == "openai":
        embedder = OpenAIEmbedder(settings)
    else:
        embedder = BGEM3Embedder(settings, worker, cache_dir=embedding_cache_dir)
    store = TenantScopedVectorStore.from_settings(
        settings, dense_dim=embedder.dimension, location=qdrant_location
    )
    reranker: Reranker
    if settings.reranker_provider == "nvidia":
        reranker = NvidiaReranker(settings)
    elif settings.reranker_provider == "none":
        reranker = NoReranker()
    else:
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
