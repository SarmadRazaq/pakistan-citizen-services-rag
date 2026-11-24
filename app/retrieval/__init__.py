"""Retrieval package. `build_ingest_components` is the hook scripts/seed_corpus.py loads."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.config import Settings
    from app.protocols import Embedder
    from app.retrieval.vector_store import TenantScopedVectorStore


def build_ingest_components(settings: Settings) -> tuple[TenantScopedVectorStore, Embedder]:
    """(store, embedder) for offline seeding. The caller awaits `store.ensure_collection()`."""
    from app.retrieval.factory import build_retrieval

    bundle = build_retrieval(settings)
    return bundle.store, bundle.embedder
