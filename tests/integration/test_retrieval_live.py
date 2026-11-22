"""Live smoke test: loads the real BGE-M3 and bge-reranker-v2-m3 weights (~4.5 GB download on first run).

Skipped unless the ML stack is installed (`requirements-ml.in`) AND `RUN_LIVE_MODELS=1` is set, so a
plain `pytest` never triggers a multi-GB download:

    RUN_LIVE_MODELS=1 .venv/Scripts/python -m pytest -m live tests/integration/test_retrieval_live.py
"""

from __future__ import annotations

import os

import pytest

from app.config import Settings
from app.core.device import log_device, max_memory_allocated_mb
from app.retrieval.embedder import BGEM3Embedder
from app.retrieval.gpu import GpuWorker
from app.retrieval.reranker import BGEReranker
from app.retrieval.vector_store import cosine

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("RUN_LIVE_MODELS") != "1", reason="set RUN_LIVE_MODELS=1"),
]


async def test_real_models_smoke(settings: Settings) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("FlagEmbedding")
    log_device(settings.device)
    worker = GpuWorker(settings.gpu_max_concurrent)
    embedder = BGEM3Embedder(settings, worker)
    reranker = BGEReranker(settings, worker)

    passage = "The fee for an urgent machine readable passport is 5,000 rupees."
    other = "Smart national identity cards are delivered within fifteen working days."
    docs = await embedder.embed_documents([passage, other])
    query = await embedder.embed_query("فوری پاسپورٹ کی فیس کتنی ہے؟")  # Urdu: urgent passport fee?

    assert all(len(d.dense) == 1024 for d in docs)
    assert docs[0].sparse.indices and list(docs[0].sparse.indices) == sorted(docs[0].sparse.indices)
    assert cosine(query.dense, docs[0].dense) > cosine(query.dense, docs[1].dense)

    scores = await reranker.rerank("How much is the urgent passport fee?", [passage, other])
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert scores[0] > scores[1]
    print("peak VRAM MB:", max_memory_allocated_mb())
    worker.shutdown()
