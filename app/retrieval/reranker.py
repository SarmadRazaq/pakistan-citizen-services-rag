"""bge-reranker-v2-m3 cross-encoder (blueprint §3.6). Scores are sigmoid-normalised into [0, 1].

Lazily loaded on first use and run through the shared `GpuWorker`, so embedding and reranking never
compete for VRAM at the same time. The FlagEmbedding call (`FlagReranker.compute_score(pairs,
normalize=True)`) follows its documented API and is exercised live only by the smoke test.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable, Sequence
from typing import Any, Protocol

import structlog

from app.config import Settings
from app.core.device import resolve_device
from app.retrieval.gpu import GpuWorker

log = structlog.get_logger(__name__)

# Module-level defaults (not in Settings; see the RET report).
RERANK_BATCH_SIZE = 32
RERANK_MAX_LENGTH = 1024


class CrossEncoderModel(Protocol):
    """The slice of `FlagEmbedding.FlagReranker` this module uses."""

    def compute_score(
        self,
        sentence_pairs: list[list[str]],
        batch_size: int = ...,
        max_length: int = ...,
        normalize: bool = ...,
    ) -> Any: ...


def default_reranker_factory(settings: Settings) -> Callable[[], CrossEncoderModel]:
    def load() -> CrossEncoderModel:
        from FlagEmbedding import FlagReranker  # type: ignore[import-not-found, unused-ignore]

        device = resolve_device(settings.device)
        use_fp16 = settings.gpu_fp16 and device.startswith("cuda")
        log.info("reranker.loading", model=settings.reranker_model, device=device, fp16=use_fp16)
        model: CrossEncoderModel = FlagReranker(settings.reranker_model, use_fp16=use_fp16, devices=[device])
        return model

    return load


def _as_unit_scores(raw: Any, expected: int) -> list[float]:
    """`compute_score` returns a float for one pair and a list otherwise; clamp into [0, 1]."""
    values = [raw] if isinstance(raw, int | float) else list(raw)
    if len(values) != expected:
        raise RuntimeError("reranker returned an unexpected number of scores")
    out = []
    for value in values:
        score = float(value)
        out.append(0.0 if not math.isfinite(score) else min(1.0, max(0.0, score)))
    return out


class BGEReranker:
    def __init__(
        self,
        settings: Settings,
        worker: GpuWorker,
        *,
        model_factory: Callable[[], CrossEncoderModel] | None = None,
        batch_size: int = RERANK_BATCH_SIZE,
        max_length: int = RERANK_MAX_LENGTH,
    ) -> None:
        self.model_name = settings.reranker_model
        self._worker = worker
        self._factory = model_factory or default_reranker_factory(settings)
        self._model: CrossEncoderModel | None = None
        self._load_lock = asyncio.Lock()
        self._batch_size = batch_size
        self._max_length = max_length

    @property
    def loaded(self) -> bool:
        return self._model is not None

    async def _get_model(self) -> CrossEncoderModel:
        if self._model is None:
            async with self._load_lock:
                if self._model is None:
                    self._model = await self._worker.run(self._factory)
                    log.info("reranker.loaded", model=self.model_name)
        return self._model

    async def rerank(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        model = await self._get_model()
        pairs = [[query, passage] for passage in passages]

        def call() -> Any:
            return model.compute_score(
                pairs, batch_size=self._batch_size, max_length=self._max_length, normalize=True
            )

        raw = await self._worker.run(call)
        return _as_unit_scores(raw, len(pairs))

    async def warmup(self) -> None:
        await self.rerank("warm-up", ["warm-up"])

    def unload(self) -> None:
        self._model = None
