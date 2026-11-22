"""BGE-M3 embedder: 1024-d dense + learned sparse lexical weights in one pass (blueprint §3.4).

- The model is loaded lazily, on first use, through the shared GPU worker.
- Every blocking `encode` call goes through `GpuWorker` (semaphore + executor), in batches of
  `settings.embed_batch_size`.
- Two caches:
  * an LRU of query embeddings (short questions repeat a lot in the eval and the demo);
  * the chunk embedding cache keyed `(model, owner_namespace, chunk_hash)` (§5.8.5), used through
    `embed_chunks`. `owner_namespace` is `u_public` for public content and the user id for private
    content, so one tenant's cache hit can never reveal another tenant's upload.

FlagEmbedding calls follow its documented API (`BGEM3FlagModel(...).encode(..., return_dense=True,
return_sparse=True)` → `dense_vecs`, `lexical_weights`). They are exercised live only by the
`@pytest.mark.live` smoke test.
"""

from __future__ import annotations

import asyncio
import base64
import json
import math
import re
import struct
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import structlog

from app.config import Settings
from app.core.device import resolve_device
from app.protocols import Embedding, SparseVector
from app.retrieval.gpu import GpuWorker

log = structlog.get_logger(__name__)

BGE_M3_DIMENSION = 1024
# Module-level defaults (not in Settings; see the RET report).
QUERY_CACHE_SIZE = 2048
DOC_MAX_LENGTH = 1024  # chunks are ≤ 512 tokens (§3.3); headroom for long table rows
QUERY_MAX_LENGTH = 512

_SAFE_SEGMENT = re.compile(r"[^A-Za-z0-9_.-]")


class M3Model(Protocol):
    """The slice of `FlagEmbedding.BGEM3FlagModel` this module uses."""

    def encode(
        self,
        sentences: list[str],
        batch_size: int = ...,
        max_length: int = ...,
        return_dense: bool = ...,
        return_sparse: bool = ...,
        return_colbert_vecs: bool = ...,
    ) -> Mapping[str, Any]: ...


def default_m3_factory(settings: Settings) -> Callable[[], M3Model]:
    def load() -> M3Model:
        from FlagEmbedding import BGEM3FlagModel  # type: ignore[import-not-found, unused-ignore]

        device = resolve_device(settings.device)
        use_fp16 = settings.gpu_fp16 and device.startswith("cuda")
        log.info("embedder.loading", model=settings.embedding_model, device=device, fp16=use_fp16)
        model: M3Model = BGEM3FlagModel(
            settings.embedding_model,
            use_fp16=use_fp16,
            devices=[device],
            normalize_embeddings=True,
        )
        return model

    return load


def _normalise(vec: Sequence[float]) -> tuple[float, ...]:
    values = [float(v) for v in vec]
    norm = math.sqrt(sum(v * v for v in values))
    if norm == 0.0:
        return tuple(values)
    return tuple(v / norm for v in values)


def to_sparse(weights: Mapping[Any, Any]) -> SparseVector:
    """BGE-M3 `lexical_weights` ({token_id: weight}, ids as str or int) → SparseVector, sorted by id.

    Duplicate ids are summed and non-positive weights dropped, so Qdrant gets a valid sparse vector.
    """
    merged: dict[int, float] = {}
    for key, value in weights.items():
        weight = float(value)
        if weight <= 0.0 or not math.isfinite(weight):
            continue
        idx = int(key)
        merged[idx] = merged.get(idx, 0.0) + weight
    items = sorted(merged.items())
    return SparseVector(tuple(i for i, _ in items), tuple(w for _, w in items))


class EmbeddingCache:
    """In-memory dict keyed `(model, owner_namespace, chunk_hash)`, optionally mirrored to disk."""

    def __init__(self, disk_dir: Path | None = None) -> None:
        self._mem: dict[tuple[str, str, str], Embedding] = {}
        self._dir = disk_dir

    def _path(self, key: tuple[str, str, str]) -> Path | None:
        if self._dir is None:
            return None
        model, owner, chunk_hash = (_SAFE_SEGMENT.sub("_", part) for part in key)
        return self._dir / model / owner / f"{chunk_hash}.json"

    def get(self, key: tuple[str, str, str]) -> Embedding | None:
        hit = self._mem.get(key)
        if hit is not None:
            return hit
        path = self._path(key)
        if path is None or not path.is_file():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            dense_bytes = base64.b64decode(raw["dense"])
            dense = struct.unpack(f"<{len(dense_bytes) // 4}f", dense_bytes)
            emb = Embedding(
                tuple(float(v) for v in dense),
                SparseVector(tuple(int(i) for i in raw["si"]), tuple(float(v) for v in raw["sv"])),
            )
        except (OSError, ValueError, KeyError, struct.error):
            log.warning("embedder.cache_entry_unreadable")
            return None
        self._mem[key] = emb
        return emb

    def put(self, key: tuple[str, str, str], emb: Embedding) -> None:
        self._mem[key] = emb
        path = self._path(key)
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            dense = base64.b64encode(struct.pack(f"<{len(emb.dense)}f", *emb.dense)).decode("ascii")
            body = {"dense": dense, "si": list(emb.sparse.indices), "sv": list(emb.sparse.values)}
            path.write_text(json.dumps(body), encoding="utf-8")
        except OSError:
            log.warning("embedder.cache_write_failed")

    def drop_namespace(self, model: str, owner_namespace: str) -> None:
        """Forget a tenant's cached embeddings (account deletion, §5.8.7)."""
        for key in [k for k in self._mem if k[0] == model and k[1] == owner_namespace]:
            del self._mem[key]
        if self._dir is not None:
            folder = self._dir / _SAFE_SEGMENT.sub("_", model) / _SAFE_SEGMENT.sub("_", owner_namespace)
            if folder.is_dir():
                for file in folder.glob("*.json"):
                    file.unlink(missing_ok=True)

    def __len__(self) -> int:
        return len(self._mem)


class BGEM3Embedder:
    """`Embedder` over BAAI/bge-m3. Construction is cheap; the weights load on the first call."""

    dimension: int = BGE_M3_DIMENSION

    def __init__(
        self,
        settings: Settings,
        worker: GpuWorker,
        *,
        model_factory: Callable[[], M3Model] | None = None,
        cache_dir: Path | None = None,
        query_cache_size: int = QUERY_CACHE_SIZE,
        dimension: int = BGE_M3_DIMENSION,
    ) -> None:
        self.model_name = settings.embedding_model
        self.dimension = dimension
        self._batch_size = max(1, settings.embed_batch_size)
        self._worker = worker
        self._factory = model_factory or default_m3_factory(settings)
        self._model: M3Model | None = None
        self._load_lock = asyncio.Lock()
        self._query_cache: OrderedDict[str, Embedding] = OrderedDict()
        self._query_cache_size = query_cache_size
        self.cache = EmbeddingCache(cache_dir)

    @property
    def loaded(self) -> bool:
        return self._model is not None

    async def _get_model(self) -> M3Model:
        if self._model is None:
            async with self._load_lock:
                if self._model is None:
                    self._model = await self._worker.run(self._factory)
                    log.info("embedder.loaded", model=self.model_name)
        return self._model

    def _convert(self, output: Mapping[str, Any], expected: int) -> list[Embedding]:
        dense_vecs = output.get("dense_vecs")
        lexical = output.get("lexical_weights")
        if dense_vecs is None or lexical is None or len(dense_vecs) != expected or len(lexical) != expected:
            raise RuntimeError("BGE-M3 returned an unexpected output shape")
        out: list[Embedding] = []
        for dense, weights in zip(dense_vecs, lexical, strict=True):
            vec = _normalise(list(dense))
            if len(vec) != self.dimension:
                raise RuntimeError(f"expected {self.dimension}-d dense vectors, got {len(vec)}")
            out.append(Embedding(vec, to_sparse(weights)))
        return out

    async def _encode(self, texts: Sequence[str], max_length: int) -> list[Embedding]:
        if not texts:
            return []
        model = await self._get_model()
        out: list[Embedding] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])

            def call(batch: list[str] = batch) -> Mapping[str, Any]:
                return model.encode(
                    batch,
                    batch_size=len(batch),
                    max_length=max_length,
                    return_dense=True,
                    return_sparse=True,
                    return_colbert_vecs=False,
                )

            output = await self._worker.run(call)
            out.extend(self._convert(output, len(batch)))
        return out

    async def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        """Embed chunk texts with no cache (the cache needs an owner namespace: see `embed_chunks`)."""
        return await self._encode(texts, DOC_MAX_LENGTH)

    async def embed_chunks(
        self, owner_namespace: str, chunks: Sequence[tuple[str, str]]
    ) -> tuple[list[Embedding], int]:
        """Embed `(chunk_hash, text)` pairs through the per-owner cache (§5.8.5).

        Returns the embeddings in input order and how many were freshly computed
        (`chunks_embedded` in the upload response).
        """
        results: list[Embedding | None] = []
        missing: dict[str, list[int]] = {}
        texts: dict[str, str] = {}
        for i, (chunk_hash, text) in enumerate(chunks):
            hit = self.cache.get((self.model_name, owner_namespace, chunk_hash))
            results.append(hit)
            if hit is None:
                missing.setdefault(chunk_hash, []).append(i)
                texts[chunk_hash] = text
        if missing:
            order = list(missing)
            fresh = await self._encode([texts[h] for h in order], DOC_MAX_LENGTH)
            for chunk_hash, emb in zip(order, fresh, strict=True):
                self.cache.put((self.model_name, owner_namespace, chunk_hash), emb)
                for i in missing[chunk_hash]:
                    results[i] = emb
        final = [r for r in results if r is not None]
        assert len(final) == len(chunks)
        return final, len(missing)

    async def embed_query(self, text: str) -> Embedding:
        cached = self._query_cache.get(text)
        if cached is not None:
            self._query_cache.move_to_end(text)
            return cached
        (emb,) = await self._encode([text], QUERY_MAX_LENGTH)
        self._query_cache[text] = emb
        if len(self._query_cache) > self._query_cache_size:
            self._query_cache.popitem(last=False)
        return emb

    async def warmup(self) -> None:
        await self.embed_query("warm-up")

    def unload(self) -> None:
        self._model = None
