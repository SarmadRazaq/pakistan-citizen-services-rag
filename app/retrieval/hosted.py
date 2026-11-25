"""Hosted retrieval models on the NVIDIA API catalog: embedder, reranker and jailbreak classifier.

- `NvidiaEmbedder` calls `baai/bge-m3` (`POST /v1/embeddings`) for the dense vector. The hosted API has
  no sparse output, so the sparse half of every `Embedding` comes from `LexicalSparseEncoder`.
- `NvidiaReranker` calls the NeMo Retriever reranking NIM and turns each logit into a score in [0, 1].
- `NvidiaJailbreakClassifier` calls NemoGuard jailbreak-detect and implements `InjectionClassifier`.

All three share one HTTP helper: Bearer key, per-request timeout, up to two retries on 429 / 5xx /
transport errors (exponential backoff with jitter, honouring a capped `Retry-After`), no retry on other
4xx, and every failure surfaces as `AppError(DEPENDENCY_UNAVAILABLE)`. Only the model, status, latency
and attempt number are logged: never the key, the texts or the response bodies, since a Chunk of a
private Document may be among them.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import random
import re
import time
import unicodedata
from collections import Counter, OrderedDict
from collections.abc import Sequence
from typing import Any

import httpx
import structlog

from app.config import Settings
from app.core.errors import AppError, ErrorCode
from app.core.resilience import Sleep, backoff_delay, parse_retry_after
from app.guardrails.patterns import unify_urdu
from app.ingestion.language import normalize_digits, strip_diacritics
from app.protocols import Embedding, SparseVector

log = structlog.get_logger(__name__)

# Module-level defaults (not in Settings).
MAX_RETRIES = 2
QUERY_CACHE_SIZE = 2048
RERANK_MAX_PASSAGES = 512  # the reranking NIM accepts at most 512 passages per request
SPARSE_INDEX_MASK = 0x7FFFFFFF  # 31 bits: valid for Qdrant's unsigned 32-bit sparse indices
BM25_K1 = 1.2
BM25_B = 0.75
BM25_AVGDL = 256.0

_WORD = re.compile(r"\w+")

_STOPWORDS_EN = (
    "a an and are as at be but by for from has have he her his i in is it its of on or she that the their "
    "them they this to was we were what when where which who why will with you your"
)
_STOPWORDS_ROMAN_URDU = (
    "ka ki ke ko se me mein hai hain tha thi the ho hota hoti hota hoga hogi aur ya par pe jo yeh ye woh wo "
    "kya kab kahan kaun kyun kyon kaise is us in un na nahi nahin bhi to toh"
)
_STOPWORDS_URDU = (
    "کا کی کے کو سے میں ہے ہیں تھا تھی تھے ہو اور یا پر جو یہ وہ کیا کب کہاں کون کیوں کیسے اس ان نہیں بھی تو"
)


def _normalise(text: str) -> str:
    text = unify_urdu(unicodedata.normalize("NFKC", text))
    return strip_diacritics(normalize_digits(text)).casefold()


_STOPWORDS = frozenset(
    _normalise(word)
    for group in (_STOPWORDS_EN, _STOPWORDS_ROMAN_URDU, _STOPWORDS_URDU)
    for word in group.split()
)


class LexicalSparseEncoder:
    """Deterministic hashed bag-of-words sparse vectors (the sparse half of hosted hybrid search).

    Text is NFKC-normalised, Urdu letter variants are unified (Arabic kaf/yeh/heh),
    Urdu and Arabic-Indic digits become ASCII, Urdu diacritics are stripped (as ingestion's
    `normalize_for_index` does), and case is folded.
    Tokens are Unicode word runs; stopwords (English, Roman Urdu, Urdu) and one-character non-digit
    tokens are dropped. A token's index is a 31-bit blake2b hash, so it is identical across processes
    and between Documents and queries. A Chunk weighs each token with BM25-style term-frequency
    saturation; a query weighs each unique token 1.0.

    There is no IDF on purpose (blueprint §5.8.5): a collection-wide IDF would let other tenants'
    Documents shift this tenant's scores, leaking information about what they uploaded.
    """

    def encode_document(self, text: str) -> SparseVector:
        counts = Counter(self._tokens(text))
        length = sum(counts.values())
        norm = BM25_K1 * (1 - BM25_B + BM25_B * length / BM25_AVGDL)
        return self._vector((tok, tf * (BM25_K1 + 1) / (tf + norm)) for tok, tf in counts.items())

    def encode_query(self, text: str) -> SparseVector:
        return self._vector((tok, 1.0) for tok in set(self._tokens(text)))

    @staticmethod
    def _tokens(text: str) -> list[str]:
        tokens = _WORD.findall(_normalise(text))
        return [t for t in tokens if t not in _STOPWORDS and (len(t) > 1 or t.isdigit())]

    @staticmethod
    def _vector(weighted: Any) -> SparseVector:
        merged: dict[int, float] = {}
        for token, weight in weighted:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") & SPARSE_INDEX_MASK
            merged[index] = merged.get(index, 0.0) + weight  # collisions sum
        items = sorted(merged.items())
        return SparseVector(tuple(i for i, _ in items), tuple(w for _, w in items))


class _HostedClient:
    """Shared HTTP plumbing for the NVIDIA retrieval endpoints."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: httpx.AsyncClient | None = None,
        sleep: Sleep = asyncio.sleep,
        rng: random.Random | None = None,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        self._key = settings.nvidia_api_key.get_secret_value().strip() if settings.nvidia_api_key else ""
        self._timeout_s = settings.hosted_retrieval_timeout_s
        self._client = client or httpx.AsyncClient()
        self._owns_client = client is None
        self._sleep = sleep
        self._rng = rng or random.Random()  # noqa: S311 (backoff jitter, not cryptography)
        self._max_retries = max_retries

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _post(self, url: str, body: dict[str, Any], *, model: str) -> Any:
        """POST `body` as JSON and return the decoded response; any failure is DEPENDENCY_UNAVAILABLE."""
        if not self._key:
            raise AppError(ErrorCode.DEPENDENCY_UNAVAILABLE, "NVIDIA_API_KEY is not set.")
        headers = {"Authorization": f"Bearer {self._key}", "Accept": "application/json"}
        retries = 0
        while True:
            sent = time.perf_counter()
            retry_after: float | None = None
            try:
                response = await self._client.post(
                    url, json=body, headers=headers, timeout=httpx.Timeout(self._timeout_s)
                )
            except httpx.HTTPError as exc:
                status: int | str = type(exc).__name__
                transient = True
            else:
                status = response.status_code
                if status == 200:
                    self._log(model, status, sent, retries + 1)
                    try:
                        return response.json()
                    except ValueError as exc:
                        raise AppError(
                            ErrorCode.DEPENDENCY_UNAVAILABLE, "A hosted model returned an unreadable reply."
                        ) from exc
                transient = status == 429 or status >= 500
                retry_after = parse_retry_after(response.headers.get("Retry-After"))
            self._log(model, status, sent, retries + 1)
            if status in (400, 422):
                raise _RequestRejectedError(
                    ErrorCode.DEPENDENCY_UNAVAILABLE, "A hosted model rejected the request."
                )
            if not transient or retries >= self._max_retries:
                raise AppError(ErrorCode.DEPENDENCY_UNAVAILABLE, "A hosted model is unavailable right now.")
            await self._sleep(backoff_delay(retries, retry_after_s=retry_after, rng=self._rng))
            retries += 1

    @staticmethod
    def _log(model: str, status: int | str, sent: float, attempt: int) -> None:
        log.info(
            "hosted.attempt",
            model=model,
            status=status,
            latency_ms=round((time.perf_counter() - sent) * 1000),
            attempt=attempt,
        )


class _RequestRejectedError(AppError):
    """A 400/422 from a hosted model: the request shape was refused (not a transient failure)."""


def _bad_reply() -> AppError:
    return AppError(ErrorCode.DEPENDENCY_UNAVAILABLE, "A hosted model returned an unexpected reply.")


class NvidiaEmbedder(_HostedClient):
    """`Embedder` over the hosted BGE-M3: dense from NVIDIA, sparse from `LexicalSparseEncoder`."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: httpx.AsyncClient | None = None,
        sparse: LexicalSparseEncoder | None = None,
        sleep: Sleep = asyncio.sleep,
        rng: random.Random | None = None,
    ) -> None:
        super().__init__(settings, client=client, sleep=sleep, rng=rng)
        self.model_name = settings.hosted_embedding_model
        self.dimension = settings.embedding_dimension
        self._url = f"{settings.nvidia_base_url.rstrip('/')}/embeddings"
        self._batch_size = max(1, settings.embed_batch_size)
        self._sparse = sparse or LexicalSparseEncoder()
        self._query_cache: OrderedDict[str, Embedding] = OrderedDict()
        # Asymmetric models (NV-Embed-QA, E5) need `input_type`; symmetric ones such as BGE-M3 may
        # reject it. Send it first; on a 400/422 retry once without it and remember (like JSON mode
        # in the LLM router).
        self._send_input_type = True

    @property
    def loaded(self) -> bool:
        return True  # nothing to load: the model runs on NVIDIA's side

    async def warmup(self) -> None:
        return None

    def unload(self) -> None:
        return None

    async def _dense(self, texts: Sequence[str], input_type: str) -> list[tuple[float, ...]]:
        body: dict[str, Any] = {
            "model": self.model_name,
            "input": list(texts),
            "encoding_format": "float",
            "truncate": "END",
        }
        if self._send_input_type:
            try:
                data = await self._post(self._url, {**body, "input_type": input_type}, model=self.model_name)
            except _RequestRejectedError:
                log.info("hosted.embedder_input_type_unsupported", model=self.model_name)
                self._send_input_type = False
                data = await self._post(self._url, body, model=self.model_name)
        else:
            data = await self._post(self._url, body, model=self.model_name)
        try:
            rows = sorted(data["data"], key=lambda row: row["index"])
            vectors = [tuple(float(v) for v in row["embedding"]) for row in rows]
        except (KeyError, TypeError, ValueError) as exc:
            raise _bad_reply() from exc
        if len(vectors) != len(texts) or any(len(v) != self.dimension for v in vectors):
            raise _bad_reply()
        return vectors

    async def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        out: list[Embedding] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            vectors = await self._dense(batch, "passage")
            out.extend(
                Embedding(vec, self._sparse.encode_document(text))
                for vec, text in zip(vectors, batch, strict=True)
            )
        return out

    async def embed_query(self, text: str) -> Embedding:
        cached = self._query_cache.get(text)
        if cached is not None:
            self._query_cache.move_to_end(text)
            return cached
        (vec,) = await self._dense([text], "query")
        emb = Embedding(vec, self._sparse.encode_query(text))
        self._query_cache[text] = emb
        if len(self._query_cache) > QUERY_CACHE_SIZE:
            self._query_cache.popitem(last=False)
        return emb


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


class NvidiaReranker(_HostedClient):
    """`Reranker` over the hosted NeMo Retriever reranking NIM; logits are sigmoid-normalised."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: httpx.AsyncClient | None = None,
        sleep: Sleep = asyncio.sleep,
        rng: random.Random | None = None,
    ) -> None:
        super().__init__(settings, client=client, sleep=sleep, rng=rng)
        self.model_name = settings.hosted_reranker_model
        self._url = settings.reranker_url

    async def warmup(self) -> None:
        return None

    def unload(self) -> None:
        return None

    async def rerank(self, query: str, passages: Sequence[str]) -> list[float]:
        scores: list[float] = []
        for start in range(0, len(passages), RERANK_MAX_PASSAGES):
            batch = passages[start : start + RERANK_MAX_PASSAGES]
            body = {
                "model": self.model_name,
                "query": {"text": query},
                "passages": [{"text": p} for p in batch],
                "truncate": "END",
            }
            data = await self._post(self._url, body, model=self.model_name)
            logits: dict[int, float] = {}
            try:
                for row in data["rankings"]:
                    logits[int(row["index"])] = float(row["logit"])
            except (KeyError, TypeError, ValueError) as exc:
                raise _bad_reply() from exc
            if set(logits) != set(range(len(batch))):
                raise _bad_reply()
            scores.extend(_sigmoid(logits[i]) for i in range(len(batch)))
        return scores


class NvidiaJailbreakClassifier(_HostedClient):
    """`InjectionClassifier` over NemoGuard jailbreak-detect.

    No retries: the input guard fails open and logs when this raises, and a Question must not wait on
    a slow classifier.
    """

    def __init__(self, settings: Settings, *, client: httpx.AsyncClient | None = None) -> None:
        super().__init__(settings, client=client, max_retries=0)
        self._url = settings.jailbreak_detect_url
        self._model = "nemoguard-jailbreak-detect"

    async def score(self, text: str) -> float:
        data = await self._post(self._url, {"input": text}, model=self._model)
        raw = data.get("score") if isinstance(data, dict) else None
        if isinstance(raw, int | float) and not isinstance(raw, bool):
            return min(1.0, max(0.0, float(raw)))
        flag = data.get("jailbreak") if isinstance(data, dict) else None
        if isinstance(flag, bool):
            return 1.0 if flag else 0.0
        raise _bad_reply()
