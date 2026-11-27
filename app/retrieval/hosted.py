"""Hosted retrieval models: the OpenAI embedder and the lexical sparse encoder.

- `OpenAIEmbedder` calls OpenAI `text-embedding-3-*` (`POST /v1/embeddings`, `dimensions` =
  EMBEDDING_DIMENSION) for the dense vector. The hosted API has no sparse output, so the sparse half of
  every `Embedding` comes from `LexicalSparseEncoder`.

The embedder shares one HTTP helper: Bearer key, per-request timeout, up to two retries on 429 / 5xx /
transport errors (exponential backoff with jitter, honouring a capped `Retry-After`), no retry on other
4xx, and every failure surfaces as `AppError(DEPENDENCY_UNAVAILABLE)`. Only the model, status, latency
and attempt number are logged: never the key, the texts or the response bodies, since a Chunk of a
private Document may be among them.
"""

from __future__ import annotations

import asyncio
import hashlib
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
OPENAI_MAX_INPUTS = 2048  # OpenAI's per-request limit on embedding inputs
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


# A digit-group separator between digits: "4,500", "1,00,000" (lakh grouping), "۴٬۵۰۰" (Arabic separator).
# Dropping it makes "Rs 4,500" in a fee table and "4500" in a question the same token.
_DIGIT_GROUP_SEP = re.compile(r"(?<=\d)[,٬](?=\d{2,3}(?!\d))")


def _normalise(text: str) -> str:
    text = unify_urdu(unicodedata.normalize("NFKC", text))
    text = _DIGIT_GROUP_SEP.sub("", normalize_digits(text))
    return strip_diacritics(text).casefold()


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
    """Shared HTTP plumbing for the hosted retrieval endpoints."""

    _KEY_NAME = "OPENAI_API_KEY"

    def __init__(
        self,
        settings: Settings,
        *,
        client: httpx.AsyncClient | None = None,
        sleep: Sleep = asyncio.sleep,
        rng: random.Random | None = None,
        max_retries: int = MAX_RETRIES,
        api_key: str | None = None,
    ) -> None:
        self._key = settings.openai_key if api_key is None else api_key.strip()
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
            raise AppError(ErrorCode.DEPENDENCY_UNAVAILABLE, f"{self._KEY_NAME} is not set.")
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


def _bad_reply() -> AppError:
    return AppError(ErrorCode.DEPENDENCY_UNAVAILABLE, "A hosted model returned an unexpected reply.")


class _HostedEmbedder(_HostedClient):
    """Shared `Embedder` logic: batching, query cache, vector validation, lexical sparse half.

    Subclasses set `model_name` and `_url`, and may add body fields (`_extra_body`).
    """

    def __init__(
        self,
        settings: Settings,
        *,
        client: httpx.AsyncClient | None = None,
        sparse: LexicalSparseEncoder | None = None,
        sleep: Sleep = asyncio.sleep,
        rng: random.Random | None = None,
        api_key: str | None = None,
    ) -> None:
        super().__init__(settings, client=client, sleep=sleep, rng=rng, api_key=api_key)
        self.model_name = ""
        self.dimension = settings.embedding_dimension
        self._url = ""
        self._batch_size = max(1, settings.embed_batch_size)
        self._sparse = sparse or LexicalSparseEncoder()
        self._query_cache: OrderedDict[str, Embedding] = OrderedDict()

    @property
    def loaded(self) -> bool:
        return True  # nothing to load: the model runs on the provider's side

    async def warmup(self) -> None:
        return None

    def unload(self) -> None:
        return None

    def _extra_body(self) -> dict[str, Any]:
        return {}

    async def _dense(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        body: dict[str, Any] = {
            "model": self.model_name,
            "input": list(texts),
            "encoding_format": "float",
            **self._extra_body(),
        }
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
            vectors = await self._dense(batch)
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
        (vec,) = await self._dense([text])
        emb = Embedding(vec, self._sparse.encode_query(text))
        self._query_cache[text] = emb
        if len(self._query_cache) > QUERY_CACHE_SIZE:
            self._query_cache.popitem(last=False)
        return emb


class OpenAIEmbedder(_HostedEmbedder):
    """`Embedder` over OpenAI `POST /embeddings` (text-embedding-3-*): dense from OpenAI, sparse lexical.

    `dimensions` asks the text-embedding-3 models for EMBEDDING_DIMENSION (1024) values, so the Qdrant
    collection stays at that size. OpenAI allows 2048
    inputs per request; EMBED_BATCH_SIZE (32) stays far below that.
    """

    _KEY_NAME = "OPENAI_API_KEY"

    def __init__(
        self,
        settings: Settings,
        *,
        client: httpx.AsyncClient | None = None,
        sparse: LexicalSparseEncoder | None = None,
        sleep: Sleep = asyncio.sleep,
        rng: random.Random | None = None,
    ) -> None:
        super().__init__(
            settings, client=client, sparse=sparse, sleep=sleep, rng=rng, api_key=settings.openai_key
        )
        self.model_name = settings.openai_embedding_model
        self._url = f"{settings.openai_base_url.rstrip('/')}/embeddings"
        self._batch_size = min(self._batch_size, OPENAI_MAX_INPUTS)

    def _extra_body(self) -> dict[str, Any]:
        return {"dimensions": self.dimension}
