"""Interfaces between components: the frozen internal contract (blueprint §2.3, §15 P0).

Every external dependency sits behind one of these `Protocol`s. Tests inject fakes, and the
evaluation swaps implementations through config alone. Data-access methods take a
`TenantContext` as their first argument, so tenant isolation cannot be skipped (§5.8.2).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from app.core.eval_overrides import EvalOverrides
from app.core.tenancy import TenantContext

# ---------------------------------------------------------------- extraction


@dataclass(frozen=True, slots=True)
class ExtractedBlock:
    """One structural unit of a document: a paragraph, heading, table, or OCR'd page region."""

    text: str
    kind: Literal["heading", "paragraph", "table", "list", "ocr"]
    page: int | None = None  # 1-based; slide number for PPTX; 1 for images
    section_path: tuple[str, ...] = ()
    char_start: int = 0
    is_ocr: bool = False
    ocr_confidence: float | None = None


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    blocks: tuple[ExtractedBlock, ...]
    page_count: int
    title: str | None = None
    language_hint: str | None = None
    hidden_text_removed: int = 0
    ocr_pages: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Extractor(Protocol):
    """Turns one validated file into structural blocks. Raises AppError for unusable files."""

    file_types: frozenset[str]

    def extract(self, path: Path) -> ExtractedDocument: ...


# ---------------------------------------------------------------- chunks and vectors


@dataclass(frozen=True, slots=True)
class SparseVector:
    indices: tuple[int, ...]
    values: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class Embedding:
    dense: tuple[float, ...]
    sparse: SparseVector


@dataclass(frozen=True, slots=True)
class ChunkPayload:
    """The Qdrant payload (§5.3). `user_id` and `visibility` are set by the store from the context."""

    doc_id: str
    doc_version: int
    file_name: str
    file_type: str
    chunk_hash: str
    chunk_index: int
    chunk_count: int
    text: str
    token_count: int
    language: str | None = None
    script: str | None = None
    agency: str | None = None
    jurisdiction: str | None = None
    topic: str | None = None
    effective_date: date | None = None
    doc_type: str | None = None
    source_url: str | None = None
    title: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    section_path: tuple[str, ...] = ()
    char_start: int = 0
    char_end: int = 0
    is_table: bool = False
    is_ocr: bool = False
    injection_score: float = 0.0
    injection_flag: bool = False


@dataclass(frozen=True, slots=True)
class ChunkPoint:
    point_id: str  # uuid5(doc_id:chunk_hash:occurrence), deterministic (§5.2)
    embedding: Embedding
    payload: ChunkPayload


@dataclass(frozen=True, slots=True)
class SearchFilters:
    doc_ids: tuple[str, ...] | None = None
    agencies: tuple[str, ...] | None = None
    jurisdictions: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class ScoredPoint:
    point_id: str
    owner_id: str
    payload: ChunkPayload
    similarity: float  # dense cosine, always present
    sparse_score: float | None = None
    rrf_score: float | None = None


@runtime_checkable
class Embedder(Protocol):
    model_name: str
    dimension: int

    async def embed_documents(self, texts: Sequence[str]) -> list[Embedding]: ...

    async def embed_query(self, text: str) -> Embedding: ...


@runtime_checkable
class VectorStore(Protocol):
    """Implemented only by TenantScopedVectorStore, which adds the tenant condition to every call."""

    async def upsert(self, ctx: TenantContext, owner_id: str, points: Sequence[ChunkPoint]) -> None: ...

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
        """`mode` is an eval override: `dense` / `sparse` search a single leg (no fused `rrf_score`)."""
        ...

    async def point_ids(self, ctx: TenantContext, doc_id: str) -> set[str]: ...

    async def set_payload(
        self, ctx: TenantContext, point_ids: Sequence[str], payload: dict[str, Any]
    ) -> None: ...

    async def delete_points(self, ctx: TenantContext, point_ids: Sequence[str]) -> None: ...

    async def delete_document(self, ctx: TenantContext, doc_id: str) -> int:
        """Delete every point of a document (wait=true) and return the remaining count (must be 0)."""
        ...

    async def count(self, ctx: TenantContext, doc_id: str | None = None) -> int: ...

    async def ready(self) -> bool: ...


@runtime_checkable
class Reranker(Protocol):
    model_name: str

    async def rerank(self, query: str, passages: Sequence[str]) -> list[float]:
        """Return one score in [0, 1] per passage, in input order."""
        ...


# ---------------------------------------------------------------- guardrails


@runtime_checkable
class InjectionClassifier(Protocol):
    async def score(self, text: str) -> float:
        """Probability in [0, 1] that `text` is a prompt-injection or jailbreak attempt."""
        ...


# ---------------------------------------------------------------- LLMs


@dataclass(frozen=True, slots=True)
class LLMMessage:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True, slots=True)
class LLMResult:
    text: str
    endpoint: Literal["hosted", "ollama"]
    model: str
    seller: str | None = None
    attempts: int = 1
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    degraded: bool = False  # True when the local fallback answered


@runtime_checkable
class LLMClient(Protocol):
    """The LLMRouter: hosted model chain, then Ollama; raises AppError(LLM_UNAVAILABLE) at the end."""

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        purpose: Literal["generate", "helper", "judge"],
        max_tokens: int = 1024,
        temperature: float = 0.0,
        model: str | None = None,
    ) -> LLMResult:
        """`model` (eval override, `generate` only) restricts the hosted chain to that one model."""
        ...

    async def available_models(self) -> list[str]: ...


# ---------------------------------------------------------------- retrieval (rev 5 contract)


@dataclass(frozen=True, slots=True)
class RetrievalQuery:
    """What the ask pipeline hands to the retriever: one or more query variants (original, English,
    Urdu script) and the filters. Variants are searched in parallel and fused with RRF (§4)."""

    variants: tuple[str, ...]
    filters: SearchFilters = SearchFilters()
    candidates_k: int = 20
    context_k: int = 5
    overrides: EvalOverrides | None = None  # eval-only: retrieval_mode / rerank


@dataclass(frozen=True, slots=True)
class Candidate:
    point: ScoredPoint
    rank: int  # 1-based, after fusion + rerank
    rerank_score: float | None = None  # sigmoid of the cross-encoder logit, in [0, 1]
    used_in_context: bool = False


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    candidates: tuple[Candidate, ...]  # everything fetched (shown in the chunks panel)
    context: tuple[Candidate, ...]  # the subset passed to the LLM
    max_rerank: float | None
    latency_ms: dict[str, int] = field(default_factory=dict)


@runtime_checkable
class Retriever(Protocol):
    async def retrieve(self, ctx: TenantContext, query: RetrievalQuery) -> RetrievalResult: ...


# ---------------------------------------------------------------- services used by the routers


@dataclass(frozen=True, slots=True)
class UploadedFile:
    """An upload already spooled to disk by the router (streaming size check done there)."""

    file_name: str
    path: Path
    size_bytes: int
    content_type: str | None = None


if TYPE_CHECKING:  # schemas import tenancy only; this keeps protocols importable from schemas-free code
    from app.schemas import (
        AskRequest,
        AskResponse,
        ChunkOut,
        DocumentMetadata,
        DocumentOut,
        FeedbackOut,
        FeedbackRequest,
        Page,
        SearchRequest,
        SearchResponse,
    )


@runtime_checkable
class DocumentService(Protocol):
    """Implemented by app/ingestion/service.py. `accepted=True` means 202 (background ingestion)."""

    async def upload(
        self,
        ctx: TenantContext,
        file: UploadedFile,
        *,
        visibility: str,
        metadata: DocumentMetadata,
    ) -> tuple[DocumentOut, bool]: ...

    async def list(
        self,
        ctx: TenantContext,
        *,
        agency: str | None,
        jurisdiction: str | None,
        topic: str | None,
        language: str | None,
        status: str | None,
        limit: int,
        cursor: str | None,
    ) -> Page[DocumentOut]: ...

    async def get(self, ctx: TenantContext, doc_id: str) -> DocumentOut: ...

    async def patch(self, ctx: TenantContext, doc_id: str, metadata: DocumentMetadata) -> DocumentOut: ...

    async def chunks(
        self, ctx: TenantContext, doc_id: str, *, limit: int, cursor: str | None
    ) -> Page[ChunkOut]: ...

    async def replace(
        self, ctx: TenantContext, doc_id: str, file: UploadedFile
    ) -> tuple[DocumentOut, bool]: ...

    async def delete(self, ctx: TenantContext, doc_id: str) -> None: ...


@runtime_checkable
class AskService(Protocol):
    """Implemented by app/generation/service.py (LangGraph pipeline)."""

    async def ask(
        self, ctx: TenantContext, request: AskRequest, *, overrides: EvalOverrides | None = None
    ) -> AskResponse: ...

    async def search(
        self, ctx: TenantContext, request: SearchRequest, *, overrides: EvalOverrides | None = None
    ) -> SearchResponse: ...

    async def feedback(self, ctx: TenantContext, request: FeedbackRequest) -> FeedbackOut: ...
