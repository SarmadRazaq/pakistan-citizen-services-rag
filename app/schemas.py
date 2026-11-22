"""Pydantic request/response models: the frozen API contract (blueprint §8).

The frontend generates its TypeScript types from the OpenAPI spec built from these models,
so a change here is a contract change: add optional fields freely, never rename or remove.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from app.core.tenancy import Role, Scope, Visibility

# ---------------------------------------------------------------- shared enums


class Jurisdiction(StrEnum):
    FEDERAL = "federal"
    PUNJAB = "punjab"
    SINDH = "sindh"
    KP = "kp"


class DocumentStatus(StrEnum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    READY = "READY"
    FAILED = "FAILED"
    UPDATING = "UPDATING"
    READY_STALE = "READY_STALE"
    DELETING = "DELETING"


class FileType(StrEnum):
    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    HTML = "html"
    TXT = "txt"
    MD = "md"
    JPG = "jpg"
    PNG = "png"


class LanguageCode(StrEnum):
    EN = "en"
    UR = "ur"  # Urdu in Arabic (Nastaliq) script
    ROMAN_UR = "roman_ur"


class AnswerLanguage(StrEnum):
    AUTO = "auto"
    EN = "en"
    UR = "ur"
    ROMAN_UR = "roman_ur"


class RefusalReason(StrEnum):
    NO_DOCUMENTS = "no_documents"
    OFF_TOPIC = "off_topic"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    GUARDRAIL_OUTPUT = "guardrail_output"
    PARTIAL = "partial"
    LLM_UNAVAILABLE_DEGRADED = "llm_unavailable_degraded"


class NoticeCode(StrEnum):
    NOT_ADVICE = "not_advice"  # GO-9
    REFERRAL = "referral"  # GO-9
    DATED_INFORMATION = "dated_information"  # GO-10
    EMERGENCY = "emergency"  # GO-11


class Base(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# ---------------------------------------------------------------- errors


class FieldError(Base):
    loc: list[str]
    msg: str
    type: str


class Problem(Base):
    """RFC 9457 problem details, as returned for every error."""

    type: str = Field(examples=["https://rag.example/errors/duplicate_document"])
    title: str = Field(examples=["Duplicate document"])
    status: int = Field(examples=[409])
    code: str = Field(examples=["duplicate_document"])
    detail: str = Field(examples=["This file was already uploaded as 'Passport-Fee-Schedule.pdf'."])
    request_id: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
    errors: list[FieldError] = Field(default_factory=list)


# ---------------------------------------------------------------- auth


class SignupRequest(Base):
    email: EmailStr
    password: Annotated[str, Field(min_length=10, max_length=128)]
    display_name: Annotated[str | None, Field(max_length=80)] = None
    # Deliberately no `role` field: admin is granted only by scripts/create_admin.py (§5.8.1).


class TokenResponse(Base):
    access_token: str
    token_type: Literal["bearer"] = "bearer"  # noqa: S105 (OAuth2 token type, not a secret)
    expires_in: int = Field(description="Seconds until the token expires.")


class QuotaUsage(Base):
    documents: int
    documents_limit: int
    storage_mb: float
    storage_mb_limit: int
    chunks: int
    chunks_limit: int


class UserOut(Base):
    id: str
    email: EmailStr
    display_name: str | None = None
    role: Role
    created_at: datetime
    quota: QuotaUsage | None = None


# ---------------------------------------------------------------- documents


class DocumentMetadata(Base):
    """Editable metadata (POST form fields and PATCH body)."""

    title: Annotated[str | None, Field(max_length=300)] = None
    agency: Annotated[str | None, Field(max_length=80, examples=["DGIP"])] = None
    jurisdiction: Jurisdiction | None = None
    topic: Annotated[str | None, Field(max_length=60, examples=["passport"])] = None
    effective_date: date | None = None


class IngestReport(Base):
    chunks_added: int = 0
    chunks_kept: int = 0
    chunks_removed: int = 0
    chunks_embedded: int = 0
    hidden_text_removed: int = 0
    ocr_pages: int = 0


class IngestProgress(Base):
    pages_done: int
    pages_total: int


class DocumentOut(Base):
    id: str
    file_name: str
    file_type: FileType
    size_bytes: int
    status: DocumentStatus
    visibility: Visibility
    version: int
    title: str | None = None
    agency: str | None = None
    jurisdiction: Jurisdiction | None = None
    topic: str | None = None
    effective_date: date | None = None
    language: LanguageCode | None = None
    pages: int | None = None
    chunk_count: int | None = None
    is_ocr: bool = False
    source_url: str | None = None
    progress: IngestProgress | None = None
    last_ingest: IngestReport | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime


class Page[T](Base):
    items: list[T]
    next_cursor: str | None = None


class ChunkOut(Base):
    chunk_id: str
    chunk_index: int
    page_start: int | None = None
    page_end: int | None = None
    section_path: list[str] = Field(default_factory=list)
    text: str
    language: LanguageCode | None = None
    is_table: bool = False
    is_ocr: bool = False
    injection_flag: bool = False
    token_count: int


# ---------------------------------------------------------------- ask / search


class AskRequest(Base):
    question: Annotated[str, Field(min_length=1, examples=["Urgent passport banwane ki fee kitni hai?"])]
    scope: Scope = Scope.ALL
    doc_ids: list[str] | None = None
    agencies: list[str] | None = None
    jurisdictions: list[Jurisdiction] | None = None
    top_k: Annotated[int, Field(ge=1, le=10)] = 5
    answer_language: AnswerLanguage = AnswerLanguage.AUTO

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be empty")
        return value


class SearchRequest(Base):
    query: Annotated[str, Field(min_length=1)]
    scope: Scope = Scope.ALL
    doc_ids: list[str] | None = None
    agencies: list[str] | None = None
    jurisdictions: list[Jurisdiction] | None = None
    top_k: Annotated[int, Field(ge=1, le=50)] = 20


class Notice(Base):
    code: NoticeCode
    message: str
    source_url: str | None = None
    contact: str | None = None  # office / helpline from referrals.yaml (never model-generated)


class LanguageInfo(Base):
    question: LanguageCode
    answer: LanguageCode


class Citation(Base):
    marker: Annotated[str, Field(pattern=r"^S\d+$", examples=["S1"])]
    chunk_id: str
    document_id: str
    scope: Visibility
    file_name: str
    agency: str | None = None
    jurisdiction: Jurisdiction | None = None
    effective_date: date | None = None
    page: int | None = None
    section: str | None = None
    source_url: str | None = None
    quote: str


class RetrievedChunk(Base):
    rank: int
    chunk_id: str
    document_id: str
    scope: Visibility
    file_name: str
    agency: str | None = None
    jurisdiction: Jurisdiction | None = None
    effective_date: date | None = None
    page: int | None = None
    chunk_index: int
    language: LanguageCode | None = None
    is_table: bool = False
    is_ocr: bool = False
    text: str
    similarity: float = Field(description="Dense cosine similarity, computed for every candidate.")
    sparse_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    used_in_context: bool = False
    cited: bool = False
    injection_flag: bool = False


class RetrievalInfo(Base):
    query_variants: list[str]
    sub_queries: list[str] = Field(default_factory=list)
    agency_filter: list[str] = Field(default_factory=list)
    jurisdiction_filter: list[Jurisdiction] = Field(default_factory=list)
    advice_request: bool = False
    candidates_fetched: int
    chunks_used: int
    threshold: float
    chunks: list[RetrievedChunk]


class GuardrailInfo(Base):
    input: Literal["passed", "blocked", "sanitized"] = "passed"
    output: Literal["passed", "repaired", "refused"] = "passed"
    flags: list[str] = Field(default_factory=list)


class ProviderInfo(Base):
    endpoint: Literal["antseed", "ollama", "none"]
    model: str | None = None
    seller: str | None = None
    attempts: int = 0


class Usage(Base):
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    list_price_equivalent_usd: float = 0.0


class AskResponse(Base):
    request_id: str
    trace_id: str | None = None
    answer: str
    answerable: bool
    refusal_reason: RefusalReason | None = None
    unanswered_parts: list[str] = Field(default_factory=list)
    degraded: bool = False
    notices: list[Notice] = Field(default_factory=list)
    language: LanguageInfo
    citations: list[Citation] = Field(default_factory=list)
    retrieval: RetrievalInfo
    guardrails: GuardrailInfo = Field(default_factory=GuardrailInfo)
    provider: ProviderInfo
    usage: Usage = Field(default_factory=Usage)
    latency_ms: dict[str, int] = Field(default_factory=dict)


class SearchResponse(Base):
    request_id: str
    retrieval: RetrievalInfo
    latency_ms: dict[str, int] = Field(default_factory=dict)


# ---------------------------------------------------------------- feedback / metrics / health


class FeedbackRequest(Base):
    request_id: str
    rating: Literal["up", "down"]
    comment: Annotated[str | None, Field(max_length=1000)] = None


class FeedbackOut(Base):
    id: str
    request_id: str
    rating: Literal["up", "down"]
    created_at: datetime


class MetricsSummary(Base):
    window: Literal["1h", "24h", "7d"]
    scope: Literal["own", "global"]
    requests: int
    p50_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    avg_input_tokens: float | None = None
    avg_output_tokens: float | None = None
    cost_per_question_usd: float = 0.0
    list_price_per_question_usd: float = 0.0
    error_rate: float = 0.0
    fallback_rate: float = 0.0
    provider_mix: dict[str, int] | None = None  # admin only


class ComponentHealth(Base):
    name: str
    ok: bool
    required: bool = True
    detail: str | None = None
    latency_ms: int | None = None


class HealthResponse(Base):
    status: Literal["ok", "degraded", "unavailable"]
    version: str
    components: list[ComponentHealth] = Field(default_factory=list)
