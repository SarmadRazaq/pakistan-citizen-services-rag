"""SQLAlchemy models: frozen shared contract for ingestion, auth, ask and metrics (blueprint §5, §5.8).

Every owned table carries `user_id` and inherits `OwnedMixin`. That makes the session-level tenant
filter in `app/db/session.py` apply to it automatically.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any, ClassVar

from sqlalchemy import JSON, Boolean, Date, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy import UniqueConstraint as Unique
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    type_annotation_map: ClassVar[dict[Any, Any]] = {dict[str, Any]: JSON}


class OwnedMixin:
    """Marker: rows of this table belong to `user_id` and are tenant-filtered on every ORM query."""

    user_id: Mapped[str] = mapped_column(String(40), index=True)


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)  # "u_<uuid7 hex>"
    email: Mapped[str] = mapped_column(String(254), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    display_name: Mapped[str | None] = mapped_column(String(80))
    role: Mapped[str] = mapped_column(String(10), default="user")  # user | admin (CLI only)
    token_version: Mapped[int] = mapped_column(Integer, default=0)  # bumped → all JWTs die (E42)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Document(OwnedMixin, Base):
    __tablename__ = "documents"
    __table_args__ = (
        Unique("user_id", "file_sha256", name="uq_documents_owner_file"),  # E3 / E16
        Index("ix_documents_owner_text", "user_id", "text_sha256"),  # E4
    )

    id: Mapped[str] = mapped_column(String(40), primary_key=True)  # "d_<uuid7 hex>"
    visibility: Mapped[str] = mapped_column(String(10), default="private")  # public ⇔ user_id == u_public
    file_name: Mapped[str] = mapped_column(String(255))
    file_type: Mapped[str] = mapped_column(String(10))
    size_bytes: Mapped[int] = mapped_column(Integer)
    file_sha256: Mapped[str] = mapped_column(String(64))
    text_sha256: Mapped[str | None] = mapped_column(String(64))
    storage_path: Mapped[str | None] = mapped_column(String(500))
    status: Mapped[str] = mapped_column(String(12), default="PENDING", index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    title: Mapped[str | None] = mapped_column(String(300))
    agency: Mapped[str | None] = mapped_column(String(80))
    jurisdiction: Mapped[str | None] = mapped_column(String(20))
    topic: Mapped[str | None] = mapped_column(String(60))
    effective_date: Mapped[date | None] = mapped_column(Date)
    doc_type: Mapped[str | None] = mapped_column(String(40))
    source_url: Mapped[str | None] = mapped_column(String(1000))
    language: Mapped[str | None] = mapped_column(String(10))
    pages: Mapped[int | None] = mapped_column(Integer)
    chunk_count: Mapped[int | None] = mapped_column(Integer)
    is_ocr: Mapped[bool] = mapped_column(Boolean, default=False)
    pages_done: Mapped[int] = mapped_column(Integer, default=0)
    pages_total: Mapped[int] = mapped_column(Integer, default=0)
    last_ingest: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class RequestLog(OwnedMixin, Base):
    """One row per /ask or /search call; feeds /metrics/summary (B3)."""

    __tablename__ = "request_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    route: Mapped[str] = mapped_column(String(40))
    status: Mapped[int] = mapped_column(Integer)
    latency_ms: Mapped[int] = mapped_column(Integer)
    error_code: Mapped[str | None] = mapped_column(String(40))
    llm_endpoint: Mapped[str | None] = mapped_column(String(20))
    llm_model: Mapped[str | None] = mapped_column(String(120))
    degraded: Mapped[bool] = mapped_column(Boolean, default=False)
    answerable: Mapped[bool | None] = mapped_column(Boolean)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    list_price_usd: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class Feedback(OwnedMixin, Base):
    __tablename__ = "feedback"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    request_id: Mapped[str] = mapped_column(ForeignKey("request_log.request_id"), index=True)
    rating: Mapped[str] = mapped_column(String(4))  # up | down
    comment: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
