"""`IngestionService`: the `DocumentService` implementation (blueprint §5).

Upload: validate → hash → dedupe (file, public corpus, content) → extract → clean → chunk →
PENDING row → PROCESSING → embed in batches → upsert deterministic points → verify the point count
→ READY (or FAILED, with partial points deleted). Ingestion is synchronous for the MVP (`accepted`
is always False, so the router answers 201).

Replace (`PUT`): the §5.4 diff upsert. Point ids are `uuid5(doc_id:chunk_hash:occurrence)`, so
unchanged chunks keep their ids: only `add` is embedded, `keep` gets `set_payload`, `remove` is
deleted. Delete: DELETING first, then the vectors (remaining must be 0, else 503 and the row stays
DELETING), then the row. One asyncio lock per document serialises writes to it.

Chunk listings are served from a small JSON manifest stored next to the original file, because the
`VectorStore` contract has no payload scroll.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import re
import shutil
from collections import Counter
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path, PurePath
from typing import Any

import structlog
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.core.errors import AppError, ErrorCode
from app.core.ids import new_document_id, point_id, sha256_file, sha256_text
from app.core.tenancy import TenantContext, Visibility
from app.db.models import Document
from app.db.session import Database
from app.ingestion.chunking import (
    Chunk,
    HFTokenCounter,
    TokenCounter,
    chunk_blocks,
    context_header,
    embedding_text,
)
from app.ingestion.cleaning import clean_blocks, join_blocks
from app.ingestion.extractors import default_extractors, extractor_for
from app.ingestion.guards import INJECTION_FLAG_THRESHOLD, heuristic_injection_score
from app.ingestion.language import majority_language, normalize_for_index
from app.ingestion.metadata import find_effective_date, infer_agency, infer_jurisdiction, infer_topic
from app.ingestion.ocr import NoOcr, PageOcr, TesseractOcr
from app.ingestion.validation import validate_file
from app.protocols import (
    ChunkPayload,
    ChunkPoint,
    Embedder,
    ExtractedDocument,
    Extractor,
    InjectionClassifier,
    UploadedFile,
    VectorStore,
)
from app.schemas import (
    ChunkOut,
    DocumentMetadata,
    DocumentOut,
    DocumentStatus,
    FileType,
    IngestProgress,
    IngestReport,
    Jurisdiction,
    LanguageCode,
    Page,
)

log = structlog.get_logger(__name__)

MIN_EXTRACTED_CHARS = 50  # GD-3: less than this after extraction (and OCR) → no_extractable_text
SET_PAYLOAD_CONCURRENCY = 16
_PATCHABLE_PAYLOAD = ("title", "agency", "jurisdiction", "topic", "effective_date")


@dataclass(slots=True)
class _Prepared:
    """Everything derived from one file before anything is written."""

    file_type: str
    file_sha256: str
    text_sha256: str
    extracted: ExtractedDocument
    chunks: list[Chunk]
    full_text: str
    language: str | None
    is_ocr: bool


@dataclass(slots=True)
class _Meta:
    title: str | None
    agency: str | None
    jurisdiction: str | None
    topic: str | None
    effective_date: date | None
    source_url: str | None
    doc_type: str | None


@dataclass(slots=True)
class _PlannedPoint:
    point_id: str
    embed_text: str
    payload: ChunkPayload


@dataclass(slots=True)
class _IndexReport:
    added: int = 0
    kept: int = 0
    removed: int = 0
    embedded: int = 0
    written: list[str] = field(default_factory=list)


def _encode_cursor(offset: int) -> str:
    return base64.urlsafe_b64encode(f"o:{offset}".encode()).decode().rstrip("=")


def _decode_cursor(cursor: str | None) -> int:
    if not cursor:
        return 0
    try:
        raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)).decode()
        prefix, _, value = raw.partition(":")
        offset = int(value)
        if prefix != "o" or offset < 0:
            raise ValueError(cursor)
        return offset
    except (ValueError, binascii.Error, UnicodeDecodeError):
        raise AppError(ErrorCode.VALIDATION_FAILED, "Invalid cursor.") from None


def _title_from_file_name(name: str) -> str:
    """'ID-05__nadra-registration-policy-6-0-2-en.pdf' → 'nadra registration policy 6 0 2 en'."""
    stem = re.sub(r"^[A-Z]{2}-\d{2}[a-z]?__", "", PurePath(name).stem)
    return " ".join(re.sub(r"[_\-]+", " ", stem).split()) or PurePath(name).stem


def _safe_file_name(name: str) -> str:
    base = PurePath(name.replace("\\", "/")).name.strip() or "upload"
    return base[:255]


class IngestionService:
    """Implements `app.protocols.DocumentService`."""

    def __init__(
        self,
        settings: Settings,
        db: Database,
        store: VectorStore,
        embedder: Embedder,
        *,
        token_counter: TokenCounter | None = None,
        ocr: PageOcr | None = None,
        extractors: dict[str, Extractor] | None = None,
        injection_classifier: InjectionClassifier | None = None,
    ) -> None:
        self.settings = settings
        self.db = db
        self.store = store
        self.embedder = embedder
        self.counter: TokenCounter = token_counter or HFTokenCounter(settings.embedding_model)
        if ocr is None:
            ocr = TesseractOcr(settings.ocr_langs) if settings.ocr_enabled else NoOcr()
        self.extractors = extractors or default_extractors(
            ocr=ocr,
            min_script_ratio=settings.text_quality_min_script_ratio,
            min_dict_hit=settings.text_quality_min_dict_hit,
        )
        self.injection_classifier = injection_classifier
        self._locks: dict[str, asyncio.Lock] = {}
        self._jobs: Counter[str] = Counter()

    # ------------------------------------------------------------ concurrency helpers

    def _lock(self, doc_id: str) -> asyncio.Lock:
        return self._locks.setdefault(doc_id, asyncio.Lock())

    @asynccontextmanager
    async def _job_slot(self, ctx: TenantContext, owner: str) -> AsyncIterator[None]:
        limited = owner != ctx.public_owner_id
        if limited and self._jobs[owner] >= self.settings.max_ingest_jobs_per_user:
            raise AppError(
                ErrorCode.TOO_MANY_INGEST_JOBS,
                "Too many uploads are being processed for this account. Try again shortly.",
                extra={"limit": self.settings.max_ingest_jobs_per_user},
                headers={"Retry-After": "5"},
            )
        self._jobs[owner] += 1
        try:
            yield
        finally:
            self._jobs[owner] -= 1

    # ------------------------------------------------------------ DB helpers

    async def _load(self, session: AsyncSession, ctx: TenantContext, doc_id: str) -> Document:
        row = (await session.scalars(select(Document).where(Document.id == doc_id))).first()
        if row is None:
            raise AppError(ErrorCode.NOT_FOUND, "Document not found.")
        ctx.ensure_readable(row.user_id)
        return row

    async def _update(self, ctx: TenantContext, doc_id: str, **values: Any) -> Document:
        async with self.db.session(ctx) as s:
            row = await self._load(s, ctx, doc_id)
            for key, value in values.items():
                setattr(row, key, value)
        return row

    async def _check_quota(self, ctx: TenantContext, owner: str, size_bytes: int) -> None:
        if owner == ctx.public_owner_id:
            return  # the admin public corpus is exempt from per-user quotas (§5.8.6)
        async with self.db.session(ctx) as s:
            docs, stored = (
                await s.execute(
                    select(func.count(Document.id), func.coalesce(func.sum(Document.size_bytes), 0)).where(
                        Document.user_id == owner
                    )
                )
            ).one()
        if docs >= self.settings.max_docs_per_user:
            raise AppError(
                ErrorCode.QUOTA_EXCEEDED,
                "Document quota reached.",
                extra={"used": docs, "limit": self.settings.max_docs_per_user},
            )
        limit_bytes = self.settings.max_storage_mb_per_user * 1024 * 1024
        if stored + size_bytes > limit_bytes:
            raise AppError(
                ErrorCode.QUOTA_EXCEEDED,
                "Storage quota reached.",
                extra={"used": round(stored / 1048576, 2), "limit": self.settings.max_storage_mb_per_user},
            )

    async def _check_chunk_quota(
        self, ctx: TenantContext, owner: str, new_chunks: int, exclude: str | None
    ) -> None:
        if owner == ctx.public_owner_id:
            return
        async with self.db.session(ctx) as s:
            stmt = select(func.coalesce(func.sum(Document.chunk_count), 0)).where(Document.user_id == owner)
            if exclude:
                stmt = stmt.where(Document.id != exclude)
            used = int((await s.execute(stmt)).scalar_one() or 0)
        if used + new_chunks > self.settings.max_chunks_per_user:
            raise AppError(
                ErrorCode.QUOTA_EXCEEDED,
                "Chunk quota reached.",
                extra={"used": used, "limit": self.settings.max_chunks_per_user},
            )

    async def _check_file_duplicate(
        self, ctx: TenantContext, owner: str, file_sha: str, exclude: str | None = None
    ) -> None:
        async with self.db.session(ctx) as s:
            stmt = select(Document).where(Document.user_id == owner, Document.file_sha256 == file_sha)
            if exclude:
                stmt = stmt.where(Document.id != exclude)
            existing = (await s.scalars(stmt)).first()
            failed_id = (
                existing.id if existing is not None and existing.status == DocumentStatus.FAILED else None
            )
            if existing is not None and failed_id is None:
                raise AppError(
                    ErrorCode.DUPLICATE_DOCUMENT,
                    f"This file was already uploaded as '{existing.file_name}'.",
                    extra={"existing_document_id": existing.id},
                )
            if owner != ctx.public_owner_id:
                public = (
                    await s.scalars(
                        select(Document).where(
                            Document.user_id == ctx.public_owner_id, Document.file_sha256 == file_sha
                        )
                    )
                ).first()
                if public is not None:
                    raise AppError(
                        ErrorCode.DUPLICATE_PUBLIC_DOCUMENT,
                        "This file is already part of the official corpus.",
                        extra={"existing_document_id": public.id},
                    )
        if failed_id is not None:  # a failed earlier attempt must not block a retry
            await self._purge(ctx, failed_id)

    async def _check_content_duplicate(
        self, ctx: TenantContext, owner: str, text_sha: str, exclude: str | None = None
    ) -> None:
        async with self.db.session(ctx) as s:
            stmt = select(Document).where(
                Document.user_id == owner,
                Document.text_sha256 == text_sha,
                Document.status != DocumentStatus.FAILED.value,
            )
            if exclude:
                stmt = stmt.where(Document.id != exclude)
            existing = (await s.scalars(stmt)).first()
            if existing is not None:
                raise AppError(
                    ErrorCode.DUPLICATE_CONTENT,
                    f"A document with the same text already exists ('{existing.file_name}').",
                    extra={"existing_document_id": existing.id},
                )
            if owner != ctx.public_owner_id:
                public = (
                    await s.scalars(
                        select(Document).where(
                            Document.user_id == ctx.public_owner_id,
                            Document.text_sha256 == text_sha,
                            Document.status != DocumentStatus.FAILED.value,
                        )
                    )
                ).first()
                if public is not None:
                    raise AppError(
                        ErrorCode.DUPLICATE_PUBLIC_DOCUMENT,
                        "The same text is already part of the official corpus.",
                        extra={"existing_document_id": public.id},
                    )

    async def _purge(self, ctx: TenantContext, doc_id: str) -> None:
        """Remove a FAILED document completely (vectors, files, row)."""
        async with self._lock(doc_id):
            await self.store.delete_document(ctx, doc_id)
            async with self.db.session(ctx) as s:
                row = await self._load(s, ctx, doc_id)
                owner = row.user_id
                await s.delete(row)
            await asyncio.to_thread(shutil.rmtree, self._doc_dir(owner, doc_id), True)
        self._locks.pop(doc_id, None)

    # ------------------------------------------------------------ storage

    def _doc_dir(self, owner: str, doc_id: str) -> Path:
        return self.settings.storage_dir / "documents" / owner / doc_id

    def _store_files(
        self, owner: str, doc_id: str, source: Path, file_type: str, chunks: list[dict[str, Any]]
    ) -> str:
        directory = self._doc_dir(owner, doc_id)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"original.{file_type}"
        for old in directory.glob("original.*"):
            if old != target:
                old.unlink(missing_ok=True)
        if source.resolve() != target.resolve():
            shutil.copyfile(source, target)
        (directory / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
        return str(target)

    def _read_chunks(self, owner: str, doc_id: str) -> list[dict[str, Any]]:
        path = self._doc_dir(owner, doc_id) / "chunks.json"
        if not path.exists():
            return []
        data: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
        return data

    # ------------------------------------------------------------ pipeline

    def _prepare_sync(self, path: Path, file_type: str, file_sha: str) -> _Prepared:
        extracted = extractor_for(self.extractors, file_type).extract(path)
        blocks = clean_blocks(extracted.blocks)
        full_text = join_blocks(blocks)
        if len("".join(full_text.split())) < MIN_EXTRACTED_CHARS:
            detail = "No readable text was found in the file."
            if extracted.metadata.get("unusable_pages"):
                detail += " Its text layer is unreadable and OCR is not available."
            raise AppError(ErrorCode.NO_EXTRACTABLE_TEXT, detail, extra={"pages": extracted.page_count})
        chunks = chunk_blocks(
            blocks,
            counter=self.counter,
            chunk_size=self.settings.chunk_size_tokens,
            overlap=self.settings.chunk_overlap_tokens,
        )
        if not chunks:
            raise AppError(ErrorCode.NO_EXTRACTABLE_TEXT, "No readable text was found in the file.")
        return _Prepared(
            file_type=file_type,
            file_sha256=file_sha,
            text_sha256=sha256_text(normalize_for_index(full_text)),
            extracted=extracted,
            chunks=chunks,
            full_text=full_text,
            language=majority_language([c.language for c in chunks]),
            is_ocr=extracted.ocr_pages > 0 or any(c.is_ocr for c in chunks),
        )

    async def _validate(self, file: UploadedFile) -> tuple[str, str]:
        """(file type, file SHA-256) after every pre-parse guardrail passed."""
        file_type = await asyncio.to_thread(validate_file, file.path, file.file_name, self.settings)
        return file_type, await asyncio.to_thread(sha256_file, file.path)

    async def _extract(self, file: UploadedFile, file_type: str, file_sha: str) -> _Prepared:
        return await asyncio.to_thread(self._prepare_sync, file.path, file_type, file_sha)

    def _infer_meta(
        self,
        requested: DocumentMetadata,
        prepared: _Prepared,
        file_name: str,
        source_url: str | None,
        doc_type: str | None,
    ) -> _Meta:
        head = prepared.full_text[:20_000]
        agency = requested.agency or infer_agency(file_name, head)
        jurisdiction = (
            requested.jurisdiction.value if requested.jurisdiction else infer_jurisdiction(agency, head)
        )
        extracted_date = prepared.extracted.metadata.get("effective_date")
        effective = (
            requested.effective_date
            or (extracted_date if isinstance(extracted_date, date) else None)
            or find_effective_date(head)
        )
        title = requested.title or prepared.extracted.title or _title_from_file_name(file_name)
        return _Meta(
            title=title[:300] if title else None,
            agency=agency,
            jurisdiction=jurisdiction if jurisdiction in {j.value for j in Jurisdiction} else None,
            topic=requested.topic or infer_topic(file_name, head),
            effective_date=effective,
            source_url=source_url or prepared.extracted.metadata.get("source_url"),
            doc_type=doc_type,
        )

    async def _score_injection(self, texts: Sequence[str]) -> list[float]:
        if self.injection_classifier is None:
            return [heuristic_injection_score(t) for t in texts]
        classifier = self.injection_classifier
        scores = await asyncio.gather(*(classifier.score(t) for t in texts))
        return [max(float(s), heuristic_injection_score(t)) for s, t in zip(scores, texts, strict=True)]

    async def _plan_points(
        self,
        doc_id: str,
        version: int,
        file_name: str,
        file_type: str,
        prepared: _Prepared,
        meta: _Meta,
    ) -> list[_PlannedPoint]:
        chunks = prepared.chunks
        scores = await self._score_injection([c.text for c in chunks])
        occurrences: Counter[str] = Counter()
        planned: list[_PlannedPoint] = []
        for chunk, score in zip(chunks, scores, strict=True):
            chunk_hash = sha256_text(normalize_for_index(chunk.text))
            occurrence = occurrences[chunk_hash]
            occurrences[chunk_hash] += 1
            payload = ChunkPayload(
                doc_id=doc_id,
                doc_version=version,
                file_name=file_name,
                file_type=file_type,
                chunk_hash=chunk_hash,
                chunk_index=chunk.index,
                chunk_count=len(chunks),
                text=chunk.text,
                token_count=chunk.token_count,
                language=chunk.language,
                script=chunk.script,
                agency=meta.agency,
                jurisdiction=meta.jurisdiction,
                topic=meta.topic,
                effective_date=meta.effective_date,
                doc_type=meta.doc_type,
                source_url=meta.source_url,
                title=meta.title,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                section_path=chunk.section_path,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
                is_table=chunk.is_table,
                is_ocr=chunk.is_ocr,
                injection_score=round(score, 4),
                injection_flag=score >= INJECTION_FLAG_THRESHOLD,
            )
            header = context_header(meta.agency, meta.jurisdiction, meta.title, chunk.section_path)
            planned.append(
                _PlannedPoint(
                    point_id(doc_id, chunk_hash, occurrence), embedding_text(header, chunk.text), payload
                )
            )
        return planned

    async def _embed_and_upsert(
        self, ctx: TenantContext, owner: str, points: Sequence[_PlannedPoint], report: _IndexReport
    ) -> None:
        batch = max(1, self.settings.embed_batch_size)
        for start in range(0, len(points), batch):
            group = points[start : start + batch]
            embeddings = await self.embedder.embed_documents([p.embed_text for p in group])
            if len(embeddings) != len(group):
                raise RuntimeError("embedder returned a different number of vectors")
            await self.store.upsert(
                ctx,
                owner,
                [ChunkPoint(p.point_id, e, p.payload) for p, e in zip(group, embeddings, strict=True)],
            )
            report.written.extend(p.point_id for p in group)
            report.embedded += len(group)

    async def _verify(self, ctx: TenantContext, doc_id: str, expected: int) -> None:
        actual = await self.store.count(ctx, doc_id)
        if actual != expected:
            raise AppError(
                ErrorCode.DEPENDENCY_UNAVAILABLE,
                "The vector store did not confirm every chunk.",
                extra={"expected": expected, "actual": actual},
            )

    @staticmethod
    def _chunk_manifest(points: Sequence[_PlannedPoint]) -> list[dict[str, Any]]:
        return [
            {
                "chunk_id": p.point_id,
                "chunk_hash": p.payload.chunk_hash,
                "chunk_index": p.payload.chunk_index,
                "page_start": p.payload.page_start,
                "page_end": p.payload.page_end,
                "section_path": list(p.payload.section_path),
                "text": p.payload.text,
                "language": p.payload.language,
                "is_table": p.payload.is_table,
                "is_ocr": p.payload.is_ocr,
                "injection_flag": p.payload.injection_flag,
                "token_count": p.payload.token_count,
            }
            for p in points
        ]

    @staticmethod
    def _report_dict(prepared: _Prepared, report: _IndexReport) -> dict[str, Any]:
        return IngestReport(
            chunks_added=report.added,
            chunks_kept=report.kept,
            chunks_removed=report.removed,
            chunks_embedded=report.embedded,
            hidden_text_removed=prepared.extracted.hidden_text_removed,
            ocr_pages=prepared.extracted.ocr_pages,
        ).model_dump()

    # ------------------------------------------------------------ DocumentService

    async def upload(
        self,
        ctx: TenantContext,
        file: UploadedFile,
        *,
        visibility: str,
        metadata: DocumentMetadata,
        source_url: str | None = None,
        doc_type: str | None = None,
    ) -> tuple[DocumentOut, bool]:
        owner = ctx.write_owner_for(Visibility(visibility))
        file_name = _safe_file_name(file.file_name)
        await self._check_quota(ctx, owner, file.size_bytes)
        async with self._job_slot(ctx, owner):
            file_type, file_sha = await self._validate(file)
            await self._check_file_duplicate(ctx, owner, file_sha)
            prepared = await self._extract(file, file_type, file_sha)
            await self._check_content_duplicate(ctx, owner, prepared.text_sha256)
            await self._check_chunk_quota(ctx, owner, len(prepared.chunks), None)
            meta = self._infer_meta(metadata, prepared, file_name, source_url, doc_type)

            doc_id = new_document_id()
            row = Document(
                id=doc_id,
                user_id=owner,
                visibility=Visibility.PUBLIC.value
                if owner == ctx.public_owner_id
                else Visibility.PRIVATE.value,
                file_name=file_name,
                file_type=prepared.file_type,
                size_bytes=file.size_bytes,
                file_sha256=prepared.file_sha256,
                text_sha256=prepared.text_sha256,
                status=DocumentStatus.PENDING.value,
                version=1,
                title=meta.title,
                agency=meta.agency,
                jurisdiction=meta.jurisdiction,
                topic=meta.topic,
                effective_date=meta.effective_date,
                doc_type=meta.doc_type,
                source_url=meta.source_url,
                language=prepared.language,
                pages=prepared.extracted.page_count,
                is_ocr=prepared.is_ocr,
                pages_total=prepared.extracted.page_count,
            )
            try:
                async with self.db.session(ctx) as s:
                    s.add(row)
            except IntegrityError:  # a concurrent identical upload won the unique constraint (E16)
                await self._check_file_duplicate(ctx, owner, prepared.file_sha256)
                raise AppError(ErrorCode.DUPLICATE_DOCUMENT, "This file was already uploaded.") from None

            async with self._lock(doc_id):
                await self._update(ctx, doc_id, status=DocumentStatus.PROCESSING.value)
                report = _IndexReport()
                try:
                    planned = await self._plan_points(
                        doc_id, 1, file_name, prepared.file_type, prepared, meta
                    )
                    await self._embed_and_upsert(ctx, owner, planned, report)
                    await self._verify(ctx, doc_id, len(planned))
                    report.added = len(planned)
                    storage_path = await asyncio.to_thread(
                        self._store_files,
                        owner,
                        doc_id,
                        file.path,
                        prepared.file_type,
                        self._chunk_manifest(planned),
                    )
                    row = await self._update(
                        ctx,
                        doc_id,
                        status=DocumentStatus.READY.value,
                        chunk_count=len(planned),
                        storage_path=storage_path,
                        pages_done=prepared.extracted.page_count,
                        last_ingest=self._report_dict(prepared, report),
                        error=None,
                    )
                except Exception as exc:
                    await self._fail(ctx, doc_id, exc)
                    if isinstance(exc, AppError):
                        raise
                    raise AppError(
                        ErrorCode.DEPENDENCY_UNAVAILABLE, "Indexing failed; the document was not stored."
                    ) from exc
        log.info(
            "ingest.ready",
            doc_id=doc_id,
            file_type=prepared.file_type,
            chunks=report.added,
            ocr_pages=prepared.extracted.ocr_pages,
        )
        return self._out(row), False

    async def _fail(self, ctx: TenantContext, doc_id: str, exc: BaseException) -> None:
        """PROCESSING → FAILED: delete partial points (compensation), keep the row with the error."""
        log.warning("ingest.failed", doc_id=doc_id, error=type(exc).__name__)
        try:
            await self.store.delete_document(ctx, doc_id)
        except Exception as cleanup:
            log.error("ingest.cleanup_failed", doc_id=doc_id, error=type(cleanup).__name__)
        detail = exc.detail if isinstance(exc, AppError) else type(exc).__name__
        await self._update(ctx, doc_id, status=DocumentStatus.FAILED.value, error=str(detail)[:500])

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
    ) -> Page[DocumentOut]:
        offset = _decode_cursor(cursor)
        stmt = select(Document)
        if agency:
            stmt = stmt.where(func.lower(Document.agency) == agency.lower())
        if jurisdiction:
            stmt = stmt.where(Document.jurisdiction == str(jurisdiction))
        if topic:
            stmt = stmt.where(Document.topic == topic)
        if language:
            stmt = stmt.where(Document.language == str(language))
        if status:
            stmt = stmt.where(Document.status == str(status))
        stmt = stmt.order_by(Document.created_at.desc(), Document.id.desc()).offset(offset).limit(limit + 1)
        async with self.db.session(ctx) as s:
            rows = list((await s.scalars(stmt)).all())
        more = len(rows) > limit
        return Page[DocumentOut](
            items=[self._out(r) for r in rows[:limit]],
            next_cursor=_encode_cursor(offset + limit) if more else None,
        )

    async def get(self, ctx: TenantContext, doc_id: str) -> DocumentOut:
        async with self.db.session(ctx) as s:
            row = await self._load(s, ctx, doc_id)
        return self._out(row)

    async def patch(self, ctx: TenantContext, doc_id: str, metadata: DocumentMetadata) -> DocumentOut:
        changes = metadata.model_dump(exclude_unset=True)
        if "jurisdiction" in changes and changes["jurisdiction"] is not None:
            changes["jurisdiction"] = Jurisdiction(changes["jurisdiction"]).value
        async with self._lock(doc_id), self.db.session(ctx) as s:
            row = await self._load(s, ctx, doc_id)
            ctx.ensure_writable(row.user_id)
            if row.status == DocumentStatus.DELETING.value:
                raise AppError(ErrorCode.NOT_FOUND, "Document not found.")
            payload = {k: changes[k] for k in _PATCHABLE_PAYLOAD if k in changes}
            if payload and row.chunk_count:
                try:
                    ids = await self.store.point_ids(ctx, doc_id)
                    if ids:
                        await self.store.set_payload(ctx, sorted(ids), payload)
                except AppError:
                    raise
                except Exception as exc:
                    raise AppError(
                        ErrorCode.DEPENDENCY_UNAVAILABLE, "The vector store is unavailable."
                    ) from exc
            for key, value in changes.items():
                setattr(row, key, value)
        return self._out(row)

    async def chunks(
        self, ctx: TenantContext, doc_id: str, *, limit: int, cursor: str | None
    ) -> Page[ChunkOut]:
        offset = _decode_cursor(cursor)
        async with self.db.session(ctx) as s:
            row = await self._load(s, ctx, doc_id)
        items = await asyncio.to_thread(self._read_chunks, row.user_id, doc_id)
        items.sort(key=lambda c: int(c["chunk_index"]))
        window = items[offset : offset + limit]
        out = [
            ChunkOut(
                chunk_id=c["chunk_id"],
                chunk_index=c["chunk_index"],
                page_start=c.get("page_start"),
                page_end=c.get("page_end"),
                section_path=list(c.get("section_path") or []),
                text=c["text"],
                language=LanguageCode(c["language"]) if c.get("language") else None,
                is_table=bool(c.get("is_table")),
                is_ocr=bool(c.get("is_ocr")),
                injection_flag=bool(c.get("injection_flag")),
                token_count=int(c["token_count"]),
            )
            for c in window
        ]
        more = offset + limit < len(items)
        return Page[ChunkOut](items=out, next_cursor=_encode_cursor(offset + limit) if more else None)

    async def replace(self, ctx: TenantContext, doc_id: str, file: UploadedFile) -> tuple[DocumentOut, bool]:
        file_name = _safe_file_name(file.file_name)
        async with self._lock(doc_id):
            async with self.db.session(ctx) as s:
                row = await self._load(s, ctx, doc_id)
                ctx.ensure_writable(row.user_id)
            if row.status == DocumentStatus.DELETING.value:
                raise AppError(ErrorCode.NOT_FOUND, "Document not found.")
            owner = row.user_id
            async with self._job_slot(ctx, owner):
                file_type, file_sha = await self._validate(file)
                if file_sha == row.file_sha256:
                    raise AppError(
                        ErrorCode.NO_CHANGE,
                        "The uploaded file is identical to the current version.",
                        extra={"document_id": doc_id, "version": row.version},
                    )
                await self._check_file_duplicate(ctx, owner, file_sha, exclude=doc_id)
                prepared = await self._extract(file, file_type, file_sha)
                await self._check_content_duplicate(ctx, owner, prepared.text_sha256, exclude=doc_id)
                await self._check_chunk_quota(ctx, owner, len(prepared.chunks), doc_id)
                previous_status = row.status
                await self._update(ctx, doc_id, status=DocumentStatus.UPDATING.value)
                report = _IndexReport()
                try:
                    inferred = self._infer_meta(
                        DocumentMetadata(), prepared, file_name, row.source_url, row.doc_type
                    )
                    meta = _Meta(
                        title=row.title or inferred.title,
                        agency=row.agency or inferred.agency,
                        jurisdiction=row.jurisdiction or inferred.jurisdiction,
                        topic=row.topic or inferred.topic,
                        effective_date=inferred.effective_date or row.effective_date,
                        source_url=row.source_url or inferred.source_url,
                        doc_type=row.doc_type,
                    )
                    version = row.version + 1
                    planned = await self._plan_points(
                        doc_id, version, file_name, prepared.file_type, prepared, meta
                    )
                    old_ids = await self.store.point_ids(ctx, doc_id)
                    new_ids = {p.point_id for p in planned}
                    add = [p for p in planned if p.point_id not in old_ids]
                    keep = [p for p in planned if p.point_id in old_ids]
                    remove = sorted(old_ids - new_ids)
                    await self._embed_and_upsert(ctx, owner, add, report)
                    await self._set_payloads(ctx, keep)
                    if remove:
                        await self.store.delete_points(ctx, remove)
                    await self._verify(ctx, doc_id, len(planned))
                    report.added, report.kept, report.removed = len(add), len(keep), len(remove)
                    storage_path = await asyncio.to_thread(
                        self._store_files,
                        owner,
                        doc_id,
                        file.path,
                        prepared.file_type,
                        self._chunk_manifest(planned),
                    )
                    row = await self._update(
                        ctx,
                        doc_id,
                        status=DocumentStatus.READY.value,
                        version=version,
                        file_name=file_name,
                        file_type=prepared.file_type,
                        size_bytes=file.size_bytes,
                        file_sha256=prepared.file_sha256,
                        text_sha256=prepared.text_sha256,
                        storage_path=storage_path,
                        title=meta.title,
                        agency=meta.agency,
                        jurisdiction=meta.jurisdiction,
                        topic=meta.topic,
                        effective_date=meta.effective_date,
                        language=prepared.language,
                        pages=prepared.extracted.page_count,
                        pages_total=prepared.extracted.page_count,
                        pages_done=prepared.extracted.page_count,
                        chunk_count=len(planned),
                        is_ocr=prepared.is_ocr,
                        last_ingest=self._report_dict(prepared, report),
                        error=None,
                    )
                except Exception as exc:
                    await self._compensate_update(ctx, doc_id, report, previous_status, exc)
                    if isinstance(exc, AppError):
                        raise
                    raise AppError(
                        ErrorCode.DEPENDENCY_UNAVAILABLE,
                        "The update failed; the previous version is still served.",
                    ) from exc
        log.info(
            "ingest.replaced",
            doc_id=doc_id,
            version=row.version,
            added=report.added,
            kept=report.kept,
            removed=report.removed,
            embedded=report.embedded,
        )
        return self._out(row), False

    async def _set_payloads(self, ctx: TenantContext, points: Sequence[_PlannedPoint]) -> None:
        semaphore = asyncio.Semaphore(SET_PAYLOAD_CONCURRENCY)

        async def one(p: _PlannedPoint) -> None:
            async with semaphore:
                await self.store.set_payload(ctx, [p.point_id], asdict(p.payload))

        await asyncio.gather(*(one(p) for p in points))

    async def _compensate_update(
        self, ctx: TenantContext, doc_id: str, report: _IndexReport, previous_status: str, exc: BaseException
    ) -> None:
        """UPDATING → READY_STALE: delete the points this attempt added; the old version keeps serving."""
        log.warning("ingest.update_failed", doc_id=doc_id, error=type(exc).__name__)
        try:
            if report.written:
                await self.store.delete_points(ctx, report.written)
        except Exception as cleanup:
            log.error("ingest.cleanup_failed", doc_id=doc_id, error=type(cleanup).__name__)
        detail = exc.detail if isinstance(exc, AppError) else type(exc).__name__
        status = (
            DocumentStatus.READY_STALE.value
            if previous_status != DocumentStatus.FAILED.value
            else previous_status
        )
        await self._update(ctx, doc_id, status=status, error=str(detail)[:500])

    async def delete(self, ctx: TenantContext, doc_id: str) -> None:
        async with self._lock(doc_id):
            async with self.db.session(ctx) as s:
                row = await self._load(s, ctx, doc_id)
                ctx.ensure_writable(row.user_id)
                owner = row.user_id
                row.status = DocumentStatus.DELETING.value  # layer 1: excluded from retrieval at once
            try:
                remaining = await self.store.delete_document(ctx, doc_id)
            except Exception as exc:
                raise AppError(
                    ErrorCode.DEPENDENCY_UNAVAILABLE,
                    "The vector store is unavailable; the document is hidden and will be removed later.",
                ) from exc
            if remaining != 0:
                raise AppError(
                    ErrorCode.DEPENDENCY_UNAVAILABLE,
                    "Some vectors could not be deleted yet; the document stays hidden.",
                    extra={"remaining": remaining},
                )
            await asyncio.to_thread(shutil.rmtree, self._doc_dir(owner, doc_id), True)
            async with self.db.session(ctx) as s:
                row = await self._load(s, ctx, doc_id)
                await s.delete(row)
        self._locks.pop(doc_id, None)
        log.info("ingest.deleted", doc_id=doc_id)

    # ------------------------------------------------------------ mapping

    @staticmethod
    def _out(row: Document) -> DocumentOut:
        report = IngestReport(**row.last_ingest) if row.last_ingest else None
        progress = (
            IngestProgress(pages_done=row.pages_done, pages_total=row.pages_total)
            if row.pages_total
            else None
        )
        return DocumentOut(
            id=row.id,
            file_name=row.file_name,
            file_type=FileType(row.file_type),
            size_bytes=row.size_bytes,
            status=DocumentStatus(row.status),
            visibility=Visibility(row.visibility),
            version=row.version,
            title=row.title,
            agency=row.agency,
            jurisdiction=Jurisdiction(row.jurisdiction) if row.jurisdiction else None,
            topic=row.topic,
            effective_date=row.effective_date,
            language=LanguageCode(row.language) if row.language else None,
            pages=row.pages,
            chunk_count=row.chunk_count,
            is_ocr=row.is_ocr,
            source_url=row.source_url,
            progress=progress,
            last_ingest=report,
            error=row.error,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )
