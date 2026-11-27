from __future__ import annotations

import uuid
from datetime import date
from pathlib import Path, PurePath
from typing import Annotated

import anyio
import structlog
from fastapi import APIRouter, File, Form, Query, Response, UploadFile, status

from app.api.deps import DocumentServiceDep, SettingsDep, TenantDep
from app.api.responses import problems
from app.config import Settings
from app.core.errors import AppError, ErrorCode
from app.core.tenancy import Scope, Visibility
from app.protocols import UploadedFile
from app.schemas import (
    ChunkOut,
    DocumentMetadata,
    DocumentOut,
    DocumentStatus,
    Jurisdiction,
    LanguageCode,
    Page,
)

router = APIRouter(prefix="/documents", tags=["documents"])
log = structlog.get_logger(__name__)

_SPOOL_CHUNK = 1024 * 1024


async def _spool(upload: UploadFile, settings: Settings) -> UploadedFile:
    """Copy the upload to a temp file under `storage_dir`, counting bytes as they stream (GD-2)."""
    limit = settings.max_upload_bytes
    declared = upload.size
    if declared is not None and declared > limit:
        raise AppError(
            ErrorCode.FILE_TOO_LARGE,
            f"The file is larger than {settings.max_upload_mb} MB.",
            extra={"limit_bytes": limit},
        )
    tmp_dir = anyio.Path(settings.storage_dir) / "tmp"
    await tmp_dir.mkdir(parents=True, exist_ok=True)
    file_name = PurePath((upload.filename or "upload").replace("\\", "/")).name or "upload"
    target = tmp_dir / f"{uuid.uuid4().hex}{PurePath(file_name).suffix.lower()[:10]}"
    size = 0
    try:
        async with await anyio.open_file(target, "wb") as out:
            while chunk := await upload.read(_SPOOL_CHUNK):
                size += len(chunk)
                if size > limit:
                    raise AppError(
                        ErrorCode.FILE_TOO_LARGE,
                        f"The file is larger than {settings.max_upload_mb} MB.",
                        extra={"limit_bytes": limit},
                    )
                await out.write(chunk)
    except BaseException:
        await target.unlink(missing_ok=True)
        raise
    return UploadedFile(
        file_name=file_name, path=Path(target), size_bytes=size, content_type=upload.content_type
    )


async def _discard(file: UploadedFile) -> None:
    """Best effort: a temp file that cannot be removed yet (locked on Windows) must not fail the request."""
    try:
        await anyio.Path(file.path).unlink(missing_ok=True)
    except OSError:
        log.warning("upload.temp_cleanup_failed")


def _respond(response: Response, settings: Settings, doc: DocumentOut, accepted: bool) -> DocumentOut:
    if accepted:
        response.status_code = status.HTTP_202_ACCEPTED
        response.headers["Location"] = f"{settings.api_prefix}/documents/{doc.id}"
    return doc


@router.post(
    "",
    response_model=DocumentOut,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest one file (201 inline, 202 for large files)",
    responses={
        202: {"model": DocumentOut, "description": "Accepted; poll the Location URL for progress"},
        **problems(400, 401, 403, 409, 413, 415, 422, 429, 503),
    },
)
async def upload_document(
    tenant: TenantDep,
    service: DocumentServiceDep,
    settings: SettingsDep,
    response: Response,
    file: Annotated[UploadFile, File(description="PDF, DOCX, PPTX, HTML, TXT, MD, JPG or PNG")],
    visibility: Annotated[Visibility, Form(description="`public` is admin only")] = Visibility.PRIVATE,
    title: Annotated[str | None, Form()] = None,
    agency: Annotated[str | None, Form()] = None,
    jurisdiction: Annotated[Jurisdiction | None, Form()] = None,
    topic: Annotated[str | None, Form()] = None,
    effective_date: Annotated[date | None, Form()] = None,
) -> DocumentOut:
    tenant.write_owner_for(visibility)  # 403 before reading the body into storage
    fields = {
        "title": title,
        "agency": agency,
        "jurisdiction": jurisdiction,
        "topic": topic,
        "effective_date": effective_date,
    }
    metadata = DocumentMetadata(**{k: v for k, v in fields.items() if v not in (None, "")})
    uploaded = await _spool(file, settings)
    try:
        doc, accepted = await service.upload(tenant, uploaded, visibility=visibility.value, metadata=metadata)
    finally:
        await _discard(uploaded)
    return _respond(response, settings, doc, accepted)


@router.get(
    "",
    response_model=Page[DocumentOut],
    summary="List documents I can read",
    responses=problems(401, 422),
)
async def list_documents(
    tenant: TenantDep,
    service: DocumentServiceDep,
    scope: Scope = Scope.ALL,
    agency: str | None = None,
    jurisdiction: Jurisdiction | None = None,
    topic: str | None = None,
    language: LanguageCode | None = None,
    doc_status: Annotated[DocumentStatus | None, Query(alias="status")] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    cursor: str | None = None,
) -> Page[DocumentOut]:
    return await service.list(
        tenant.with_scope(scope),
        agency=agency,
        jurisdiction=jurisdiction.value if jurisdiction else None,
        topic=topic,
        language=language.value if language else None,
        status=doc_status.value if doc_status else None,
        limit=limit,
        cursor=cursor,
    )


@router.get(
    "/{document_id}",
    response_model=DocumentOut,
    summary="Document detail and ingestion progress",
    responses=problems(401, 404),
)
async def get_document(tenant: TenantDep, service: DocumentServiceDep, document_id: str) -> DocumentOut:
    return await service.get(tenant, document_id)


@router.patch(
    "/{document_id}",
    response_model=DocumentOut,
    summary="Edit metadata (agency, jurisdiction, topic, effective date, title)",
    responses=problems(401, 403, 404, 422),
)
async def patch_document(
    tenant: TenantDep, service: DocumentServiceDep, document_id: str, body: DocumentMetadata
) -> DocumentOut:
    return await service.patch(tenant, document_id, body)


@router.get(
    "/{document_id}/chunks",
    response_model=Page[ChunkOut],
    summary="Inspect stored chunks",
    responses=problems(401, 404),
)
async def list_chunks(
    tenant: TenantDep,
    service: DocumentServiceDep,
    document_id: str,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    cursor: str | None = None,
) -> Page[ChunkOut]:
    return await service.chunks(tenant, document_id, limit=limit, cursor=cursor)


@router.put(
    "/{document_id}",
    response_model=DocumentOut,
    summary="Replace content with a new version (diff upsert, §5.4)",
    responses={
        202: {"model": DocumentOut, "description": "Accepted; large file processed in the background"},
        **problems(400, 401, 403, 404, 409, 413, 415, 503),
    },
)
async def replace_document(
    tenant: TenantDep,
    service: DocumentServiceDep,
    settings: SettingsDep,
    response: Response,
    document_id: str,
    file: Annotated[UploadFile, File()],
) -> DocumentOut:
    uploaded = await _spool(file, settings)
    try:
        doc, accepted = await service.replace(tenant, document_id, uploaded)
    finally:
        await _discard(uploaded)
    return _respond(response, settings, doc, accepted)


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document and all its vectors (§5.5)",
    responses=problems(401, 403, 404, 503),
)
async def delete_document(tenant: TenantDep, service: DocumentServiceDep, document_id: str) -> None:
    await service.delete(tenant, document_id)
