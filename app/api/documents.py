from __future__ import annotations

from datetime import date
from typing import Annotated

from fastapi import APIRouter, File, Form, Query, UploadFile, status

from app.api._stub import not_implemented
from app.api.deps import TenantDep
from app.api.responses import problems
from app.core.tenancy import Scope, Visibility
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
    file: Annotated[UploadFile, File(description="PDF, DOCX, PPTX, HTML, TXT, MD, JPG or PNG")],
    visibility: Annotated[Visibility, Form(description="`public` is admin only")] = Visibility.PRIVATE,
    title: Annotated[str | None, Form()] = None,
    agency: Annotated[str | None, Form()] = None,
    jurisdiction: Annotated[Jurisdiction | None, Form()] = None,
    topic: Annotated[str | None, Form()] = None,
    effective_date: Annotated[date | None, Form()] = None,
) -> DocumentOut:
    not_implemented("A1")


@router.get(
    "",
    response_model=Page[DocumentOut],
    summary="List documents I can read",
    responses=problems(401, 422),
)
async def list_documents(
    tenant: TenantDep,
    scope: Scope = Scope.ALL,
    agency: str | None = None,
    jurisdiction: Jurisdiction | None = None,
    topic: str | None = None,
    language: LanguageCode | None = None,
    doc_status: Annotated[DocumentStatus | None, Query(alias="status")] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    cursor: str | None = None,
) -> Page[DocumentOut]:
    not_implemented("A1")


@router.get(
    "/{document_id}",
    response_model=DocumentOut,
    summary="Document detail and ingestion progress",
    responses=problems(401, 404),
)
async def get_document(tenant: TenantDep, document_id: str) -> DocumentOut:
    not_implemented("A1")


@router.patch(
    "/{document_id}",
    response_model=DocumentOut,
    summary="Edit metadata (agency, jurisdiction, topic, effective date, title)",
    responses=problems(401, 403, 404, 422),
)
async def patch_document(tenant: TenantDep, document_id: str, body: DocumentMetadata) -> DocumentOut:
    not_implemented("A1")


@router.get(
    "/{document_id}/chunks",
    response_model=Page[ChunkOut],
    summary="Inspect stored chunks",
    responses=problems(401, 404),
)
async def list_chunks(
    tenant: TenantDep,
    document_id: str,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    cursor: str | None = None,
) -> Page[ChunkOut]:
    not_implemented("A1")


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
    tenant: TenantDep, document_id: str, file: Annotated[UploadFile, File()]
) -> DocumentOut:
    not_implemented("A1")


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document and all its vectors (§5.5)",
    responses=problems(401, 403, 404, 503),
)
async def delete_document(tenant: TenantDep, document_id: str) -> None:
    not_implemented("A1")
