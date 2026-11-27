"""Public Documents must use a Topic from the closed list; private Documents keep free text."""

from __future__ import annotations

import itertools
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from app.config import Settings
from app.core.errors import AppError, ErrorCode
from app.core.tenancy import Role, Scope, TenantContext
from app.db.session import Database
from app.ingestion.chunking import RegexTokenCounter
from app.ingestion.metadata import TOPICS
from app.ingestion.ocr import NoOcr
from app.ingestion.service import IngestionService
from app.protocols import UploadedFile
from app.schemas import DocumentMetadata
from tests.fakes import FakeEmbedder, InMemoryVectorStore

USER = TenantContext.for_user("u_a", Role.USER, Scope.ALL)
ADMIN = TenantContext.for_user("u_admin", Role.ADMIN, Scope.ALL)

BODY = (
    "## Eligibility\n\nEvery citizen who has attained the age of eighteen years may apply for the card.\n\n"
    "## Fees\n\nThe fee for the normal category is paid at the counter and the urgent category costs more.\n"
)
_numbers = itertools.count(1)


@pytest.fixture
async def db() -> AsyncIterator[Database]:
    database = Database("sqlite+aiosqlite:///:memory:")
    await database.create_all()
    yield database
    await database.dispose()


@pytest.fixture
def service(tmp_path: Path, db: Database) -> IngestionService:
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None,
        app_env="test",
        storage_dir=tmp_path / "storage",
        chunk_size_tokens=60,
        chunk_overlap_tokens=8,
        ocr_enabled=False,
    )
    return IngestionService(
        settings, db, InMemoryVectorStore(), FakeEmbedder(), token_counter=RegexTokenCounter(), ocr=NoOcr()
    )


def _file(tmp_path: Path) -> UploadedFile:
    n = next(_numbers)
    path = tmp_path / f"guide{n}.md"
    path.write_text(f"# Guide {n}\n\n{BODY}\nUnique note {n}.\n", encoding="utf-8")
    return UploadedFile(file_name=path.name, path=path, size_bytes=path.stat().st_size)


async def _upload(
    service: IngestionService,
    tmp_path: Path,
    visibility: str,
    topic: str | None,
    ctx: TenantContext = ADMIN,
) -> Any:
    doc, _ = await service.upload(
        ctx, _file(tmp_path), visibility=visibility, metadata=DocumentMetadata(topic=topic)
    )
    return doc


def _assert_rejected(err: AppError) -> None:
    assert err.code is ErrorCode.VALIDATION_FAILED
    assert err.extra["field"] == "topic"
    assert sorted(err.extra["allowed"]) == sorted(TOPICS)


# ---------------------------------------------------------------- upload


async def test_public_upload_normalises_a_known_topic(service: IngestionService, tmp_path: Path) -> None:
    doc = await _upload(service, tmp_path, "public", "Passport")
    assert doc.topic == "passport"


async def test_public_upload_normalises_spaced_topic(service: IngestionService, tmp_path: Path) -> None:
    doc = await _upload(service, tmp_path, "public", " Social Protection ")
    assert doc.topic == "social_protection"


async def test_public_upload_with_unknown_topic_is_rejected_and_creates_no_row(
    service: IngestionService, tmp_path: Path
) -> None:
    with pytest.raises(AppError) as info:
        await _upload(service, tmp_path, "public", "cnic")
    _assert_rejected(info.value)
    listed = await service.list(
        ADMIN.with_scope(Scope.PUBLIC),
        agency=None,
        jurisdiction=None,
        topic=None,
        language=None,
        status=None,
        limit=10,
        cursor=None,
    )
    assert listed.items == []


async def test_private_upload_keeps_free_text_topic(service: IngestionService, tmp_path: Path) -> None:
    doc = await _upload(service, tmp_path, "private", "my tenancy stuff", ctx=USER)
    assert doc.topic == "my tenancy stuff"


# ---------------------------------------------------------------- patch


async def test_patch_public_known_topic_is_stored_normalised(
    service: IngestionService, tmp_path: Path
) -> None:
    doc = await _upload(service, tmp_path, "public", "tax")
    patched = await service.patch(ADMIN, doc.id, DocumentMetadata(topic="Driving-Licence"))
    assert patched.topic == "driving_licence"


async def test_patch_public_unknown_topic_is_rejected_and_topic_unchanged(
    service: IngestionService, tmp_path: Path
) -> None:
    doc = await _upload(service, tmp_path, "public", "tax")
    with pytest.raises(AppError) as info:
        await service.patch(ADMIN, doc.id, DocumentMetadata(topic="cnic"))
    _assert_rejected(info.value)
    assert (await service.get(ADMIN, doc.id)).topic == "tax"


async def test_patch_public_may_clear_the_topic(service: IngestionService, tmp_path: Path) -> None:
    doc = await _upload(service, tmp_path, "public", "tax")
    patched = await service.patch(ADMIN, doc.id, DocumentMetadata(topic=None))
    assert patched.topic is None


async def test_patch_private_free_text_topic_is_stored_as_given(
    service: IngestionService, tmp_path: Path
) -> None:
    doc = await _upload(service, tmp_path, "private", None, ctx=USER)
    patched = await service.patch(USER, doc.id, DocumentMetadata(topic="My Notes"))
    assert patched.topic == "My Notes"
