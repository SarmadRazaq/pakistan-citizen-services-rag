"""Background ingestion for large uploads and replacements (ISSUE-011: E5, E6, E28; blueprint §5.7).

Every test forces the background path with `sync_ingest_max_mb=0` (or a one-page sync limit for the
page-count trigger). Jobs are held with asyncio.Event gates and finished with `await service.drain()`.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from app.config import Settings
from app.core.errors import AppError, ErrorCode
from app.core.tenancy import TenantContext
from app.db.session import Database
from app.ingestion.chunking import RegexTokenCounter
from app.ingestion.ocr import NoOcr
from app.ingestion.service import IngestionService
from app.protocols import Embedding
from app.schemas import DocumentMetadata, DocumentOut, DocumentStatus, Jurisdiction
from tests.fakes import FakeEmbedder, InMemoryVectorStore
from tests.fixtures.files import factories as f
from tests.integration.test_ingestion_service import (
    ADMIN,
    SECTION_TEXT,
    USER_A,
    USER_B,
    CountingEmbedder,
    markdown,
    upload_file,
    write,
)

ServiceFactory = Callable[..., IngestionService]
WAIT = 10  # seconds; only ever reached when a test is about to fail


class StepEmbedder(FakeEmbedder):
    """Every embed call waits for one `release()`; `waiting` is set whenever a call is blocked."""

    def __init__(self) -> None:
        self.sem = asyncio.Semaphore(0)
        self.waiting = asyncio.Event()
        self.calls = 0

    async def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        self.calls += 1
        self.waiting.set()
        await self.sem.acquire()
        return await super().embed_documents(texts)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Background path for every non-empty file (`sync_ingest_max_mb=0`)."""
    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        app_env="test",
        storage_dir=tmp_path / "storage",
        chunk_size_tokens=60,
        chunk_overlap_tokens=8,
        embed_batch_size=2,
        ocr_enabled=False,
        sync_ingest_max_mb=0,
    )


@pytest.fixture
async def db() -> AsyncIterator[Database]:
    database = Database("sqlite+aiosqlite:///:memory:")
    await database.create_all()
    yield database
    await database.dispose()


@pytest.fixture
def store() -> InMemoryVectorStore:
    return InMemoryVectorStore()


@pytest.fixture
def build(settings: Settings, db: Database, store: InMemoryVectorStore) -> ServiceFactory:
    def factory(embedder: Any = None, **overrides: Any) -> IngestionService:
        return IngestionService(
            settings.model_copy(update=overrides),
            db,
            store,
            embedder if embedder is not None else FakeEmbedder(),
            token_counter=RegexTokenCounter(),
            ocr=NoOcr(),
        )

    return factory


def unique_pdf(path: Path, pages: int = 3, tag: str = "Alpha") -> Path:
    """A multi-page PDF whose every paragraph is different, so nothing is stripped as boilerplate."""

    def paragraph(page: int, part: int) -> str:
        rules = " ".join(
            f"Rule {tag}{page}{part}{k} requires the applicant to submit form {page}{part}{k} at the office."
            for k in range(4)
        )
        return f"{tag} page {page} part {part}. {rules}"

    return f.make_pdf(path, [[paragraph(p, 1), paragraph(p, 2)] for p in range(1, pages + 1)])


def long_markdown(tmp_path: Path, name: str = "long.md") -> Path:
    sections = {f"Section {i}": SECTION_TEXT["Fees"] + f" Variant {i}." for i in range(12)}
    return write(tmp_path / name, markdown(sections))


async def _bg_upload(
    service: IngestionService, ctx: TenantContext, path: Path, *, visibility: str = "private", **meta: Any
) -> DocumentOut:
    doc, accepted = await service.upload(
        ctx, upload_file(path), visibility=visibility, metadata=DocumentMetadata(**meta)
    )
    assert accepted is True, "a large upload must be accepted for background ingestion"
    return doc


async def _error(coro: Any) -> AppError:
    with pytest.raises(AppError) as info:
        await asyncio.wait_for(coro, WAIT)
    return info.value


async def _held(embedder: CountingEmbedder) -> None:
    """Wait until a background job is inside the embedder (so it is mid-way and holds its slot)."""
    await asyncio.wait_for(embedder.called.wait(), WAIT)


async def _list(
    service: IngestionService, ctx: TenantContext, status: str | None = None
) -> list[DocumentOut]:
    page = await service.list(
        ctx,
        agency=None,
        jurisdiction=None,
        topic=None,
        language=None,
        status=status,
        limit=50,
        cursor=None,
    )
    return page.items


# ------------------------------------------------------------------ criteria 2, 3, 5, 6


async def test_large_upload_is_accepted_and_ingested_in_background(
    tmp_path: Path, build: ServiceFactory, store: InMemoryVectorStore
) -> None:
    gate = asyncio.Event()
    embedder = CountingEmbedder(gate=gate)
    service = build(embedder)
    try:
        doc = await _bg_upload(service, USER_A, unique_pdf(tmp_path / "big.pdf"))
        assert doc.status in (DocumentStatus.PENDING, DocumentStatus.PROCESSING)
        assert doc.visibility.value == "private"
        await _held(embedder)
        assert (await service.get(USER_A, doc.id)).status is DocumentStatus.PROCESSING
    finally:
        gate.set()
        await service.drain()

    done = await service.get(USER_A, doc.id)
    assert done.status is DocumentStatus.READY and done.error is None
    assert done.chunk_count and done.chunk_count > 0
    assert await store.count(USER_A, doc.id) == done.chunk_count
    chunks = await service.chunks(USER_A, doc.id, limit=100, cursor=None)
    assert len(chunks.items) == done.chunk_count
    assert done.progress is not None and done.progress.pages_total == 3
    assert done.progress.pages_done == done.progress.pages_total


async def test_page_count_triggers_background_even_for_a_small_file(
    tmp_path: Path, build: ServiceFactory
) -> None:
    service = build(sync_ingest_max_mb=10, sync_ingest_max_pages=1)
    try:
        small = await service.upload(
            USER_A,
            upload_file(unique_pdf(tmp_path / "one.pdf", pages=1, tag="Beta")),
            visibility="private",
            metadata=DocumentMetadata(),
        )
        assert small[1] is False and small[0].status is DocumentStatus.READY  # at the limit: inline

        many = await _bg_upload(service, USER_A, unique_pdf(tmp_path / "many.pdf", pages=3, tag="Gamma"))
        assert many.status in (DocumentStatus.PENDING, DocumentStatus.PROCESSING)
    finally:
        await service.drain()
    assert (await service.get(USER_A, many.id)).status is DocumentStatus.READY


async def test_background_progress_reaches_total(tmp_path: Path, build: ServiceFactory) -> None:
    embedder = StepEmbedder()
    service = build(embedder)
    seen: list[tuple[DocumentStatus, int, int]] = []

    async def poll(doc_id: str) -> None:
        got = await service.get(USER_A, doc_id)
        assert got.progress is not None
        seen.append((got.status, got.progress.pages_done, got.progress.pages_total))

    try:
        doc = await _bg_upload(service, USER_A, unique_pdf(tmp_path / "big.pdf", pages=4))
        await asyncio.wait_for(embedder.waiting.wait(), WAIT)  # blocked on the first batch
        await poll(doc.id)
        assert seen[0][0] is DocumentStatus.PROCESSING and seen[0][2] == 4
        for _ in range(2):  # let two batches through, polling between them
            embedder.waiting.clear()
            embedder.sem.release()
            await asyncio.wait_for(embedder.waiting.wait(), WAIT)
            await poll(doc.id)
    finally:
        for _ in range(10_000):  # Semaphore.release() takes no count: open the gate for every batch
            embedder.sem.release()
        await service.drain()
    await poll(doc.id)

    assert embedder.calls >= 3
    for status, done, total in seen:
        assert status is DocumentStatus.PROCESSING or status is DocumentStatus.READY
        assert total == 4 and 0 <= done <= total
    dones = [done for _, done, _ in seen]
    assert dones == sorted(dones)  # never goes backwards
    assert seen[-1] == (DocumentStatus.READY, 4, 4)


async def test_background_job_survives_temp_file_removal(
    tmp_path: Path, build: ServiceFactory, settings: Settings
) -> None:
    gate = asyncio.Event()
    embedder = CountingEmbedder(gate=gate)
    service = build(embedder)
    path = unique_pdf(tmp_path / "spooled.pdf")
    try:
        doc = await _bg_upload(service, USER_A, path)
        path.unlink()  # the router deletes the spooled upload as soon as `upload` returns
        await _held(embedder)
    finally:
        gate.set()
        await service.drain()

    done = await service.get(USER_A, doc.id)
    assert done.status is DocumentStatus.READY, done.error
    assert done.chunk_count and done.chunk_count > 0
    stored = list((settings.storage_dir / "documents" / "u_a" / doc.id).iterdir())
    assert stored  # the original was kept even though its spooled copy was removed


async def test_background_request_metadata_wins_over_inference(tmp_path: Path, build: ServiceFactory) -> None:
    service = build()
    try:
        doc = await _bg_upload(
            service,
            USER_A,
            write(tmp_path / "note.md", markdown(SECTION_TEXT)),
            title="My note",
            agency="DGIP",
            jurisdiction="sindh",
            topic="passport",
            effective_date="2025-01-02",
        )
    finally:
        await service.drain()
    done = await service.get(USER_A, doc.id)
    assert done.status is DocumentStatus.READY
    assert (done.title, done.agency, done.jurisdiction, str(done.effective_date)) == (
        "My note",
        "DGIP",
        Jurisdiction.SINDH,
        "2025-01-02",
    )
    assert done.topic == "passport"


async def test_background_workers_cap_concurrency(tmp_path: Path, build: ServiceFactory) -> None:
    gate = asyncio.Event()
    embedder = CountingEmbedder(gate=gate)
    service = build(embedder, background_ingest_workers=1)
    try:
        first = await _bg_upload(service, USER_A, long_markdown(tmp_path, "one.md"))
        await _held(embedder)
        second = await _bg_upload(
            service, USER_B, write(tmp_path / "two.md", markdown(SECTION_TEXT, "Other guide"))
        )
        for _ in range(20):
            await asyncio.sleep(0)
        assert (await service.get(USER_A, first.id)).status is DocumentStatus.PROCESSING
        assert (await service.get(USER_B, second.id)).status is DocumentStatus.PENDING  # waits for a worker
    finally:
        gate.set()
        await service.drain()
    assert (await service.get(USER_A, first.id)).status is DocumentStatus.READY
    assert (await service.get(USER_B, second.id)).status is DocumentStatus.READY


# ------------------------------------------------------------------ criterion 4


async def test_inline_checks_still_fail_before_202(
    tmp_path: Path, build: ServiceFactory, store: InMemoryVectorStore
) -> None:
    service = build(max_docs_per_user=1)
    try:
        spoofed = tmp_path / "virus.pdf"
        spoofed.write_bytes(b"MZ\x90\x00" + b"\x00" * 300)
        err = await _error(
            service.upload(USER_B, upload_file(spoofed), visibility="private", metadata=DocumentMetadata())
        )
        assert err.code is ErrorCode.UNSUPPORTED_MEDIA_TYPE

        photo = tmp_path / "photo.jpg"
        photo.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
        err = await _error(
            service.upload(USER_B, upload_file(photo), visibility="private", metadata=DocumentMetadata())
        )
        assert err.code is ErrorCode.UNSUPPORTED_MEDIA_TYPE

        empty = tmp_path / "empty.txt"
        empty.write_bytes(b"")
        err = await _error(
            service.upload(USER_B, upload_file(empty), visibility="private", metadata=DocumentMetadata())
        )
        assert err.code is ErrorCode.EMPTY_FILE

        err = await _error(
            service.upload(
                USER_B,
                upload_file(write(tmp_path / "p.md", markdown(SECTION_TEXT, "Official"))),
                visibility="public",
                metadata=DocumentMetadata(),
            )
        )
        assert err.code is ErrorCode.ADMIN_REQUIRED
        assert await _list(service, USER_B) == []  # none of the above left a row

        # file duplicate, duplicate of the public corpus and the document quota
        path = write(tmp_path / "a.md", markdown(SECTION_TEXT))
        first = await _bg_upload(service, USER_A, path)
        err = await _error(
            service.upload(USER_A, upload_file(path), visibility="private", metadata=DocumentMetadata())
        )
        assert err.code is ErrorCode.DUPLICATE_DOCUMENT and err.extra["existing_document_id"] == first.id

        official = write(tmp_path / "official.md", markdown(SECTION_TEXT, "Official guide"))
        public = await _bg_upload(service, ADMIN, official, visibility="public")
        await service.drain()
        err = await _error(
            service.upload(USER_B, upload_file(official), visibility="private", metadata=DocumentMetadata())
        )
        assert (
            err.code is ErrorCode.DUPLICATE_PUBLIC_DOCUMENT and err.extra["existing_document_id"] == public.id
        )

        other = write(tmp_path / "b.md", markdown(SECTION_TEXT, "Different"))
        err = await _error(
            service.upload(USER_A, upload_file(other), visibility="private", metadata=DocumentMetadata())
        )
        assert err.code is ErrorCode.QUOTA_EXCEEDED
    finally:
        await service.drain()
    assert {d.id for d in await _list(service, USER_B)} == {public.id}  # only the public row is visible
    assert store.points  # the accepted documents were indexed; refused ones added nothing


# ------------------------------------------------------------------ criteria 7, 8, 9


async def test_background_failure_ends_failed_without_points(
    tmp_path: Path, build: ServiceFactory, store: InMemoryVectorStore
) -> None:
    path = long_markdown(tmp_path)
    service = build(CountingEmbedder(fail_on_batch=2), max_ingest_jobs_per_user=1)
    try:
        doc = await _bg_upload(service, USER_A, path)
    finally:
        await service.drain()

    failed = await service.get(USER_A, doc.id)
    assert failed.status is DocumentStatus.FAILED
    assert failed.error
    assert store.points == {}  # the first batch was compensated

    service.embedder = FakeEmbedder()  # type: ignore[attr-defined]
    try:
        again = await _bg_upload(service, USER_A, path)  # slot released, failed row purged
        assert again.id != doc.id
    finally:
        await service.drain()
    assert (await service.get(USER_A, again.id)).status is DocumentStatus.READY
    assert await store.count(USER_A, again.id) > 0


async def test_background_content_duplicate_ends_failed(
    tmp_path: Path, build: ServiceFactory, store: InMemoryVectorStore
) -> None:
    service = build()
    try:
        first = await _bg_upload(service, USER_A, write(tmp_path / "a.md", markdown(SECTION_TEXT)))
        await service.drain()
        points_before = await store.point_ids(USER_A, first.id)
        copy = write(tmp_path / "copy.md", markdown(SECTION_TEXT) + "\n\n")  # same text, other bytes
        second = await _bg_upload(service, USER_A, copy)
    finally:
        await service.drain()

    failed = await service.get(USER_A, second.id)
    assert failed.status is DocumentStatus.FAILED
    assert failed.error and "already" in failed.error.lower()
    assert await store.count(USER_A, second.id) == 0
    assert await store.point_ids(USER_A, first.id) == points_before
    assert (await service.get(USER_A, first.id)).status is DocumentStatus.READY


async def test_background_job_holds_the_user_slot(tmp_path: Path, build: ServiceFactory) -> None:
    gate = asyncio.Event()
    embedder = CountingEmbedder(gate=gate)
    service = build(embedder, max_ingest_jobs_per_user=1)
    try:
        first = await _bg_upload(service, USER_A, long_markdown(tmp_path))
        await _held(embedder)

        blocked = write(tmp_path / "b.md", markdown(SECTION_TEXT, "Other"))
        err = await _error(
            service.upload(USER_A, upload_file(blocked), visibility="private", metadata=DocumentMetadata())
        )
        assert err.code is ErrorCode.TOO_MANY_INGEST_JOBS and err.code.status == 429
        assert err.headers == {"Retry-After": "5"} and err.extra == {"limit": 1}

        # The official corpus is exempt from the per-user limit.
        public = await _bg_upload(
            service,
            ADMIN,
            write(tmp_path / "official.md", markdown(SECTION_TEXT, "Official guide")),
            visibility="public",
        )
    finally:
        gate.set()
        await service.drain()

    assert (await service.get(USER_A, first.id)).status is DocumentStatus.READY
    assert (await service.get(ADMIN, public.id)).status is DocumentStatus.READY
    again = await _bg_upload(service, USER_A, blocked)  # the slot is free again
    await service.drain()
    assert (await service.get(USER_A, again.id)).status is DocumentStatus.READY


async def test_background_queue_is_bounded(
    tmp_path: Path, build: ServiceFactory, store: InMemoryVectorStore
) -> None:
    gate = asyncio.Event()
    embedder = CountingEmbedder(gate=gate)
    service = build(embedder, background_ingest_workers=1, background_ingest_queue_max=1)
    try:
        first = await _bg_upload(service, USER_A, long_markdown(tmp_path))
        await _held(embedder)

        refused = write(tmp_path / "b.md", markdown(SECTION_TEXT, "Other"))
        err = await _error(
            service.upload(USER_B, upload_file(refused), visibility="private", metadata=DocumentMetadata())
        )
        assert err.code is ErrorCode.TOO_MANY_INGEST_JOBS and err.code.status == 429
        assert "Retry-After" in err.headers
        assert await _list(service, USER_B) == []  # nothing was stored for the refused upload
    finally:
        gate.set()
        await service.drain()

    assert (await service.get(USER_A, first.id)).status is DocumentStatus.READY
    assert await _list(service, USER_B) == []
    again = await _bg_upload(service, USER_B, refused)  # room again once the queue drained
    await service.drain()
    assert (await service.get(USER_B, again.id)).status is DocumentStatus.READY


# ------------------------------------------------------------------ criteria 10 to 13 (replace)

CHANGED = dict(
    SECTION_TEXT,
    Fees="The fee for the normal category was revised in 2026 and urgent costs double.",
    Requirements="Applicants now bring two photographs and the old card to the counter.",
)


async def test_large_replace_runs_in_background(
    tmp_path: Path, build: ServiceFactory, store: InMemoryVectorStore
) -> None:
    service = build()
    doc = await _bg_upload(service, USER_A, write(tmp_path / "v1.md", markdown(SECTION_TEXT)))
    await service.drain()
    old_ids = await store.point_ids(USER_A, doc.id)
    assert old_ids

    gate = asyncio.Event()
    embedder = CountingEmbedder(gate=gate)
    service.embedder = embedder  # type: ignore[attr-defined]
    try:
        updating, accepted = await service.replace(
            USER_A, doc.id, upload_file(write(tmp_path / "v2.md", markdown(CHANGED)))
        )
        assert accepted is True
        assert updating.status is DocumentStatus.UPDATING and updating.version == 1
        await _held(embedder)
        mid = await service.get(USER_A, doc.id)
        assert mid.status is DocumentStatus.UPDATING and mid.version == 1
        assert old_ids <= await store.point_ids(USER_A, doc.id)  # the old version is still served
    finally:
        gate.set()
        await service.drain()

    new = await service.get(USER_A, doc.id)
    assert new.status is DocumentStatus.READY and new.version == 2 and new.file_name == "v2.md"
    texts = " ".join(s.point.payload.text for s in store.points.values())
    assert "revised in 2026" in texts and "urgent category costs more" not in texts
    assert "two photographs" in texts and "birth certificate" not in texts
    assert len(await store.point_ids(USER_A, doc.id)) == new.chunk_count


async def test_large_replace_failure_keeps_old_version(
    tmp_path: Path, build: ServiceFactory, store: InMemoryVectorStore
) -> None:
    service = build(embed_batch_size=1)
    doc = await _bg_upload(service, USER_A, write(tmp_path / "v1.md", markdown(SECTION_TEXT)))
    await service.drain()
    before = await store.point_ids(USER_A, doc.id)

    service.embedder = CountingEmbedder(fail_on_batch=2)  # type: ignore[attr-defined]
    try:
        _, accepted = await service.replace(
            USER_A, doc.id, upload_file(write(tmp_path / "v2.md", markdown(CHANGED)))
        )
        assert accepted is True
    finally:
        await service.drain()

    current = await service.get(USER_A, doc.id)
    assert current.status is DocumentStatus.READY_STALE and current.version == 1
    assert current.error
    assert await store.point_ids(USER_A, doc.id) == before  # old points kept, this attempt's removed


async def test_large_replace_no_change_and_duplicate_stay_inline(
    tmp_path: Path, build: ServiceFactory
) -> None:
    service = build()
    one = write(tmp_path / "one.md", markdown(SECTION_TEXT))
    two = write(tmp_path / "two.md", markdown(CHANGED, "Second guide"))
    doc = await _bg_upload(service, USER_A, one)
    other = await _bg_upload(service, USER_A, two)
    await service.drain()

    err = await _error(service.replace(USER_A, doc.id, upload_file(one)))
    assert err.code is ErrorCode.NO_CHANGE
    err = await _error(service.replace(USER_A, doc.id, upload_file(two)))
    assert err.code is ErrorCode.DUPLICATE_DOCUMENT and err.extra["existing_document_id"] == other.id
    current = await service.get(USER_A, doc.id)
    assert current.status is DocumentStatus.READY and current.version == 1


# ------------------------------------------------------------------ criteria 14, 15 (E28)


async def test_patch_and_replace_during_job_are_409(tmp_path: Path, build: ServiceFactory) -> None:
    gate = asyncio.Event()
    embedder = CountingEmbedder(gate=gate)
    service = build(embedder)
    try:
        doc = await _bg_upload(service, USER_A, long_markdown(tmp_path))
        await _held(embedder)

        assert ErrorCode.INGEST_IN_PROGRESS.status == 409
        assert ErrorCode.INGEST_IN_PROGRESS.title == "Ingestion in progress"
        assert ErrorCode.INGEST_IN_PROGRESS.slug == "ingest_in_progress"

        err = await _error(service.patch(USER_A, doc.id, DocumentMetadata(agency="DGIP")))
        assert err.code is ErrorCode.INGEST_IN_PROGRESS
        other = write(tmp_path / "other.md", markdown(CHANGED, "Replacement"))
        err = await _error(service.replace(USER_A, doc.id, upload_file(other)))
        assert err.code is ErrorCode.INGEST_IN_PROGRESS
        assert (await service.get(USER_A, doc.id)).agency != "DGIP"
    finally:
        gate.set()
        await service.drain()
    assert (await service.get(USER_A, doc.id)).status is DocumentStatus.READY


async def test_large_replace_in_progress_blocks_patch_and_replace(
    tmp_path: Path, build: ServiceFactory
) -> None:
    service = build()
    doc = await _bg_upload(service, USER_A, write(tmp_path / "v1.md", markdown(SECTION_TEXT)))
    await service.drain()
    gate = asyncio.Event()
    embedder = CountingEmbedder(gate=gate)
    service.embedder = embedder  # type: ignore[attr-defined]
    try:
        _, accepted = await service.replace(
            USER_A, doc.id, upload_file(write(tmp_path / "v2.md", markdown(CHANGED)))
        )
        assert accepted is True
        await _held(embedder)
        err = await _error(service.patch(USER_A, doc.id, DocumentMetadata(agency="DGIP")))
        assert err.code is ErrorCode.INGEST_IN_PROGRESS
        v3 = write(tmp_path / "v3.md", markdown(CHANGED, "Third"))
        err = await _error(service.replace(USER_A, doc.id, upload_file(v3)))
        assert err.code is ErrorCode.INGEST_IN_PROGRESS
    finally:
        gate.set()
        await service.drain()


async def test_delete_cancels_background_job(
    tmp_path: Path, build: ServiceFactory, store: InMemoryVectorStore, settings: Settings
) -> None:  # E28
    gate = asyncio.Event()
    embedder = CountingEmbedder(gate=gate)
    service = build(embedder, max_ingest_jobs_per_user=1)
    path = long_markdown(tmp_path)
    try:
        doc = await _bg_upload(service, USER_A, path)
        await _held(embedder)

        # The job is still blocked inside the embedder: delete must cancel it, not wait for it.
        assert await asyncio.wait_for(service.delete(USER_A, doc.id), WAIT) is None
        assert (await _error(service.get(USER_A, doc.id))).code is ErrorCode.NOT_FOUND
        assert store.points == {}
        assert not (settings.storage_dir / "documents" / "u_a" / doc.id).exists()
    finally:
        gate.set()
        await service.drain()

    # The cancelled job recreated nothing.
    assert (await _error(service.get(USER_A, doc.id))).code is ErrorCode.NOT_FOUND
    assert await _list(service, USER_A) == []
    assert store.points == {}
    assert not (settings.storage_dir / "documents" / "u_a" / doc.id).exists()

    # Its job slot was released: the same file can be uploaded again.
    service.embedder = FakeEmbedder()  # type: ignore[attr-defined]
    again = await _bg_upload(service, USER_A, path)
    await service.drain()
    assert (await service.get(USER_A, again.id)).status is DocumentStatus.READY


# ------------------------------------------------------------------ criteria 16, 17


async def test_aclose_marks_running_jobs_failed(
    tmp_path: Path, build: ServiceFactory, store: InMemoryVectorStore
) -> None:
    gate = asyncio.Event()
    embedder = CountingEmbedder(gate=gate)
    service = build(embedder)
    try:
        running = await _bg_upload(service, USER_A, long_markdown(tmp_path))
        await _held(embedder)
        await asyncio.wait_for(service.aclose(), WAIT)
    finally:
        gate.set()

    failed = await service.get(USER_A, running.id)
    assert failed.status is DocumentStatus.FAILED
    assert failed.error and "interrupted" in failed.error.lower()
    assert await store.count(USER_A, running.id) == 0
    assert store.points == {}
    await asyncio.wait_for(service.drain(), WAIT)  # nothing is left running


async def test_drain_without_jobs_returns(build: ServiceFactory) -> None:
    service = build()
    assert await asyncio.wait_for(service.drain(), WAIT) is None
    await asyncio.wait_for(service.aclose(), WAIT)  # also a no-op
