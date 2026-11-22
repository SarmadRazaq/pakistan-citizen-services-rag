"""/api/v1/documents through httpx.ASGITransport with the real IngestionService and fake stores."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from app.api.deps import get_tenant
from app.config import Settings
from app.core.health import HealthService
from app.core.tenancy import Role, Scope, TenantContext
from app.db.session import Database
from app.ingestion.chunking import RegexTokenCounter
from app.ingestion.ocr import NoOcr
from app.ingestion.service import IngestionService
from app.main import create_app
from app.protocols import UploadedFile
from app.schemas import DocumentMetadata, DocumentOut
from tests.conftest import ok_probe
from tests.fakes import FakeEmbedder, InMemoryVectorStore
from tests.fixtures.files import factories as f

DOCS = "/api/v1/documents"
TEXT = (
    "# Passport Guide\n\n## Fees\n\n"
    + f.ENGLISH_PARAGRAPH
    + "\n\n## Documents\n\nBring the CNIC and the old passport.\n"
)


class Tenant:
    """Mutable tenant holder so one app can act as different users."""

    def __init__(self) -> None:
        self.ctx = TenantContext.for_user("u_a", Role.USER, Scope.ALL)

    def __call__(self) -> TenantContext:
        return self.ctx

    def act_as(self, user_id: str, role: Role = Role.USER) -> None:
        self.ctx = TenantContext.for_user(user_id, role, Scope.ALL)


@pytest.fixture
def api_settings(tmp_path: Path) -> Settings:
    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        app_env="test",
        log_level="WARNING",
        storage_dir=tmp_path / "storage",
        max_upload_mb=1,
        chunk_size_tokens=80,
        chunk_overlap_tokens=8,
        ocr_enabled=False,
    )


@pytest.fixture
def tenant() -> Tenant:
    return Tenant()


@pytest.fixture
async def app(api_settings: Settings, tenant: Tenant) -> AsyncIterator[FastAPI]:
    db = Database("sqlite+aiosqlite:///:memory:")
    await db.create_all()
    application = create_app(api_settings, health_service=HealthService([ok_probe("qdrant")], timeout_s=1.0))
    application.state.db = db
    application.state.documents = IngestionService(
        api_settings,
        db,
        InMemoryVectorStore(),
        FakeEmbedder(),
        token_counter=RegexTokenCounter(),
        ocr=NoOcr(),
    )
    application.dependency_overrides[get_tenant] = tenant
    yield application
    await db.dispose()


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def files(name: str, content: bytes | str, mime: str = "application/octet-stream") -> dict[str, Any]:
    data = content.encode() if isinstance(content, str) else content
    return {"file": (name, data, mime)}


async def test_upload_list_get_chunks_patch_put_delete(
    client: httpx.AsyncClient, api_settings: Settings
) -> None:
    r = await client.post(DOCS, files=files("guide.md", TEXT), data={"agency": "DGIP", "topic": "passport"})
    assert r.status_code == 201, r.text
    doc = r.json()
    assert doc["status"] == "READY" and doc["agency"] == "DGIP" and doc["file_type"] == "md"
    doc_id = doc["id"]

    listed = await client.get(DOCS, params={"scope": "mine", "agency": "DGIP", "status": "READY"})
    assert listed.status_code == 200 and [d["id"] for d in listed.json()["items"]] == [doc_id]
    assert (await client.get(DOCS, params={"scope": "public"})).json()["items"] == []

    got = await client.get(f"{DOCS}/{doc_id}")
    assert got.status_code == 200 and got.json()["chunk_count"] == doc["chunk_count"]

    chunks = await client.get(f"{DOCS}/{doc_id}/chunks", params={"limit": 1})
    assert chunks.status_code == 200
    body = chunks.json()
    assert len(body["items"]) == 1 and body["items"][0]["chunk_index"] == 0

    patched = await client.patch(
        f"{DOCS}/{doc_id}", json={"jurisdiction": "punjab", "effective_date": "2026-01-31"}
    )
    assert patched.status_code == 200
    assert (patched.json()["jurisdiction"], patched.json()["effective_date"]) == ("punjab", "2026-01-31")

    same = await client.put(f"{DOCS}/{doc_id}", files=files("guide.md", TEXT))
    assert same.status_code == 409 and same.json()["code"] == "no_change"
    new = await client.put(
        f"{DOCS}/{doc_id}", files=files("guide-v2.md", TEXT.replace("old passport", "expired passport"))
    )
    assert new.status_code == 200, new.text
    assert new.json()["version"] == 2 and new.json()["last_ingest"]["chunks_kept"] >= 1

    deleted = await client.delete(f"{DOCS}/{doc_id}")
    assert deleted.status_code == 204 and deleted.content == b""
    assert (await client.get(f"{DOCS}/{doc_id}")).status_code == 404
    assert list((api_settings.storage_dir / "tmp").iterdir()) == []  # spooled uploads are cleaned up


async def test_duplicate_returns_409_with_existing_id(client: httpx.AsyncClient) -> None:  # E3
    first = await client.post(DOCS, files=files("a.md", TEXT))
    second = await client.post(DOCS, files=files("copy.md", TEXT))
    assert second.status_code == 409
    problem = second.json()
    assert problem["code"] == "duplicate_document"
    assert problem["extra"]["existing_document_id"] == first.json()["id"]
    assert second.headers["content-type"].startswith("application/problem+json")


async def test_too_large_is_413(client: httpx.AsyncClient, api_settings: Settings) -> None:  # E5
    big = b"a " * (api_settings.max_upload_bytes // 2 + 10)
    r = await client.post(DOCS, files=files("big.txt", big))
    assert r.status_code == 413 and r.json()["code"] == "file_too_large"
    tmp = api_settings.storage_dir / "tmp"
    assert not tmp.exists() or list(tmp.iterdir()) == []  # nothing left behind


@pytest.mark.parametrize(
    ("name", "content", "status", "code"),
    [
        ("virus.pdf", b"MZ\x90\x00" + b"\x00" * 300, 415, "unsupported_media_type"),  # E18
        ("photo.jpg", b"\xff\xd8\xff\xe0" + b"\x00" * 100, 415, "unsupported_media_type"),
        ("empty.pdf", b"", 400, "empty_file"),  # E2
        ("broken.pdf", b"%PDF-1.4\n garbage", 400, "corrupt_file"),  # E20
        ("macro.docm", b"PK\x03\x04", 400, "active_content_rejected"),  # E21
    ],
)
async def test_rejections(
    client: httpx.AsyncClient, name: str, content: bytes, status: int, code: str
) -> None:
    r = await client.post(DOCS, files=files(name, content))
    assert r.status_code == status and r.json()["code"] == code


async def test_encrypted_pdf_is_400(client: httpx.AsyncClient, tmp_path: Path) -> None:  # E19
    pdf = f.make_encrypted_pdf(tmp_path / "e.pdf").read_bytes()
    r = await client.post(DOCS, files=files("e.pdf", pdf, "application/pdf"))
    assert r.status_code == 400 and r.json()["code"] == "encrypted_document"


async def test_tenant_rules(client: httpx.AsyncClient, tenant: Tenant) -> None:  # E30, E33
    mine = (await client.post(DOCS, files=files("a.md", TEXT))).json()["id"]
    assert (
        await client.post(DOCS, files=files("p.md", TEXT + "x"), data={"visibility": "public"})
    ).status_code == 403

    tenant.act_as("u_admin", Role.ADMIN)
    public = await client.post(
        DOCS, files=files("p.md", TEXT + "\n\nOfficial copy."), data={"visibility": "public"}
    )
    assert public.status_code == 201 and public.json()["visibility"] == "public"
    public_id = public.json()["id"]

    tenant.act_as("u_b")
    for method, url in [
        ("GET", f"{DOCS}/{mine}"),
        ("GET", f"{DOCS}/{mine}/chunks"),
        ("DELETE", f"{DOCS}/{mine}"),
    ]:
        assert (await client.request(method, url)).status_code == 404
    assert (await client.patch(f"{DOCS}/{mine}", json={"agency": "X"})).status_code == 404
    assert (await client.get(f"{DOCS}/{public_id}")).status_code == 200
    assert (await client.delete(f"{DOCS}/{public_id}")).status_code == 403
    assert (await client.patch(f"{DOCS}/{public_id}", json={"agency": "X"})).status_code == 403
    assert {d["id"] for d in (await client.get(DOCS)).json()["items"]} == {public_id}


async def test_validation_errors_are_422(client: httpx.AsyncClient) -> None:
    assert (await client.get(DOCS, params={"limit": 0})).status_code == 422
    assert (await client.post(DOCS, data={"visibility": "private"})).status_code == 422  # no file
    assert (await client.get(DOCS, params={"cursor": "%%%"})).status_code == 422


class AcceptingService:
    """A DocumentService double that reports background ingestion (accepted=True)."""

    def __init__(self, doc: DocumentOut) -> None:
        self.doc = doc
        self.seen: list[UploadedFile] = []

    async def upload(
        self, ctx: TenantContext, file: UploadedFile, *, visibility: str, metadata: DocumentMetadata
    ) -> tuple[DocumentOut, bool]:
        self.seen.append(file)
        assert file.path.exists() and file.size_bytes == len(TEXT)
        return self.doc, True

    async def replace(self, ctx: TenantContext, doc_id: str, file: UploadedFile) -> tuple[DocumentOut, bool]:
        return self.doc, True

    async def list(self, ctx: TenantContext, **kwargs: Any) -> Any:  # pragma: no cover - unused
        raise NotImplementedError

    async def get(self, ctx: TenantContext, doc_id: str) -> DocumentOut:  # pragma: no cover - unused
        return self.doc

    async def patch(
        self, ctx: TenantContext, doc_id: str, metadata: DocumentMetadata
    ) -> DocumentOut:  # pragma: no cover
        return self.doc

    async def chunks(
        self, ctx: TenantContext, doc_id: str, **kwargs: Any
    ) -> Any:  # pragma: no cover - unused
        raise NotImplementedError

    async def delete(self, ctx: TenantContext, doc_id: str) -> None:  # pragma: no cover - unused
        return None


async def test_accepted_upload_returns_202_with_location(app: FastAPI, client: httpx.AsyncClient) -> None:
    real = (await client.post(DOCS, files=files("a.md", TEXT))).json()
    fake = AcceptingService(DocumentOut.model_validate(real))
    app.state.documents = fake
    r = await client.post(DOCS, files=files("b.md", TEXT))
    assert r.status_code == 202
    assert r.headers["location"] == f"{DOCS}/{real['id']}"
    put = await client.put(f"{DOCS}/{real['id']}", files=files("b.md", TEXT))
    assert put.status_code == 202 and put.headers["location"].endswith(real["id"])
    assert not fake.seen[0].path.exists()  # the router removed its temp file


def test_routes_keep_documented_errors(app: FastAPI) -> None:
    spec: Callable[[], dict[str, Any]] = app.openapi
    paths = spec()["paths"]
    assert set(paths[DOCS]["post"]["responses"]) >= {"201", "202", "400", "409", "413", "415"}
    assert set(paths[DOCS + "/{document_id}"]["delete"]["responses"]) >= {"204", "403", "404", "503"}
