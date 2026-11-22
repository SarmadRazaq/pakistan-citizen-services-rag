"""Auth endpoints, token resolution, account deletion and rate limits (§5.8.1, §5.8.6, §5.8.7, E41, E42)."""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import select, update

from app.config import Settings
from app.core.auth_service import AuthService
from app.core.errors import AppError, ErrorCode
from app.core.health import HealthService
from app.core.ratelimit import install_rate_limits
from app.core.security import create_access_token, decode_access_token, hash_password, verify_password
from app.core.tenancy import PUBLIC_OWNER_ID, Role, TenantContext, system_context
from app.db.models import Document, User
from app.db.session import Database
from app.main import create_app
from app.protocols import DocumentService, UploadedFile
from app.schemas import (
    ChunkOut,
    DocumentMetadata,
    DocumentOut,
    DocumentStatus,
    FileType,
    Page,
    SignupRequest,
)
from tests.conftest import ok_probe

SECRET = "test-jwt-secret-that-is-long-enough-for-hs256-0123456789"
PASSWORD = "correct horse battery"


def auth_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = dict(
        _env_file=None,
        app_env="test",
        log_level="WARNING",
        jwt_secret=SECRET,
        rate_limit_login="1000/minute",
        rate_limit_ask="1000/minute",
        rate_limit_upload="1000/minute",
    )
    values.update(overrides)
    return Settings(**values)


@dataclass
class World:
    app: FastAPI
    db: Database
    auth: AuthService
    settings: Settings

    def client(self, ip: str = "10.0.0.1") -> httpx.AsyncClient:
        transport = httpx.ASGITransport(app=self.app, raise_app_exceptions=False, client=(ip, 5000))
        return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.fixture
def make_world() -> Callable[..., Any]:
    created: list[Database] = []

    async def factory(**overrides: Any) -> World:
        settings = auth_settings(**overrides)
        db = Database("sqlite+aiosqlite:///:memory:")
        await db.create_all()
        created.append(db)
        app = create_app(settings, health_service=HealthService([ok_probe("qdrant")], timeout_s=1.0))
        auth = AuthService(settings, db)
        app.state.db = db
        app.state.auth = auth
        install_rate_limits(app, settings)
        return World(app, db, auth, settings)

    return factory


@pytest.fixture
async def world(make_world: Callable[..., Any]) -> AsyncIterator[World]:
    w: World = await make_world()
    yield w
    await w.db.dispose()


@pytest.fixture
async def client(world: World) -> AsyncIterator[httpx.AsyncClient]:
    async with world.client() as c:
        yield c


async def signup(client: httpx.AsyncClient, email: str = "Asma@Example.ORG") -> httpx.Response:
    return await client.post(
        "/api/v1/auth/signup", json={"email": email, "password": PASSWORD, "display_name": "Asma"}
    )


async def login(client: httpx.AsyncClient, email: str = "asma@example.org", password: str = PASSWORD) -> str:
    r = await client.post("/api/v1/auth/login", data={"username": email, "password": password})
    assert r.status_code == 200, r.text
    token: str = r.json()["access_token"]
    return token


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------- signup / login / me


async def test_signup_login_me_happy_path(client: httpx.AsyncClient) -> None:
    r = await signup(client)
    assert r.status_code == 201, r.text
    created = r.json()
    assert created["email"] == "asma@example.org"  # normalised to lowercase
    assert created["role"] == "user"
    assert created["id"].startswith("u_")

    r = await client.post("/api/v1/auth/login", data={"username": "ASMA@example.org", "password": PASSWORD})
    assert r.status_code == 200
    body = r.json()
    assert body["token_type"] == "bearer"
    assert body["expires_in"] == 30 * 60

    r = await client.get("/api/v1/auth/me", headers=bearer(body["access_token"]))
    assert r.status_code == 200
    me = r.json()
    assert me["id"] == created["id"]
    assert me["display_name"] == "Asma"
    assert me["quota"] == {
        "documents": 0,
        "documents_limit": 50,
        "storage_mb": 0.0,
        "storage_mb_limit": 200,
        "chunks": 0,
        "chunks_limit": 20_000,
    }


async def test_me_quota_counts_only_my_documents(world: World, client: httpx.AsyncClient) -> None:
    user_id = (await signup(client)).json()["id"]
    async with world.db.session(system_context("test seed")) as s:
        s.add_all(
            [
                _doc("d_1", user_id, size=3 * 1024 * 1024, chunks=40),
                _doc("d_2", user_id, size=1024 * 1024, chunks=2),
                _doc("d_other", "u_other", size=5, chunks=9),
                _doc("d_pub", PUBLIC_OWNER_ID, size=5, chunks=9),
            ]
        )
    r = await client.get("/api/v1/auth/me", headers=bearer(await login(client)))
    quota = r.json()["quota"]
    assert (quota["documents"], quota["storage_mb"], quota["chunks"]) == (2, 4.0, 42)


async def test_signup_cannot_choose_a_role(client: httpx.AsyncClient) -> None:
    r = await client.post(
        "/api/v1/auth/signup", json={"email": "x@example.org", "password": PASSWORD, "role": "admin"}
    )
    assert r.status_code == 422


async def test_duplicate_email_is_409_case_insensitive(client: httpx.AsyncClient) -> None:
    assert (await signup(client, "dup@example.org")).status_code == 201
    r = await signup(client, "DUP@example.org")
    assert r.status_code == 409
    assert r.headers["content-type"].startswith("application/problem+json")
    assert r.json()["code"] == "email_taken"


async def test_bad_password_and_unknown_email_get_the_same_401(client: httpx.AsyncClient) -> None:
    await signup(client)
    wrong = await client.post(
        "/api/v1/auth/login", data={"username": "asma@example.org", "password": "wrong password!"}
    )
    unknown = await client.post(
        "/api/v1/auth/login", data={"username": "nobody@example.org", "password": PASSWORD}
    )
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json()["code"] == unknown.json()["code"] == "invalid_credentials"
    assert wrong.json()["detail"] == unknown.json()["detail"]


def test_unknown_user_still_runs_argon2() -> None:
    """The timing equaliser: a missing hash verifies against a dummy hash instead of returning early."""
    assert verify_password(PASSWORD, None) == (False, None)
    assert verify_password(PASSWORD, "not-a-hash") == (False, None)
    valid, _ = verify_password(PASSWORD, hash_password(PASSWORD))
    assert valid
    assert hash_password(PASSWORD).startswith("$argon2id$")


# ---------------------------------------------------------------- tokens


async def test_missing_or_malformed_header_is_401(client: httpx.AsyncClient) -> None:
    for headers in ({}, {"Authorization": "Basic abc"}, {"Authorization": "Bearer "}):
        r = await client.get("/api/v1/auth/me", headers=headers)
        assert r.status_code == 401
        assert r.headers["www-authenticate"] == "Bearer"


async def test_tampered_token_is_401(client: httpx.AsyncClient) -> None:
    await signup(client)
    token = await login(client)
    header, payload, signature = token.split(".")
    claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    claims["sub"] = "u_someone_else"
    forged_payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode()
    flipped_sig = ("A" if signature[0] != "A" else "B") + signature[1:]
    for forged in (f"{header}.{forged_payload}.{signature}", f"{header}.{payload}.{flipped_sig}"):
        r = await client.get("/api/v1/auth/me", headers=bearer(forged))
        assert r.status_code == 401
        assert r.json()["code"] == "unauthorized"


async def test_alg_none_and_wrong_secret_tokens_are_401(world: World, client: httpx.AsyncClient) -> None:
    import jwt

    user_id = (await signup(client)).json()["id"]
    now = int(datetime.now(UTC).timestamp())
    claims = {"sub": user_id, "jti": "x", "ver": 0, "iat": now, "exp": now + 600}
    unsigned = jwt.encode(claims, key="", algorithm="none")
    other_secret = jwt.encode(claims, key="another-secret-that-is-long-enough-0123456789", algorithm="HS256")
    for token in (unsigned, other_secret):
        assert (await client.get("/api/v1/auth/me", headers=bearer(token))).status_code == 401


async def test_expired_token_is_401(world: World, client: httpx.AsyncClient) -> None:
    user_id = (await signup(client)).json()["id"]
    token, _ = create_access_token(world.settings, user_id, 0, now=datetime.now(UTC) - timedelta(minutes=31))
    r = await client.get("/api/v1/auth/me", headers=bearer(token))
    assert r.status_code == 401
    assert "expired" in r.json()["detail"]


def test_token_carries_the_required_claims() -> None:
    settings = auth_settings()
    token, expires_in = create_access_token(settings, "u_abc", 4)
    claims = decode_access_token(settings, token)
    assert (claims.sub, claims.ver, expires_in) == ("u_abc", 4, 1800)
    assert claims.exp - claims.iat == 1800
    assert len(claims.jti) == 32


async def test_token_version_bump_kills_old_tokens(world: World, client: httpx.AsyncClient) -> None:
    """E42: bumping users.token_version revokes every token issued before."""
    user_id = (await signup(client)).json()["id"]
    old = await login(client)
    assert (await client.get("/api/v1/auth/me", headers=bearer(old))).status_code == 200

    async with world.db.session(system_context("test")) as s:
        await s.execute(update(User).where(User.id == user_id).values(token_version=User.token_version + 1))

    assert (await client.get("/api/v1/auth/me", headers=bearer(old))).status_code == 401
    fresh = await login(client)
    assert (await client.get("/api/v1/auth/me", headers=bearer(fresh))).status_code == 200


async def test_role_is_read_from_the_db_not_the_token(world: World, client: httpx.AsyncClient) -> None:
    """E42: a demoted admin's still-valid token resolves to the `user` role on the next request."""
    user_id = (await signup(client)).json()["id"]
    token = await login(client)

    async def set_role(role: str) -> None:
        async with world.db.session(system_context("test")) as s:
            await s.execute(update(User).where(User.id == user_id).values(role=role))

    await set_role("admin")
    ctx = await world.auth.resolve(token, request_id="r1")
    assert ctx.role is Role.ADMIN and ctx.request_id == "r1"
    assert ctx.readable_owner_ids == frozenset({user_id, PUBLIC_OWNER_ID})

    await set_role("user")  # demoted without a version bump: same token, new role
    ctx = await world.auth.resolve(token)
    assert ctx.role is Role.USER
    assert not ctx.is_admin


async def test_demote_admin_bumps_version(world: World, client: httpx.AsyncClient) -> None:
    await world.auth.create_or_promote_admin("boss@example.org", PASSWORD)
    await world.auth.create_or_promote_admin("deputy@example.org", PASSWORD)
    token = await login(client, "boss@example.org")
    assert (await client.get("/api/v1/auth/me", headers=bearer(token))).json()["role"] == "admin"

    demoted = await world.auth.demote_admin("boss@example.org")
    assert demoted.role is Role.USER
    assert (await client.get("/api/v1/auth/me", headers=bearer(token))).status_code == 401

    with pytest.raises(AppError) as info:
        await world.auth.demote_admin("deputy@example.org")  # the last admin
    assert info.value.code is ErrorCode.LAST_ADMIN


async def test_get_tenant_without_auth_service_is_401(make_app: Callable[..., FastAPI]) -> None:
    app = make_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
    ) as c:
        r = await c.get("/api/v1/auth/me", headers=bearer("whatever"))
    assert r.status_code == 401


def test_prod_refuses_the_default_jwt_secret() -> None:
    db = Database("sqlite+aiosqlite:///:memory:")
    with pytest.raises(RuntimeError, match="JWT_SECRET"):
        AuthService(Settings(_env_file=None, app_env="prod"), db)  # type: ignore[call-arg]
    AuthService(auth_settings(app_env="prod"), db)  # a real secret is fine


# ---------------------------------------------------------------- account deletion (E41)


def _doc(doc_id: str, owner: str, *, size: int = 10, chunks: int = 1) -> Document:
    return Document(
        id=doc_id,
        user_id=owner,
        visibility="public" if owner == PUBLIC_OWNER_ID else "private",
        file_name=f"{doc_id}.pdf",
        file_type="pdf",
        size_bytes=size,
        file_sha256=doc_id.ljust(64, "0"),
        status="READY",
        chunk_count=chunks,
    )


def _out(doc_id: str) -> DocumentOut:
    now = datetime.now(UTC)
    return DocumentOut(
        id=doc_id,
        file_name=f"{doc_id}.pdf",
        file_type=FileType.PDF,
        size_bytes=10,
        status=DocumentStatus.READY,
        visibility="private",  # type: ignore[arg-type]
        version=1,
        created_at=now,
        updated_at=now,
    )


@dataclass
class FakeDocumentService:
    """Owns `docs` (doc_id → owner); delete records the call, as the real service would purge vectors."""

    docs: dict[str, str] = field(default_factory=dict)
    deleted: list[tuple[str, str]] = field(default_factory=list)
    fail_on: set[str] = field(default_factory=set)

    async def upload(
        self, ctx: TenantContext, file: UploadedFile, *, visibility: str, metadata: DocumentMetadata
    ) -> tuple[DocumentOut, bool]:
        raise NotImplementedError

    async def list(self, ctx: TenantContext, **kwargs: Any) -> Page[DocumentOut]:
        mine = sorted(d for d, owner in self.docs.items() if owner in ctx.readable_owner_ids)
        return Page[DocumentOut](items=[_out(d) for d in mine])

    async def get(self, ctx: TenantContext, doc_id: str) -> DocumentOut:
        return _out(doc_id)

    async def patch(self, ctx: TenantContext, doc_id: str, metadata: DocumentMetadata) -> DocumentOut:
        return _out(doc_id)

    async def chunks(
        self, ctx: TenantContext, doc_id: str, *, limit: int, cursor: str | None
    ) -> Page[ChunkOut]:
        return Page[ChunkOut](items=[])

    async def replace(self, ctx: TenantContext, doc_id: str, file: UploadedFile) -> tuple[DocumentOut, bool]:
        raise NotImplementedError

    async def delete(self, ctx: TenantContext, doc_id: str) -> None:
        if doc_id in self.fail_on:
            raise AppError(ErrorCode.DEPENDENCY_UNAVAILABLE)
        ctx.ensure_writable(self.docs[doc_id])
        self.deleted.append((ctx.user_id, doc_id))
        del self.docs[doc_id]


async def test_delete_me_cascades_through_the_document_service(
    world: World, client: httpx.AsyncClient
) -> None:
    me_id = (await signup(client)).json()["id"]
    await signup(client, "other@example.org")
    async with world.db.session(system_context("test seed")) as s:
        s.add_all([_doc("d_mine1", me_id), _doc("d_mine2", me_id), _doc("d_other", "u_other")])
    docs = FakeDocumentService(
        {"d_mine1": me_id, "d_mine2": me_id, "d_other": "u_other", "d_pub": PUBLIC_OWNER_ID}
    )
    assert isinstance(docs, DocumentService)
    world.app.state.documents = docs
    token = await login(client)

    r = await client.delete("/api/v1/auth/me", headers=bearer(token))

    assert r.status_code == 204, r.text
    assert sorted(d for _, d in docs.deleted) == ["d_mine1", "d_mine2"]
    assert set(docs.docs) == {"d_other", "d_pub"}  # nobody else's data, and never the public corpus
    assert (await client.get("/api/v1/auth/me", headers=bearer(token))).status_code == 401  # old token dies
    gone = await client.post(
        "/api/v1/auth/login", data={"username": "asma@example.org", "password": PASSWORD}
    )
    assert gone.status_code == 401
    async with world.db.session(system_context("test")) as s:
        assert (await s.scalars(select(Document.id))).all() == ["d_other"]
        assert await s.get(User, me_id) is None
    await login(client, "other@example.org")


async def test_delete_me_failure_is_503_and_tokens_still_die(world: World, client: httpx.AsyncClient) -> None:
    me_id = (await signup(client)).json()["id"]
    world.app.state.documents = FakeDocumentService({"d_1": me_id}, fail_on={"d_1"})
    token = await login(client)

    r = await client.delete("/api/v1/auth/me", headers=bearer(token))

    assert r.status_code == 503
    assert (await client.get("/api/v1/auth/me", headers=bearer(token))).status_code == 401
    retry = await login(client)  # the account is still there, so the user can retry
    world.app.state.documents = FakeDocumentService({"d_1": me_id})
    assert (await client.delete("/api/v1/auth/me", headers=bearer(retry))).status_code == 204


async def test_delete_me_with_documents_but_no_document_service_is_503(
    world: World, client: httpx.AsyncClient
) -> None:
    me_id = (await signup(client)).json()["id"]
    async with world.db.session(system_context("test seed")) as s:
        s.add(_doc("d_1", me_id))
    r = await client.delete("/api/v1/auth/me", headers=bearer(await login(client)))
    assert r.status_code == 503
    async with world.db.session(system_context("test")) as s:
        assert (await s.get(Document, "d_1")).status == "DELETING"  # type: ignore[union-attr]


async def test_delete_me_without_documents_needs_no_document_service(client: httpx.AsyncClient) -> None:
    await signup(client)
    assert (await client.delete("/api/v1/auth/me", headers=bearer(await login(client)))).status_code == 204


async def test_last_admin_cannot_be_deleted(world: World, client: httpx.AsyncClient) -> None:
    await world.auth.create_or_promote_admin("boss@example.org", PASSWORD)
    token = await login(client, "boss@example.org")

    r = await client.delete("/api/v1/auth/me", headers=bearer(token))
    assert r.status_code == 409
    assert r.json()["code"] == "last_admin"
    assert (await client.get("/api/v1/auth/me", headers=bearer(token))).status_code == 200  # nothing changed

    await world.auth.create_or_promote_admin("deputy@example.org", PASSWORD)
    assert (await client.delete("/api/v1/auth/me", headers=bearer(token))).status_code == 204


async def test_create_or_promote_admin(world: World) -> None:
    await world.auth.signup(SignupRequest(email="plain@example.org", password=PASSWORD))
    promoted, created = await world.auth.create_or_promote_admin("PLAIN@example.org", None)
    assert (promoted.role, created) == (Role.ADMIN, False)
    with pytest.raises(ValueError, match="password"):
        await world.auth.create_or_promote_admin("new@example.org", None)
    new, created = await world.auth.create_or_promote_admin("new@example.org", PASSWORD, display_name="Ops")
    assert (new.role, new.display_name, created) == (Role.ADMIN, "Ops", True)


# ---------------------------------------------------------------- rate limits (§5.8.6)


async def test_login_burst_is_429_with_retry_after(make_world: Callable[..., Any]) -> None:
    world: World = await make_world(rate_limit_login="5/minute")
    async with world.client() as c:
        statuses = [
            (
                await c.post("/api/v1/auth/login", data={"username": f"u{i}@example.org", "password": "x"})
            ).status_code
            for i in range(5)
        ]
        assert statuses == [401] * 5
        r = await c.post("/api/v1/auth/login", data={"username": "u9@example.org", "password": "x"})
    assert r.status_code == 429
    assert r.headers["content-type"].startswith("application/problem+json")
    assert r.json()["code"] == "rate_limited"
    assert 1 <= int(r.headers["retry-after"]) <= 60
    assert r.headers["x-request-id"] == r.json()["request_id"]
    await world.db.dispose()


async def test_login_is_also_limited_per_email_across_ips(make_world: Callable[..., Any]) -> None:
    world: World = await make_world(rate_limit_login="3/minute")
    form = {"username": "Target@example.org", "password": "guess"}
    for ip in ("10.0.0.1", "10.0.0.2", "10.0.0.3"):
        async with world.client(ip) as c:
            assert (await c.post("/api/v1/auth/login", data=form)).status_code == 401
    async with world.client("10.0.0.4") as c:
        assert (await c.post("/api/v1/auth/login", data=form)).status_code == 429  # same email, fresh IP
        other = {"username": "someone@example.org", "password": "guess"}
        assert (await c.post("/api/v1/auth/login", data=other)).status_code == 401  # IP itself is fine
    await world.db.dispose()


async def test_ask_is_limited_per_user(make_world: Callable[..., Any]) -> None:
    world: World = await make_world(rate_limit_ask="2/minute")
    async with world.client() as c:
        await signup(c, "a@example.org")
        await signup(c, "b@example.org")
        token_a, token_b = await login(c, "a@example.org"), await login(c, "b@example.org")
        body = {"question": "Passport fee?"}
        first = [
            (await c.post("/api/v1/ask", json=body, headers=bearer(token_a))).status_code for _ in range(2)
        ]
        assert 429 not in first
        assert (await c.post("/api/v1/ask", json=body, headers=bearer(token_a))).status_code == 429
        assert (await c.post("/api/v1/ask", json=body, headers=bearer(token_b))).status_code != 429
    await world.db.dispose()


async def test_install_rate_limits_after_start_is_refused(world: World, client: httpx.AsyncClient) -> None:
    await client.get("/health/live")
    with pytest.raises(RuntimeError):
        install_rate_limits(world.app, world.settings)
