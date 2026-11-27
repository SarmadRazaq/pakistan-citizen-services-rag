"""Accounts: signup, login, current user, account deletion and token resolution (§5.8.1, §5.8.7, §12.6).

The `users` table is not tenant-owned, so its queries run in a session bound to `_AUTH_CTX`, a
context that can read **no** owned rows. Owned tables (documents, feedback, request log) are only
touched through the caller's own `TenantContext`, so the ORM tenant filter still applies to them.
"""

from __future__ import annotations

import asyncio

import structlog
from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import IntegrityError

from app.config import Settings
from app.core.errors import AppError, ErrorCode
from app.core.ids import new_user_id
from app.core.security import (
    BEARER_CHALLENGE,
    create_access_token,
    decode_access_token,
    ensure_secure_secret,
    hash_password,
    verify_password,
)
from app.core.tenancy import Role, Scope, TenantContext
from app.db.models import Document, Feedback, RequestLog, User
from app.db.session import Database
from app.protocols import DocumentService
from app.schemas import QuotaUsage, SignupRequest, TokenResponse, UserOut

log = structlog.get_logger(__name__)

# Reads nothing owned (empty readable set): used only for `users` table work.
_AUTH_CTX = TenantContext(user_id="u_auth", role=Role.USER, readable_owner_ids=frozenset())
INVALID_CREDENTIALS_DETAIL = "Incorrect email or password."
DELETE_PAGE_SIZE = 100  # documents listed per page while deleting an account


def normalize_email(email: str) -> str:
    return email.strip().lower()


def _user_out(user: User, quota: QuotaUsage | None = None) -> UserOut:
    return UserOut(
        id=user.id,
        email=user.email,
        display_name=user.display_name,
        role=Role(user.role),
        created_at=user.created_at,
        quota=quota,
    )


class AuthService:
    def __init__(self, settings: Settings, db: Database) -> None:
        ensure_secure_secret(settings)
        self._settings = settings
        self._db = db

    # ------------------------------------------------------------ signup / login

    async def signup(self, request: SignupRequest) -> UserOut:
        """Create a `user` account. The role is never taken from the request (§5.8.1)."""
        email = normalize_email(str(request.email))
        password_hash = await asyncio.to_thread(hash_password, request.password)
        async with self._db.session(_AUTH_CTX) as s:
            if await s.scalar(select(User.id).where(User.email == email)) is not None:
                raise AppError(ErrorCode.EMAIL_TAKEN, "An account with this email already exists.")
        user = User(
            id=new_user_id(),
            email=email,
            password_hash=password_hash,
            display_name=request.display_name,
            role=Role.USER.value,
            token_version=0,
        )
        try:
            async with self._db.session(_AUTH_CTX) as s:
                s.add(user)
        except IntegrityError as exc:  # lost a race with a concurrent signup for the same email
            raise AppError(ErrorCode.EMAIL_TAKEN, "An account with this email already exists.") from exc
        log.info("auth.signup", user_id=user.id)
        return _user_out(user)

    async def login(self, email: str, password: str) -> TokenResponse:
        """Same 401 (detail and argon2 work) whether the email is unknown or the password is wrong."""
        normalized = normalize_email(email)
        async with self._db.session(_AUTH_CTX) as s:
            user = await s.scalar(select(User).where(User.email == normalized))
        valid, new_hash = await asyncio.to_thread(
            verify_password, password, user.password_hash if user else None
        )
        if user is None or not valid:
            raise AppError(ErrorCode.INVALID_CREDENTIALS, INVALID_CREDENTIALS_DETAIL)
        if new_hash:  # argon2 parameters changed since this hash was made
            async with self._db.session(_AUTH_CTX) as s:
                await s.execute(update(User).where(User.id == user.id).values(password_hash=new_hash))
        token, expires_in = create_access_token(self._settings, user.id, user.token_version)
        log.info("auth.login", user_id=user.id)
        return TokenResponse(access_token=token, expires_in=expires_in)

    # ------------------------------------------------------------ token → tenant

    async def resolve(self, token: str, *, request_id: str | None = None) -> TenantContext:
        """JWT → TenantContext. `ver` must match the DB, and the role comes from the DB (E42)."""
        claims = decode_access_token(self._settings, token)
        async with self._db.session(_AUTH_CTX) as s:
            user = await s.get(User, claims.sub)
        if user is None or user.token_version != claims.ver:
            raise AppError(
                ErrorCode.UNAUTHORIZED,
                "The access token is no longer valid. Sign in again.",
                headers=dict(BEARER_CHALLENGE),
            )
        try:
            role = Role(user.role)
        except ValueError as exc:
            raise AppError(ErrorCode.UNAUTHORIZED, headers=dict(BEARER_CHALLENGE)) from exc
        return TenantContext.for_user(
            user.id,
            role,
            Scope.ALL,
            request_id=request_id,
            public_owner_id=self._settings.public_owner_id,
        )

    # ------------------------------------------------------------ current user

    async def me(self, ctx: TenantContext) -> UserOut:
        async with self._db.session(_AUTH_CTX) as s:
            user = await s.get(User, ctx.user_id)
        if user is None:
            raise AppError(ErrorCode.UNAUTHORIZED, headers=dict(BEARER_CHALLENGE))
        mine = ctx.with_scope(Scope.MINE)
        async with self._db.session(mine) as s:
            row = (
                await s.execute(
                    select(
                        func.count(Document.id),
                        func.coalesce(func.sum(Document.size_bytes), 0),
                        func.coalesce(func.sum(Document.chunk_count), 0),
                    ).where(Document.user_id == ctx.user_id)
                )
            ).one()
        quota = QuotaUsage(
            documents=int(row[0] or 0),
            documents_limit=self._settings.max_docs_per_user,
            storage_mb=round(int(row[1] or 0) / (1024 * 1024), 2),
            storage_mb_limit=self._settings.max_storage_mb_per_user,
            chunks=int(row[2] or 0),
            chunks_limit=self._settings.max_chunks_per_user,
        )
        return _user_out(user, quota)

    # ------------------------------------------------------------ account deletion (§5.8.7, E41)

    async def delete_me(self, ctx: TenantContext, documents: DocumentService | None) -> None:
        """Bump `token_version`, delete every document through the DocumentService, then the user.

        The version bump is committed first, so every token dies even if a later step fails. A
        failure surfaces as 503; the user can sign in again and retry (the reconciler also finishes).
        """
        mine = ctx.with_scope(Scope.MINE)
        async with self._db.session(_AUTH_CTX) as s:
            user = await s.get(User, ctx.user_id)
            if user is None:
                raise AppError(ErrorCode.UNAUTHORIZED, headers=dict(BEARER_CHALLENGE))
            if user.role == Role.ADMIN.value:
                admins = await s.scalar(select(func.count(User.id)).where(User.role == Role.ADMIN.value))
                if (admins or 0) <= 1:
                    raise AppError(ErrorCode.LAST_ADMIN, "The last admin account cannot be deleted.")
            user.token_version += 1
        # Same step, per §5.8.7: the user's documents are marked DELETING so they stop being served.
        async with self._db.session(mine) as s:
            await s.execute(update(Document).where(Document.user_id == ctx.user_id).values(status="DELETING"))

        doc_ids = await self._document_ids(ctx, documents)
        if doc_ids and documents is None:
            log.error("auth.delete_me.no_document_service", user_id=ctx.user_id, documents=len(doc_ids))
            raise AppError(ErrorCode.DEPENDENCY_UNAVAILABLE, "Document storage is unavailable. Try again.")
        failed = 0
        for doc_id in doc_ids:
            assert documents is not None
            try:
                await documents.delete(ctx, doc_id)
            except AppError as exc:
                if exc.code is not ErrorCode.NOT_FOUND:  # already gone counts as deleted
                    failed += 1
            except Exception:
                failed += 1
        if failed:
            log.error("auth.delete_me.partial", user_id=ctx.user_id, failed=failed, total=len(doc_ids))
            raise AppError(
                ErrorCode.DEPENDENCY_UNAVAILABLE, "Some documents could not be deleted. Try again."
            )

        async with self._db.session(mine) as s:
            await s.execute(delete(Feedback).where(Feedback.user_id == ctx.user_id))
            await s.execute(delete(RequestLog).where(RequestLog.user_id == ctx.user_id))
            await s.execute(delete(Document).where(Document.user_id == ctx.user_id))
        async with self._db.session(_AUTH_CTX) as s:
            await s.execute(delete(User).where(User.id == ctx.user_id))
        log.info("auth.delete_me", user_id=ctx.user_id, documents=len(doc_ids))

    async def _document_ids(self, ctx: TenantContext, documents: DocumentService | None) -> list[str]:
        """Every document the user owns: from the DocumentService when present, plus the DB rows."""
        mine = ctx.with_scope(Scope.MINE)
        ids: list[str] = []
        if documents is not None:
            cursor: str | None = None
            while True:
                page = await documents.list(
                    mine,
                    agency=None,
                    jurisdiction=None,
                    topic=None,
                    language=None,
                    status=None,
                    limit=DELETE_PAGE_SIZE,
                    cursor=cursor,
                )
                ids.extend(item.id for item in page.items)
                cursor = page.next_cursor
                if not cursor or not page.items:
                    break
        async with self._db.session(mine) as s:
            rows = (await s.scalars(select(Document.id).where(Document.user_id == ctx.user_id))).all()
        seen = set(ids)
        ids.extend(doc_id for doc_id in rows if doc_id not in seen)
        return ids

    # ------------------------------------------------------------ admin CLI (scripts/create_admin.py)

    async def find_user(self, email: str) -> UserOut | None:
        async with self._db.session(_AUTH_CTX) as s:
            user = await s.scalar(select(User).where(User.email == normalize_email(email)))
        return _user_out(user) if user else None

    async def create_or_promote_admin(
        self, email: str, password: str | None, *, display_name: str | None = None
    ) -> tuple[UserOut, bool]:
        """Promote an existing account, or create a new admin (then `password` is required).

        Returns (user, created). This is the only way to grant the admin role (§5.8.1).
        """
        normalized = normalize_email(email)
        async with self._db.session(_AUTH_CTX) as s:
            user = await s.scalar(select(User).where(User.email == normalized))
            if user is not None:
                user.role = Role.ADMIN.value
                log.warning("auth.admin_promoted", user_id=user.id)
                return _user_out(user), False
        if not password:
            raise ValueError("a password is required to create a new admin")
        SignupRequest(email=normalized, password=password, display_name=display_name)  # same rules
        user = User(
            id=new_user_id(),
            email=normalized,
            password_hash=await asyncio.to_thread(hash_password, password),
            display_name=display_name,
            role=Role.ADMIN.value,
            token_version=0,
        )
        async with self._db.session(_AUTH_CTX) as s:
            s.add(user)
        log.warning("auth.admin_created", user_id=user.id)
        return _user_out(user), True

    async def demote_admin(self, email: str) -> UserOut:
        """Admin → user. Bumps `token_version`, so the old tokens die at once (E42)."""
        async with self._db.session(_AUTH_CTX) as s:
            user = await s.scalar(select(User).where(User.email == normalize_email(email)))
            if user is None:
                raise AppError(ErrorCode.NOT_FOUND, "No account with this email.")
            if user.role == Role.ADMIN.value:
                admins = await s.scalar(select(func.count(User.id)).where(User.role == Role.ADMIN.value))
                if (admins or 0) <= 1:
                    raise AppError(ErrorCode.LAST_ADMIN, "The last admin cannot be demoted.")
            user.role = Role.USER.value
            user.token_version += 1
            log.warning("auth.admin_demoted", user_id=user.id)
            return _user_out(user)
