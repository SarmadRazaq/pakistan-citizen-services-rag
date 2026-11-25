"""Async engine/session factory with the structural SQL tenant filter (blueprint §5.8.2, layer 3).

Usage:
    async with db.session(ctx) as s:      # ctx: TenantContext
        rows = (await s.scalars(select(Document))).all()   # already filtered to ctx.readable_owner_ids

A session opened with a non-system context can only *read* rows whose `user_id` is readable.
Writes are checked by the services through `ctx.ensure_writable(...)`.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import ORMExecuteState, Session, with_loader_criteria
from sqlalchemy.pool import StaticPool

from app.core.tenancy import TenantContext
from app.db.models import Base, OwnedMixin

TENANT_KEY = "tenant"


@event.listens_for(Session, "do_orm_execute")
def _apply_tenant_filter(state: ORMExecuteState) -> None:
    ctx: TenantContext | None = state.session.info.get(TENANT_KEY)
    if ctx is None or ctx.is_system:
        return
    if state.is_select:
        owners = tuple(ctx.readable_owner_ids)
    elif state.is_update or state.is_delete:
        # Bulk writes only ever touch rows the tenant may *write*: their own, plus the public
        # corpus for admins. Reading the public corpus never grants writing it.
        owners = (ctx.user_id, ctx.public_owner_id) if ctx.is_admin else (ctx.user_id,)
    else:
        return
    state.statement = state.statement.options(
        with_loader_criteria(OwnedMixin, lambda cls: cls.user_id.in_(owners), include_aliases=True)
    )


class Database:
    def __init__(self, url: str) -> None:
        if url.startswith("sqlite") and ":///" in url:
            path = url.split(":///", 1)[1]
            if path and path != ":memory:":
                Path(path).parent.mkdir(parents=True, exist_ok=True)
        kwargs: dict[str, Any] = {}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
        self.engine: AsyncEngine = create_async_engine(url, **kwargs)
        self._factory = async_sessionmaker(self.engine, expire_on_commit=False)
        # In-memory SQLite shares ONE connection between all sessions, so one session's rollback or
        # close would undo another's uncommitted write. Serialise sessions there (background jobs
        # and requests overlap). File databases and Postgres give each session its own connection.
        self._shared_connection: asyncio.Lock | None = (
            asyncio.Lock() if isinstance(self.engine.pool, StaticPool) else None
        )

    async def create_all(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def dispose(self) -> None:
        await self.engine.dispose()

    @asynccontextmanager
    async def session(self, ctx: TenantContext) -> AsyncIterator[AsyncSession]:
        """A session bound to one tenant. Commits on success, rolls back on error."""
        if self._shared_connection is None:
            async with self._open(ctx) as s:
                yield s
        else:
            async with self._shared_connection, self._open(ctx) as s:
                yield s

    @asynccontextmanager
    async def _open(self, ctx: TenantContext) -> AsyncIterator[AsyncSession]:
        async with self._factory() as s:
            s.sync_session.info[TENANT_KEY] = ctx
            try:
                yield s
                await s.commit()
            except Exception:
                await s.rollback()
                raise
