from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from sqlalchemy import delete, select

from app.core.tenancy import PUBLIC_OWNER_ID, Role, Scope, TenantContext, system_context
from app.db.models import Document
from app.db.session import Database


def _doc(doc_id: str, owner: str) -> Document:
    return Document(
        id=doc_id,
        user_id=owner,
        visibility="public" if owner == PUBLIC_OWNER_ID else "private",
        file_name=f"{doc_id}.pdf",
        file_type="pdf",
        size_bytes=10,
        file_sha256=doc_id.ljust(64, "0"),
    )


@pytest.fixture
async def db() -> AsyncIterator[Database]:
    database = Database("sqlite+aiosqlite:///:memory:")
    await database.create_all()
    async with database.session(system_context("test seed")) as s:
        s.add_all([_doc("d_pub", PUBLIC_OWNER_ID), _doc("d_a", "u_a"), _doc("d_b", "u_b")])
    yield database
    await database.dispose()


async def _ids(db: Database, ctx: TenantContext) -> set[str]:
    async with db.session(ctx) as s:
        return set((await s.scalars(select(Document.id))).all())


async def test_select_is_filtered_per_scope(db: Database) -> None:
    assert await _ids(db, TenantContext.for_user("u_a", Role.USER, Scope.ALL)) == {"d_pub", "d_a"}
    assert await _ids(db, TenantContext.for_user("u_a", Role.USER, Scope.MINE)) == {"d_a"}
    assert await _ids(db, TenantContext.for_user("u_a", Role.USER, Scope.PUBLIC)) == {"d_pub"}


async def test_get_by_id_of_other_users_document_returns_nothing(db: Database) -> None:
    ctx = TenantContext.for_user("u_a", Role.USER)
    async with db.session(ctx) as s:
        assert (await s.scalars(select(Document).where(Document.id == "d_b"))).first() is None


async def test_bulk_delete_cannot_touch_other_tenants_or_public(db: Database) -> None:
    ctx = TenantContext.for_user("u_a", Role.USER, Scope.ALL)  # can READ public, must not delete it
    async with db.session(ctx) as s:
        await s.execute(delete(Document))  # deliberately unfiltered by the caller
    assert await _ids(db, system_context("check")) == {"d_pub", "d_b"}


async def test_admin_bulk_delete_reaches_public_but_not_private(db: Database) -> None:
    admin = TenantContext.for_user("u_admin", Role.ADMIN, Scope.ALL)
    async with db.session(admin) as s:
        await s.execute(delete(Document))
    assert await _ids(db, system_context("check")) == {"d_a", "d_b"}
