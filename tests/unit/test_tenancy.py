from __future__ import annotations

import dataclasses

import pytest
from hypothesis import given
from hypothesis import strategies as st

from app.core.errors import AppError, ErrorCode
from app.core.tenancy import (
    PUBLIC_OWNER_ID,
    SYSTEM_USER_ID,
    Role,
    Scope,
    TenantContext,
    Visibility,
    system_context,
)


@pytest.mark.parametrize(
    ("scope", "expected"),
    [
        (Scope.ALL, {"u_a", PUBLIC_OWNER_ID}),
        (Scope.MINE, {"u_a"}),
        (Scope.PUBLIC, {PUBLIC_OWNER_ID}),
    ],
)
def test_readable_owner_ids_per_scope(scope: Scope, expected: set[str]) -> None:
    ctx = TenantContext.for_user("u_a", Role.USER, scope)
    assert ctx.readable_owner_ids == frozenset(expected)


def test_context_is_immutable(user_a: TenantContext) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        user_a.user_id = "u_b"  # type: ignore[misc]


@pytest.mark.parametrize("reserved", ["", PUBLIC_OWNER_ID, SYSTEM_USER_ID])
def test_reserved_ids_cannot_be_request_tenants(reserved: str) -> None:
    with pytest.raises(ValueError, match="reserved"):
        TenantContext.for_user(reserved, Role.ADMIN)


def test_user_cannot_read_other_private_and_gets_404(user_a: TenantContext) -> None:
    assert not user_a.can_read("u_b")
    with pytest.raises(AppError) as err:
        user_a.ensure_readable("u_b")
    assert err.value.code is ErrorCode.NOT_FOUND


def test_user_reads_public_but_cannot_write_it(user_a: TenantContext) -> None:
    user_a.ensure_readable(PUBLIC_OWNER_ID)
    with pytest.raises(AppError) as err:
        user_a.ensure_writable(PUBLIC_OWNER_ID)
    assert err.value.code is ErrorCode.ADMIN_REQUIRED


def test_admin_writes_public(admin: TenantContext) -> None:
    admin.ensure_writable(PUBLIC_OWNER_ID)
    assert admin.write_owner_for(Visibility.PUBLIC) == PUBLIC_OWNER_ID


def test_non_admin_public_upload_is_403(user_a: TenantContext) -> None:
    with pytest.raises(AppError) as err:
        user_a.write_owner_for(Visibility.PUBLIC)
    assert err.value.code is ErrorCode.ADMIN_REQUIRED
    assert user_a.write_owner_for(Visibility.PRIVATE) == "u_a"


def test_admin_cannot_read_other_users_private_data(admin: TenantContext) -> None:
    # Admins manage the public corpus; they do not get access to private uploads.
    assert not admin.can_read("u_b")


def test_system_context_reads_and_writes_everything() -> None:
    ctx = system_context("unit test")
    assert ctx.is_system
    assert ctx.can_read("u_anyone")
    assert ctx.can_write(PUBLIC_OWNER_ID)


owner_ids = st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789_", min_size=1, max_size=12)


@given(me=owner_ids, other=owner_ids, scope=st.sampled_from(list(Scope)))
def test_never_reads_another_users_private_data(me: str, other: str, scope: Scope) -> None:
    if me in {PUBLIC_OWNER_ID, SYSTEM_USER_ID} or other in {me, PUBLIC_OWNER_ID}:
        return
    ctx = TenantContext.for_user(me, Role.USER, scope)
    assert not ctx.can_read(other)
    assert not ctx.can_write(other)
