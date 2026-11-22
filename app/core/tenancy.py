"""Tenant identity and access rules (blueprint §5.8).

`TenantContext` is a required positional argument on every service, repository and store method.
Isolation is structural: no data-access method exists without one. The only bypass is
`system_context()`, for the reconciler and the seed script, and every use of it is logged.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import structlog

from app.core.errors import AppError, ErrorCode

log = structlog.get_logger(__name__)

PUBLIC_OWNER_ID = "u_public"  # overridable through Settings.public_owner_id at wiring time
SYSTEM_USER_ID = "u_system"


class Role(StrEnum):
    USER = "user"
    ADMIN = "admin"


class Scope(StrEnum):
    """Which corpus a request reads from."""

    ALL = "all"  # the official corpus plus my uploads (default)
    MINE = "mine"  # my uploads only
    PUBLIC = "public"  # the official corpus only


class Visibility(StrEnum):
    PUBLIC = "public"
    PRIVATE = "private"


@dataclass(frozen=True, slots=True)
class TenantContext:
    """Who is asking, and which owners' data they may read."""

    user_id: str
    role: Role
    readable_owner_ids: frozenset[str]
    request_id: str | None = None
    public_owner_id: str = PUBLIC_OWNER_ID
    is_system: bool = False

    @classmethod
    def for_user(
        cls,
        user_id: str,
        role: Role,
        scope: Scope = Scope.ALL,
        *,
        request_id: str | None = None,
        public_owner_id: str = PUBLIC_OWNER_ID,
    ) -> TenantContext:
        if not user_id or user_id in {public_owner_id, SYSTEM_USER_ID}:
            raise ValueError("reserved or empty user_id cannot be used as a request tenant")
        readable = {
            Scope.ALL: frozenset({user_id, public_owner_id}),
            Scope.MINE: frozenset({user_id}),
            Scope.PUBLIC: frozenset({public_owner_id}),
        }[scope]
        return cls(
            user_id=user_id,
            role=role,
            readable_owner_ids=readable,
            request_id=request_id,
            public_owner_id=public_owner_id,
        )

    @property
    def is_admin(self) -> bool:
        return self.role is Role.ADMIN

    def can_read(self, owner_id: str) -> bool:
        return self.is_system or owner_id in self.readable_owner_ids

    def can_write(self, owner_id: str) -> bool:
        """Owners write their own data; only admins write the public corpus."""
        if self.is_system:
            return True
        if owner_id == self.public_owner_id:
            return self.is_admin
        return owner_id == self.user_id

    def write_owner_for(self, visibility: Visibility) -> str:
        """The `user_id` a new document gets. Raises 403 when a non-admin targets the public corpus."""
        if visibility is Visibility.PUBLIC:
            if not (self.is_admin or self.is_system):
                raise AppError(ErrorCode.ADMIN_REQUIRED, "Only admins can add to the official corpus.")
            return self.public_owner_id
        return self.user_id

    def ensure_readable(self, owner_id: str) -> None:
        """Another user's private resource is reported as 404, never 403, so its existence stays hidden."""
        if not self.can_read(owner_id):
            raise AppError(ErrorCode.NOT_FOUND, "Resource not found.")

    def ensure_writable(self, owner_id: str) -> None:
        self.ensure_readable(owner_id)
        if not self.can_write(owner_id):
            raise AppError(ErrorCode.ADMIN_REQUIRED, "Only admins can change the official corpus.")


def system_context(reason: str, *, public_owner_id: str = PUBLIC_OWNER_ID) -> TenantContext:
    """Full-access context for the reconciler and seed script only. Every use is logged."""
    log.warning("security.system_context_used", reason=reason)
    return TenantContext(
        user_id=SYSTEM_USER_ID,
        role=Role.ADMIN,
        readable_owner_ids=frozenset(),
        public_owner_id=public_owner_id,
        is_system=True,
    )
