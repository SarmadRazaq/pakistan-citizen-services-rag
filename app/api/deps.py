"""Shared FastAPI dependencies: settings, services and the current tenant."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from app.config import Settings
from app.core.errors import AppError, ErrorCode
from app.core.health import HealthService
from app.core.tenancy import TenantContext


def get_app_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_health_service(request: Request) -> HealthService:
    service: HealthService = request.app.state.health
    return service


async def get_tenant(request: Request) -> TenantContext:
    """Resolve the JWT into a TenantContext.

    Contract only: work package A5 implements it (JWT `ver` check, role read from the DB,
    scope from the request). Until then every protected route answers 401.
    Tests override this dependency.
    """
    raise AppError(
        ErrorCode.UNAUTHORIZED,
        "Authentication is not implemented yet.",
        headers={"WWW-Authenticate": "Bearer"},
    )


SettingsDep = Annotated[Settings, Depends(get_app_settings)]
HealthDep = Annotated[HealthService, Depends(get_health_service)]
TenantDep = Annotated[TenantContext, Depends(get_tenant)]
