"""Shared FastAPI dependencies: settings, services and the current tenant.

Services are built once in `app/main.py` and stored on `app.state`. Routers depend on the
protocol types from `app/protocols.py`, never on concrete classes.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request, Response

from app.config import Settings
from app.core.errors import AppError, ErrorCode
from app.core.eval_overrides import APPLIED_HEADER, EVAL_HEADER, EvalOverrides, parse_eval_header
from app.core.health import HealthService
from app.core.tenancy import TenantContext
from app.db.session import Database
from app.observability.metrics import MetricsService
from app.protocols import AskService, DocumentService


def get_app_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_health_service(request: Request) -> HealthService:
    service: HealthService = request.app.state.health
    return service


def _state(request: Request, name: str, work_package: str) -> object:
    service = getattr(request.app.state, name, None)
    if service is None:
        raise AppError(ErrorCode.NOT_IMPLEMENTED, f"Implemented in work package {work_package}.")
    return service


def get_db(request: Request) -> Database:
    db = _state(request, "db", "foundations")
    assert isinstance(db, Database)
    return db


def get_document_service(request: Request) -> DocumentService:
    service = _state(request, "documents", "ingestion")
    assert isinstance(service, DocumentService)
    return service


def get_ask_service(request: Request) -> AskService:
    service = _state(request, "ask", "ask")
    assert isinstance(service, AskService)
    return service


def get_metrics_service(request: Request) -> MetricsService:
    service = _state(request, "metrics", "observability")
    assert isinstance(service, MetricsService)
    return service


async def get_tenant(request: Request) -> TenantContext:
    """Resolve the JWT into a TenantContext (scope ALL; services narrow it with ctx.with_scope).

    Contract: the auth work package replaces this body (JWT `ver` check, role read from the DB).
    Until then every protected route answers 401. Tests override this dependency.
    """
    from app.core.context import get_request_id

    auth = getattr(request.app.state, "auth", None)
    scheme, _, token = request.headers.get("Authorization", "").partition(" ")
    if auth is None or scheme.lower() != "bearer" or not token.strip():
        raise AppError(
            ErrorCode.UNAUTHORIZED,
            "Sign in to continue." if auth is not None else "Authentication is not configured.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    ctx: TenantContext = await auth.resolve(token.strip(), request_id=get_request_id())
    return ctx


def get_eval_overrides(
    request: Request,
    response: Response,
    settings: Annotated[Settings, Depends(get_app_settings)],
    tenant: Annotated[TenantContext, Depends(get_tenant)],
) -> EvalOverrides | None:
    """The `X-Eval-Config` overrides, or None.

    Honoured only when `EVAL_OVERRIDES_ENABLED=true` (default false, also in prod) and the caller is an
    admin; otherwise the header is ignored entirely (not even validated, no echo). When honoured, an
    invalid header is a 422 and `X-Eval-Config-Applied` carries the normalised overrides.
    """
    raw = request.headers.get(EVAL_HEADER)
    if raw is None or not settings.eval_overrides_enabled or not tenant.is_admin:
        return None
    overrides = parse_eval_header(raw, settings.generation_models)
    response.headers[APPLIED_HEADER] = overrides.applied_json()
    return overrides


SettingsDep = Annotated[Settings, Depends(get_app_settings)]
HealthDep = Annotated[HealthService, Depends(get_health_service)]
TenantDep = Annotated[TenantContext, Depends(get_tenant)]
DbDep = Annotated[Database, Depends(get_db)]
DocumentServiceDep = Annotated[DocumentService, Depends(get_document_service)]
AskServiceDep = Annotated[AskService, Depends(get_ask_service)]
EvalOverridesDep = Annotated[EvalOverrides | None, Depends(get_eval_overrides)]
MetricsServiceDep = Annotated[MetricsService, Depends(get_metrics_service)]
