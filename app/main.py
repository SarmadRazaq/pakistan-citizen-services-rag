"""App factory: wiring only, no business logic (blueprint §2.3)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api import ask as ask_api
from app.api import auth as auth_api
from app.api import documents as documents_api
from app.api import health as health_api
from app.api import metrics as metrics_api
from app.config import Settings, get_settings
from app.core.auth_service import AuthService
from app.core.errors import install_error_handlers
from app.core.health import HealthService, default_probes
from app.core.logging import configure_logging
from app.core.middleware import RequestContextMiddleware, SecurityHeadersMiddleware
from app.core.ratelimit import install_rate_limits
from app.db.session import Database
from app.observability.metrics import MetricsService

log = structlog.get_logger(__name__)


async def _wire_runtime(app: FastAPI, settings: Settings, db: Database) -> list[object]:
    """Build the heavy services (Qdrant, models, LLM clients). Returns objects to close on shutdown.

    Imported lazily so tests and tooling that only need the API schema never load model code.
    """
    from app.generation.llm import build_llm_clients
    from app.generation.service import build_ask_service
    from app.guardrails.input_guards import build_injection_classifier
    from app.retrieval.factory import build_retrieval

    closables: list[object] = []
    retrieval = build_retrieval(settings)
    try:
        await retrieval.startup()
    except Exception as exc:  # Qdrant down at boot: keep serving, readiness reports it
        log.warning("startup.retrieval_unavailable", error=type(exc).__name__)
    closables.append(retrieval)
    app.state.retrieval = retrieval

    hosted, local = build_llm_clients(settings)  # never eval_cache=True for serving
    closables.extend([hosted, local])
    app.state.llm, app.state.llm_local = hosted, local

    classifier = build_injection_classifier(settings)
    if classifier is not None:
        closables.append(classifier)
    app.state.ask = build_ask_service(
        settings, db, retrieval.retriever, llm_hosted=hosted, llm_local=local, classifier=classifier
    )

    from app.ingestion.service import IngestionService

    documents = IngestionService(settings, db, retrieval.store, retrieval.embedder)
    app.state.documents = documents
    closables.append(documents)  # cancels running background jobs at shutdown
    return closables


async def _close(objs: list[object]) -> None:
    for obj in reversed(objs):
        for name in ("shutdown", "aclose"):
            fn = getattr(obj, name, None)
            if fn is not None:
                try:
                    await fn()
                except Exception as exc:
                    log.warning("shutdown.close_failed", obj=type(obj).__name__, error=type(exc).__name__)
                break


def create_app(
    settings: Settings | None = None,
    *,
    health_service: HealthService | None = None,
    wire_services: bool | None = None,
) -> FastAPI:
    """Build the app.

    Tests pass their own settings and fake services (`health_service=...`). That turns off service
    wiring, so the tests set `app.state.<service>` themselves. The real app (`app_factory`) wires
    the DB, auth, rate limits, retrieval, LLM clients, ingestion and the ask pipeline.
    """
    settings = settings or get_settings()
    wire = (health_service is None) if wire_services is None else wire_services
    configure_logging(settings.log_level, json=settings.app_env != "dev")

    db = Database(settings.database_url) if wire else None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        closables: list[object] = []
        async with httpx.AsyncClient(timeout=settings.health_probe_timeout_s) as client:
            if health_service is None:
                app.state.health = HealthService(
                    default_probes(settings, client), settings.health_probe_timeout_s
                )
            if db is not None:
                await db.create_all()
                closables = await _wire_runtime(app, settings, db)
            try:
                yield
            finally:
                await _close(closables)
                if db is not None:
                    await db.dispose()

    app = FastAPI(
        title="Citizen Services RAG",
        version=__version__,
        summary="Cited answers about Pakistani citizen services and rights, in English, Urdu and Roman Urdu.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    if health_service is not None:
        app.state.health = health_service
    if db is not None:
        app.state.db = db
        app.state.metrics = MetricsService(db)
        app.state.auth = AuthService(settings, db)  # refuses the default JWT secret in prod

    install_error_handlers(app)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "Location", "Retry-After"],
    )
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(SecurityHeadersMiddleware, hsts=settings.app_env == "prod")  # outermost
    if wire:
        install_rate_limits(app, settings)

    app.include_router(health_api.router)
    for router_module in (auth_api, documents_api, ask_api, metrics_api):
        app.include_router(router_module.router, prefix=settings.api_prefix)
    return app


def app_factory() -> FastAPI:
    """Entry point for `uvicorn app.main:app_factory --factory`."""
    return create_app()
