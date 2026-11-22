"""App factory: wiring only, no business logic (blueprint §2.3)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api import ask as ask_api
from app.api import auth as auth_api
from app.api import documents as documents_api
from app.api import health as health_api
from app.api import metrics as metrics_api
from app.config import Settings, get_settings
from app.core.errors import install_error_handlers
from app.core.health import HealthService, default_probes
from app.core.logging import configure_logging
from app.core.middleware import RequestContextMiddleware


def create_app(settings: Settings | None = None, *, health_service: HealthService | None = None) -> FastAPI:
    """Build the app. Tests pass their own settings and fake services."""
    settings = settings or get_settings()
    configure_logging(settings.log_level, json=settings.app_env != "dev")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with httpx.AsyncClient(timeout=settings.health_probe_timeout_s) as client:
            if health_service is None:
                app.state.health = HealthService(
                    default_probes(settings, client), settings.health_probe_timeout_s
                )
            yield

    app = FastAPI(
        title="Citizen Services RAG",
        version=__version__,
        summary="Cited answers about Pakistani citizen services and rights, in English, Urdu and Roman Urdu.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    if health_service is not None:
        app.state.health = health_service

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

    app.include_router(health_api.router)
    for router_module in (auth_api, documents_api, ask_api, metrics_api):
        app.include_router(router_module.router, prefix=settings.api_prefix)
    return app


def app_factory() -> FastAPI:
    """Entry point for `uvicorn app.main:app_factory --factory`."""
    return create_app()
