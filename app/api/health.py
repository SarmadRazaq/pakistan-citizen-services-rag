from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app import __version__
from app.api.deps import HealthDep
from app.schemas import HealthResponse

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live", response_model=HealthResponse, summary="Liveness")
async def live() -> HealthResponse:
    """The process is up. Checks nothing external."""
    return HealthResponse(status="ok", version=__version__)


@router.get(
    "/ready",
    response_model=HealthResponse,
    summary="Readiness",
    responses={503: {"model": HealthResponse, "description": "A required dependency is down"}},
)
async def ready(health: HealthDep) -> JSONResponse:
    """Qdrant, the NVIDIA API (key + model list) and Ollama. 503 when a required one is down."""
    report = await health.readiness()
    status_code = 503 if report.status == "unavailable" else 200
    return JSONResponse(status_code=status_code, content=report.model_dump(mode="json"))
