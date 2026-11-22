from __future__ import annotations

from typing import Literal

from fastapi import APIRouter

from app.api._stub import not_implemented
from app.api.deps import TenantDep
from app.api.responses import problems
from app.schemas import MetricsSummary

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get(
    "/summary",
    response_model=MetricsSummary,
    summary="Latency, tokens, cost and error rates (own for users, global for admins)",
    responses=problems(401),
)
async def summary(tenant: TenantDep, window: Literal["1h", "24h", "7d"] = "24h") -> MetricsSummary:
    not_implemented("A5")
