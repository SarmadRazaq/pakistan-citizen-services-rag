"""`MetricsService`: the numbers behind GET /metrics/summary (bonus B3).

Everything is computed from `request_log` rows in the window. Percentiles are computed in Python
over the window's latencies (linear interpolation): at this scale (thousands of rows per week)
that is cheaper than portable SQL percentile functions across SQLite and Postgres.

Definitions:
- requests: every /ask and /search row in the window.
- p50 / p95: of `latency_ms` over those requests.
- error_rate: share of requests whose status is 5xx (server-side failures, including "all models
  unavailable"). 4xx are caller mistakes (validation, auth, rate limit) and are not errors here.
- fallback_rate: share of /ask completions answered in degraded mode (local fallback model).
- avg tokens, cost and list price per question: over completed /ask requests (status 200), refusals
  included because they still spent tokens. cost = actual spend (0 on the NVIDIA free tier);
  list price = tokens x configured model prices ("what it would cost").
- provider_mix (admins only): completed requests per `llm_model`.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from typing import Literal

from sqlalchemy import select

from app.core.tenancy import TenantContext, system_context
from app.db.models import RequestLog, utcnow
from app.db.session import Database
from app.schemas import MetricsSummary

Window = Literal["1h", "24h", "7d"]
WINDOWS: dict[str, timedelta] = {
    "1h": timedelta(hours=1),
    "24h": timedelta(hours=24),
    "7d": timedelta(days=7),
}


def percentile(values: Sequence[float], q: float) -> float | None:
    """`q` in [0, 100], linear interpolation between closest ranks; None for no data."""
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo))


class MetricsService:
    def __init__(self, db: Database, clock: Callable[[], datetime] = utcnow) -> None:
        self.db = db
        self.clock = clock

    async def summary(self, ctx: TenantContext, window: Window = "24h") -> MetricsSummary:
        since = self.clock() - WINDOWS[window]
        scope: Literal["own", "global"] = "global" if ctx.is_admin else "own"
        query = select(
            RequestLog.route,
            RequestLog.status,
            RequestLog.latency_ms,
            RequestLog.degraded,
            RequestLog.llm_model,
            RequestLog.input_tokens,
            RequestLog.output_tokens,
            RequestLog.cost_usd,
            RequestLog.list_price_usd,
        ).where(RequestLog.created_at >= since)
        if scope == "own":
            query = query.where(RequestLog.user_id == ctx.user_id)
        # Admins read every user's rows, which the per-tenant SQL filter would hide.
        reader = system_context("metrics.global") if scope == "global" else ctx
        async with self.db.session(reader) as s:
            rows = (await s.execute(query)).all()

        total = len(rows)
        if total == 0:
            return MetricsSummary(
                window=window, scope=scope, requests=0, provider_mix={} if scope == "global" else None
            )

        completed = [r for r in rows if r.route == "ask" and r.status == 200]
        n = len(completed)
        latencies = [float(r.latency_ms) for r in rows]
        return MetricsSummary(
            window=window,
            scope=scope,
            requests=total,
            p50_latency_ms=percentile(latencies, 50),
            p95_latency_ms=percentile(latencies, 95),
            avg_input_tokens=sum(r.input_tokens for r in completed) / n if n else None,
            avg_output_tokens=sum(r.output_tokens for r in completed) / n if n else None,
            cost_per_question_usd=sum(r.cost_usd for r in completed) / n if n else 0.0,
            list_price_per_question_usd=sum(r.list_price_usd for r in completed) / n if n else 0.0,
            error_rate=sum(1 for r in rows if r.status >= 500) / total,
            fallback_rate=sum(1 for r in completed if r.degraded) / n if n else 0.0,
            provider_mix=dict(Counter(r.llm_model for r in completed if r.llm_model))
            if scope == "global"
            else None,
        )
