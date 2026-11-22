"""Readiness probes for external dependencies (blueprint §8.2 `/health/ready`).

Work packages register more probes here (database, GPU models loaded) as they land.
A probe returns (ok, detail); a probe that raises or times out counts as not ok.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import httpx

from app import __version__
from app.config import Settings
from app.schemas import ComponentHealth, HealthResponse

ProbeFn = Callable[[], Awaitable[tuple[bool, str | None]]]


@dataclass(frozen=True, slots=True)
class Probe:
    name: str
    check: ProbeFn
    required: bool = True


class HealthService:
    def __init__(self, probes: list[Probe], timeout_s: float) -> None:
        self._probes = probes
        self._timeout_s = timeout_s

    def register(self, probe: Probe) -> None:
        self._probes.append(probe)

    async def _run(self, probe: Probe) -> ComponentHealth:
        started = time.perf_counter()
        try:
            ok, detail = await asyncio.wait_for(probe.check(), timeout=self._timeout_s)
        except TimeoutError:
            ok, detail = False, f"timed out after {self._timeout_s:.1f}s"
        except Exception as exc:  # a probe must never crash readiness
            ok, detail = False, f"{type(exc).__name__}"
        return ComponentHealth(
            name=probe.name,
            ok=ok,
            required=probe.required,
            detail=detail,
            latency_ms=round((time.perf_counter() - started) * 1000),
        )

    async def readiness(self) -> HealthResponse:
        components = list(await asyncio.gather(*(self._run(p) for p in self._probes)))
        if any(c.required and not c.ok for c in components):
            status = "unavailable"
        elif any(not c.ok for c in components):
            status = "degraded"
        else:
            status = "ok"
        return HealthResponse(status=status, version=__version__, components=components)


def _http_probe(client: httpx.AsyncClient, url: str, *, models: bool = False) -> ProbeFn:
    async def check() -> tuple[bool, str | None]:
        response = await client.get(url)
        if response.status_code != 200:
            return False, f"HTTP {response.status_code}"
        if models:
            data = response.json().get("data", [])
            return (len(data) > 0, f"{len(data)} models listed")
        return True, None

    return check


def default_probes(settings: Settings, client: httpx.AsyncClient) -> list[Probe]:
    return [
        Probe("qdrant", _http_probe(client, f"{settings.qdrant_url.rstrip('/')}/readyz")),
        Probe(
            "antseed_proxy",
            _http_probe(client, f"{settings.antseed_base_url.rstrip('/')}/models", models=True),
        ),
        Probe(
            "ollama",
            _http_probe(client, f"{settings.ollama_base_url.rstrip('/')}/models", models=True),
            required=settings.ollama_required,
        ),
    ]
