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


def _http_probe(
    client: httpx.AsyncClient,
    url: str,
    *,
    models: bool = False,
    headers: dict[str, str] | None = None,
) -> ProbeFn:
    async def check() -> tuple[bool, str | None]:
        response = await client.get(url, headers=headers)
        if response.status_code != 200:
            return False, f"HTTP {response.status_code}"
        if models:
            data = response.json().get("data", [])
            return (len(data) > 0, f"{len(data)} models listed")
        return True, None

    return check


def _missing(detail: str) -> ProbeFn:
    async def check() -> tuple[bool, str | None]:
        return False, detail

    return check


def default_probes(settings: Settings, client: httpx.AsyncClient) -> list[Probe]:
    key = settings.nvidia_api_key
    nvidia_check = (
        _http_probe(
            client,
            f"{settings.nvidia_base_url.rstrip('/')}/models",
            models=True,
            headers={"Authorization": f"Bearer {key.get_secret_value()}"},
        )
        if key is not None and key.get_secret_value()
        else _missing("NVIDIA_API_KEY is not set")
    )
    return [
        Probe("qdrant", _http_probe(client, f"{settings.qdrant_url.rstrip('/')}/readyz")),
        Probe("nvidia_api", nvidia_check),
        Probe(
            "ollama",
            _http_probe(client, f"{settings.ollama_base_url.rstrip('/')}/models", models=True),
            required=settings.ollama_required,
        ),
    ]
