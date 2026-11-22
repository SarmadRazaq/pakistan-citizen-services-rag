"""Compute-device selection for the local models (blueprint §3.4, §12.9).

`torch` is imported lazily, inside the functions, so the base install (CI, the CPU image) works
without the multi-GB ML stack from `requirements-ml.in`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog

from app.config import Device

log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    device: str  # "cuda:0" or "cpu"
    name: str | None = None  # GPU name, e.g. "NVIDIA GeForce RTX 3050 Laptop GPU"
    vram_total_mb: int | None = None
    vram_free_mb: int | None = None
    torch_available: bool = False

    @property
    def is_cuda(self) -> bool:
        return self.device.startswith("cuda")


def _import_torch() -> Any | None:
    try:
        import torch  # type: ignore[import-not-found, unused-ignore]
    except ImportError:
        return None
    return torch


def resolve_device(setting: Device) -> str:
    """`auto` → `cuda:0` when torch sees a GPU, else `cpu`. An explicit `cuda` without a GPU is an error."""
    if setting == "cpu":
        return "cpu"
    torch = _import_torch()
    has_cuda = bool(torch is not None and torch.cuda.is_available())
    if setting == "cuda":
        if not has_cuda:
            raise RuntimeError(
                "DEVICE=cuda but no CUDA device is available (is the CUDA build of torch installed? "
                "see requirements-ml.in)"
            )
        return "cuda:0"
    return "cuda:0" if has_cuda else "cpu"


def device_info(setting: Device) -> DeviceInfo:
    """The resolved device plus VRAM figures, for the startup log and notebook 03."""
    device = resolve_device(setting)
    torch = _import_torch()
    if torch is None or not device.startswith("cuda"):
        return DeviceInfo(device=device, torch_available=torch is not None)
    index = int(device.split(":", 1)[1]) if ":" in device else 0
    free_b, total_b = torch.cuda.mem_get_info(index)
    return DeviceInfo(
        device=device,
        name=str(torch.cuda.get_device_name(index)),
        vram_total_mb=int(total_b) // (1024 * 1024),
        vram_free_mb=int(free_b) // (1024 * 1024),
        torch_available=True,
    )


def log_device(setting: Device) -> DeviceInfo:
    info = device_info(setting)
    log.info(
        "device.selected",
        device=info.device,
        gpu=info.name,
        vram_total_mb=info.vram_total_mb,
        vram_free_mb=info.vram_free_mb,
    )
    return info


def max_memory_allocated_mb() -> int | None:
    """Peak VRAM used by this process so far (§12.9 measurements); None without CUDA."""
    torch = _import_torch()
    if torch is None or not torch.cuda.is_available():
        return None
    return int(torch.cuda.max_memory_allocated()) // (1024 * 1024)


def empty_cuda_cache() -> None:
    torch = _import_torch()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
