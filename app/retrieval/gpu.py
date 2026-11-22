"""The GPU worker: serialises blocking model calls off the event loop (blueprint §12.3, §12.9).

The embedder and the reranker share one worker, so at most `gpu_max_concurrent` forward passes run
at once on the 6 GB RTX 3050. Each call runs in a dedicated thread pool, so the event loop stays free.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

T = TypeVar("T")


class GpuWorker:
    def __init__(self, max_concurrent: int = 1) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent, thread_name_prefix="gpu")
        self._closed = False

    async def run(self, fn: Callable[[], T]) -> T:
        """Run one blocking call (model load, encode, compute_score) under the semaphore."""
        if self._closed:
            raise RuntimeError("GpuWorker is shut down")
        async with self._semaphore:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self._executor, fn)

    def shutdown(self) -> None:
        self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)
