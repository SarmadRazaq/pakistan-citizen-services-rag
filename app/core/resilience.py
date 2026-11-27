"""Small resilience primitives for outbound calls (blueprint §3.7, §12.2).

- `CircuitBreaker`: opens after N consecutive failures, half-opens after a cool-down.
- `TokenBucket`: client-side request-rate limiter (reservation style, so it never busy-waits).
- `backoff_delay` / `parse_retry_after`: exponential backoff with jitter that honours `Retry-After`.

Clocks and sleeps are injectable, so tests run instantly and deterministically.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

Clock = Callable[[], float]
Sleep = Callable[[float], Awaitable[None]]


class CircuitBreaker:
    """closed → (N consecutive failures) → open → (cool-down elapsed) → half-open → one trial call.

    A successful trial closes the breaker; a failed trial re-opens it for another cool-down.
    """

    def __init__(
        self, failure_threshold: int = 3, reset_after_s: float = 60.0, clock: Clock = time.monotonic
    ) -> None:
        self.failure_threshold = failure_threshold
        self.reset_after_s = reset_after_s
        self._clock = clock
        self._failures = 0
        self._opened_at: float | None = None
        self._trial_in_flight = False

    @property
    def state(self) -> str:
        if self._opened_at is None:
            return "closed"
        if self._clock() - self._opened_at >= self.reset_after_s:
            return "half_open"
        return "open"

    def allow(self) -> bool:
        state = self.state
        if state == "closed":
            return True
        if state == "half_open" and not self._trial_in_flight:
            self._trial_in_flight = True
            return True
        return False

    def seconds_until_retry(self) -> float:
        if self._opened_at is None:
            return 0.0
        return max(0.0, self.reset_after_s - (self._clock() - self._opened_at))

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None
        self._trial_in_flight = False

    def record_failure(self) -> None:
        self._failures += 1
        if self._trial_in_flight or self._failures >= self.failure_threshold:
            self._opened_at = self._clock()
        self._trial_in_flight = False


class TokenBucket:
    """`rate_per_minute` tokens per minute with a small burst. `acquire()` reserves a token and
    sleeps until it is due, so concurrent callers queue fairly instead of spinning."""

    def __init__(
        self,
        rate_per_minute: float,
        burst: int = 5,
        *,
        clock: Clock = time.monotonic,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        if rate_per_minute <= 0:
            raise ValueError("rate_per_minute must be positive")
        self._rate_per_s = rate_per_minute / 60.0
        self._capacity = float(max(1, burst))
        self._tokens = self._capacity
        self._clock = clock
        self._sleep = sleep
        self._updated = clock()
        self._lock = asyncio.Lock()

    async def acquire(self) -> float:
        """Take one token; returns the seconds waited."""
        async with self._lock:
            now = self._clock()
            self._tokens = min(self._capacity, self._tokens + (now - self._updated) * self._rate_per_s)
            self._updated = now
            self._tokens -= 1.0
            wait = 0.0 if self._tokens >= 0 else -self._tokens / self._rate_per_s
        if wait > 0:
            await self._sleep(wait)
        return wait


def parse_retry_after(value: str | None, *, now: datetime | None = None) -> float | None:
    """`Retry-After` as delta-seconds or an HTTP date → seconds (None when absent or unparseable)."""
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - (now or datetime.now(UTC))).total_seconds())


def backoff_delay(
    retry: int,
    *,
    base_s: float = 0.5,
    cap_s: float = 8.0,
    retry_after_s: float | None = None,
    max_retry_after_s: float = 30.0,
    rng: random.Random | None = None,
) -> float:
    """Delay before retry number `retry` (0-based): `Retry-After` when given (capped), otherwise
    exponential backoff with ±50% jitter."""
    if retry_after_s is not None:
        return min(retry_after_s, max_retry_after_s)
    jitter = (rng or random).uniform(0.5, 1.5)
    return float(min(cap_s, base_s * (2**retry) * jitter))
