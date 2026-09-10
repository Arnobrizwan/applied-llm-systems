"""Retry with full-jitter exponential backoff, plus a circuit breaker.

Full jitter (sleep = random(0, base * 2**attempt)) rather than fixed backoff:
when a provider rate-limits a burst of callers, fixed backoff reconverges them
into a second identical burst. This is the same failure shape as a batch job
that retries on a fixed schedule and re-triggers the 429 it was retrying.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, Type


@dataclass
class RetryPolicy:
    attempts: int = 4
    base_delay: float = 0.05
    max_delay: float = 2.0
    jitter: str = "full"  # "full" | "none"

    def delay_for(self, attempt: int, rng: Optional[random.Random] = None) -> float:
        raw = min(self.max_delay, self.base_delay * (2 ** attempt))
        if self.jitter == "none":
            return raw
        return (rng or random).uniform(0.0, raw)


class RetryExhausted(RuntimeError):
    def __init__(self, attempts: int, last_error: BaseException):
        super().__init__(f"giving up after {attempts} attempt(s): {last_error}")
        self.attempts = attempts
        self.last_error = last_error


def retry(fn: Callable[[], Any], policy: Optional[RetryPolicy] = None,
          retry_on: Tuple[Type[BaseException], ...] = (Exception,),
          should_retry: Optional[Callable[[BaseException], bool]] = None,
          on_retry: Optional[Callable[[int, BaseException, float], None]] = None,
          sleep: Callable[[float], None] = time.sleep,
          rng: Optional[random.Random] = None) -> Any:
    """Call `fn`, retrying on `retry_on`. Non-retryable errors surface at once."""
    pol = policy or RetryPolicy()
    last: Optional[BaseException] = None
    for attempt in range(pol.attempts):
        try:
            return fn()
        except retry_on as exc:  # type: ignore[misc]
            last = exc
            retryable = should_retry(exc) if should_retry else getattr(exc, "retryable", True)
            if not retryable or attempt == pol.attempts - 1:
                break
            delay = pol.delay_for(attempt, rng)
            if on_retry:
                on_retry(attempt + 1, exc, delay)
            sleep(delay)
    raise RetryExhausted(pol.attempts, last or RuntimeError("unknown"))


class CircuitBreaker:
    """Closed -> open after N consecutive failures -> half-open after cooldown."""

    def __init__(self, failure_threshold: int = 5, cooldown_s: float = 30.0,
                 clock: Callable[[], float] = time.monotonic):
        self.failure_threshold = failure_threshold
        self.cooldown_s = cooldown_s
        self.clock = clock
        self.failures = 0
        self.opened_at: Optional[float] = None

    @property
    def state(self) -> str:
        if self.opened_at is None:
            return "closed"
        return "half_open" if self.clock() - self.opened_at >= self.cooldown_s else "open"

    def allow(self) -> bool:
        return self.state != "open"

    def record_success(self) -> None:
        self.failures = 0
        self.opened_at = None

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.opened_at = self.clock()
