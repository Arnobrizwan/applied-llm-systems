"""Per-tenant rate limiting with a token bucket.

A token bucket, not a fixed window counter. A fixed window of 60 requests per
minute lets a caller send 60 at 11:59:59 and 60 more at 12:00:00, so the real
worst case is 120 requests inside one second and the backend sees a burst the
limit was supposed to prevent. A bucket separates the two things an operator
actually cares about: sustained rate (the refill) and burst tolerance (the
capacity). A sliding window log gives the same smoothness but stores a
timestamp per request, which is unbounded memory per tenant under attack; the
bucket is two floats.

The bucket also computes Retry-After. Returning 429 without telling the client
when to come back guarantees a hot retry loop, which is exactly the traffic
the limiter was trying to shed.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Tuple


@dataclass
class RateLimitConfig:
    burst: int          # bucket capacity, the largest instantaneous spike allowed
    per_second: float   # sustained refill rate

    def __post_init__(self) -> None:
        if self.burst <= 0:
            raise ValueError("burst must be positive")
        if self.per_second <= 0:
            raise ValueError("per_second must be positive")


class TokenBucket:
    """One bucket. Thread-safe because the HTTP server is threaded."""

    def __init__(self, config: RateLimitConfig, clock: Callable[[], float] = time.monotonic):
        self.config = config
        self.clock = clock
        self.tokens = float(config.burst)
        self.updated_at = clock()
        self._lock = threading.Lock()

    def _refill(self, now: float) -> None:
        elapsed = max(0.0, now - self.updated_at)
        self.tokens = min(float(self.config.burst), self.tokens + elapsed * self.config.per_second)
        self.updated_at = now

    def try_consume(self, cost: float = 1.0) -> Tuple[bool, float]:
        """Take `cost` tokens. Returns (allowed, retry_after_seconds).

        retry_after is 0.0 when allowed. When denied it is the exact time until
        the bucket holds enough tokens, so a well-behaved client wakes up once
        rather than polling.
        """
        with self._lock:
            now = self.clock()
            self._refill(now)
            if self.tokens >= cost:
                self.tokens -= cost
                return True, 0.0
            deficit = cost - self.tokens
            return False, deficit / self.config.per_second

    def peek(self) -> float:
        with self._lock:
            self._refill(self.clock())
            return self.tokens


class RateLimiter:
    """Bucket per tenant, created on first sight."""

    def __init__(self, default: RateLimitConfig, clock: Callable[[], float] = time.monotonic):
        self.default = default
        self.clock = clock
        self._overrides: Dict[str, RateLimitConfig] = {}
        self._buckets: Dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    def configure(self, tenant_id: str, config: RateLimitConfig) -> None:
        with self._lock:
            self._overrides[tenant_id] = config
            self._buckets.pop(tenant_id, None)

    def bucket(self, tenant_id: str) -> TokenBucket:
        with self._lock:
            b = self._buckets.get(tenant_id)
            if b is None:
                b = TokenBucket(self._overrides.get(tenant_id, self.default), self.clock)
                self._buckets[tenant_id] = b
            return b

    def check(self, tenant_id: str, cost: float = 1.0) -> Tuple[bool, int]:
        """Returns (allowed, retry_after_whole_seconds).

        Retry-After is an integer header, and it is rounded *up*: rounding down
        would tell the client to retry fractionally before the bucket has
        refilled, producing a second 429 and an unnecessary round trip.
        """
        allowed, wait = self.bucket(tenant_id).try_consume(cost)
        return allowed, 0 if allowed else max(1, int(math.ceil(wait)))

    def remaining(self, tenant_id: str) -> float:
        return self.bucket(tenant_id).peek()
