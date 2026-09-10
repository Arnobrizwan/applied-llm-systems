"""Streaming latency measurement: TTFT and inter-token gaps.

Time to first token is the only latency number a user of a streaming interface
can feel. Total completion time is what a load test reports and it is close to
irrelevant here: a 4 second response that starts printing after 200ms feels
fast, and a 2 second response that arrives all at once after 2 seconds of a
spinner feels slow. Both numbers are collected, and TTFT is the one reported
first.

Inter-token latency is the second half of perceived quality. A stream with a
good average but a 900ms stall in the middle reads as broken, which is why the
p95 gap is tracked separately rather than folded into a mean. Percentiles come
from llmkit.tracing.percentile so this project reports them the same way the
rest of the repo does.

The clock is time.perf_counter, a monotonic high-resolution clock. time.time()
would be wrong here: it can step backwards on an NTP correction and produce a
negative inter-token gap.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from llmkit import percentile


@dataclass
class StreamMetrics:
    """Records one stream's timing. Reusable on both the client and the server."""

    label: str = "stream"
    clock: Callable[[], float] = time.perf_counter
    started_at: Optional[float] = None
    first_token_at: Optional[float] = None
    ended_at: Optional[float] = None
    token_times: List[float] = field(default_factory=list)
    reconnects: int = 0
    heartbeats: int = 0

    def start(self) -> "StreamMetrics":
        self.started_at = self.clock()
        return self

    def record_token(self) -> None:
        now = self.clock()
        if self.started_at is None:
            self.started_at = now
        if self.first_token_at is None:
            self.first_token_at = now
        self.token_times.append(now)

    def record_heartbeat(self) -> None:
        self.heartbeats += 1

    def record_reconnect(self) -> None:
        self.reconnects += 1

    def stop(self) -> "StreamMetrics":
        self.ended_at = self.clock()
        return self

    # -- derived ---------------------------------------------------------
    @property
    def ttft_ms(self) -> float:
        if self.started_at is None or self.first_token_at is None:
            return 0.0
        return (self.first_token_at - self.started_at) * 1000.0

    @property
    def total_ms(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.ended_at if self.ended_at is not None else self.clock()
        return (end - self.started_at) * 1000.0

    @property
    def gaps_ms(self) -> List[float]:
        """Inter-token deltas. Excludes TTFT, which is a different measurement."""
        return [(b - a) * 1000.0 for a, b in zip(self.token_times, self.token_times[1:])]

    @property
    def tokens(self) -> int:
        return len(self.token_times)

    @property
    def tokens_per_second(self) -> float:
        if self.tokens < 2 or self.first_token_at is None:
            return 0.0
        span = self.token_times[-1] - self.first_token_at
        return (self.tokens - 1) / span if span > 0 else 0.0

    def summary(self) -> Dict[str, float]:
        gaps = self.gaps_ms
        return {
            "tokens": self.tokens,
            "ttft_ms": round(self.ttft_ms, 2),
            "total_ms": round(self.total_ms, 2),
            "gap_p50_ms": round(percentile(gaps, 50), 2),
            "gap_p95_ms": round(percentile(gaps, 95), 2),
            "gap_max_ms": round(max(gaps), 2) if gaps else 0.0,
            "tokens_per_s": round(self.tokens_per_second, 2),
            "reconnects": self.reconnects,
            "heartbeats": self.heartbeats,
        }


def aggregate(runs: List[StreamMetrics]) -> Dict[str, float]:
    """Percentiles across several streams, which is what an SLO is written against."""
    ttfts = [m.ttft_ms for m in runs if m.tokens]
    all_gaps: List[float] = []
    for m in runs:
        all_gaps.extend(m.gaps_ms)
    return {
        "streams": len(runs),
        "ttft_p50_ms": round(percentile(ttfts, 50), 2),
        "ttft_p95_ms": round(percentile(ttfts, 95), 2),
        "gap_p50_ms": round(percentile(all_gaps, 50), 2),
        "gap_p95_ms": round(percentile(all_gaps, 95), 2),
        "total_tokens": sum(m.tokens for m in runs),
    }
