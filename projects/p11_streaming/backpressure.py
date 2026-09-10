"""A bounded channel between the token producer and the socket writer.

Why this exists at all: a model generates tokens at its own pace and a client
consumes them at its own pace, and those two rates are unrelated. The naive
implementation writes each delta straight to the socket from the generation
loop, which means one slow reader stalls generation and holds a worker thread.
The next-most-naive implementation puts an unbounded `queue.Queue` in between,
which does not stall, but a client that reads at half the generation rate now
accumulates the difference in RAM for the whole life of the stream. With a
thousand concurrent streams that is not a slow client, it is an out-of-memory
kill of the whole process. Unbounded buffering does not remove backpressure,
it converts it into a memory leak.

So the queue is bounded, and when it is full there are exactly two honest
choices. Both are implemented and the policy is configurable:

  BLOCK  the producer waits for room. Nothing is lost, the stream is complete,
         but generation runs at the consumer's speed and a worker stays busy.
         Correct for anything where every token matters (a chat completion the
         user is reading, an answer being persisted).

  DROP   the producer discards the delta and records a gap. Generation runs at
         full speed and memory is bounded, but the stream has holes. Only
         acceptable when the payload is refreshable state rather than an
         append-only token sequence: live progress percentages, telemetry, a
         partial-render preview. This implementation makes the loss explicit by
         counting gaps and emitting a `gap` event, because silently dropping
         tokens is how you ship a truncated answer and never find out.

A third option, close the stream with an error when the buffer fills, is a
reasonable production choice and is deliberately not implemented here: it is
just DROP with a lower tolerance, and it hides the interesting tradeoff.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, List, Optional

POLICY_BLOCK = "block"
POLICY_DROP = "drop"
POLICIES = (POLICY_BLOCK, POLICY_DROP)


@dataclass
class ChannelStats:
    accepted: int = 0
    dropped: int = 0
    gaps: int = 0                 # runs of consecutive drops, not individual drops
    max_depth: int = 0
    blocked_s: float = 0.0
    block_events: int = 0

    def to_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "dropped": self.dropped,
            "gaps": self.gaps,
            "max_depth": self.max_depth,
            "blocked_s": round(self.blocked_s, 4),
            "block_events": self.block_events,
        }


class BoundedChannel:
    """Single-producer, single-consumer channel with an explicit full-buffer policy.

    Written on a Condition rather than `queue.Queue` because the BLOCK policy
    needs to wake on either "room is available" or "the stream was cancelled",
    and `queue.Queue.put(timeout=...)` can only express the first. Polling a
    Queue with a short timeout would work but adds latency to every token in
    the common case where the buffer is not full.
    """

    def __init__(self, maxsize: int = 8, policy: str = POLICY_BLOCK):
        if maxsize <= 0:
            raise ValueError("maxsize must be positive")
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")
        self.maxsize = maxsize
        self.policy = policy
        self.stats = ChannelStats()
        self._items: List[Any] = []
        self._closed = False
        self._cancelled = False
        self._last_was_drop = False
        self._cond = threading.Condition()

    # -- producer side ---------------------------------------------------
    def put(self, item: Any, timeout: Optional[float] = None) -> bool:
        """Offer an item. Returns True if it was buffered, False if dropped.

        Under BLOCK this waits and returns True (or False if the stream was
        cancelled while waiting, which is how a disconnect unblocks a stalled
        producer instead of leaking the thread).
        """
        with self._cond:
            if self._closed or self._cancelled:
                return False
            if len(self._items) >= self.maxsize:
                if self.policy == POLICY_DROP:
                    self.stats.dropped += 1
                    if not self._last_was_drop:
                        self.stats.gaps += 1
                    self._last_was_drop = True
                    return False
                started = time.perf_counter()
                self.stats.block_events += 1
                deadline = None if timeout is None else started + timeout
                while len(self._items) >= self.maxsize and not (self._closed or self._cancelled):
                    remaining = None if deadline is None else max(0.0, deadline - time.perf_counter())
                    if remaining == 0.0:
                        break
                    self._cond.wait(remaining if remaining is not None else 0.25)
                self.stats.blocked_s += time.perf_counter() - started
                if self._closed or self._cancelled or len(self._items) >= self.maxsize:
                    return False
            self._items.append(item)
            self._last_was_drop = False
            self.stats.accepted += 1
            self.stats.max_depth = max(self.stats.max_depth, len(self._items))
            self._cond.notify_all()
            return True

    def close(self) -> None:
        """Producer is finished. The consumer still drains what is buffered."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def cancel(self) -> None:
        """Client is gone. Unblocks a waiting producer and abandons the buffer."""
        with self._cond:
            self._cancelled = True
            self._items.clear()
            self._cond.notify_all()

    # -- consumer side ---------------------------------------------------
    def get(self, timeout: Optional[float] = 0.1) -> Any:
        """Pop the next item, or None if nothing arrived before `timeout`.

        None means "nothing yet", which is what triggers a heartbeat upstream.
        Use `finished` to distinguish an idle stream from a completed one.
        """
        with self._cond:
            if not self._items and not (self._closed or self._cancelled):
                self._cond.wait(timeout)
            if self._items:
                item = self._items.pop(0)
                self._cond.notify_all()
                return item
            return None

    @property
    def finished(self) -> bool:
        with self._cond:
            return (self._closed and not self._items) or self._cancelled

    @property
    def cancelled(self) -> bool:
        with self._cond:
            return self._cancelled

    def depth(self) -> int:
        with self._cond:
            return len(self._items)
