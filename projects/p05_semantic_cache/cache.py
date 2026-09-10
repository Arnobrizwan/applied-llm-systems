"""A semantic cache with TTL, LRU eviction, tenant isolation and a safety gate.

An exact-match cache on the query string has a hit rate close to zero on natural
language, because two users never type the same sentence. A semantic cache
matches on embedding similarity instead, which is what makes it useful and what
makes it dangerous: the same fuzziness that lets it serve "what's the per
workspace rate limit" from "what is the default rate limit per workspace" will
also serve "how long is the paid trial" from "how long is the free trial".

The design here treats a false hit as a much more expensive event than a miss. A
miss costs one model call. A false hit puts a confident, well-formed, wrong
answer in front of a user with no signal that anything went wrong, and it keeps
doing it for the whole TTL.

Eviction and isolation
----------------------
The entry cap is **per namespace**, not global. A global LRU cap looks simpler and
introduces a noisy-neighbour bug: a high-traffic tenant fills the cache and
evicts a quiet tenant's entries, so the quiet tenant's costs go up because of
someone else's traffic, and nothing in their own metrics explains it. Per
namespace caps make each tenant's cache behaviour a function of their own load.

Namespaces are also the prompt-version boundary. A namespace of
`{tenant}:{prompt_version}` means shipping a new system prompt cannot serve
answers generated under the old one, which otherwise happens silently and is
extremely hard to diagnose from the outside.

Lookup is a linear cosine scan within one namespace. At the sizes a cache like
this holds per tenant that is fine and it is honest; past roughly ten thousand
entries per namespace it wants an ANN index, and the `lookup` signature does not
change when it gets one.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from llmkit import Embedder, cosine, count_tokens, get_embedder

from .salience import SalienceGuard, SalienceReport


@dataclass
class CacheEntry:
    namespace: str
    query: str
    answer: str
    vector: List[float]
    created_at: float
    last_used_at: float
    hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class Lookup:
    """The result of one cache probe, including why a near hit was refused."""

    hit: bool
    reason: str
    similarity: float = 0.0
    entry: Optional[CacheEntry] = None
    candidate: Optional[CacheEntry] = None
    salience: Optional[SalienceReport] = None
    lookup_ms: float = 0.0

    @property
    def answer(self) -> Optional[str]:
        return self.entry.answer if self.entry else None


@dataclass
class CacheStats:
    lookups: int = 0
    hits: int = 0
    misses_below_threshold: int = 0
    misses_empty: int = 0
    blocked_by_salience: int = 0
    expired: int = 0
    evicted_lru: int = 0
    writes: int = 0
    tokens_saved: int = 0
    latency_saved_ms: float = 0.0
    lookup_ms_total: float = 0.0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0

    @property
    def avg_lookup_ms(self) -> float:
        return self.lookup_ms_total / self.lookups if self.lookups else 0.0

    def as_lines(self) -> List[str]:
        return [
            f"lookups              : {self.lookups}",
            f"hits                 : {self.hits} ({self.hit_rate:.2%})",
            f"misses below threshold: {self.misses_below_threshold}",
            f"blocked by salience  : {self.blocked_by_salience}",
            f"empty namespace      : {self.misses_empty}",
            f"expired entries      : {self.expired}",
            f"evicted (LRU)        : {self.evicted_lru}",
            f"tokens saved         : {self.tokens_saved}",
            f"model latency saved  : {self.latency_saved_ms:.1f} ms",
            f"avg lookup cost      : {self.avg_lookup_ms:.3f} ms",
        ]


class SemanticCache:
    def __init__(
        self,
        threshold: float = 0.85,
        ttl_s: float = 3600.0,
        max_entries: int = 256,
        embedder: Optional[Embedder] = None,
        guard: Optional[SalienceGuard] = None,
        use_guard: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ):
        if not -1.0 <= threshold <= 1.0:
            raise ValueError("threshold must be a cosine value in [-1, 1]")
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self.threshold = threshold
        self.ttl_s = ttl_s
        self.max_entries = max_entries
        self.embedder = embedder or get_embedder()
        self.guard = guard or SalienceGuard()
        self.use_guard = use_guard
        self.clock = clock
        self.stats = CacheStats()
        self._namespaces: Dict[str, "OrderedDict[str, CacheEntry]"] = {}

    # -- introspection ---------------------------------------------------
    def __len__(self) -> int:
        return sum(len(ns) for ns in self._namespaces.values())

    def size(self, namespace: str) -> int:
        return len(self._namespaces.get(namespace, ()))

    def namespaces(self) -> List[str]:
        return sorted(self._namespaces)

    def get_entry(self, namespace: str, query: str) -> Optional[CacheEntry]:
        """Exact-key fetch, for tests and for accounting what a write cost."""
        return self._namespaces.get(namespace, {}).get(query)

    # -- writes ----------------------------------------------------------
    def put(
        self,
        namespace: str,
        query: str,
        answer: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        latency_ms: float = 0.0,
        vector: Optional[List[float]] = None,
    ) -> CacheEntry:
        bucket = self._namespaces.setdefault(namespace, OrderedDict())
        now = self.clock()
        entry = CacheEntry(
            namespace=namespace,
            query=query,
            answer=answer,
            vector=vector if vector is not None else self.embedder.embed_one(query),
            created_at=now,
            last_used_at=now,
            prompt_tokens=prompt_tokens or count_tokens(query),
            completion_tokens=completion_tokens or count_tokens(answer),
            latency_ms=latency_ms,
        )
        # Keyed on the exact query text so a re-ask of the identical question
        # refreshes rather than duplicating. Semantically similar variants are
        # stored separately: collapsing them would throw away the wording that
        # the salience guard needs to compare against later.
        bucket[query] = entry
        bucket.move_to_end(query)
        self.stats.writes += 1
        self._evict(namespace)
        return entry

    def _evict(self, namespace: str) -> None:
        bucket = self._namespaces.get(namespace)
        if bucket is None:
            return
        while len(bucket) > self.max_entries:
            bucket.popitem(last=False)  # oldest touched entry
            self.stats.evicted_lru += 1

    def sweep(self) -> int:
        """Drop expired entries everywhere. Returns how many were removed."""
        now = self.clock()
        removed = 0
        for bucket in self._namespaces.values():
            for key in [k for k, e in bucket.items() if now - e.created_at >= self.ttl_s]:
                del bucket[key]
                removed += 1
        self.stats.expired += removed
        return removed

    def invalidate(self, namespace: str) -> int:
        """Drop one tenant's entries, for example after their documents change."""
        bucket = self._namespaces.pop(namespace, OrderedDict())
        return len(bucket)

    # -- reads -----------------------------------------------------------
    def lookup(self, namespace: str, query: str) -> Lookup:
        start = time.perf_counter()
        self.stats.lookups += 1
        bucket = self._namespaces.get(namespace)

        if not bucket:
            return self._finish(Lookup(False, "empty_namespace"), start, "misses_empty")

        # TTL is enforced on read as well as by sweep(). A cache that only expires
        # on a background sweep serves stale answers for however long the sweep
        # interval is, which is the one thing TTL exists to prevent.
        now = self.clock()
        stale = [k for k, e in bucket.items() if now - e.created_at >= self.ttl_s]
        for key in stale:
            del bucket[key]
        self.stats.expired += len(stale)
        if not bucket:
            return self._finish(Lookup(False, "all_entries_expired"), start, "misses_empty")

        qvec = self.embedder.embed_one(query)
        best_entry: Optional[CacheEntry] = None
        best_score = -1.0
        for entry in bucket.values():
            score = cosine(qvec, entry.vector)
            if score > best_score:
                best_score, best_entry = score, entry

        if best_entry is None or best_score < self.threshold:
            return self._finish(
                Lookup(False, "below_threshold", similarity=best_score, candidate=best_entry),
                start, "misses_below_threshold",
            )

        if self.use_guard:
            report = self.guard.compare(query, best_entry.query)
            if not report.ok:
                # Refusing here rather than falling through to the next-best entry
                # is deliberate. The runner-up is by definition less similar, so
                # if the closest match is semantically incompatible the others are
                # not better candidates, they are worse ones that happen to lack a
                # token the guard knows how to check.
                return self._finish(
                    Lookup(False, "blocked_by_salience", similarity=best_score,
                           candidate=best_entry, salience=report),
                    start, "blocked_by_salience",
                )

        best_entry.hits += 1
        best_entry.last_used_at = self.clock()
        bucket.move_to_end(best_entry.query)
        self.stats.hits += 1
        self.stats.tokens_saved += best_entry.total_tokens
        self.stats.latency_saved_ms += best_entry.latency_ms
        return self._finish(
            Lookup(True, "hit", similarity=best_score, entry=best_entry), start, None
        )

    def _finish(self, result: Lookup, start: float, counter: Optional[str]) -> Lookup:
        result.lookup_ms = (time.perf_counter() - start) * 1000.0
        self.stats.lookup_ms_total += result.lookup_ms
        if counter:
            setattr(self.stats, counter, getattr(self.stats, counter) + 1)
        return result

    # -- convenience -----------------------------------------------------
    def get_or_call(
        self,
        namespace: str,
        query: str,
        call: Callable[[str], Tuple[str, int, int, float]],
    ) -> Tuple[str, Lookup]:
        """Serve from cache or invoke `call` and store the result.

        `call` returns (answer, prompt_tokens, completion_tokens, latency_ms) so
        the cache records what a hit actually saved, measured from the real call
        rather than estimated afterwards.
        """
        result = self.lookup(namespace, query)
        if result.hit and result.entry is not None:
            return result.entry.answer, result
        answer, prompt_tokens, completion_tokens, latency_ms = call(query)
        self.put(namespace, query, answer, prompt_tokens, completion_tokens, latency_ms)
        return answer, result
