"""Measurement: hit rate, false-hit rate, the threshold sweep, and real savings.

The metric that matters is not hit rate. A cache with the threshold at 0.0 has a
hit rate of 1.0 and is a random answer generator. The pair that matters is:

  recall     fraction of genuine paraphrases served from cache (the saving)
  precision  fraction of served hits that were genuinely the same question

and the sweep exists to show that on a realistic fixture there is no threshold
where both are acceptable, which is what justifies the salience guard.

False-hit rate is reported against the labelled near-miss set rather than against
production traffic, because in production a false hit is invisible: the user gets
a fluent answer and nobody logs that it was wrong. Labelled near-misses are the
only way to measure this before shipping.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from llmkit import Embedder, LLMProvider, get_embedder, get_llm, user

from .cache import SemanticCache
from .fixtures import BASE_ANSWERS, NAMESPACE, NEAR_MISSES, PARAPHRASES, Pair
from .salience import SalienceGuard


@dataclass
class SweepRow:
    threshold: float
    guard: bool
    true_hits: int
    missed_paraphrases: int
    false_hits: int
    blocked: int

    @property
    def recall(self) -> float:
        total = self.true_hits + self.missed_paraphrases
        return self.true_hits / total if total else 0.0

    @property
    def precision(self) -> float:
        served = self.true_hits + self.false_hits
        return self.true_hits / served if served else 0.0

    @property
    def false_hit_rate(self) -> float:
        return self.false_hits / len(NEAR_MISSES) if NEAR_MISSES else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def seed_cache(cache: SemanticCache, namespace: str = NAMESPACE) -> SemanticCache:
    """Populate the cache with one entry per base query."""
    for query, answer in BASE_ANSWERS.items():
        cache.put(namespace, query, answer)
    return cache


def run_probes(
    threshold: float,
    guard_enabled: bool,
    embedder: Optional[Embedder] = None,
    guard: Optional[SalienceGuard] = None,
) -> SweepRow:
    """Score the labelled fixture at one threshold, with the guard on or off.

    A fresh cache per configuration on purpose: reusing one would let LRU order
    and hit counts from a previous threshold leak into the next measurement.
    """
    cache = seed_cache(
        SemanticCache(
            threshold=threshold,
            ttl_s=1e9,
            max_entries=1000,
            embedder=embedder or get_embedder(),
            guard=guard or SalienceGuard(ignore_entities=("meridian",)),
            use_guard=guard_enabled,
        )
    )

    true_hits = missed = false_hits = blocked = 0
    for pair in PARAPHRASES:
        result = cache.lookup(NAMESPACE, pair.probe)
        # A hit on the wrong base entry is not a true hit. Counting any hit as
        # correct would let a cache that always returns its first entry look
        # perfect.
        if result.hit and result.entry is not None and result.entry.query == pair.base:
            true_hits += 1
        else:
            missed += 1
    for pair in NEAR_MISSES:
        result = cache.lookup(NAMESPACE, pair.probe)
        if result.hit:
            false_hits += 1
        elif result.reason == "blocked_by_salience":
            blocked += 1

    return SweepRow(threshold, guard_enabled, true_hits, missed, false_hits, blocked)


def blocked_paraphrases(
    threshold: float,
    embedder: Optional[Embedder] = None,
) -> List[Tuple[Pair, str]]:
    """Genuine paraphrases the guard refuses. This is the guard's cost, itemised.

    A guard that blocks nothing is not doing anything; a guard that blocks real
    paraphrases is charging the user an extra model call. Both numbers have to be
    on the table, so this returns the specific pairs rather than a count.
    """
    cache = seed_cache(
        SemanticCache(threshold=threshold, ttl_s=1e9, max_entries=1000,
                      embedder=embedder or get_embedder(),
                      guard=SalienceGuard(ignore_entities=("meridian",)))
    )
    out: List[Tuple[Pair, str]] = []
    for pair in PARAPHRASES:
        result = cache.lookup(NAMESPACE, pair.probe)
        if result.reason == "blocked_by_salience" and result.salience is not None:
            out.append((pair, result.salience.describe()))
    return out


def false_hits_at(
    threshold: float,
    embedder: Optional[Embedder] = None,
) -> List[Tuple[Pair, float, str]]:
    """Near-misses that clear the threshold, with the reason the guard refuses.

    These are the entries a cosine-only cache would have served: high enough
    similarity to look like a hit, incompatible enough to be the wrong answer.
    """
    loose = seed_cache(
        SemanticCache(threshold=threshold, ttl_s=1e9, max_entries=1000,
                      embedder=embedder or get_embedder(), use_guard=False)
    )
    strict = seed_cache(
        SemanticCache(threshold=threshold, ttl_s=1e9, max_entries=1000,
                      embedder=embedder or get_embedder(),
                      guard=SalienceGuard(ignore_entities=("meridian",)))
    )
    out: List[Tuple[Pair, float, str]] = []
    for pair in NEAR_MISSES:
        served = loose.lookup(NAMESPACE, pair.probe)
        if not served.hit:
            continue
        refused = strict.lookup(NAMESPACE, pair.probe)
        detail = refused.salience.describe() if refused.salience else refused.reason
        out.append((pair, served.similarity, detail))
    return out


def sweep(
    thresholds: Sequence[float] = (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95),
    embedder: Optional[Embedder] = None,
) -> List[SweepRow]:
    shared = embedder or get_embedder()
    rows: List[SweepRow] = []
    for threshold in thresholds:
        rows.append(run_probes(threshold, False, shared))
        rows.append(run_probes(threshold, True, shared))
    return rows


def format_sweep(rows: Sequence[SweepRow]) -> str:
    header = (
        f"{'threshold':>10}{'guard':>7}{'served':>8}{'true':>6}{'false':>7}"
        f"{'blocked':>9}{'precision':>11}{'recall':>8}{'F1':>7}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row.threshold:>10.2f}{('on' if row.guard else 'off'):>7}"
            f"{row.true_hits + row.false_hits:>8}{row.true_hits:>6}{row.false_hits:>7}"
            f"{row.blocked:>9}{row.precision:>11.2f}{row.recall:>8.2f}{row.f1:>7.2f}"
        )
    return "\n".join(lines)


def separability(embedder: Optional[Embedder] = None) -> Dict[str, float]:
    """How far the two labelled distributions overlap.

    If the highest-scoring near-miss outscores the lowest-scoring paraphrase then
    no single threshold can separate them, and the size of that inversion is the
    quantitative case for the guard.
    """
    from llmkit import cosine

    embed = embedder or get_embedder()
    base_vectors = {q: embed.embed_one(q) for q in BASE_ANSWERS}

    def best_for(pair: Pair) -> float:
        return cosine(embed.embed_one(pair.probe), base_vectors[pair.base])

    para = [best_for(p) for p in PARAPHRASES]
    near = [best_for(p) for p in NEAR_MISSES]
    return {
        "paraphrase_min": min(para),
        "paraphrase_max": max(para),
        "paraphrase_mean": sum(para) / len(para),
        "near_miss_min": min(near),
        "near_miss_max": max(near),
        "near_miss_mean": sum(near) / len(near),
        "overlap": max(near) - min(para),
    }


@dataclass
class SavingsReport:
    requests: int
    served_from_cache: int
    model_calls: int
    tokens_with_cache: int
    tokens_without_cache: int
    model_ms_with_cache: float
    model_ms_without_cache: float
    cache_overhead_ms: float

    @property
    def token_saving(self) -> float:
        if not self.tokens_without_cache:
            return 0.0
        return 1.0 - self.tokens_with_cache / self.tokens_without_cache

    @property
    def hit_rate(self) -> float:
        return self.served_from_cache / self.requests if self.requests else 0.0


def measure_savings(
    workload: Sequence[Tuple[str, str]],
    cache: Optional[SemanticCache] = None,
    llm: Optional[LLMProvider] = None,
) -> Tuple[SavingsReport, SemanticCache]:
    """Run a (namespace, query) workload through the cache and without it.

    Both arms call the same provider with the same prompts, so the token
    difference is the cache's doing and not a prompt change. Latency is recorded
    from the actual calls; see the README on why the absolute latency figure is
    not the interesting one with an offline provider.
    """
    cache = cache or SemanticCache(threshold=0.80, guard=SalienceGuard(ignore_entities=("meridian",)))
    model = llm or get_llm()

    # One throwaway call before either arm is timed. Without it the first arm
    # absorbs interpreter and import warm-up and the comparison is meaningless.
    model.complete([user("warm up")])

    def call(query: str) -> Tuple[str, int, int, float]:
        started = time.perf_counter()
        response = model.complete([user(query)])
        elapsed = (time.perf_counter() - started) * 1000.0
        return response.text, response.prompt_tokens, response.completion_tokens, elapsed

    tokens_with = 0
    model_ms_with = 0.0
    overhead_ms = 0.0
    served = 0
    calls = 0
    for namespace, query in workload:
        _answer, result = cache.get_or_call(namespace, query, call)
        overhead_ms += result.lookup_ms
        if result.hit:
            served += 1
            continue
        calls += 1
        entry = cache.get_entry(namespace, query)
        if entry is not None:
            tokens_with += entry.total_tokens
            model_ms_with += entry.latency_ms

    tokens_without = 0
    model_ms_without = 0.0
    for _namespace, query in workload:
        started = time.perf_counter()
        response = model.complete([user(query)])
        model_ms_without += (time.perf_counter() - started) * 1000.0
        tokens_without += response.total_tokens

    report = SavingsReport(
        requests=len(workload),
        served_from_cache=served,
        model_calls=calls,
        tokens_with_cache=tokens_with,
        tokens_without_cache=tokens_without,
        model_ms_with_cache=model_ms_with,
        model_ms_without_cache=model_ms_without,
        cache_overhead_ms=overhead_ms,
    )
    return report, cache
