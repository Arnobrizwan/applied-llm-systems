"""End-to-end demo for the semantic cache layer.

Run:  python3 projects/p05_semantic_cache/demo.py
Every number printed here is measured at run time. Nothing is hard coded.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from llmkit import get_embedder  # noqa: E402

from projects.p05_semantic_cache.cache import SemanticCache  # noqa: E402
from projects.p05_semantic_cache.fixtures import (  # noqa: E402
    BASE_ANSWERS, NAMESPACE, NEAR_MISSES, PARAPHRASES,
)
from projects.p05_semantic_cache.metrics import (  # noqa: E402
    blocked_paraphrases, false_hits_at, format_sweep, measure_savings, run_probes,
    separability, sweep,
)
from projects.p05_semantic_cache.salience import SalienceGuard  # noqa: E402


def rule(title: str) -> None:
    print("\n" + title)
    print("=" * len(title))


def build_cache(threshold: float = 0.80, use_guard: bool = True, **kwargs) -> SemanticCache:
    return SemanticCache(
        threshold=threshold,
        use_guard=use_guard,
        guard=SalienceGuard(ignore_entities=("meridian",)),
        **kwargs,
    )


def main() -> None:
    embedder = get_embedder()

    rule("1. Why a threshold alone cannot work")
    spread = separability(embedder)
    print(f"  paraphrases  min={spread['paraphrase_min']:.3f} "
          f"mean={spread['paraphrase_mean']:.3f} max={spread['paraphrase_max']:.3f}")
    print(f"  near misses  min={spread['near_miss_min']:.3f} "
          f"mean={spread['near_miss_mean']:.3f} max={spread['near_miss_max']:.3f}")
    print(f"  the best near miss outscores the worst paraphrase by "
          f"{spread['overlap']:.3f} cosine")
    print("  so no single cut point separates the two labelled sets")

    threshold = 0.80

    rule("2. Every near miss a cosine-only cache would serve")
    for pair, similarity, detail in false_hits_at(threshold, embedder):
        print(f"  cached  : {pair.base}")
        print(f"  incoming: {pair.probe}")
        print(f"  cosine {similarity:.3f} clears the {threshold:.2f} threshold, "
              "so similarity alone serves the cached answer")
        print(f"  guard refuses -> {detail}\n")

    rule("3. Guard on and off at the shipped threshold")
    off = run_probes(threshold, False, embedder)
    on = run_probes(threshold, True, embedder)
    print(f"  threshold {threshold:.2f}, {len(PARAPHRASES)} paraphrases, "
          f"{len(NEAR_MISSES)} near misses\n")
    for row, label in ((off, "guard off"), (on, "guard on ")):
        print(f"  {label}: hit_rate={row.recall:.2f} false_hits={row.false_hits} "
              f"false_hit_rate={row.false_hit_rate:.2f} precision={row.precision:.2f} "
              f"F1={row.f1:.2f}")
    print(f"\n  false hits prevented by the guard: {off.false_hits - on.false_hits} "
          f"of {off.false_hits}")
    print(f"  genuine paraphrases lost to the guard: {on.missed_paraphrases - off.missed_paraphrases}")

    rule("4. Threshold sweep")
    rows = sweep(embedder=embedder)
    print(format_sweep(rows))
    best_off = max((r for r in rows if not r.guard), key=lambda r: r.f1)
    best_on = max((r for r in rows if r.guard), key=lambda r: r.f1)
    print(f"\n  best F1 without the guard: {best_off.f1:.2f} at threshold "
          f"{best_off.threshold:.2f} ({best_off.false_hits} false hits)")
    print(f"  best F1 with the guard   : {best_on.f1:.2f} at threshold "
          f"{best_on.threshold:.2f} ({best_on.false_hits} false hits)")

    rule("5. Tenant isolation")
    probe = "What is the default per workspace rate limit?"
    shared = build_cache()
    shared.put("tenant-a:v3", "What is the default rate limit per workspace?",
               "Tenant A is on a custom limit of 5000 rpm.")
    shared.put("tenant-b:v3", "What is the default rate limit per workspace?",
               "Tenant B is on the standard 600 rpm.")
    print(f"  probe: {probe}")
    for namespace in shared.namespaces():
        result = shared.lookup(namespace, probe)
        print(f"  {namespace:<12} hit={result.hit} sim={result.similarity:.3f} "
              f"answer={result.answer!r}")
    fresh = shared.lookup("tenant-c:v3", probe)
    print(f"  {'tenant-c:v3':<12} hit={fresh.hit} reason={fresh.reason}")
    versioned = shared.lookup("tenant-a:v4", probe)
    print(f"  {'tenant-a:v4':<12} hit={versioned.hit} reason={versioned.reason}")
    print("  a new prompt version is a new namespace, so it cannot serve answers")
    print("  that were generated under the old prompt")

    rule("6. TTL and LRU")
    now = [1000.0]
    ttl_cache = build_cache(ttl_s=60.0, max_entries=3, clock=lambda: now[0])
    ttl_cache.put(NAMESPACE, "How long are request logs retained?", "30 days.")
    print(f"  t=0    hit={ttl_cache.lookup(NAMESPACE, 'How long are request logs retained?').hit}")
    now[0] += 61.0
    expired = ttl_cache.lookup(NAMESPACE, "How long are request logs retained?")
    print(f"  t=61s  hit={expired.hit} reason={expired.reason}")

    lru_cache = build_cache(max_entries=3)
    for query, answer in list(BASE_ANSWERS.items())[:5]:
        lru_cache.put(NAMESPACE, query, answer)
    print(f"  wrote 5 entries with max_entries=3 -> size={lru_cache.size(NAMESPACE)}, "
          f"evicted={lru_cache.stats.evicted_lru}")

    rule("7. Measured savings on a repeated workload")
    workload = []
    for pair in PARAPHRASES:
        workload.append((NAMESPACE, pair.base))
        workload.append((NAMESPACE, pair.probe))
    for pair in NEAR_MISSES:
        workload.append((NAMESPACE, pair.probe))
    report, used = measure_savings(workload, cache=build_cache(threshold=threshold))
    print(f"  requests            : {report.requests}")
    print(f"  served from cache   : {report.served_from_cache} ({report.hit_rate:.2%})")
    print(f"  model calls made    : {report.model_calls}")
    print(f"  tokens with cache   : {report.tokens_with_cache}")
    print(f"  tokens without cache: {report.tokens_without_cache}")
    print(f"  token saving        : {report.token_saving:.2%}")
    print("\n  wall clock against the offline EchoLLM rule engine:")
    print(f"    model ms without cache: {report.model_ms_without_cache:.2f}")
    print(f"    model ms with cache   : {report.model_ms_with_cache:.2f}")
    print(f"    cache lookup overhead : {report.cache_overhead_ms:.2f}")
    print("    a local rule engine answers faster than the cache can look it up,")
    print("    so the transferable figure here is the token saving, not this one")

    rule("8. What the guard costs")
    for cut in (0.60, threshold):
        rejected = blocked_paraphrases(cut, embedder)
        print(f"  threshold {cut:.2f}: {len(rejected)} genuine paraphrase(s) refused")
        for pair, detail in rejected:
            print(f"    probe : {pair.probe}")
            print(f"    cached: {pair.base}")
            print(f"    {detail}")

    print("\n  cache counters from section 7:")
    for line in used.stats.as_lines():
        print("    " + line)


if __name__ == "__main__":
    main()
