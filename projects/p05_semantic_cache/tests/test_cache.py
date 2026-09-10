"""Cache mechanics: hits, isolation, TTL, LRU and the savings accounting."""
import pytest

from llmkit import get_embedder

from projects.p05_semantic_cache.cache import SemanticCache
from projects.p05_semantic_cache.fixtures import BASE_ANSWERS, NAMESPACE, PARAPHRASES
from projects.p05_semantic_cache.metrics import measure_savings, seed_cache
from projects.p05_semantic_cache.salience import SalienceGuard


@pytest.fixture(scope="module")
def embedder():
    return get_embedder()


def build(**kwargs) -> SemanticCache:
    kwargs.setdefault("guard", SalienceGuard(ignore_entities=("meridian",)))
    kwargs.setdefault("threshold", 0.80)
    return SemanticCache(**kwargs)


def test_paraphrase_is_served_and_the_saving_is_recorded(embedder):
    cache = seed_cache(build(embedder=embedder))
    pair = PARAPHRASES[1]
    result = cache.lookup(NAMESPACE, pair.probe)

    assert result.hit
    assert result.entry.query == pair.base
    assert result.answer == BASE_ANSWERS[pair.base]
    assert result.similarity >= cache.threshold
    # The saving is the tokens the cached entry cost, not an estimate made later.
    assert cache.stats.tokens_saved == result.entry.total_tokens
    assert result.entry.hits == 1


def test_unrelated_question_misses_rather_than_returning_the_nearest_thing(embedder):
    cache = seed_cache(build(embedder=embedder))
    result = cache.lookup(NAMESPACE, "What is the capital city of Iceland?")
    assert not result.hit
    assert result.reason == "below_threshold"
    # The candidate is still reported, which is what makes a threshold tunable
    # from production logs instead of by guessing.
    assert result.candidate is not None
    assert result.similarity < cache.threshold


def test_namespaces_cannot_read_each_other(embedder):
    """The multi-tenant property. A shared cache with a tenant column is a leak."""
    question = "What is the default rate limit per workspace?"
    cache = build(embedder=embedder)
    cache.put("tenant-a:v3", question, "Tenant A is on 5000 rpm.")
    cache.put("tenant-b:v3", question, "Tenant B is on 600 rpm.")

    assert cache.lookup("tenant-a:v3", question).answer == "Tenant A is on 5000 rpm."
    assert cache.lookup("tenant-b:v3", question).answer == "Tenant B is on 600 rpm."
    # A prompt-version bump is a new namespace, so it cannot serve answers that
    # were generated under the old system prompt.
    assert cache.lookup("tenant-a:v4", question).reason == "empty_namespace"
    assert cache.lookup("tenant-c:v3", question).reason == "empty_namespace"
    assert cache.size("tenant-a:v3") == 1


def test_ttl_expires_on_read_not_only_on_sweep(embedder):
    now = [100.0]
    cache = build(embedder=embedder, ttl_s=60.0, clock=lambda: now[0])
    cache.put(NAMESPACE, "How long are request logs retained?", "30 days.")

    assert cache.lookup(NAMESPACE, "How long are request logs retained?").hit
    now[0] += 61.0
    stale = cache.lookup(NAMESPACE, "How long are request logs retained?")
    assert not stale.hit and stale.reason == "all_entries_expired"
    assert cache.size(NAMESPACE) == 0, "an expired entry must be dropped, not just skipped"


def test_lru_evicts_least_recently_used_not_least_recently_written(embedder):
    """Eviction order has to follow reads, or the hottest entry gets thrown out."""
    cache = build(embedder=embedder, max_entries=3)
    queries = list(BASE_ANSWERS)[:3]
    for query in queries:
        cache.put(NAMESPACE, query, BASE_ANSWERS[query])

    assert cache.lookup(NAMESPACE, queries[0]).hit  # touch the oldest write

    fourth = list(BASE_ANSWERS)[3]
    cache.put(NAMESPACE, fourth, BASE_ANSWERS[fourth])

    assert cache.size(NAMESPACE) == 3
    assert cache.stats.evicted_lru == 1
    assert cache.get_entry(NAMESPACE, queries[0]) is not None, "recently read entry survived"
    assert cache.get_entry(NAMESPACE, queries[1]) is None, "untouched entry was evicted"


def test_entry_cap_is_per_namespace_so_one_tenant_cannot_evict_another(embedder):
    cache = build(embedder=embedder, max_entries=2)
    for query in list(BASE_ANSWERS)[:5]:
        cache.put("noisy", query, "noise")
    cache.put("quiet", "How long are request logs retained?", "30 days.")
    for query in list(BASE_ANSWERS)[5:10]:
        cache.put("noisy", query, "more noise")

    assert cache.size("noisy") == 2
    assert cache.size("quiet") == 1
    assert cache.lookup("quiet", "How long are request logs retained?").hit


def test_get_or_call_only_invokes_the_model_on_a_miss(embedder):
    calls = []

    def call(query):
        calls.append(query)
        return f"answer for {query}", 100, 20, 5.0

    cache = build(embedder=embedder)
    first, first_lookup = cache.get_or_call(NAMESPACE, "How long are request logs retained?", call)
    second, second_lookup = cache.get_or_call(
        NAMESPACE, "How long are the request logs retained for?", call
    )

    assert len(calls) == 1, "the paraphrase must not reach the model"
    assert first == second
    assert not first_lookup.hit and second_lookup.hit
    assert cache.stats.tokens_saved == 120


def test_savings_are_measured_against_the_same_provider(embedder):
    workload = [(NAMESPACE, p.base) for p in PARAPHRASES] + \
               [(NAMESPACE, p.probe) for p in PARAPHRASES]
    report, cache = measure_savings(workload, cache=build(embedder=embedder))

    assert report.requests == len(workload)
    assert report.served_from_cache > 0
    assert report.model_calls == report.requests - report.served_from_cache
    assert report.tokens_with_cache < report.tokens_without_cache
    assert 0.0 < report.token_saving < 1.0
    assert cache.stats.hits == report.served_from_cache


def test_invalid_configuration_is_rejected_at_construction():
    with pytest.raises(ValueError):
        SemanticCache(threshold=1.5)
    with pytest.raises(ValueError):
        SemanticCache(max_entries=0)


def test_invalidate_drops_one_tenant_only(embedder):
    cache = build(embedder=embedder)
    cache.put("a", "q", "answer a")
    cache.put("b", "q", "answer b")
    assert cache.invalidate("a") == 1
    assert cache.lookup("a", "q").reason == "empty_namespace"
    assert cache.lookup("b", "q").hit
