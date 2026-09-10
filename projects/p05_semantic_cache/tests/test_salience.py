"""The safety gate: what it catches, what it costs, and what it cannot catch."""
import pytest

from llmkit import get_embedder

from projects.p05_semantic_cache.cache import SemanticCache
from projects.p05_semantic_cache.fixtures import NAMESPACE, NEAR_MISSES, PARAPHRASES
from projects.p05_semantic_cache.metrics import (
    blocked_paraphrases, false_hits_at, run_probes, separability, seed_cache, sweep,
)
from projects.p05_semantic_cache.salience import SalienceGuard

THRESHOLD = 0.80


@pytest.fixture(scope="module")
def embedder():
    return get_embedder()


@pytest.fixture(scope="module")
def guard():
    return SalienceGuard(ignore_entities=("meridian",))


@pytest.mark.parametrize(
    "incoming,cached,expected_check",
    [
        ("How do I rotate a token after 30 days?",
         "How do I rotate a token after 90 days?", "numbers"),
        ("What is the uptime target on the Starter plan?",
         "What is the uptime target on the Growth plan?", "entities"),
        ("Which role is not allowed to delete a workspace?",
         "Which role is allowed to delete a workspace?", "negation"),
        ("How long is the paid trial?", "How long is the free trial?", "contrastive:cost"),
        ("How do I disable single sign-on?", "How do I enable single sign-on?",
         "contrastive:enablement"),
        ("What is the minimum page size?", "What is the maximum page size?",
         "contrastive:bound"),
        ("What happens after scheduled maintenance?",
         "What happens before scheduled maintenance?", "contrastive:sequence"),
        ("How do I read events from production?", "How do I read events from sandbox?",
         "contrastive:environment"),
    ],
)
def test_each_guard_class_refuses_its_own_failure_mode(guard, incoming, cached, expected_check):
    report = guard.compare(incoming, cached)
    assert not report.ok
    assert report.check == expected_check
    assert report.describe() != report.reason


def test_the_free_versus_paid_trial_case_is_invisible_to_the_other_checks(guard):
    """The canonical example, and why the contrastive list has to exist.

    No number, no capitalised entity, no negation. Only the contrastive group
    separates these two questions, which is the entire reason that check was
    added rather than stopping at the obvious three.
    """
    a, b = "How long is the free trial?", "How long is the paid trial?"
    assert guard.numbers(a) == guard.numbers(b)
    assert guard.entities(a) == guard.entities(b)
    assert guard.negations(a) == guard.negations(b)
    assert not guard.compare(a, b).ok


def test_genuine_paraphrases_survive_the_guard(guard):
    survivors = [p for p in PARAPHRASES if guard.compare(p.probe, p.base).ok]
    assert len(survivors) >= len(PARAPHRASES) - 1, \
        "a guard that refuses paraphrases is just a slower exact-match cache"


def test_sentence_initial_capitals_are_not_treated_as_entities(guard):
    """Otherwise every question starting with a different word looks different.

    The cost of this rule is real and is asserted here rather than glossed: an
    entity that happens to start the sentence is invisible to the entity check.
    The contrastive plan group catches this particular one, which is why the two
    checks overlap on purpose.
    """
    assert guard.entities("What is the Growth plan limit?") == {"growth"}
    assert guard.entities("Growth plan limits are what?") == set()
    assert guard.compare("What is the limit?", "Which is the limit?").ok
    assert not guard.compare("Growth plan limits?", "Starter plan limits?").ok


def test_no_threshold_separates_the_two_labelled_sets(embedder):
    """The measurement that justifies the guard existing at all."""
    spread = separability(embedder)
    assert spread["near_miss_max"] > spread["paraphrase_min"], (
        "if the sets were separable a threshold would be enough and this whole "
        "module would be unnecessary"
    )
    assert spread["overlap"] > 0.1


def test_guard_removes_every_false_hit_at_the_shipped_threshold(embedder):
    off = run_probes(THRESHOLD, False, embedder)
    on = run_probes(THRESHOLD, True, embedder)

    assert off.false_hits > 0, "the fixture must contain hits a cosine cache would serve"
    assert on.false_hits == 0
    assert on.blocked == off.false_hits
    assert on.recall == off.recall, "at this threshold the guard costs no paraphrases"
    assert on.precision == 1.0 > off.precision


def test_the_guard_lets_a_looser_threshold_be_safe(embedder):
    """The practical payoff: precision stops being a function of the threshold."""
    rows = sweep(thresholds=(0.60, 0.70, 0.80), embedder=embedder)
    guarded = [r for r in rows if r.guard]
    unguarded = [r for r in rows if not r.guard]

    assert all(r.precision == 1.0 for r in guarded)
    assert all(r.precision < 1.0 for r in unguarded)
    assert max(r.f1 for r in guarded) > max(r.f1 for r in unguarded)


def test_the_guards_own_cost_is_visible_and_bounded(embedder):
    """It over-triggers, and the demo says so rather than hiding it."""
    rejected = blocked_paraphrases(0.60, embedder)
    assert len(rejected) <= 2, "over-triggering has to stay rare to be worth it"
    if rejected:
        _pair, detail = rejected[0]
        assert "contrastive" in detail or "entities" in detail


def test_blocked_hit_is_refused_outright_not_downgraded_to_a_worse_entry(embedder):
    """Falling through to runner-up entries would serve something even less alike."""
    cache = seed_cache(
        SemanticCache(threshold=0.50, ttl_s=1e9, embedder=embedder,
                      guard=SalienceGuard(ignore_entities=("meridian",)))
    )
    result = cache.lookup(NAMESPACE, "Which role is not allowed to delete a workspace?")
    assert not result.hit
    assert result.reason == "blocked_by_salience"
    assert result.entry is None
    assert result.candidate is not None and result.similarity >= 0.50


def test_every_near_miss_that_clears_the_threshold_is_reported_with_a_reason(embedder):
    served = false_hits_at(THRESHOLD, embedder)
    assert served, "nothing to prevent means nothing to prove"
    assert len(served) <= len(NEAR_MISSES)
    for _pair, similarity, detail in served:
        assert similarity >= THRESHOLD
        assert ":" in detail, "a refusal must name the check that fired"
