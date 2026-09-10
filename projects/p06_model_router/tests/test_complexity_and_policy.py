"""Complexity features, task classification and policy precedence."""
import pytest

from projects.p06_model_router.complexity import (
    TASK_DIFFICULTY, classify_task, has_explicit_task_signal, measure_ambiguity,
    required_output_tokens, score_complexity,
)
from projects.p06_model_router.policy import (
    RoutingPolicy, TenantOverride, clamp_tier, default_policy, fixed_policy,
)


def test_scoring_returns_explainable_features_not_just_a_number():
    f = score_complexity("Write a Python function that parses a Retry-After header")
    assert f.task_type == "code" and f.score > 0
    assert f.contributions["task_type"] > 0
    assert pytest.approx(sum(f.contributions.values()), abs=1e-9) == f.score
    assert "code" in f.explain() and "score" in f.explain()


def test_task_patterns_are_ordered_hardest_first():
    """A request that is both code and reasoning must route as the harder one."""
    assert classify_task("Explain why this SQL query is slow and how to fix it") == "reasoning"
    assert classify_task("Write a SQL query that lists failed payments") == "code"
    assert classify_task("Classify this ticket as billing or technical") == "classification"
    assert classify_task("Extract the invoice number as JSON") == "extraction"


def test_an_unmatched_request_falls_back_and_says_so():
    assert not has_explicit_task_signal("it broke again, sort it out somehow")
    assert has_explicit_task_signal("summarise the incident report")
    # The fallback is mid-difficulty, not the cheapest tier, because an
    # unrecognised request is not evidence that the request is easy.
    fallback = classify_task("it broke again")
    assert TASK_DIFFICULTY[fallback] > TASK_DIFFICULTY["classification"]


def test_explicit_length_instructions_beat_the_task_default():
    assert required_output_tokens("Summarise this in 100 words", "summarisation") == 130
    assert required_output_tokens("Give me 5 bullets", "summarisation") == 70
    assert required_output_tokens("Summarise this", "summarisation") == 200


def test_ambiguity_separates_a_vague_request_from_a_precise_one():
    assert measure_ambiguity("fix it") > 0.9
    assert measure_ambiguity("Write a Python function that parses a Retry-After header") == 0.0
    assert measure_ambiguity("") == 1.0


def test_a_harder_task_scores_higher_than_an_easier_one_of_the_same_length():
    easy = score_complexity("Classify the sentiment of this review text please now")
    hard = score_complexity("Explain the trade-offs of this review process design")
    assert hard.score > easy.score


def test_math_and_tool_flags_add_to_the_score():
    plain = score_complexity("Summarise the plan tiers")
    mathy = score_complexity("Summarise the plan tiers and calculate the 99.82 percent SLA credit")
    tooled = score_complexity("Summarise the latest status of the eu-west region")
    assert mathy.has_math and mathy.score > plain.score
    assert tooled.needs_tools and tooled.score > plain.score


def test_policy_precedence_pin_then_endpoint_then_tenant_then_band():
    policy = default_policy()
    easy = score_complexity("Classify this ticket as billing or technical")
    hard = score_complexity("Explain the trade-offs between cursor and offset pagination")

    assert policy.decide(easy, "acme", "/chat").tier == "small", "band"
    assert policy.decide(easy, "enterprise", "/chat").tier == "medium", "tenant floor beats band"
    assert policy.decide(hard, "batch", "/chat").tier == "small", "tenant ceiling beats band"
    assert policy.decide(easy, "enterprise", "/codegen").tier == "large", "endpoint beats tenant"
    assert policy.decide(hard, "enterprise", "/codegen", pin="small").tier == "small", "pin wins"


def test_the_decision_records_which_rules_fired():
    decision = default_policy().decide(
        score_complexity("Classify this ticket"), "enterprise", "/chat")
    assert decision.band_tier == "small" and decision.tier == "medium"
    assert decision.applied == ["band:small", "tenant:floor:medium"]
    assert "enterprise tier floor" in decision.reason


def test_clamp_respects_both_bounds_and_rejects_an_unknown_pin():
    assert clamp_tier("large", ceiling="small") == "small"
    assert clamp_tier("small", floor="medium") == "medium"
    assert clamp_tier("medium", floor="small", ceiling="large") == "medium"
    with pytest.raises(ValueError):
        default_policy().decide(score_complexity("hi"), pin="enormous")
    with pytest.raises(ValueError):
        fixed_policy("gigantic")


def test_a_tenant_can_have_a_floor_and_a_ceiling_at_once():
    policy = RoutingPolicy(tenant_overrides={
        "pinned": TenantOverride(min_tier="medium", max_tier="medium")})
    for prompt in ("Classify this", "Explain the trade-offs of this architecture in detail"):
        assert policy.decide(score_complexity(prompt), "pinned").tier == "medium"
