"""Routing, budget guarding, fallback, circuit breaking and escalation."""
import pytest

from llmkit import EchoLLM, FailingLLM, ScriptedLLM

from projects.p06_model_router.budget import (
    BudgetGuard, BudgetPolicy, CostLedger,
)
from projects.p06_model_router.gateway import (
    ModelRouter, QualityJudge, Request, default_backends, fallback_chain,
)
from projects.p06_model_router.policy import default_policy, fixed_policy
from projects.p06_model_router.workload import WORKLOAD, run_workload, without_pins

EASY = Request("q1", "Classify this ticket as billing or technical", "acme", "/chat")
HARD = Request("q2", "Explain the trade-offs between cursor and offset pagination", "acme", "/chat")


def router(**kwargs):
    ledger = kwargs.pop("ledger", None) or CostLedger()
    backends = kwargs.pop("backends", None) or default_backends()
    return ModelRouter(backends, policy=kwargs.pop("policy", None) or default_policy(),
                       ledger=ledger, **kwargs), ledger


def test_a_cheap_request_goes_cheap_and_an_expensive_one_does_not():
    r, ledger = router()
    assert r.handle(EASY).tier_used == "small"
    assert r.handle(HARD).tier_used == "large"
    assert ledger.cost_for("q2") > ledger.cost_for("q1")


def test_router_refuses_to_start_without_a_backend_for_every_tier():
    with pytest.raises(ValueError) as exc:
        ModelRouter({"small": EchoLLM()})
    assert "medium" in str(exc.value) and "large" in str(exc.value)


def test_fallback_order_prefers_the_nearest_capability_not_the_cheapest():
    assert fallback_chain("small") == ["small", "medium", "large"]
    assert fallback_chain("medium") == ["medium", "large", "small"], "falling to small is a quality cut"
    assert fallback_chain("large") == ["large", "medium", "small"]


def test_a_tier_outage_does_not_fail_the_request():
    backends = default_backends()
    backends["large"] = FailingLLM(fail_times=10 ** 6, status=503)
    r, _ = router(backends=backends)
    result = r.handle(HARD)
    assert result.ok and result.tier_used == "medium"
    assert [a.outcome for a in result.attempts] == ["failed", "ok"]
    assert "503" in result.attempts[0].error


def test_the_breaker_stops_paying_for_a_dead_tier_on_every_request():
    backends = default_backends()
    dead = FailingLLM(fail_times=10 ** 6, status=503)
    backends["large"] = dead
    r, _ = router(backends=backends, failure_threshold=2)
    outage = [r.handle(Request(f"h{i}", HARD.prompt, "acme", "/chat")) for i in range(6)]
    assert all(x.ok for x in outage), "every request still answered"
    assert r.breakers["large"].state == "open"
    skipped = [a for x in outage for a in x.attempts if a.outcome == "circuit_open"]
    assert len(skipped) == 4, "two real failures opened the breaker, the rest were skipped"
    assert dead.attempts == 4, "2 requests x 2 retries, then no more calls at all"


def test_every_tier_down_is_a_failure_not_a_silent_success():
    backends = {t: FailingLLM(fail_times=10 ** 6) for t in ("small", "medium", "large")}
    r, _ = router(backends=backends)
    result = r.handle(EASY)
    assert not result.ok and not result.refused
    assert result.tier_used is None and result.cost_usd == 0.0
    assert "every candidate tier failed" in result.reason


def test_budget_guard_allows_then_downgrades_then_refuses():
    ledger = CostLedger()
    guard = BudgetGuard(ledger, policies={"acme": BudgetPolicy(daily_usd=0.0006,
                                                              downgrade_at=0.5)})
    r, _ = router(ledger=ledger, guard=guard)
    seen = []
    for i in range(12):
        result = r.handle(Request(f"b{i}", HARD.prompt, "acme", "/chat"))
        seen.append(result.budget.action)
        if result.refused:
            break
    assert seen[0] == "allow"
    assert "downgrade" in seen
    assert seen[-1] == "refuse"


def test_a_refusal_is_a_distinct_outcome_that_costs_nothing():
    """Not an exception, not a silent downgrade, and no upstream call."""
    ledger = CostLedger()
    ledger.charge("prior", "acme", "large", 100_000, 100_000)  # burn the budget
    guard = BudgetGuard(ledger, policies={"acme": BudgetPolicy(daily_usd=0.01)})
    backends = default_backends()
    r, _ = router(ledger=ledger, guard=guard, backends=backends)
    before = backends["small"].call_count

    result = r.handle(EASY)

    assert result.refused and not result.ok
    assert result.tier_used is None and result.cost_usd == 0.0
    assert backends["small"].call_count == before, "a refused request must not call a model"
    assert "daily budget" in result.reason


def test_downgrade_moves_the_tier_and_records_why():
    ledger = CostLedger()
    ledger.charge("prior", "acme", "large", 2500, 0)  # 0.0075, 75 percent of 0.01
    guard = BudgetGuard(ledger, policies={"acme": BudgetPolicy(daily_usd=0.01, downgrade_at=0.7)})
    r, _ = router(ledger=ledger, guard=guard)
    result = r.handle(HARD)
    assert result.decision.tier == "large" and result.tier_used == "small"
    assert result.budget.action == "downgrade" and "of budget" in result.budget.reason


def test_escalation_only_happens_when_the_judge_rejects():
    accepting = QualityJudge(llm=ScriptedLLM(
        ['{"faithfulness": 5, "relevance": 5, "completeness": 5, "reason": "good"}']))
    r, _ = router(judge=accepting)
    kept = r.handle(EASY)
    assert kept.tier_used == "small" and not kept.escalated
    assert kept.judge_score == 1.0

    rejecting = QualityJudge(llm=ScriptedLLM(
        ['{"faithfulness": 1, "relevance": 1, "completeness": 1, "reason": "bad"}']))
    r2, ledger2 = router(judge=rejecting)
    moved = r2.handle(Request("q3", EASY.prompt, "acme", "/chat"))
    assert moved.escalated and moved.escalated_from == "small" and moved.tier_used == "medium"
    assert "escalated small to medium" in moved.reason
    assert len(ledger2.entries) == 2, "the cheap call is paid for as well as the escalation"


def test_the_top_tier_is_never_judged_because_there_is_nowhere_to_escalate():
    judge = QualityJudge(llm=ScriptedLLM(
        ['{"faithfulness": 1, "relevance": 1, "completeness": 1, "reason": "bad"}']))
    r, _ = router(judge=judge)
    result = r.handle(HARD)
    assert result.tier_used == "large"
    assert result.judge_score is None and judge.calls == 0


def test_the_ledger_attributes_cost_by_request_tenant_and_tier():
    r, ledger = router()
    r.handle(EASY)
    r.handle(HARD)
    r.handle(Request("q4", HARD.prompt, "globex", "/chat"))
    assert set(ledger.by_tenant()) == {"acme", "globex"}
    assert ledger.by_tier()["large"]["calls"] == 2
    assert ledger.total == pytest.approx(sum(ledger.by_tenant().values()), abs=1e-6)
    assert ledger.cost_for("q1") > 0


def test_routing_is_cheaper_than_always_large_on_the_real_workload():
    routed_router, _ = router()
    routed = run_workload(routed_router, WORKLOAD, "routed")
    big_router, _ = router(policy=fixed_policy("large"))
    always_large = run_workload(big_router, without_pins(), "always-large")

    assert routed.failures == 0 and routed.refusals == 0
    assert routed.total_cost < always_large.total_cost
    dist = routed.tier_distribution()
    assert all(count > 0 for count in dist.values()), "all three tiers get used"
    assert sum(dist.values()) == len(WORKLOAD)


def test_stripping_pins_is_what_makes_a_baseline_a_baseline():
    assert any(r.pin for r in WORKLOAD)
    assert not any(r.pin for r in without_pins())
    r, _ = router(policy=fixed_policy("small"))
    report = run_workload(r, without_pins(), "always-small")
    assert report.tier_distribution() == {"small": len(WORKLOAD), "medium": 0, "large": 0}
