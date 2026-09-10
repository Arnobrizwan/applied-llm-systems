"""End to end demo: route 44 requests, break a tier, then run out of budget.

Run it with:  python3 projects/p06_model_router/demo.py
Every number printed is measured in this process.
"""
import os
import sys
from collections import Counter
from dataclasses import replace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from llmkit import FailingLLM  # noqa: E402

from projects.p06_model_router.budget import (  # noqa: E402
    BudgetGuard, BudgetPolicy, CostLedger,
)
from projects.p06_model_router.complexity import (  # noqa: E402
    has_explicit_task_signal, score_complexity,
)
from projects.p06_model_router.gateway import (  # noqa: E402
    ModelRouter, QualityJudge, default_backends,
)
from projects.p06_model_router.policy import default_policy, fixed_policy  # noqa: E402
from projects.p06_model_router.workload import (  # noqa: E402
    WORKLOAD, run_workload, without_pins, workload_profile,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(ROOT, "artifacts", "p06_model_router")

SHOWCASE = ["r02", "r15", "r22", "r30", "r18", "r43", "r40"]


def rule(title):
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def build_router(**kwargs):
    ledger = kwargs.pop("ledger", None) or CostLedger()
    return ModelRouter(kwargs.pop("backends", None) or default_backends(),
                       policy=kwargs.pop("policy", None) or default_policy(),
                       ledger=ledger, **kwargs), ledger


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    rule("1. THE WORKLOAD")
    profile = workload_profile()
    print(f"{profile['requests']} requests   task mix: " +
          ", ".join(f"{k} {v}" for k, v in profile["by_task"].items()))
    print(f"complexity score: min {profile['min_score']}, median {profile['median_score']}, "
          f"max {profile['max_score']}")
    inferred = [r for r in WORKLOAD if not has_explicit_task_signal(r.prompt)]
    print(f"requests with no task pattern match, routed on length and ambiguity alone: "
          f"{len(inferred)} of {len(WORKLOAD)} ({', '.join(r.id for r in inferred)})")
    print(f"tenants: {dict(Counter(r.tenant for r in WORKLOAD))}")

    rule("2. EXPLAINABLE ROUTING DECISIONS")
    router, _ = build_router()
    for rid in SHOWCASE:
        request = next(r for r in WORKLOAD if r.id == rid)
        features = score_complexity(request.prompt, request.task_type)
        decision = router.policy.decide(features, request.tenant, request.endpoint, request.pin)
        print(f"\n{rid} [{request.tenant} {request.endpoint}] {request.prompt[:64]}")
        print(f"   {features.explain()}")
        print(f"   -> {decision.explain()}")

    rule("3. TIER DISTRIBUTION AND COST AGAINST BASELINES")
    routed_router, routed_ledger = build_router()
    routed = run_workload(routed_router, WORKLOAD, "routed")
    baselines = {}
    for tier in ("small", "large"):
        base_router, base_ledger = build_router(policy=fixed_policy(tier))
        baselines[tier] = (run_workload(base_router, without_pins(), f"always-{tier}"), base_ledger)

    print(f"{'strategy':<16}{'small':>7}{'medium':>8}{'large':>7}{'cost usd':>12}{'vs large':>10}")
    print("-" * 60)
    large_cost = baselines["large"][0].total_cost
    for label, report in (("routed", routed), ("always-small", baselines["small"][0]),
                          ("always-large", baselines["large"][0])):
        dist = report.tier_distribution()
        saved = (large_cost - report.total_cost) / large_cost * 100 if large_cost else 0.0
        print(f"{label:<16}{dist['small']:>7}{dist['medium']:>8}{dist['large']:>7}"
              f"{report.total_cost:>12.6f}{saved:>9.1f}%")
    print(f"\nrouting saved ${large_cost - routed.total_cost:.6f} of ${large_cost:.6f}, "
          f"{(large_cost - routed.total_cost) / large_cost:.1%}, "
          f"at {routed.total_cost / baselines['small'][0].total_cost:.1f}x the always-small bill")
    print("cost by tier: " + ", ".join(
        f"{tier} {agg['calls']} calls ${agg['cost_usd']:.6f}"
        for tier, agg in routed_ledger.by_tier().items()))
    print("cost by tenant: " + ", ".join(f"{k} ${v:.6f}" for k, v in routed_ledger.by_tenant().items()))

    rule("4. QUALITY-AWARE ESCALATION")
    judge = QualityJudge(threshold=0.6)
    esc_router, esc_ledger = build_router(judge=judge)
    escalated = run_workload(esc_router, WORKLOAD, "routed+judge")
    eligible = [r for r in escalated.results if r.judge_score is not None]
    print(f"{len(eligible)} of {len(escalated.results)} answers were served below the top tier "
          f"and went to the judge ({judge.calls} judge calls)")
    print(f"escalated: {escalated.escalations} ({escalated.escalation_rate:.1%} of eligible)")
    print(f"tier distribution after escalation: {escalated.tier_distribution()}")
    print(f"cost with escalation ${escalated.total_cost:.6f} vs ${routed.total_cost:.6f} without, "
          f"still {(large_cost - escalated.total_cost) / large_cost:.1%} below always-large")

    rule("5. FALLBACK WHEN A TIER GOES DOWN")
    broken = default_backends()
    broken["large"] = FailingLLM(fail_times=10 ** 6, status=503)
    down_router, _ = build_router(backends=broken)
    outage = run_workload(down_router, WORKLOAD, "large-tier-outage")
    outcomes = Counter(a.outcome for r in outage.results for a in r.attempts)
    print(f"the large tier is returning HTTP 503 for every call")
    print(f"requests served: {len(outage.served)} of {len(outage.results)}, "
          f"failures: {outage.failures}, refusals: {outage.refusals}")
    print(f"attempt outcomes: {dict(outcomes)}")
    print(f"breaker states: {{{', '.join(f'{t}: {b.state}' for t, b in down_router.breakers.items())}}}")
    print(f"tier distribution during the outage: {outage.tier_distribution()}")
    print(f"only {outcomes['failed']} requests paid for a real failed call; the other "
          f"{outcomes['circuit_open']} skipped the dead tier without one")
    example = next(r for r in outage.results if r.fallback_events)
    print(f"example {example.request_id}: " +
          " then ".join(f"{a.tier}={a.outcome}" for a in example.attempts))

    rule("6. BUDGET GUARD: DOWNGRADE, THEN REFUSE")
    acme_day = [replace(r, id=f"{r.id}-{n}") for n in range(3)
                for r in WORKLOAD if r.tenant == "acme"]
    ledger = CostLedger()
    guard = BudgetGuard(ledger, policies={"acme": BudgetPolicy(daily_usd=0.003,
                                                              downgrade_at=0.8)})
    guarded_router, _ = build_router(ledger=ledger, guard=guard)
    guarded = run_workload(guarded_router, acme_day, "acme-day")
    actions = Counter(r.budget.action for r in guarded.results if r.budget)
    print(f"acme daily budget $0.003000, {len(acme_day)} requests in one day")
    print(f"outcomes: allowed {actions['allow']}, downgraded {actions['downgrade']}, "
          f"refused {actions['refuse']}")
    print(f"spend stopped at ${ledger.spend('acme'):.6f} of $0.003000")
    first_downgrade = next((r for r in guarded.results if r.budget
                            and r.budget.action == "downgrade"), None)
    first_refusal = next((r for r in guarded.results if r.refused), None)
    if first_downgrade:
        print(f"first downgrade  {first_downgrade.request_id}: {first_downgrade.budget.reason}")
    if first_refusal:
        print(f"first refusal    {first_refusal.request_id}: {first_refusal.reason}")
        print(f"refused requests are a distinct outcome: ok={first_refusal.ok}, "
              f"refused={first_refusal.refused}, tier_used={first_refusal.tier_used}, "
              f"cost=${first_refusal.cost_usd:.6f}")

    rule("SUMMARY")
    print(f"{'measure':<34}{'value':>22}")
    print("-" * 56)
    for label, value in (
        ("requests", f"{len(WORKLOAD)}"),
        ("tier split small/medium/large", "/".join(str(v) for v in routed.tier_distribution().values())),
        ("routed cost", f"${routed.total_cost:.6f}"),
        ("always-large cost", f"${large_cost:.6f}"),
        ("always-small cost", f"${baselines['small'][0].total_cost:.6f}"),
        ("saved vs always-large", f"{(large_cost - routed.total_cost) / large_cost:.1%}"),
        ("escalation rate", f"{escalated.escalation_rate:.1%}"),
        ("fallback events in the outage", f"{outage.fallback_events}"),
        ("requests lost to the outage", f"{outage.failures}"),
        ("budget refusals in the acme day", f"{actions['refuse']}"),
    ):
        print(f"{label:<34}{value:>22}")

    import json
    path = os.path.join(OUT_DIR, "workload_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"routed": routed.to_dict(), "with_judge": escalated.to_dict(),
                   "always_small": baselines["small"][0].to_dict(),
                   "always_large": baselines["large"][0].to_dict(),
                   "outage": outage.to_dict(),
                   "cost_by_tier": routed_ledger.by_tier(),
                   "cost_by_tenant": routed_ledger.by_tenant()}, f, indent=2, sort_keys=True)
    print(f"\nreport written to {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
