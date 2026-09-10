"""Project 06: model routing gateway.

Complexity-based routing across three price tiers, with per-tenant and
per-endpoint policy, cost tracking, a budget guard, a fallback chain and
quality-aware escalation. Public API:

    from projects.p06_model_router import (
        ModelRouter, Request, default_backends, default_policy,
        CostLedger, BudgetGuard, BudgetPolicy, QualityJudge, score_complexity,
    )

    ledger = CostLedger()
    router = ModelRouter(default_backends(), policy=default_policy(), ledger=ledger,
                         guard=BudgetGuard(ledger, {"acme": BudgetPolicy(daily_usd=5.0)}),
                         judge=QualityJudge(threshold=0.6))
    result = router.handle(Request("r1", "Classify this ticket", tenant="acme"))
    print(result.tier_used, result.cost_usd, result.decision.explain())

The escalation judge is `RubricJudge` from project 04, imported rather than
reimplemented.
"""
from .budget import BudgetDecision, BudgetGuard, BudgetPolicy, CostLedger, LedgerEntry
from .complexity import (
    ComplexityFeatures, classify_task, has_explicit_task_signal, measure_ambiguity,
    required_output_tokens, score_complexity,
)
from .gateway import (
    Attempt, GatewayResult, ModelRouter, QualityJudge, Request, default_backends, fallback_chain,
)
from .policy import (
    RoutingDecision, RoutingPolicy, TenantOverride, clamp_tier, default_policy, fixed_policy,
)
from .workload import WORKLOAD, WorkloadReport, run_workload, without_pins, workload_profile

__all__ = [
    "ComplexityFeatures", "score_complexity", "classify_task", "measure_ambiguity",
    "required_output_tokens", "has_explicit_task_signal",
    "RoutingPolicy", "RoutingDecision", "TenantOverride", "default_policy", "fixed_policy",
    "clamp_tier",
    "CostLedger", "LedgerEntry", "BudgetGuard", "BudgetPolicy", "BudgetDecision",
    "ModelRouter", "Request", "GatewayResult", "Attempt", "QualityJudge",
    "default_backends", "fallback_chain",
    "WORKLOAD", "WorkloadReport", "run_workload", "workload_profile", "without_pins",
]
