"""Cost tracking per request and per tenant, and the guard that acts on it.

Two things a budget guard has to get right.

The first is that it must have a distinct refusal outcome. A guard that silently
downgrades forever turns a budget problem into a quality problem that nobody
attributes to the budget, and the tenant experiences it as "the product got
worse this month" rather than "we hit the cap we set". Refusal is a separate,
testable result with its own reason string, and it is what the caller sees.

The second is that downgrade has to come before refusal. Cutting a tenant off at
100 percent with no warning shelf means the last request before the cap is full
quality and the next one is an error. Degrading to the cheap tier at 80 percent
buys a band where the service is worse but alive, which is almost always the
outcome a customer would choose.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from llmkit import estimate_cost

from .policy import TIER_ORDER, tier_index


@dataclass
class BudgetPolicy:
    """Daily cap plus the fraction at which degradation starts."""

    daily_usd: float
    downgrade_at: float = 0.8
    refuse_at: float = 1.0

    def __post_init__(self) -> None:
        if self.daily_usd < 0:
            raise ValueError("daily_usd cannot be negative")
        if not 0 < self.downgrade_at <= self.refuse_at:
            raise ValueError("downgrade_at must be positive and no greater than refuse_at")


@dataclass
class LedgerEntry:
    request_id: str
    tenant: str
    tier: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


class CostLedger:
    """Every charge, attributable by request, tenant and tier.

    In memory and per process, which is the honest scope: a real deployment needs
    a shared store because the cap is per tenant across every replica, and an
    in-process ledger lets a tenant spend the cap once per replica. The
    accounting logic is the same either way and is the part worth showing.
    """

    def __init__(self) -> None:
        self.entries: List[LedgerEntry] = []
        self._by_tenant: Dict[str, float] = defaultdict(float)

    def charge(self, request_id: str, tenant: str, tier: str,
               prompt_tokens: int, completion_tokens: int) -> float:
        cost = estimate_cost(tier, prompt_tokens, completion_tokens)
        self.entries.append(LedgerEntry(request_id, tenant, tier, prompt_tokens,
                                        completion_tokens, cost))
        self._by_tenant[tenant] += cost
        return cost

    def spend(self, tenant: str) -> float:
        return self._by_tenant.get(tenant, 0.0)

    @property
    def total(self) -> float:
        return sum(self._by_tenant.values())

    def by_tier(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for e in self.entries:
            agg = out.setdefault(e.tier, {"calls": 0, "tokens": 0, "cost_usd": 0.0})
            agg["calls"] += 1
            agg["tokens"] += e.prompt_tokens + e.completion_tokens
            agg["cost_usd"] += e.cost_usd
        for agg in out.values():
            agg["cost_usd"] = round(agg["cost_usd"], 6)
        return dict(sorted(out.items(), key=lambda kv: tier_index(kv[0])))

    def by_tenant(self) -> Dict[str, float]:
        return {k: round(v, 6) for k, v in sorted(self._by_tenant.items())}

    def cost_for(self, request_id: str) -> float:
        return sum(e.cost_usd for e in self.entries if e.request_id == request_id)


@dataclass
class BudgetDecision:
    action: str          # "allow" | "downgrade" | "refuse"
    tier: Optional[str]  # None when refused
    reason: str = ""
    spent_usd: float = 0.0
    limit_usd: float = 0.0

    @property
    def refused(self) -> bool:
        return self.action == "refuse"

    def to_dict(self) -> Dict[str, object]:
        return {"action": self.action, "tier": self.tier, "reason": self.reason,
                "spent_usd": round(self.spent_usd, 6), "limit_usd": self.limit_usd}


class BudgetGuard:
    """Decides whether a tenant may spend, and at which tier.

    Checked before the call, using spend recorded so far. It cannot know what the
    pending request will cost, because completion length is not known until the
    model has answered. That means a tenant can overshoot the cap by the cost of
    one in-flight request. Reserving an estimate up front and reconciling after
    would close that gap and is the right thing at scale; it is not done here
    because it doubles the ledger's complexity for a bounded overshoot.
    """

    def __init__(self, ledger: CostLedger, policies: Optional[Dict[str, BudgetPolicy]] = None,
                 default_policy: Optional[BudgetPolicy] = None):
        self.ledger = ledger
        self.policies = dict(policies or {})
        self.default_policy = default_policy

    def policy_for(self, tenant: str) -> Optional[BudgetPolicy]:
        return self.policies.get(tenant, self.default_policy)

    def check(self, tenant: str, tier: str) -> BudgetDecision:
        policy = self.policy_for(tenant)
        if policy is None:
            return BudgetDecision("allow", tier, "no budget configured")

        spent = self.ledger.spend(tenant)
        limit = policy.daily_usd
        used = spent / limit if limit > 0 else float("inf")

        if used >= policy.refuse_at:
            return BudgetDecision(
                "refuse", None,
                f"{tenant} has spent ${spent:.4f} of a ${limit:.4f} daily budget",
                spent, limit)

        if used >= policy.downgrade_at and tier != TIER_ORDER[0]:
            return BudgetDecision(
                "downgrade", TIER_ORDER[0],
                f"{tenant} is at {used:.0%} of budget, downgraded {tier} to {TIER_ORDER[0]}",
                spent, limit)

        return BudgetDecision("allow", tier, f"{tenant} is at {used:.0%} of budget", spent, limit)
