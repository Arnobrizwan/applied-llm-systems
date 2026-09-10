"""The policy table: score bands to tiers, with overrides and a hard pin.

A router is a policy problem wearing a scoring problem's clothes. The scorer
produces a number; everything that makes the router usable in a company is in
this file: the enterprise tenant who is paying for the big model and will not
accept "the classifier said your question was easy", the internal batch endpoint
that must never touch the expensive tier, and the escape hatch for the engineer
at 2am who needs one request pinned to a known-good tier right now.

Precedence, highest first, and it is deliberately boring:

  1. hard pin        - explicit on the request, wins over everything
  2. endpoint force  - the endpoint owns the tier, no negotiation
  3. tenant floor    - "never serve me below medium"
  4. tenant ceiling  - "never spend above small on me"
  5. score band      - the default path

Rejected: a single priority number per rule, which is more flexible and which
nobody can reason about after the third rule is added.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .complexity import ComplexityFeatures

TIER_ORDER: List[str] = ["small", "medium", "large"]


def tier_index(tier: str) -> int:
    return TIER_ORDER.index(tier)


def clamp_tier(tier: str, floor: Optional[str] = None, ceiling: Optional[str] = None) -> str:
    idx = tier_index(tier)
    if floor is not None:
        idx = max(idx, tier_index(floor))
    if ceiling is not None:
        idx = min(idx, tier_index(ceiling))
    return TIER_ORDER[idx]


@dataclass
class TenantOverride:
    """Per tenant constraints. Both are optional and both can be set."""

    min_tier: Optional[str] = None
    max_tier: Optional[str] = None
    daily_budget_usd: Optional[float] = None
    note: str = ""


@dataclass
class RoutingDecision:
    tier: str
    reason: str
    band_tier: str
    score: float
    applied: List[str] = field(default_factory=list)
    features: Optional[ComplexityFeatures] = None

    def explain(self) -> str:
        chain = " -> ".join(self.applied) if self.applied else "band only"
        return f"{self.tier} ({self.reason}; {chain})"

    def to_dict(self) -> Dict[str, object]:
        return {"tier": self.tier, "reason": self.reason, "band_tier": self.band_tier,
                "score": round(self.score, 4), "applied": self.applied,
                "features": self.features.to_dict() if self.features else None}


@dataclass
class RoutingPolicy:
    """Score bands plus overrides.

    Bands are upper bounds on the complexity score, in ascending order. The
    defaults put roughly the bottom third on the cheap tier: a band edge is a
    business decision about how much quality risk a saved cent is worth, and
    0.30/0.60 is a starting point to be moved once escalation rates are known,
    not a tuned value.
    """

    bands: List[tuple] = field(default_factory=lambda: [(0.30, "small"), (0.60, "medium"),
                                                        (1.01, "large")])
    tenant_overrides: Dict[str, TenantOverride] = field(default_factory=dict)
    endpoint_overrides: Dict[str, str] = field(default_factory=dict)

    def band_for(self, score: float) -> str:
        for upper, tier in self.bands:
            if score < upper:
                return tier
        return self.bands[-1][1]

    def decide(self, features: ComplexityFeatures, tenant: str = "default",
               endpoint: str = "/chat", pin: Optional[str] = None) -> RoutingDecision:
        band_tier = self.band_for(features.score)
        applied: List[str] = [f"band:{band_tier}"]

        if pin:
            if pin not in TIER_ORDER:
                raise ValueError(f"unknown pinned tier {pin!r}")
            return RoutingDecision(tier=pin, reason="hard pin on the request",
                                   band_tier=band_tier, score=features.score,
                                   applied=applied + [f"pin:{pin}"], features=features)

        forced = self.endpoint_overrides.get(endpoint)
        if forced:
            return RoutingDecision(tier=forced, reason=f"endpoint {endpoint} is forced to {forced}",
                                   band_tier=band_tier, score=features.score,
                                   applied=applied + [f"endpoint:{forced}"], features=features)

        tier = band_tier
        reason = f"score {features.score:.2f} falls in the {band_tier} band"
        override = self.tenant_overrides.get(tenant)
        if override:
            adjusted = clamp_tier(tier, override.min_tier, override.max_tier)
            if adjusted != tier:
                bound = "floor" if tier_index(adjusted) > tier_index(tier) else "ceiling"
                applied.append(f"tenant:{bound}:{adjusted}")
                reason = (f"score {features.score:.2f} put this in {tier}, raised to {adjusted} "
                          f"by the {tenant} tier {bound}") if bound == "floor" else (
                          f"score {features.score:.2f} put this in {tier}, capped at {adjusted} "
                          f"by the {tenant} tier {bound}")
                tier = adjusted

        return RoutingDecision(tier=tier, reason=reason, band_tier=band_tier,
                               score=features.score, applied=applied, features=features)


def fixed_policy(tier: str) -> RoutingPolicy:
    """Everything goes to one tier. Used to measure the always-X baselines."""
    if tier not in TIER_ORDER:
        raise ValueError(f"unknown tier {tier!r}")
    return RoutingPolicy(bands=[(1.01, tier)])


def default_policy() -> RoutingPolicy:
    """The policy the demo measures.

    enterprise never drops below medium because they bought an SLA on answer
    quality. batch is capped at small because it is an overnight backfill where
    latency and quality both matter less than the bill. /classify is forced to
    small because the endpoint only ever emits a label, and /codegen is forced to
    large because a wrong function costs more engineer time than the tier saves.
    """
    return RoutingPolicy(
        tenant_overrides={
            "enterprise": TenantOverride(min_tier="medium", daily_budget_usd=5.0,
                                         note="contractual answer quality floor"),
            "batch": TenantOverride(max_tier="small", daily_budget_usd=0.05,
                                    note="overnight backfill, cost dominates"),
            "acme": TenantOverride(daily_budget_usd=0.20, note="standard plan"),
            "globex": TenantOverride(daily_budget_usd=0.20, note="standard plan"),
        },
        endpoint_overrides={"/classify": "small", "/codegen": "large"},
    )
