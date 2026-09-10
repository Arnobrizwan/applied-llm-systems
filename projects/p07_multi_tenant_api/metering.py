"""Per-request usage metering and the monthly billing report.

Every request produces a record, including the ones that were rejected. A
meter that only counts successes cannot answer the two questions support
actually gets asked: "why was I throttled" and "what did I pay for". Rejected
calls carry zero tokens and zero cost but still show up in the counts, so a
tenant hammering into a 429 wall is visible in their own report.

Cost uses llmkit.estimate_cost against a named price tier rather than a
hardcoded rate, so the same code produces a real invoice the moment a paid
model is plugged in. Against the default echo provider every tier resolves to
zero, which is honest: the demo genuinely costs nothing.
"""
from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from llmkit import estimate_cost

from .budget import month_key


@dataclass
class UsageRecord:
    request_id: str
    tenant_id: str
    key_id: str
    route: str
    status: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    period: str = ""
    at: float = field(default_factory=time.time)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["total_tokens"] = self.total_tokens
        d["cost_usd"] = round(self.cost_usd, 6)
        d["latency_ms"] = round(self.latency_ms, 3)
        return d


class UsageMeter:
    def __init__(self, price_tier: str = "small"):
        self.price_tier = price_tier
        self.records: List[UsageRecord] = []
        self._lock = threading.Lock()

    def record(
        self,
        request_id: str,
        tenant_id: str,
        key_id: str,
        route: str,
        status: int,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        latency_ms: float = 0.0,
    ) -> UsageRecord:
        rec = UsageRecord(
            request_id=request_id,
            tenant_id=tenant_id,
            key_id=key_id,
            route=route,
            status=status,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=estimate_cost(self.price_tier, prompt_tokens, completion_tokens),
            latency_ms=latency_ms,
            period=month_key(),
        )
        with self._lock:
            self.records.append(rec)
        return rec

    def for_tenant(self, tenant_id: str) -> List[UsageRecord]:
        with self._lock:
            return [r for r in self.records if r.tenant_id == tenant_id]

    def billing_report(self, tenant_id: str, period: Optional[str] = None) -> Dict[str, Any]:
        """One tenant's invoice for one month, broken down by route and status."""
        period = period or month_key()
        rows = [r for r in self.for_tenant(tenant_id) if r.period == period]
        by_route: Dict[str, Dict[str, Any]] = {}
        by_status: Dict[str, int] = {}
        billable = 0
        for r in rows:
            agg = by_route.setdefault(
                r.route, {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}
            )
            agg["requests"] += 1
            agg["prompt_tokens"] += r.prompt_tokens
            agg["completion_tokens"] += r.completion_tokens
            agg["cost_usd"] = round(agg["cost_usd"] + r.cost_usd, 6)
            by_status[str(r.status)] = by_status.get(str(r.status), 0) + 1
            if 200 <= r.status < 300:
                billable += 1
        return {
            "tenant_id": tenant_id,
            "period": period,
            "price_tier": self.price_tier,
            "requests": len(rows),
            "billable_requests": billable,
            "rejected_requests": len(rows) - billable,
            "prompt_tokens": sum(r.prompt_tokens for r in rows),
            "completion_tokens": sum(r.completion_tokens for r in rows),
            "total_tokens": sum(r.total_tokens for r in rows),
            "cost_usd": round(sum(r.cost_usd for r in rows), 6),
            "by_route": by_route,
            "by_status": by_status,
        }
