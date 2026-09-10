"""A mixed workload of 44 requests, and the harness that measures a router on it.

Hand written rather than generated. A generated workload measures the generator:
if the same template produces every classification request, the scorer's regex
either catches all of them or none, and the tier distribution says more about the
template than about the router. These are written to span the five task types,
four endpoints, five tenants, both override paths, the hard pin escape hatch and
a handful of genuinely ambiguous requests that should not route cleanly.

The mix is deliberately cheap-heavy (more classification and extraction than
reasoning), because that is what a support or internal-tooling workload actually
looks like and it is the shape where routing pays. A workload of nothing but
hard reasoning questions would correctly route everything to the large tier and
save nothing, which is a real result and a boring demo.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence

from .complexity import score_complexity
from .gateway import GatewayResult, ModelRouter, Request
from .policy import TIER_ORDER

WORKLOAD: List[Request] = [
    # -- classification, the cheap majority ------------------------------
    Request("r01", "Classify this ticket as billing, technical or account: my card was declined twice.", "acme", "/classify"),
    Request("r02", "Is this message spam? Win a free cruise, click here now.", "acme", "/classify"),
    Request("r03", "What is the sentiment of this review: the dashboard is fast but the export is broken.", "globex", "/classify"),
    Request("r04", "Label this support email as urgent or routine: our production webhook endpoint is down.", "acme", "/classify"),
    Request("r05", "Which category does this belong to, bug or feature request: the date picker rejects 29 February.", "globex", "/classify"),
    Request("r06", "Yes or no: does this refund request fall inside the 30 day window if the invoice is dated 12 days ago?", "acme", "/classify"),
    Request("r07", "Tag this log line as info, warning or error: retry scheduled in 4 seconds after HTTP 429.", "batch", "/classify"),
    Request("r08", "Route this enquiry to sales or support: we want to add 40 seats next quarter.", "globex", "/classify"),

    # -- extraction ------------------------------------------------------
    Request("r09", "Extract the invoice number, total and due date as JSON from: INV-4471, 320.00 USD, due 2026-10-01.", "acme", "/chat"),
    Request("r10", "Parse the workspace id and region out of this error payload and return them as fields.", "acme", "/chat"),
    Request("r11", "Find all the HTTP status codes mentioned in this incident timeline and list them.", "globex", "/chat"),
    Request("r12", "Extract every email address and phone number from the attached signature block.", "batch", "/chat"),
    Request("r13", "Pull out the retry schedule days from the billing documentation and return them as an array.", "acme", "/chat"),
    Request("r14", "Identify all the endpoints referenced in this changelog entry.", "globex", "/chat"),

    # -- summarisation ---------------------------------------------------
    Request("r15", "Summarise this incident report in 100 words for the status page.", "acme", "/chat"),
    Request("r16", "Give me a tl;dr of the last three release notes.", "globex", "/chat"),
    Request("r17", "Condense this 40 message support thread into the customer's actual problem.", "acme", "/chat"),
    Request("r18", "Summarise the key points of the enterprise SSO documentation in 5 bullets.", "enterprise", "/chat"),
    Request("r19", "Rewrite this error message so a non technical customer understands it.", "globex", "/chat"),
    Request("r20", "Shorten this changelog to one paragraph for the weekly email.", "batch", "/chat"),
    Request("r21", "Summarise the audit log retention rules for a compliance questionnaire.", "enterprise", "/chat"),

    # -- reasoning -------------------------------------------------------
    Request("r22", "Why did our p99 latency get worse after we added the response cache? Explain the likely root cause.", "acme", "/chat"),
    Request("r23", "Compare cursor pagination and offset pagination for our API and explain the trade-offs for a client doing a full scan.", "enterprise", "/chat"),
    Request("r24", "Should we move the workspace to eu-west given data residency requirements? Walk through the implications step by step.", "enterprise", "/chat"),
    Request("r25", "Design a retry strategy for webhook delivery that will not amplify an upstream outage.", "acme", "/chat"),
    Request("r26", "Our error rate rose 3 percent after a deploy. What is the most likely root cause and how would you confirm it?", "globex", "/chat"),
    Request("r27", "Explain why token rotation needs an overlap window and what breaks without one.", "enterprise", "/chat"),
    Request("r28", "Which plan should a customer doing 2 million compute-seconds a month choose, and why?", "globex", "/chat"),
    Request("r29", "Plan a migration from per-user tokens to per-workspace tokens with no downtime.", "enterprise", "/chat"),

    # -- code ------------------------------------------------------------
    Request("r30", "Write a Python function that parses a Retry-After header and returns seconds as an integer.", "acme", "/codegen"),
    Request("r31", "Refactor this handler to use exponential backoff with jitter instead of a fixed sleep.", "acme", "/codegen"),
    Request("r32", "Write a SQL query that selects every workspace with more than 5 failed payments in the last 90 days.", "globex", "/codegen"),
    Request("r33", "Debug this stack trace: KeyError 'next_cursor' raised inside the pagination loop.", "acme", "/codegen"),
    Request("r34", "Write unit tests for a function that validates an idempotency key.", "globex", "/codegen"),
    Request("r35", "Write a regex that matches a Meridian request id and nothing else.", "batch", "/codegen"),

    # -- math and tools --------------------------------------------------
    Request("r36", "Calculate the SLA credit owed for 99.82 percent uptime on a 4000 USD monthly plan.", "enterprise", "/chat"),
    Request("r37", "How much would 1.4 million compute-seconds cost on the Growth plan compared to Enterprise?", "globex", "/chat"),
    Request("r38", "Look up the current status of the eu-west region and tell me if the incident is resolved.", "acme", "/chat"),
    Request("r39", "Search the web for today's guidance on SAML assertion encryption and summarise it.", "enterprise", "/chat"),

    # -- ambiguous and degenerate ----------------------------------------
    Request("r40", "fix it", "acme", "/chat"),
    Request("r41", "it broke again, same thing as last time, can you sort it out somehow", "acme", "/chat"),
    Request("r42", "this doesn't work", "globex", "/chat"),

    # -- explicit escape hatches -----------------------------------------
    Request("r43", "Classify this ticket as billing or technical, but use the big model, this customer is escalated.",
            "acme", "/classify", pin="large"),
    Request("r44", "Summarise the outage for the executive briefing.", "batch", "/chat", pin="medium"),
]


def without_pins(requests: Optional[Sequence[Request]] = None) -> List[Request]:
    """Strip the hard pins, for measuring an always-one-tier baseline.

    A pin is an operational escape hatch that exists under any policy, so leaving
    it in would make an "always small" baseline serve two requests on a bigger
    tier and quietly flatter the routed comparison.
    """
    return [replace(r, pin=None) for r in (requests or WORKLOAD)]


@dataclass
class WorkloadReport:
    label: str
    results: List[GatewayResult] = field(default_factory=list)

    @property
    def served(self) -> List[GatewayResult]:
        return [r for r in self.results if r.ok]

    @property
    def refusals(self) -> int:
        return sum(1 for r in self.results if r.refused)

    @property
    def failures(self) -> int:
        return sum(1 for r in self.results if not r.ok and not r.refused)

    @property
    def total_cost(self) -> float:
        return sum(r.cost_usd for r in self.results)

    @property
    def escalations(self) -> int:
        return sum(1 for r in self.results if r.escalated)

    @property
    def escalation_rate(self) -> float:
        """Escalations over requests that were eligible: served below the top tier."""
        eligible = [r for r in self.results if r.ok and (r.escalated_from or r.judge_score is not None)]
        return len([r for r in eligible if r.escalated]) / len(eligible) if eligible else 0.0

    @property
    def fallback_events(self) -> int:
        return sum(r.fallback_events for r in self.results)

    def tier_distribution(self) -> Dict[str, int]:
        counts = {t: 0 for t in TIER_ORDER}
        for r in self.served:
            if r.tier_used:
                counts[r.tier_used] += 1
        return counts

    def routed_distribution(self) -> Dict[str, int]:
        """Where the policy sent it, before budget, fallback or escalation moved it."""
        counts = {t: 0 for t in TIER_ORDER}
        for r in self.results:
            if r.decision:
                counts[r.decision.tier] += 1
        return counts

    def to_dict(self) -> Dict[str, object]:
        return {"label": self.label, "requests": len(self.results),
                "served": len(self.served), "refusals": self.refusals,
                "failures": self.failures, "total_cost_usd": round(self.total_cost, 6),
                "tier_distribution": self.tier_distribution(),
                "escalations": self.escalations,
                "escalation_rate": round(self.escalation_rate, 4),
                "fallback_events": self.fallback_events}


def run_workload(router: ModelRouter, requests: Optional[Sequence[Request]] = None,
                 label: str = "routed") -> WorkloadReport:
    report = WorkloadReport(label=label)
    for request in (requests or WORKLOAD):
        report.results.append(router.handle(request))
    return report


def workload_profile(requests: Optional[Sequence[Request]] = None) -> Dict[str, object]:
    """Task type and score distribution of the workload itself, before routing."""
    features = [score_complexity(r.prompt, r.task_type) for r in (requests or WORKLOAD)]
    by_task: Dict[str, int] = {}
    for f in features:
        by_task[f.task_type] = by_task.get(f.task_type, 0) + 1
    scores = sorted(f.score for f in features)
    return {
        "requests": len(features),
        "by_task": dict(sorted(by_task.items(), key=lambda kv: -kv[1])),
        "min_score": round(scores[0], 3),
        "median_score": round(scores[len(scores) // 2], 3),
        "max_score": round(scores[-1], 3),
    }
