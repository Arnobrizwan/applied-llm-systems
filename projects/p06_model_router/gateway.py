"""The gateway: route, guard the budget, fall back on failure, escalate on quality.

Order of operations, and each step is where it is for a reason:

  score -> policy -> budget guard -> fallback chain -> quality escalation

Budget is checked before the call, not after, because a guard that discovers the
overspend afterwards is a report, not a guard. Escalation happens after the call
because it needs an answer to judge, which is also why escalation costs the cheap
call plus a judge call plus the expensive call: it is only worth it when the
cheap tier is right often enough to pay for the ones it gets wrong. The demo
reports the escalation rate precisely so that trade can be evaluated rather than
assumed.

Failure handling uses `llmkit.retry` for the transient case and
`llmkit.CircuitBreaker` for the sustained one. Retry alone turns a tier outage
into a slower tier outage: every request pays every retry before failing over.
The breaker makes the second request through a dead tier skip it immediately, and
the reason is recorded on the result rather than swallowed, because "why did this
request cost 3 cents" is a question with an answer.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from llmkit import (
    CircuitBreaker, EchoLLM, LLMProvider, RetryExhausted, RetryPolicy, retry, system, user,
)

# Cross-project reuse: project 04 ships the judge and keeps its public API small
# precisely so it can be imported here. A second, subtly different judge living
# in this project would drift from it within a month.
from projects.p04_eval_harness import EvalCase, RubricJudge

from .budget import BudgetDecision, BudgetGuard, CostLedger
from .complexity import ComplexityFeatures, score_complexity
from .policy import TIER_ORDER, RoutingDecision, RoutingPolicy, default_policy, tier_index


@dataclass
class Request:
    id: str
    prompt: str
    tenant: str = "default"
    endpoint: str = "/chat"
    pin: Optional[str] = None
    task_type: Optional[str] = None


@dataclass
class Attempt:
    tier: str
    outcome: str          # "ok" | "failed" | "circuit_open"
    tries: int = 0
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"tier": self.tier, "outcome": self.outcome, "tries": self.tries,
                "error": self.error}


@dataclass
class GatewayResult:
    request_id: str
    tenant: str
    ok: bool = False
    refused: bool = False
    text: str = ""
    tier_used: Optional[str] = None
    decision: Optional[RoutingDecision] = None
    budget: Optional[BudgetDecision] = None
    attempts: List[Attempt] = field(default_factory=list)
    escalated: bool = False
    escalated_from: Optional[str] = None
    judge_score: Optional[float] = None
    cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reason: str = ""

    @property
    def fallback_events(self) -> int:
        """Attempts that did not serve the request: failures and open breakers."""
        return sum(1 for a in self.attempts if a.outcome != "ok")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id, "tenant": self.tenant, "ok": self.ok,
            "refused": self.refused, "tier_used": self.tier_used,
            "routed_to": self.decision.tier if self.decision else None,
            "score": round(self.decision.score, 4) if self.decision else None,
            "attempts": [a.to_dict() for a in self.attempts],
            "escalated": self.escalated, "escalated_from": self.escalated_from,
            "judge_score": round(self.judge_score, 4) if self.judge_score is not None else None,
            "cost_usd": round(self.cost_usd, 6), "reason": self.reason,
        }


def fallback_chain(tier: str) -> List[str]:
    """Candidates in order: the chosen tier, then nearest capability first.

    Ties break toward the more capable tier. Falling from medium to small is a
    quality cut the caller did not ask for, so when medium is down the first
    alternative is large. Rejected: always falling to the cheapest available tier,
    which is cheaper and turns every upstream incident into a silent quality
    incident.
    """
    home = tier_index(tier)
    others = sorted((t for t in TIER_ORDER if t != tier),
                    key=lambda t: (abs(tier_index(t) - home), -tier_index(t)))
    return [tier] + others


class QualityJudge:
    """Accept or reject a cheap tier answer, using project 04's rubric judge."""

    def __init__(self, llm: Optional[LLMProvider] = None, threshold: float = 0.6):
        self.judge = RubricJudge(llm=llm or EchoLLM(model="echo-judge"))
        self.threshold = threshold
        self.calls = 0

    def accepts(self, prompt: str, answer: str) -> tuple:
        self.calls += 1
        case = EvalCase(id="routing", input=prompt, expected="",
                        metadata={"reference": prompt}, scorers=[])
        verdict = self.judge.judge(case, answer)
        return verdict.overall >= self.threshold, verdict.overall


class ModelRouter:
    """Complexity-based routing with budget guarding, fallback and escalation."""

    def __init__(self, backends: Dict[str, LLMProvider],
                 policy: Optional[RoutingPolicy] = None,
                 ledger: Optional[CostLedger] = None,
                 guard: Optional[BudgetGuard] = None,
                 judge: Optional[QualityJudge] = None,
                 retry_policy: Optional[RetryPolicy] = None,
                 failure_threshold: int = 2,
                 cooldown_s: float = 30.0,
                 clock: Optional[Callable[[], float]] = None,
                 sleep: Callable[[float], None] = lambda _: None):
        missing = [t for t in TIER_ORDER if t not in backends]
        if missing:
            raise ValueError(f"no backend configured for tier(s): {missing}")
        self.backends = backends
        self.policy = policy or default_policy()
        self.ledger = ledger or CostLedger()
        self.guard = guard
        self.judge = judge
        # attempts=2, not the default 4: a fallback tier is a better use of the
        # next 200 ms than a third retry against an upstream that just failed twice.
        self.retry_policy = retry_policy or RetryPolicy(attempts=2, base_delay=0.0)
        self.sleep = sleep
        breaker_kwargs: Dict[str, Any] = {"failure_threshold": failure_threshold,
                                          "cooldown_s": cooldown_s}
        if clock is not None:
            breaker_kwargs["clock"] = clock
        self.breakers = {tier: CircuitBreaker(**breaker_kwargs) for tier in TIER_ORDER}

    # -- internals -------------------------------------------------------
    def _call(self, tier: str, prompt: str) -> Any:
        backend = self.backends[tier]
        return backend.complete([system("You are a helpful assistant."), user(prompt)])

    def _try_chain(self, tiers: Sequence[str], prompt: str,
                   attempts: List[Attempt]) -> tuple:
        """Walk the candidate tiers until one answers. Returns (tier, response)."""
        for tier in tiers:
            breaker = self.breakers[tier]
            if not breaker.allow():
                attempts.append(Attempt(tier=tier, outcome="circuit_open", tries=0,
                                        error="breaker open, tier skipped without a call"))
                continue
            counter = {"n": 0}

            def once():
                counter["n"] += 1
                return self._call(tier, prompt)

            try:
                response = retry(once, policy=self.retry_policy, sleep=self.sleep)
            except RetryExhausted as exc:
                breaker.record_failure()
                attempts.append(Attempt(tier=tier, outcome="failed", tries=counter["n"],
                                        error=str(exc.last_error)))
                continue
            breaker.record_success()
            attempts.append(Attempt(tier=tier, outcome="ok", tries=counter["n"]))
            return tier, response
        return None, None

    # -- public ----------------------------------------------------------
    def handle(self, request: Request) -> GatewayResult:
        result = GatewayResult(request_id=request.id or uuid.uuid4().hex[:8],
                               tenant=request.tenant)
        features: ComplexityFeatures = score_complexity(request.prompt, request.task_type)
        decision = self.policy.decide(features, tenant=request.tenant,
                                      endpoint=request.endpoint, pin=request.pin)
        result.decision = decision
        target = decision.tier

        if self.guard is not None:
            budget = self.guard.check(request.tenant, target)
            result.budget = budget
            if budget.refused:
                # A distinct outcome, not an exception and not a silent downgrade.
                result.refused = True
                result.ok = False
                result.reason = budget.reason
                return result
            if budget.action == "downgrade":
                target = budget.tier or target
                result.reason = budget.reason

        tier, response = self._try_chain(fallback_chain(target), request.prompt, result.attempts)
        if response is None:
            result.ok = False
            result.reason = "every candidate tier failed or was circuit broken"
            return result

        result.ok = True
        result.tier_used = tier
        result.text = response.text
        result.prompt_tokens = response.prompt_tokens
        result.completion_tokens = response.completion_tokens
        result.cost_usd = self.ledger.charge(result.request_id, request.tenant, tier,
                                             response.prompt_tokens, response.completion_tokens)
        if not result.reason:
            result.reason = decision.reason

        if self.judge is not None and tier != TIER_ORDER[-1]:
            accepted, score = self.judge.accepts(request.prompt, response.text)
            result.judge_score = score
            if not accepted:
                result.escalated_from = tier
                higher = TIER_ORDER[tier_index(tier) + 1]
                up_tier, up_response = self._try_chain(fallback_chain(higher), request.prompt,
                                                       result.attempts)
                if up_response is not None:
                    result.escalated = True
                    result.tier_used = up_tier
                    result.text = up_response.text
                    result.prompt_tokens += up_response.prompt_tokens
                    result.completion_tokens += up_response.completion_tokens
                    result.cost_usd += self.ledger.charge(
                        result.request_id, request.tenant, up_tier,
                        up_response.prompt_tokens, up_response.completion_tokens)
                    result.reason = (f"{decision.reason}; judge scored {score:.2f} below "
                                     f"{self.judge.threshold:.2f}, escalated {tier} to {up_tier}")
        return result


def default_backends(latency_ms: float = 0.0) -> Dict[str, LLMProvider]:
    """One offline model per tier.

    The same `EchoLLM` behind all three, named differently. That is honest about
    what this demo can prove: the routing, cost accounting, fallback and
    escalation logic is real and would be unchanged with three different models,
    but the quality difference between tiers is not simulated and no claim is
    made about it.
    """
    return {
        "small": EchoLLM(model="echo-small", latency_ms=latency_ms),
        "medium": EchoLLM(model="echo-medium", latency_ms=latency_ms),
        "large": EchoLLM(model="echo-large", latency_ms=latency_ms),
    }
