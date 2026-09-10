"""The promotion gate: the thing that says no.

A gate is worth building only if it refuses. Three conditions have to hold before
a challenger is promoted, and each one exists because of a specific way prompt
experiments go wrong:

  1. Minimum sample size per arm. Prompt A/Bs are usually run by the person who
     wrote variant B, on a dashboard that updates live, and a two-arm test on 30
     requests will show a "winner" most of the time. The floor is a
     pre-registered commitment made before the numbers arrive.
  2. Statistical significance. Two-sided z-test on the success rates, alpha 0.05
     by default. Two-sided because a one-sided test doubles the false-positive
     rate in the direction the experimenter already believes.
  3. A minimum absolute lift. Significance on a large sample can certify a real
     but pointless difference. A prompt change that is significantly better by
     half a point is not worth the deploy, the rollback risk or the cache churn.

Cost and latency are reported but do not block by default, because "better and
slower" is a judgement call that belongs to a human. `max_cost_ratio` and
`max_latency_ratio` make that a policy when a team wants it enforced.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .outcomes import ArmSummary, OutcomeStore
from .stats import ProportionTest, required_sample_size, two_proportion_z_test, wilson_interval


@dataclass
class GateDecision:
    """A yes or a no, with the numbers that produced it."""

    promote: bool
    reasons: List[str] = field(default_factory=list)
    control: Optional[ArmSummary] = None
    challenger: Optional[ArmSummary] = None
    test: Optional[ProportionTest] = None
    control_interval: tuple = (0.0, 0.0)
    challenger_interval: tuple = (0.0, 0.0)
    needed_per_arm: Optional[int] = None

    @property
    def verdict(self) -> str:
        return "PROMOTE" if self.promote else "HOLD"

    def render(self) -> str:
        lines = [f"  verdict: {self.verdict}"]
        if self.control and self.challenger:
            lines.append(
                f"  control    {self.control.arm:<10} {self.control.successes}/{self.control.samples} "
                f"= {self.control.success_rate * 100:.1f}%  "
                f"95% CI [{self.control_interval[0] * 100:.1f}%, {self.control_interval[1] * 100:.1f}%]"
            )
            lines.append(
                f"  challenger {self.challenger.arm:<10} {self.challenger.successes}/{self.challenger.samples} "
                f"= {self.challenger.success_rate * 100:.1f}%  "
                f"95% CI [{self.challenger_interval[0] * 100:.1f}%, {self.challenger_interval[1] * 100:.1f}%]"
            )
        if self.test:
            lines.append(
                f"  absolute lift {self.test.absolute_lift * 100:+.1f} points, "
                f"z = {self.test.z:.3f}, p = {self.test.p_value:.5f}"
            )
        for reason in self.reasons:
            lines.append(f"  - {reason}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "promote": self.promote,
            "reasons": list(self.reasons),
            "control": vars(self.control) if self.control else None,
            "challenger": vars(self.challenger) if self.challenger else None,
            "p_value": self.test.p_value if self.test else None,
            "z": self.test.z if self.test else None,
            "absolute_lift": self.test.absolute_lift if self.test else None,
            "needed_per_arm": self.needed_per_arm,
        }


@dataclass
class PromotionGate:
    """Evaluates one challenger against one control."""

    min_samples_per_arm: int = 200
    alpha: float = 0.05
    min_absolute_lift: float = 0.05
    max_cost_ratio: Optional[float] = None
    max_latency_ratio: Optional[float] = None

    def __post_init__(self) -> None:
        if self.min_samples_per_arm < 2:
            raise ValueError("min_samples_per_arm must be at least 2")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if self.min_absolute_lift < 0:
            raise ValueError("min_absolute_lift cannot be negative")

    def evaluate(self, store: OutcomeStore, experiment: str, control: str, challenger: str) -> GateDecision:
        a = store.summarise(experiment, control)
        b = store.summarise(experiment, challenger)
        decision = GateDecision(promote=False, control=a, challenger=b)

        if a.samples < self.min_samples_per_arm or b.samples < self.min_samples_per_arm:
            decision.reasons.append(
                f"not enough data: {a.arm} has {a.samples} and {b.arm} has {b.samples}, "
                f"the gate requires {self.min_samples_per_arm} per arm"
            )
            if a.samples and b.samples:
                decision.control_interval = wilson_interval(a.successes, a.samples)
                decision.challenger_interval = wilson_interval(b.successes, b.samples)
                decision.test = two_proportion_z_test(a.successes, a.samples, b.successes, b.samples)
            decision.needed_per_arm = required_sample_size(
                a.success_rate or 0.5, max(self.min_absolute_lift, 0.01), alpha=self.alpha
            )
            return decision

        test = two_proportion_z_test(a.successes, a.samples, b.successes, b.samples)
        decision.test = test
        decision.control_interval = wilson_interval(a.successes, a.samples)
        decision.challenger_interval = wilson_interval(b.successes, b.samples)

        if test.absolute_lift <= 0:
            decision.reasons.append(
                f"challenger is not ahead: {b.success_rate * 100:.1f}% against {a.success_rate * 100:.1f}%"
            )
            return decision
        if not test.significant_at(self.alpha):
            decision.reasons.append(
                f"difference is not significant: p = {test.p_value:.4f}, threshold {self.alpha}"
            )
            decision.needed_per_arm = required_sample_size(
                a.success_rate, max(test.absolute_lift, 0.01), alpha=self.alpha
            )
            return decision
        if test.absolute_lift < self.min_absolute_lift:
            decision.reasons.append(
                f"lift of {test.absolute_lift * 100:.1f} points is real but below the "
                f"{self.min_absolute_lift * 100:.1f} point bar for a deploy"
            )
            return decision

        if self.max_cost_ratio is not None and a.cost_usd > 0:
            ratio = b.cost_usd / a.cost_usd
            if ratio > self.max_cost_ratio:
                decision.reasons.append(
                    f"cost ratio {ratio:.2f} exceeds the {self.max_cost_ratio:.2f} ceiling"
                )
                return decision
        if self.max_latency_ratio is not None and a.latency_p95 > 0:
            ratio = b.latency_p95 / a.latency_p95
            if ratio > self.max_latency_ratio:
                decision.reasons.append(
                    f"p95 latency ratio {ratio:.2f} exceeds the {self.max_latency_ratio:.2f} ceiling"
                )
                return decision

        decision.promote = True
        decision.reasons.append(
            f"{b.arm} beats {a.arm} by {test.absolute_lift * 100:.1f} points "
            f"(p = {test.p_value:.5f}) on {a.samples} and {b.samples} samples"
        )
        return decision
