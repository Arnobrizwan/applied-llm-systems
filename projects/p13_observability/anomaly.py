"""Anomaly detection and alerting.

Three problems, in the order they bite you:

1. A static latency threshold is wrong the day after you set it. Traffic mix
   changes, the prompt gets longer, a new tier is added. So the baseline is
   learned: an EWMA of the mean and the variance, which adapts without keeping
   a window of history per series.
2. One incident emits hundreds of alerts. Every request during a five minute
   outage trips the same rule. The fix is a per-fingerprint cooldown, and the
   number of suppressed events is carried on the next alert so nobody thinks it
   went quiet because it got better.
3. Alerts from different series get conflated. A fingerprint is a hash of the
   rule name plus its labels, so latency on the `generate` step and latency on
   the `judge` step are different alerts with independent cooldowns.

What is deliberately not here: paging policy, escalation chains and routing.
Those belong in an alert manager, and re-implementing a bad version of PagerDuty
inside the application is a well travelled road to nowhere.
"""
from __future__ import annotations

import hashlib
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

SEVERITIES = ("info", "warning", "critical")


class EwmaBaseline:
    """Exponentially weighted mean and variance, updated in constant memory.

    Rejected: a rolling window of the last N values. It is easier to explain and
    it needs N values in memory per series, which at a few thousand series is
    real memory for no extra accuracy. EWMA also degrades more gracefully when a
    series is bursty, because an old spike decays instead of dropping out of the
    window all at once and moving the baseline in one step.

    `alpha` is the smoothing factor. 0.2 means roughly the last ten samples
    dominate. Lower reacts slower and alerts later; higher chases the spike it is
    supposed to be detecting and stops alerting during a sustained incident.
    """

    def __init__(self, alpha: float = 0.2, min_samples: int = 8):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.min_samples = min_samples
        self.count = 0
        self.mean = 0.0
        self.variance = 0.0

    @property
    def std(self) -> float:
        return math.sqrt(max(0.0, self.variance))

    @property
    def ready(self) -> bool:
        """Not enough history is not the same as no anomaly."""
        return self.count >= self.min_samples

    def zscore(self, value: float) -> float:
        """Score `value` against the current baseline, before updating it."""
        if not self.ready:
            return 0.0
        spread = self.std
        if spread <= 1e-9:
            # A perfectly flat series would make every deviation infinite. Fall
            # back to a relative deviation so a flat series can still alert.
            return 0.0 if abs(value - self.mean) < 1e-9 else (value - self.mean) / max(abs(self.mean), 1e-9)
        return (value - self.mean) / spread

    def update(self, value: float) -> None:
        self.count += 1
        if self.count == 1:
            self.mean = value
            return
        previous = self.mean
        self.mean += self.alpha * (value - previous)
        self.variance = (1 - self.alpha) * (self.variance + self.alpha * (value - previous) ** 2)

    def observe(self, value: float) -> float:
        """Score then update. Scoring after updating hides the spike in its own
        baseline, which is the single most common bug in home grown detectors."""
        z = self.zscore(value)
        self.update(value)
        return z


class RollingRate:
    """Rate of a boolean event over the last `window` observations.

    Used for error rate, where a z-score is the wrong tool: errors are rare, the
    baseline standard deviation is near zero, and every single failure would look
    like a twenty sigma event. A rate against a flat threshold is what an on-call
    engineer actually reasons about.
    """

    def __init__(self, window: int = 20):
        self.window = window
        self._events: Deque[bool] = deque(maxlen=window)

    def observe(self, failed: bool) -> float:
        self._events.append(bool(failed))
        return self.rate

    @property
    def rate(self) -> float:
        return sum(1 for e in self._events if e) / len(self._events) if self._events else 0.0

    @property
    def samples(self) -> int:
        return len(self._events)


@dataclass
class AlertRule:
    """One rule. `mode` picks how `threshold` is interpreted."""

    name: str
    metric: str
    severity: str = "warning"
    mode: str = "zscore"          # "zscore" | "threshold"
    threshold: float = 3.0
    direction: str = "above"      # "above" | "below"
    cooldown_s: float = 300.0
    min_samples: int = 8
    description: str = ""

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(f"severity must be one of {SEVERITIES}")
        if self.mode not in ("zscore", "threshold"):
            raise ValueError("mode must be 'zscore' or 'threshold'")
        if self.direction not in ("above", "below"):
            raise ValueError("direction must be 'above' or 'below'")

    def breached(self, value: float, z: float) -> bool:
        """Compare in the rule's own units: sigma for zscore rules, the raw
        metric for threshold rules. A `below` zscore rule compares against the
        negated threshold so every rule can be written with a positive number.
        """
        subject = z if self.mode == "zscore" else value
        if self.direction == "above":
            return subject > self.threshold
        limit = -abs(self.threshold) if self.mode == "zscore" else self.threshold
        return subject < limit


@dataclass
class Alert:
    rule: str
    severity: str
    metric: str
    fingerprint: str
    value: float
    zscore: float
    baseline: float
    labels: Dict[str, str] = field(default_factory=dict)
    at: float = 0.0
    suppressed_since_last: int = 0
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"rule": self.rule, "severity": self.severity, "metric": self.metric,
                "fingerprint": self.fingerprint, "value": round(self.value, 3),
                "zscore": round(self.zscore, 3), "baseline": round(self.baseline, 3),
                "labels": self.labels, "suppressed_since_last": self.suppressed_since_last,
                "message": self.message}


def fingerprint(rule_name: str, labels: Dict[str, str]) -> str:
    """Stable identity for an alert series: the rule plus its labels.

    Not the value and not the timestamp. Including either would give every
    occurrence a new fingerprint, which is the bug that produces 400 alerts for
    one incident in the first place.
    """
    parts = [rule_name] + [f"{k}={labels[k]}" for k in sorted(labels)]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:12]


class AlertManager:
    """Evaluates rules against a stream of metric observations.

    One EWMA baseline per (metric, labels) series, one cooldown per fingerprint.
    The clock is injectable so the cooldown can be tested without sleeping.
    """

    def __init__(self, rules: List[AlertRule], clock: Callable[[], float] = time.monotonic):
        self.rules = list(rules)
        self.clock = clock
        self._baselines: Dict[str, EwmaBaseline] = {}
        self._last_fired: Dict[str, float] = {}
        self._suppressed: Dict[str, int] = {}
        self.fired: List[Alert] = []
        self.suppressed_total = 0

    def _baseline_for(self, rule: AlertRule, labels: Dict[str, str]) -> EwmaBaseline:
        key = fingerprint(rule.metric, labels)
        if key not in self._baselines:
            self._baselines[key] = EwmaBaseline(min_samples=rule.min_samples)
        return self._baselines[key]

    def observe(self, metric: str, value: float,
                labels: Optional[Dict[str, str]] = None) -> List[Alert]:
        """Feed one observation. Returns the alerts that were emitted, if any."""
        labels = dict(labels or {})
        emitted: List[Alert] = []
        for rule in self.rules:
            if rule.metric != metric:
                continue
            baseline = self._baseline_for(rule, labels)
            z = baseline.observe(value)
            if rule.mode == "zscore" and not baseline.ready:
                continue
            if not rule.breached(value, z):
                continue
            fp = fingerprint(rule.name, labels)
            now = self.clock()
            last = self._last_fired.get(fp)
            if last is not None and now - last < rule.cooldown_s:
                self._suppressed[fp] = self._suppressed.get(fp, 0) + 1
                self.suppressed_total += 1
                continue
            alert = Alert(
                rule=rule.name, severity=rule.severity, metric=metric, fingerprint=fp,
                value=value, zscore=z, baseline=baseline.mean, labels=labels, at=now,
                suppressed_since_last=self._suppressed.pop(fp, 0),
                message=self._message(rule, value, z, baseline),
            )
            self._last_fired[fp] = now
            self.fired.append(alert)
            emitted.append(alert)
        return emitted

    @staticmethod
    def _message(rule: AlertRule, value: float, z: float, baseline: EwmaBaseline) -> str:
        if rule.mode == "zscore":
            return (f"{rule.metric} {value:.1f} is {z:.1f} sigma from a baseline of "
                    f"{baseline.mean:.1f} (threshold {rule.threshold:.1f} sigma)")
        return f"{rule.metric} {value:.3f} crossed the {rule.threshold:.3f} threshold"

    def pending_suppressed(self) -> Dict[str, int]:
        """Suppressed counts not yet attached to an alert. Printed at the end of
        a run so an incident that is still inside its cooldown is not invisible."""
        return dict(self._suppressed)

    def summary(self) -> Dict[str, Any]:
        by_severity: Dict[str, int] = {}
        by_rule: Dict[str, int] = {}
        for a in self.fired:
            by_severity[a.severity] = by_severity.get(a.severity, 0) + 1
            by_rule[a.rule] = by_rule.get(a.rule, 0) + 1
        return {
            "alerts_emitted": len(self.fired),
            # suppressed_total already counts every suppression, including the
            # ones still waiting to be attached to a future alert.
            "alerts_suppressed": self.suppressed_total,
            "distinct_fingerprints": len({a.fingerprint for a in self.fired}),
            "by_severity": by_severity,
            "by_rule": by_rule,
        }


def default_rules() -> List[AlertRule]:
    """The starting set. Latency on a learned baseline, errors on a flat rate.

    Thresholds are round numbers chosen to be legible, not tuned: 3 sigma and a
    20 percent error rate. Tuning them needs production traffic, and shipping
    invented "tuned" values would be worse than shipping obvious ones.
    """
    return [
        AlertRule(name="llm_latency_spike", metric="latency_ms", severity="warning",
                  mode="zscore", threshold=3.0, cooldown_s=60.0, min_samples=8,
                  description="per step latency far outside its own learned baseline"),
        AlertRule(name="llm_latency_severe", metric="latency_ms", severity="critical",
                  mode="zscore", threshold=6.0, cooldown_s=60.0, min_samples=8,
                  description="latency so far out that a timeout is likely next"),
        AlertRule(name="error_rate_high", metric="error_rate", severity="critical",
                  mode="threshold", threshold=0.2, cooldown_s=60.0,
                  description="rolling request error rate above 20 percent"),
        AlertRule(name="cost_per_request_high", metric="cost_usd", severity="warning",
                  mode="threshold", threshold=0.02, cooldown_s=60.0,
                  description="a single request cost more than two cents"),
    ]
