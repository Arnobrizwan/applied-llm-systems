"""Outcome tracking per arm: success rate, latency, tokens, cost, sample count.

Four things are recorded rather than one, because a prompt change that raises
success by two points and doubles token spend is not a win, and a change that
looks equal on success but halves latency is. A registry that only tracks quality
produces experiments that quietly get more expensive.

Latency is summarised with percentiles, never with a mean alone. The mean of a
latency distribution with a long tail describes a request that nobody made.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from llmkit import percentile


@dataclass
class Outcome:
    """One request's result, attributed to an arm and a version."""

    experiment: str
    arm: str
    version_id: str
    unit_id: str
    success: bool
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float = 0.0
    detail: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> Dict[str, Any]:
        return dict(vars(self))


@dataclass
class ArmSummary:
    arm: str
    version_id: str
    samples: int
    successes: int
    success_rate: float
    latency_p50: float
    latency_p95: float
    mean_total_tokens: float
    total_tokens: int
    cost_usd: float

    def row(self) -> str:
        return (
            f"  {self.arm:<12}{self.version_id:<16}{self.samples:>8}{self.successes:>10}"
            f"{self.success_rate * 100:>10.1f}%{self.latency_p50:>10.2f}{self.latency_p95:>10.2f}"
            f"{self.mean_total_tokens:>10.1f}{self.cost_usd:>10.4f}"
        )

    @staticmethod
    def header() -> str:
        return (
            f"  {'arm':<12}{'version':<16}{'samples':>8}{'success':>10}{'rate':>11}"
            f"{'p50 ms':>10}{'p95 ms':>10}{'tokens':>10}{'cost usd':>10}"
        )


class OutcomeStore:
    """Append-only outcome log with per-arm rollups and JSON persistence."""

    def __init__(self, outcomes: Optional[Iterable[Outcome]] = None):
        self.outcomes: List[Outcome] = list(outcomes or [])

    def __len__(self) -> int:
        return len(self.outcomes)

    def record(self, outcome: Outcome) -> Outcome:
        self.outcomes.append(outcome)
        return outcome

    def for_experiment(self, experiment: str) -> List[Outcome]:
        return [o for o in self.outcomes if o.experiment == experiment]

    def arms(self, experiment: str) -> List[str]:
        seen: List[str] = []
        for outcome in self.for_experiment(experiment):
            if outcome.arm not in seen:
                seen.append(outcome.arm)
        return seen

    def summarise(self, experiment: str, arm: str) -> ArmSummary:
        rows = [o for o in self.for_experiment(experiment) if o.arm == arm]
        if not rows:
            return ArmSummary(arm, "", 0, 0, 0.0, 0.0, 0.0, 0.0, 0, 0.0)
        successes = sum(1 for o in rows if o.success)
        latencies = [o.latency_ms for o in rows]
        tokens = [o.total_tokens for o in rows]
        return ArmSummary(
            arm=arm,
            version_id=rows[0].version_id,
            samples=len(rows),
            successes=successes,
            success_rate=successes / len(rows),
            latency_p50=round(percentile(latencies, 50), 3),
            latency_p95=round(percentile(latencies, 95), 3),
            mean_total_tokens=round(sum(tokens) / len(tokens), 2),
            total_tokens=sum(tokens),
            cost_usd=round(sum(o.cost_usd for o in rows), 6),
        )

    def summaries(self, experiment: str) -> List[ArmSummary]:
        return [self.summarise(experiment, arm) for arm in self.arms(experiment)]

    def save(self, path: str) -> int:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump([o.to_dict() for o in self.outcomes], handle, indent=2)
        return len(self.outcomes)

    @classmethod
    def load(cls, path: str) -> "OutcomeStore":
        if not os.path.exists(path):
            return cls()
        with open(path, encoding="utf-8") as handle:
            rows = json.load(handle)
        return cls(Outcome(**row) for row in rows)
