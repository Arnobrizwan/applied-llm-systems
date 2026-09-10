"""Deterministic traffic splitting.

The requirement that makes this non-trivial: the same unit id must always land in
the same arm. A splitter that calls `random.random()` per request re-randomises
every user on every request, so a user sees variant A, then B, then A again, the
experiment measures nothing, and the support ticket says "the assistant changes
personality mid-conversation".

The assignment is `sha256(salt + ":" + experiment + ":" + unit)`, taken as an
integer, mapped into [0, 1), then walked against the cumulative arm weights.
Properties that follow:

  * Sticky. Same inputs, same arm, forever, with no state to store. A database of
    assignments would also be sticky and would put a read on the request path and
    a consistency problem behind it.
  * Independent across experiments. The experiment id is in the hash, so a user
    in arm B of one test is not systematically in arm B of the next. Hashing the
    unit id alone is the classic mistake and it correlates every concurrent
    experiment in the system.
  * Reproducible offline. Analysis can recompute assignment from a log without
    the assignment service, which is what makes an A/B result auditable.

`salt` exists so the same experiment can be re-randomised deliberately, for
example when relaunching after a bug fix, without renaming it.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

BUCKETS = 10_000


@dataclass(frozen=True)
class Arm:
    name: str
    version_id: str
    weight: float

    def __post_init__(self) -> None:
        if self.weight <= 0:
            raise ValueError(f"arm {self.name}: weight must be positive")


@dataclass
class TrafficSplitter:
    """Weighted, sticky assignment of unit ids to arms."""

    experiment: str
    arms: Sequence[Arm]
    salt: str = ""

    _cumulative: List[Tuple[float, Arm]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.arms:
            raise ValueError("a split needs at least one arm")
        names = [a.name for a in self.arms]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate arm names: {names}")
        total = sum(a.weight for a in self.arms)
        running = 0.0
        cumulative: List[Tuple[float, Arm]] = []
        for arm in self.arms:
            running += arm.weight / total
            cumulative.append((running, arm))
        # Guard against float drift leaving a sliver above the last boundary.
        cumulative[-1] = (1.0, cumulative[-1][1])
        self._cumulative = cumulative

    def bucket(self, unit_id: str) -> int:
        """Stable integer in [0, BUCKETS) for this unit in this experiment."""
        key = f"{self.salt}:{self.experiment}:{unit_id}".encode("utf-8")
        digest = hashlib.sha256(key).digest()
        return int.from_bytes(digest[:8], "big") % BUCKETS

    def position(self, unit_id: str) -> float:
        return self.bucket(unit_id) / BUCKETS

    def assign(self, unit_id: str) -> Arm:
        pos = self.position(unit_id)
        for boundary, arm in self._cumulative:
            if pos < boundary:
                return arm
        return self._cumulative[-1][1]  # pragma: no cover - unreachable, boundary is 1.0

    def assign_many(self, unit_ids: Sequence[str]) -> Dict[str, Arm]:
        return {unit: self.assign(unit) for unit in unit_ids}

    def distribution(self, unit_ids: Sequence[str]) -> Dict[str, float]:
        """Observed share per arm over a population of unit ids."""
        counts: Dict[str, int] = {a.name: 0 for a in self.arms}
        for unit in unit_ids:
            counts[self.assign(unit).name] += 1
        total = len(unit_ids) or 1
        return {name: count / total for name, count in counts.items()}

    def expected(self) -> Dict[str, float]:
        total = sum(a.weight for a in self.arms)
        return {a.name: a.weight / total for a in self.arms}

    def max_deviation(self, unit_ids: Sequence[str]) -> float:
        """Largest absolute gap between observed and intended share.

        A hash split is not exactly even on a finite population and pretending it
        is invites a false alarm the first time someone eyeballs the numbers.
        This is the number to alert on, with a threshold, not equality.
        """
        observed = self.distribution(unit_ids)
        expected = self.expected()
        return max(abs(observed[name] - expected[name]) for name in expected)
