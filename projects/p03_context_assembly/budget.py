"""The token allocator.

Three passes, in this order:

  1. Floors. Every section that declared a minimum gets it first. A guaranteed
     minimum is how a caller says "if the system prompt is not in the window the
     answer is wrong, not just worse".
  2. Water-filling by priority and marginal value. The remainder is handed out
     in proportion to priority times a diminishing marginal value, capped by each
     section's max_share and by what it actually asked for. Sections that fill up
     hand their surplus back and the loop runs again, so no tokens are stranded
     behind a cap.
  3. Shedding. If the floors alone exceed the budget the lowest-priority
     sections are dropped whole, cheapest first, until they fit.

Rejected alternative: a single proportional split of the whole budget by
priority. It is one line shorter and it fails in the two cases that matter. A
section that only needs 40 tokens still gets its proportional 900 and wastes
them, and a small high-priority section can be allocated less than its floor
because proportionality has no concept of a guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from .sections import Section

MARGINAL_DECAY = 0.5
MAX_ROUNDS = 12


@dataclass
class Allocation:
    """The budget decision for one section, kept for the receipt."""

    name: str
    priority: float
    demand: int
    floor: int
    cap: int
    granted: int = 0
    note: str = ""

    @property
    def satisfied(self) -> bool:
        return self.granted >= self.demand


@dataclass
class BudgetPlan:
    available: int
    allocations: Dict[str, Allocation] = field(default_factory=dict)

    @property
    def granted_total(self) -> int:
        return sum(a.granted for a in self.allocations.values())

    @property
    def demand_total(self) -> int:
        return sum(a.demand for a in self.allocations.values())

    @property
    def unallocated(self) -> int:
        return self.available - self.granted_total


class BudgetAllocator:
    """Turns section policy plus a hard budget into per-section token grants."""

    def __init__(self, marginal_decay: float = MARGINAL_DECAY, max_rounds: int = MAX_ROUNDS):
        if not 0.0 <= marginal_decay < 1.0:
            raise ValueError("marginal_decay must be in [0, 1)")
        self.marginal_decay = marginal_decay
        self.max_rounds = max_rounds

    def _weight(self, section: Section, granted: int, demand: int) -> float:
        """Priority scaled by how much of this section is already funded.

        Diminishing returns are the point: the tenth retrieved document is worth
        much less than the first, so a section that is already 90 percent funded
        should lose the next token to a section that is at 10 percent even if its
        raw priority is higher. Value density breaks ties between sections whose
        priorities are equal but whose items score differently this request.
        """
        fill = min(1.0, granted / demand) if demand else 1.0
        base = section.spec.priority * (1.0 + section.value_density)
        return base * (1.0 - self.marginal_decay * fill)

    def allocate(self, sections: Sequence[Section], available: int) -> BudgetPlan:
        if available < 0:
            raise ValueError("available budget cannot be negative")
        plan = BudgetPlan(available=available)

        live: List[Section] = []
        for s in sections:
            demand = s.demand
            cap = min(demand, int(s.spec.max_share * available)) if demand else 0
            floor = min(s.spec.min_tokens, cap)
            plan.allocations[s.name] = Allocation(
                name=s.name, priority=s.spec.priority, demand=demand, floor=floor, cap=cap
            )
            if demand == 0:
                plan.allocations[s.name].note = "no items supplied"
                continue
            if cap == 0:
                plan.allocations[s.name].note = "max_share rounds to zero tokens at this window size"
                continue
            live.append(s)

        # Pass 3 first, conceptually: make the floors fit before honouring them.
        live.sort(key=lambda s: (-s.spec.priority, s.name))
        while live and sum(plan.allocations[s.name].floor for s in live) > available:
            victim = live.pop()  # lowest priority, last in the sorted list
            alloc = plan.allocations[victim.name]
            alloc.floor = 0
            alloc.cap = 0
            alloc.note = "shed: guaranteed minimums of higher-priority sections filled the window"

        for s in live:
            plan.allocations[s.name].granted = plan.allocations[s.name].floor

        remainder = available - sum(plan.allocations[s.name].granted for s in live)

        for _ in range(self.max_rounds):
            if remainder <= 0:
                break
            open_sections = [s for s in live if plan.allocations[s.name].granted < plan.allocations[s.name].cap]
            if not open_sections:
                break
            weights = {
                s.name: self._weight(s, plan.allocations[s.name].granted, plan.allocations[s.name].demand)
                for s in open_sections
            }
            total_weight = sum(weights.values())
            if total_weight <= 0:
                break
            handed = 0
            for s in open_sections:
                alloc = plan.allocations[s.name]
                share = int(remainder * weights[s.name] / total_weight)
                take = max(0, min(share, alloc.cap - alloc.granted))
                alloc.granted += take
                handed += take
            if handed == 0:
                # Integer rounding stalled the loop with a few tokens left. Give
                # them to the heaviest open section rather than spinning.
                best = max(open_sections, key=lambda s: weights[s.name])
                alloc = plan.allocations[best.name]
                take = min(remainder, alloc.cap - alloc.granted)
                alloc.granted += take
                handed += take
                if take == 0:
                    break
            remainder -= handed

        for s in live:
            alloc = plan.allocations[s.name]
            if not alloc.note:
                alloc.note = "fully funded" if alloc.satisfied else "partially funded, compression required"

        # The invariant this whole module exists to hold.
        if plan.granted_total > available:
            raise AssertionError(f"allocator overran the budget: {plan.granted_total} > {available}")
        return plan
