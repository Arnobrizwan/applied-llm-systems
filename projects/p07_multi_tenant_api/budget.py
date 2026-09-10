"""Monthly token budgets with pre-flight reservation and post-call reconcile.

The failure this prevents: a tenant with 1,000 tokens left sends a request with
a 100,000 token context. If the gateway only meters after the call, the bill is
already incurred by the time the counter notices. Checking the *estimate*
before dispatch and holding a reservation is the only way a cap is actually a
cap rather than a report.

Two-phase accounting, borrowed from payment authorisation:

  reserve(estimate)  -> funds are held, available balance drops immediately
  commit(actual)     -> the hold converts to a charge at the real amount
  release()          -> the call failed, the hold evaporates

Reserving matters under concurrency. Two simultaneous requests that each check
"is there room?" against the committed total will both pass and both spend; a
hold taken under the same lock as the check cannot double-spend. The reserve
uses the *upper bound* (prompt estimate plus max_tokens) because a reservation
that under-estimates is not a limit.

Token counts come from llmkit.count_message_tokens, which is a calibrated
estimator rather than a real BPE tokenizer, so reservations are approximate on
the way in and exact on the way out. That asymmetry is deliberate and is why
commit reconciles instead of trusting the estimate.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional


class BudgetExceeded(Exception):
    """Raised when a reservation cannot fit inside the remaining monthly quota."""

    def __init__(self, tenant_id: str, requested: int, available: int, period: str):
        super().__init__(
            f"tenant {tenant_id} needs {requested} tokens but only {available} "
            f"remain in period {period}"
        )
        self.tenant_id = tenant_id
        self.requested = requested
        self.available = available
        self.period = period


def month_key(now: Optional[float] = None) -> str:
    t = time.gmtime(now if now is not None else time.time())
    return f"{t.tm_year:04d}-{t.tm_mon:02d}"


@dataclass
class Reservation:
    tenant_id: str
    period: str
    amount: int
    reservation_id: str
    settled: bool = False


@dataclass
class PeriodUsage:
    committed: int = 0
    reserved: int = 0
    calls: int = 0
    rejected: int = 0


@dataclass
class TenantBudget:
    """Monthly token quota for one tenant."""

    tenant_id: str
    monthly_tokens: int
    periods: Dict[str, PeriodUsage] = field(default_factory=dict)

    def period(self, key: str) -> PeriodUsage:
        return self.periods.setdefault(key, PeriodUsage())


class BudgetLedger:
    """All tenants' budgets. One lock, because the checks must be atomic."""

    def __init__(self, clock: Callable[[], float] = time.time):
        self.clock = clock
        self._budgets: Dict[str, TenantBudget] = {}
        self._open: Dict[str, Reservation] = {}
        self._lock = threading.Lock()
        self._counter = 0

    def set_budget(self, tenant_id: str, monthly_tokens: int) -> TenantBudget:
        if monthly_tokens < 0:
            raise ValueError("monthly_tokens cannot be negative")
        with self._lock:
            b = TenantBudget(tenant_id, monthly_tokens)
            self._budgets[tenant_id] = b
            return b

    def _budget(self, tenant_id: str) -> TenantBudget:
        b = self._budgets.get(tenant_id)
        if b is None:
            raise KeyError(f"no budget configured for tenant {tenant_id}")
        return b

    def state(self, tenant_id: str) -> Dict[str, object]:
        with self._lock:
            b = self._budget(tenant_id)
            key = month_key(self.clock())
            p = b.period(key)
            return {
                "period": key,
                "monthly_tokens": b.monthly_tokens,
                "committed": p.committed,
                "reserved": p.reserved,
                "available": max(0, b.monthly_tokens - p.committed - p.reserved),
                "calls": p.calls,
                "rejected": p.rejected,
            }

    def reserve(self, tenant_id: str, estimate: int) -> Reservation:
        """Hold `estimate` tokens or raise BudgetExceeded. Never partially holds."""
        if estimate < 0:
            raise ValueError("estimate cannot be negative")
        with self._lock:
            b = self._budget(tenant_id)
            key = month_key(self.clock())
            p = b.period(key)
            available = b.monthly_tokens - p.committed - p.reserved
            if estimate > available:
                p.rejected += 1
                raise BudgetExceeded(tenant_id, estimate, max(0, available), key)
            p.reserved += estimate
            self._counter += 1
            res = Reservation(tenant_id, key, estimate, f"res_{self._counter:06d}")
            self._open[res.reservation_id] = res
            return res

    def commit(self, reservation: Reservation, actual_tokens: int) -> int:
        """Settle a hold at the real cost. Returns the committed amount.

        The actual can exceed the reservation when the model runs past the
        estimate. We charge the real number rather than clamping to the hold:
        clamping would let a tenant systematically underpay by writing prompts
        the estimator underestimates. The overshoot is allowed to push the
        period slightly over quota, and the *next* reserve then fails, which is
        the correct behaviour for a soft monthly cap on a metered resource.
        """
        if actual_tokens < 0:
            raise ValueError("actual_tokens cannot be negative")
        with self._lock:
            if reservation.settled:
                raise ValueError(f"reservation {reservation.reservation_id} already settled")
            b = self._budget(reservation.tenant_id)
            p = b.period(reservation.period)
            p.reserved = max(0, p.reserved - reservation.amount)
            p.committed += actual_tokens
            p.calls += 1
            reservation.settled = True
            self._open.pop(reservation.reservation_id, None)
            return actual_tokens

    def release(self, reservation: Reservation) -> None:
        """Drop a hold without charging. Used when the downstream call fails."""
        with self._lock:
            if reservation.settled:
                return
            b = self._budget(reservation.tenant_id)
            p = b.period(reservation.period)
            p.reserved = max(0, p.reserved - reservation.amount)
            reservation.settled = True
            self._open.pop(reservation.reservation_id, None)

    def open_reservations(self) -> int:
        """Non-zero after a crash means holds leaked and quota is stuck."""
        with self._lock:
            return len(self._open)
