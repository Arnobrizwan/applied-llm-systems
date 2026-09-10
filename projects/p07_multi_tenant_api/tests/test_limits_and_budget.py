"""Rate limiting (429) and token budgets (402) are separate failure modes."""
import pytest

from llmkit import EchoLLM, LLMError

from projects.p07_multi_tenant_api.budget import BudgetExceeded, BudgetLedger
from projects.p07_multi_tenant_api.gateway import Gateway
from projects.p07_multi_tenant_api.limits import RateLimitConfig, RateLimiter, TokenBucket

ALL = ["documents:read", "documents:write", "llm:complete", "usage:read"]


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_bucket_absorbs_a_burst_then_refills_at_the_sustained_rate():
    clock = _Clock()
    bucket = TokenBucket(RateLimitConfig(burst=3, per_second=2.0), clock=clock)
    assert [bucket.try_consume()[0] for _ in range(4)] == [True, True, True, False]
    clock.t = 0.5           # 2 tokens/s for half a second is exactly one token
    assert bucket.try_consume()[0] is True
    assert bucket.try_consume()[0] is False
    clock.t = 100.0         # refill is capped at the burst size, not unbounded
    assert bucket.peek() == 3.0


def test_retry_after_is_rounded_up_so_the_client_does_not_wake_too_early():
    clock = _Clock()
    limiter = RateLimiter(RateLimitConfig(burst=1, per_second=1.0), clock=clock)
    assert limiter.check("t")[0] is True
    allowed, retry_after = limiter.check("t")
    assert allowed is False
    assert retry_after == 1               # 1.0s deficit, not 0
    clock.t = 0.9
    assert limiter.check("t")[1] == 1     # 0.1s left still rounds to a whole second


def test_rate_limited_request_returns_429_with_a_retry_after_header():
    g = Gateway(default_rate=RateLimitConfig(burst=2, per_second=0.1))
    _, key, _ = g.register_tenant("acme", "Acme", ALL, monthly_tokens=10_000)
    codes = [g.handle("GET", "/v1/documents", {"Authorization": f"Bearer {key}"}) for _ in range(4)]
    assert [c.status for c in codes] == [200, 200, 429, 429]
    assert int(codes[-1].headers["Retry-After"]) >= 1


def test_budget_exhaustion_returns_402_which_is_distinct_from_429():
    """The two must not be collapsed: they clear on completely different timescales."""
    g = Gateway(default_rate=RateLimitConfig(burst=100, per_second=100.0))
    _, key, _ = g.register_tenant("tiny", "Tiny", ALL, monthly_tokens=40)
    hdr = {"Authorization": f"Bearer {key}"}
    first = g.handle("POST", "/v1/complete", hdr, {"prompt": "hello there", "max_tokens": 8})
    assert first.status == 200
    statuses = {g.handle("POST", "/v1/complete", hdr,
                         {"prompt": "hello there", "max_tokens": 8}).status for _ in range(6)}
    assert 402 in statuses
    assert 429 not in statuses


def test_reservation_uses_the_upper_bound_so_one_huge_request_cannot_overshoot():
    g = Gateway(default_rate=RateLimitConfig(burst=100, per_second=100.0))
    _, key, _ = g.register_tenant("tiny", "Tiny", ALL, monthly_tokens=500)
    hdr = {"Authorization": f"Bearer {key}"}
    resp = g.handle("POST", "/v1/complete", hdr, {"prompt": "summarise this", "max_tokens": 100_000})
    assert resp.status == 402
    assert resp.body["estimated_tokens"] > 500
    # nothing was spent and nothing was left held
    assert g.budgets.state("tiny")["committed"] == 0
    assert g.budgets.open_reservations() == 0


def test_reserve_commit_and_release_keep_the_ledger_balanced():
    ledger = BudgetLedger()
    ledger.set_budget("acme", 100)
    r1 = ledger.reserve("acme", 60)
    assert ledger.state("acme")["available"] == 40
    ledger.commit(r1, 25)                       # actual came in under the estimate
    assert ledger.state("acme")["available"] == 75
    r2 = ledger.reserve("acme", 70)
    ledger.release(r2)                          # call failed, hold must evaporate
    assert ledger.state("acme")["available"] == 75
    assert ledger.open_reservations() == 0
    with pytest.raises(ValueError):
        ledger.commit(r1, 5)                    # double settlement is a bug, not a no-op
    with pytest.raises(BudgetExceeded):
        ledger.reserve("acme", 76)


def test_a_failed_upstream_call_releases_the_hold_rather_than_charging_for_it():
    class Broken(EchoLLM):
        def complete(self, messages, **kwargs):
            raise LLMError("upstream down")

    g = Gateway(llm=Broken(), default_rate=RateLimitConfig(burst=10, per_second=10.0))
    _, key, _ = g.register_tenant("acme", "Acme", ALL, monthly_tokens=1000)
    resp = g.handle("POST", "/v1/complete", {"Authorization": f"Bearer {key}"}, {"prompt": "hi"})
    assert resp.status == 502
    state = g.budgets.state("acme")
    assert state["committed"] == 0 and state["reserved"] == 0
    assert g.budgets.open_reservations() == 0
