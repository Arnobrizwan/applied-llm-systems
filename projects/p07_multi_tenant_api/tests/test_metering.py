"""Request ids, structured logs and the per-tenant billing report."""
from projects.p07_multi_tenant_api.gateway import Gateway
from projects.p07_multi_tenant_api.limits import RateLimitConfig

ALL = ["documents:read", "documents:write", "llm:complete", "usage:read"]


def _gateway():
    g = Gateway(default_rate=RateLimitConfig(burst=50, per_second=50.0))
    _, key, rec = g.register_tenant("acme", "Acme", ALL, monthly_tokens=10_000)
    return g, key, rec


def test_every_response_and_every_log_line_carries_a_request_id():
    g, key, _ = _gateway()
    responses = [
        g.handle("GET", "/v1/documents", {"Authorization": f"Bearer {key}"}),
        g.handle("GET", "/v1/nope", {"Authorization": f"Bearer {key}"}),
        g.handle("GET", "/v1/documents"),                       # unauthenticated
    ]
    ids = [r.headers["X-Request-Id"] for r in responses]
    assert all(i.startswith("req_") for i in ids)
    assert len(set(ids)) == 3                                   # unique per request
    assert all(r.body["request_id"] == r.headers["X-Request-Id"] for r in responses)
    assert [line["request_id"] for line in g.logs] == ids


def test_logs_redact_the_secret_but_keep_the_key_identifiable():
    g, key, rec = _gateway()
    g.handle("GET", "/v1/documents", {"Authorization": f"Bearer {key}"})
    line = g.logs[-1]
    assert key not in str(line)
    assert key.split("_")[-1] not in str(line)
    assert rec.public_id in line["key"]


def test_rejected_requests_are_metered_too():
    """A report that only counts successes cannot explain a throttled customer."""
    g = Gateway(default_rate=RateLimitConfig(burst=1, per_second=0.01))
    _, key, _ = g.register_tenant("acme", "Acme", ALL, monthly_tokens=10_000)
    for _ in range(3):
        g.handle("GET", "/v1/documents", {"Authorization": f"Bearer {key}"})
    report = g.meter.billing_report("acme")
    assert report["requests"] == 3
    assert report["billable_requests"] == 1
    assert report["rejected_requests"] == 2
    assert report["by_status"]["429"] == 2


def test_billing_report_sums_tokens_and_cost_by_route():
    g, key, _ = _gateway()
    hdr = {"Authorization": f"Bearer {key}"}
    g.handle("POST", "/v1/documents", hdr, {"text": "some grounding text"})
    for _ in range(2):
        g.handle("POST", "/v1/complete", hdr, {"prompt": "explain the onboarding flow"})
    report = g.meter.billing_report("acme")
    complete = report["by_route"]["POST /v1/complete"]
    assert complete["requests"] == 2
    assert complete["prompt_tokens"] > 0 and complete["completion_tokens"] > 0
    assert report["total_tokens"] == report["prompt_tokens"] + report["completion_tokens"]
    assert report["cost_usd"] > 0.0            # price tier 'small' is non-zero


def test_billing_is_scoped_to_one_tenant():
    g, key, _ = _gateway()
    _, other, _ = g.register_tenant("globex", "Globex", ALL, monthly_tokens=10_000)
    g.handle("POST", "/v1/complete", {"Authorization": f"Bearer {key}"}, {"prompt": "acme work"})
    g.handle("POST", "/v1/complete", {"Authorization": f"Bearer {other}"}, {"prompt": "globex work"})
    assert g.meter.billing_report("acme")["requests"] == 1
    assert g.meter.billing_report("globex")["requests"] == 1
    assert g.meter.billing_report("nobody")["requests"] == 0


def test_usage_endpoint_reports_budget_and_remaining_rate_limit():
    g, key, _ = _gateway()
    hdr = {"Authorization": f"Bearer {key}"}
    g.handle("POST", "/v1/complete", hdr, {"prompt": "warm the meter"})
    resp = g.handle("GET", "/v1/usage", hdr)
    assert resp.status == 200
    assert resp.body["budget"]["committed"] > 0
    assert resp.body["budget"]["monthly_tokens"] == 10_000
    assert 0.0 <= resp.body["rate_limit_tokens_remaining"] <= 50.0
    assert resp.body["billing"]["tenant_id"] == "acme"
