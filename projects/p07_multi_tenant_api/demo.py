"""End-to-end demo: three tenants against one real HTTP server.

Everything below goes over a genuine socket on an ephemeral loopback port and
is driven with urllib, not by calling Gateway.handle directly. The point is to
show that the isolation, throttling and budget behaviour survives the HTTP
layer, because "it works when I call the function" is not the same claim.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import json
import time
import urllib.error
import urllib.request

from projects.p07_multi_tenant_api.gateway import Gateway
from projects.p07_multi_tenant_api.keys import redact
from projects.p07_multi_tenant_api.limits import RateLimitConfig
from projects.p07_multi_tenant_api.server import serve

ALL_SCOPES = ["documents:read", "documents:write", "llm:complete", "usage:read"]


def call(base_url, method, path, key=None, body=None):
    """One HTTP round trip. Returns (status, parsed_json, headers)."""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(base_url + path, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if key:
        req.add_header("Authorization", "Bearer " + key)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read().decode("utf-8")), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8")), dict(e.headers)


def rule(title):
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def main():
    gateway = Gateway(default_rate=RateLimitConfig(burst=20, per_second=10.0))

    # Three tenants with deliberately different postures.
    _, acme_key, acme_rec = gateway.register_tenant(
        "acme", "Acme Corp", ALL_SCOPES, monthly_tokens=4000,
        rate=RateLimitConfig(burst=20, per_second=10.0), plan="growth")
    _, globex_key, globex_rec = gateway.register_tenant(
        "globex", "Globex", ALL_SCOPES, monthly_tokens=260,
        rate=RateLimitConfig(burst=20, per_second=10.0), plan="starter")
    _, initech_key, initech_rec = gateway.register_tenant(
        "initech", "Initech", ["documents:read", "usage:read"], monthly_tokens=1000,
        rate=RateLimitConfig(burst=3, per_second=0.5), plan="starter")

    httpd, base_url, _thread = serve(gateway)
    print("Multi-tenant LLM API listening on " + base_url)

    try:
        rule("1. KEY ISSUANCE  secret is shown once, stored as a salted hash")
        for name, key, rec in (("acme", acme_key, acme_rec),
                               ("globex", globex_key, globex_rec),
                               ("initech", initech_key, initech_rec)):
            print(f"  {name:<8} issued   {key}")
            print(f"  {'':<8} logged   {redact(key)}")
            print(f"  {'':<8} stored   key_id={rec.key_id} hash={rec.secret_hash[:16]}... "
                  f"scopes={sorted(rec.scopes)}")
        print("  the plaintext secret exists nowhere in the server after this point")

        rule("2. AUTHENTICATION")
        status, body, _ = call(base_url, "GET", "/v1/documents")
        print(f"  no key                -> {status} {body['error']}")
        status, body, _ = call(base_url, "GET", "/v1/documents", key="mtk_acme_deadbeef_" + "0" * 32)
        print(f"  forged key            -> {status} {body['error']}")
        status, body, hdrs = call(base_url, "GET", "/v1/documents", key=acme_key)
        print(f"  valid key             -> {status}  X-Request-Id={hdrs.get('X-Request-Id')}")

        rule("3. DATA ISOLATION  tenant B guesses tenant A's document id")
        status, body, _ = call(base_url, "POST", "/v1/documents", acme_key,
                               {"text": "Acme internal: Q3 severance list, do not distribute.",
                                "metadata": {"classification": "confidential"}})
        acme_doc_id = body["document"]["id"]
        print(f"  acme    POST /v1/documents            -> {status} id={acme_doc_id}")
        status, body, _ = call(base_url, "GET", f"/v1/documents/{acme_doc_id}", acme_key)
        print(f"  acme    GET  own document             -> {status} "
              f"text={body['document']['text'][:34]!r}...")
        status, body, _ = call(base_url, "GET", f"/v1/documents/{acme_doc_id}", globex_key)
        print(f"  globex  GET  acme's exact document id -> {status} {body['error']}  "
              f"(404 not 403: no existence oracle)")
        status, body, _ = call(base_url, "GET", "/v1/documents", globex_key)
        print(f"  globex  GET  /v1/documents            -> {status} count={body['count']}")

        rule("4. SCOPES  initech holds a read-only key")
        status, body, _ = call(base_url, "POST", "/v1/documents", initech_key, {"text": "attempt"})
        print(f"  initech POST /v1/documents -> {status} {body['error']}: "
              f"needs {body['required_scope']}, has {body['granted_scopes']}")

        rule("5. RATE LIMITING  initech bucket: burst 3, sustained 0.5/s")
        codes = []
        for i in range(6):
            status, body, hdrs = call(base_url, "GET", "/v1/documents", initech_key)
            codes.append(status)
            extra = f"  Retry-After: {hdrs.get('Retry-After')}s" if status == 429 else ""
            print(f"  request {i + 1}: {status}{extra}")
        first_429 = codes.index(429) + 1 if 429 in codes else None
        print(f"  burst absorbed {codes.count(200)} requests, first 429 at request {first_429}")
        wait = 2.2
        print(f"  sleeping {wait}s to let the bucket refill at 0.5 tokens/s")
        time.sleep(wait)
        status, _, _ = call(base_url, "GET", "/v1/documents", initech_key)
        print(f"  after refill: {status}")

        rule("6. TOKEN BUDGET  globex has a 260 token monthly quota")
        prompt = ("Summarise the onboarding checklist for a new workspace, including "
                  "the steps an administrator must complete before inviting members.")
        n = 0
        while True:
            n += 1
            status, body, _ = call(base_url, "POST", "/v1/complete", globex_key,
                                   {"prompt": prompt, "max_tokens": 64})
            if status == 200:
                u, b = body["usage"], body["budget"]
                print(f"  call {n}: 200  reserved {u['estimated_tokens']} "
                      f"charged {u['actual_tokens']}  committed={b['committed']}/{b['monthly_tokens']} "
                      f"available={b['available']}")
            else:
                b = body["budget"]
                print(f"  call {n}: {status} {body['error']}  needed {body['estimated_tokens']} "
                      f"but only {b['available']} available")
                break
            if n > 10:
                break
        print("  402 is distinct from 429: 429 clears in seconds, 402 clears on quota change")
        print(f"  open (leaked) reservations after settlement: {gateway.budgets.open_reservations()}")

        rule("7. REVOCATION")
        status, _, _ = call(base_url, "GET", "/v1/documents", acme_key)
        print(f"  before revoke -> {status}")
        gateway.keys.revoke(acme_rec.key_id)
        status, body, _ = call(base_url, "GET", "/v1/documents", acme_key)
        print(f"  after revoke  -> {status} {body['message']}")

        rule("8. BILLING REPORT  per tenant, current period")
        for tid, key in (("acme", None), ("globex", globex_key), ("initech", initech_key)):
            report = gateway.meter.billing_report(tid)
            budget = gateway.budgets.state(tid)
            print(f"\n  {tid}")
            print(f"    requests {report['requests']} "
                  f"(billable {report['billable_requests']}, rejected {report['rejected_requests']})")
            print(f"    tokens   prompt {report['prompt_tokens']} "
                  f"completion {report['completion_tokens']} total {report['total_tokens']}")
            print(f"    cost     ${report['cost_usd']:.6f} at price tier '{report['price_tier']}'")
            print(f"    budget   {budget['committed']}/{budget['monthly_tokens']} committed, "
                  f"{budget['available']} available, {budget['rejected']} rejected for quota")
            print(f"    statuses {report['by_status']}")

        rule("9. STRUCTURED LOG  every line carries the request id and a redacted key")
        for line in gateway.logs[-5:]:
            print("  " + json.dumps(line, separators=(",", ":")))
        print(f"\n  {len(gateway.logs)} log lines written, "
              f"{sum(1 for l in gateway.logs if l['request_id'])} with a request id")
    finally:
        httpd.shutdown()
        httpd.server_close()
    print("\nserver stopped")


if __name__ == "__main__":
    main()
