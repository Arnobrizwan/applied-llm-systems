"""Hosted adapter for the multi-tenant API gateway.

The project's own demo binds a real loopback socket and drives the gateway with
urllib. A serverless function cannot bind a socket, so this adapter calls
`Gateway.handle(method, path, headers, body)` directly. That is not a reduced
version of the system: `handle` is where authentication, scope checks, the
token bucket, the budget reservation and the metering all live, and `server.py`
is a forty-line HTTP shim on top of it.

Two other adaptations:

* The rate-limit section uses a virtual clock instead of `time.sleep`. The
  demo sleeps 2.2 real seconds to watch a bucket refill at 0.5 tokens/second.
  `RateLimiter` already accepts an injected clock, so advancing a float proves
  the same refill arithmetic in microseconds.
* Nothing is written to disk. The gateway is in-memory by construction.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

NUMBER = 7
SLUG = "multi-tenant-api"
TITLE = "Multi-Tenant LLM API Gateway"
TAGLINE = "Act as one of three tenants on a shared API and watch the gateway decide what you are allowed to do."

WHAT_IT_DOES = """Three companies share one API. Each has its own key, its own
permissions, its own request rate and its own monthly token allowance. Type a
company name and a question and the gateway runs the whole check list on your
request: is this key real, is it allowed to do this, is this company sending
requests too fast, and is there enough allowance left to pay for the answer.

The most important part is the privacy check. One company stores a confidential
document, then a second company asks for that exact document by its id. The
answer it gets back is the same "not found" it would get for an id that never
existed, so it cannot use the API to work out what the first company is storing.
The page shows both replies side by side.

You also see the money side: the allowance is reserved before the model is
called and settled at the real cost afterwards, every request is metered
including the rejected ones, and the billing summary at the bottom is the
invoice that falls out of it."""

INPUT_LABEL = "Pick a company and ask something, or type a scenario word"
PLACEHOLDER = "acme refund policy"

EXAMPLES = [
    "acme refund policy",
    "globex onboarding checklist",
    "initech refund policy",
    "isolation",
]

SOURCE = "projects/p07_multi_tenant_api"

_ALL_SCOPES = ["documents:read", "documents:write", "llm:complete", "usage:read"]
_TENANTS = ("acme", "globex", "initech")
_FICTIONAL_DOC_ID = "0" * 32


class _Clock:
    """A clock the demo can move by hand, so a refill needs no real sleep."""

    def __init__(self, start: float = 1000.0):
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def _wrap(text: str, width: int = 86, indent: str = "        ") -> str:
    words, lines, current = text.split(), [], ""
    for w in words:
        if current and len(current) + 1 + len(w) > width:
            lines.append(current)
            current = w
        else:
            current = f"{current} {w}".strip()
    if current:
        lines.append(current)
    return ("\n" + indent).join(lines)


def _parse(user_input: str) -> Tuple[str, str, str]:
    """Returns (tenant, prompt, scenario)."""
    text = (user_input or "").strip()
    if not text:
        text = EXAMPLES[0]
    lowered = text.lower()
    for word, scenario in (
        ("isolation", "isolation"), ("leak", "isolation"), ("privacy", "isolation"),
        ("rate", "ratelimit"), ("429", "ratelimit"), ("throttl", "ratelimit"),
        ("budget", "budget"), ("402", "budget"), ("quota", "budget"),
        ("scope", "scopes"), ("403", "scopes"), ("permission", "scopes"),
    ):
        if word in lowered:
            first = lowered.split()[0]
            tenant = first if first in _TENANTS else "acme"
            return tenant, "", scenario
    parts = text.split()
    if parts and parts[0].lower() in _TENANTS:
        return parts[0].lower(), " ".join(parts[1:]).strip(), ""
    return "acme", text, ""


def _body_without_request_id(resp: Any) -> Dict[str, Any]:
    body = dict(resp.body)
    body.pop("request_id", None)
    return body


def _canonical(body: Dict[str, Any], doc_id: str) -> bytes:
    """The 404 body with the id the caller itself supplied masked out.

    The message echoes the requested id, so two 404s for two different ids are
    never literally the same bytes. Masking the caller's own id is the honest
    comparison: what is left is everything the server chose to reveal, and if
    those bytes match then the response carries no information about whether
    the document exists.
    """
    text = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return text.replace(doc_id, "<the id the caller sent>").encode("utf-8")


def run(user_input: str) -> str:
    try:
        return _run(user_input)
    except Exception as exc:  # an adapter must never take the page down
        return (f"This demo hit an unexpected error and stopped: "
                f"{type(exc).__name__}: {exc}\n"
                "Nothing was left running; the gateway is in-memory only.")


def _run(user_input: str) -> str:
    from llmkit import EchoLLM

    from projects.p07_multi_tenant_api.gateway import Gateway
    from projects.p07_multi_tenant_api.keys import redact
    from projects.p07_multi_tenant_api.limits import RateLimitConfig

    tenant, prompt, scenario = _parse(user_input)
    out: List[str] = []
    add = out.append

    clock = _Clock()
    gateway = Gateway(llm=EchoLLM(), default_rate=RateLimitConfig(burst=20, per_second=10.0),
                      clock=clock)

    keys: Dict[str, str] = {}
    records: Dict[str, Any] = {}
    _, keys["acme"], records["acme"] = gateway.register_tenant(
        "acme", "Acme Corp", _ALL_SCOPES, monthly_tokens=4000,
        rate=RateLimitConfig(burst=20, per_second=10.0), plan="growth")
    _, keys["globex"], records["globex"] = gateway.register_tenant(
        "globex", "Globex", _ALL_SCOPES, monthly_tokens=260,
        rate=RateLimitConfig(burst=20, per_second=10.0), plan="starter")
    _, keys["initech"], records["initech"] = gateway.register_tenant(
        "initech", "Initech", ["documents:read", "usage:read"], monthly_tokens=1000,
        rate=RateLimitConfig(burst=3, per_second=0.5), plan="starter")

    def call(method: str, path: str, who: Optional[str], body: Optional[Dict[str, Any]] = None):
        headers = {"Authorization": "Bearer " + keys[who]} if who in keys else {}
        return gateway.handle(method, path, headers, body)

    add("MULTI-TENANT API GATEWAY  driven in process, no socket bound")
    add("=" * 78)
    add("Three tenants on one gateway. Ordering of the checks is fixed:")
    add("  authenticate -> authorise (scope) -> rate limit -> reserve budget -> model")
    add("")
    add(f"  {'tenant':<9}{'plan':<9}{'scopes on the key':<30}{'rate limit':<22}"
        f"{'monthly tokens':>14}")
    add("  " + "-" * 84)
    for name, plan, scopes, rate, budget in (
        ("acme", "growth", "read write complete usage", "20 burst, 10 per sec", 4000),
        ("globex", "starter", "read write complete usage", "20 burst, 10 per sec", 260),
        ("initech", "starter", "read usage (read only)", "3 burst, 0.5 per sec", 1000),
    ):
        add(f"  {name:<9}{plan:<9}{scopes:<30}{rate:<22}{budget:>14}")
    add("")
    add(f"  key issued to {tenant}   {keys[tenant]}")
    add(f"  written to the log as   {redact(keys[tenant])}")
    add(f"  stored on the server    salted sha256 {records[tenant].secret_hash[:20]}...")
    add("  the plaintext secret exists nowhere on the server after issue")

    # Seed each tenant's own documents before anything else, so a free-text
    # request can actually be grounded in the caller's own data and so the
    # isolation section has a real row to try to steal.
    created = call("POST", "/v1/documents", "acme",
                   {"text": "Acme internal: Q3 severance list, do not distribute.",
                    "metadata": {"classification": "confidential"}})
    acme_doc_id = created.body["document"]["id"]
    call("POST", "/v1/documents", "acme",
         {"text": "Acme refund policy: a refund is issued within fourteen days of "
                  "purchase, charged back to the original card, and the workspace "
                  "stays active until the end of the paid period."})
    call("POST", "/v1/documents", "globex",
         {"text": "Globex onboarding checklist: an administrator verifies the domain, "
                  "sets the seat count, then invites members by email."})

    # -- 1. the visitor's own request -------------------------------------
    add("")
    add("-" * 78)
    add("1. YOUR REQUEST")
    add("-" * 78)
    if prompt:
        add(f"  tenant   {tenant}")
        add(f"  request  POST /v1/complete   prompt={prompt[:60]!r}")
        before = gateway.budgets.state(tenant)
        remaining_before = gateway.limiter.remaining(tenant)
        resp = call("POST", "/v1/complete", tenant,
                    {"prompt": prompt, "max_tokens": 64, "use_documents": True})
        after = gateway.budgets.state(tenant)
        add("")
        add(f"  authentication      accepted, key {records[tenant].public_id}")
        add(f"  scope required      llm:complete   granted: "
            f"{'yes' if records[tenant].allows('llm:complete') else 'NO'}")
        add(f"  rate limit          {remaining_before:.1f} tokens in the bucket before, "
            f"{gateway.limiter.remaining(tenant):.1f} after")
        if resp.status == 429:
            add("                      (the bucket is checked after the scope check, so an "
                "unauthorised flood cannot drain a real tenant's bucket)")
        add(f"  status              {resp.status}")
        if resp.status == 200:
            grounding = gateway.store.search(tenant, prompt, limit=3)
            add(f"  grounding           {len(grounding)} of {tenant}'s own documents matched "
                f"and went to the model; no other tenant's rows are reachable")
            usage = resp.body["usage"]
            add(f"  budget reserved     {usage['estimated_tokens']} tokens "
                f"(prompt estimate plus the 64 token ceiling), held before the model ran")
            add(f"  budget settled      {usage['actual_tokens']} tokens actually used, "
                f"the hold released and recharged at the real number")
            add(f"  allowance           {before['available']} available before, "
                f"{after['available']} after, out of {after['monthly_tokens']}")
            add(f"  answer              {_wrap(resp.body['completion'][:400], 66, ' ' * 22)}")
        else:
            add(f"  refused             {resp.body.get('error')}: {resp.body.get('message')}")
            if resp.status == 403:
                add("  note                the bucket above is untouched: the scope check runs "
                    "first, so a refused request costs the tenant no rate-limit budget")
                add(f"  needed scope        {resp.body.get('required_scope')}, "
                    f"key holds {resp.body.get('granted_scopes')}")
            if resp.status == 402:
                add(f"  needed {resp.body.get('estimated_tokens')} tokens, "
                    f"{resp.body['budget']['available']} left this month")
        rec = gateway.meter.records[-1]
        add(f"  billing line        request {rec.request_id}  route {rec.route}  "
            f"status {rec.status}  tokens {rec.total_tokens}  cost ${rec.cost_usd:.6f}")
    else:
        add(f"  no free-text request given, running the {scenario} scenario below")

    # -- 2. isolation ------------------------------------------------------
    add("")
    add("-" * 78)
    add("2. TENANT ISOLATION  globex asks for a document acme owns")
    add("-" * 78)
    add(f"  acme    POST /v1/documents                 -> {created.status} "
        f"id={acme_doc_id}")
    own = call("GET", f"/v1/documents/{acme_doc_id}", "acme")
    add(f"  acme    GET  its own document              -> {own.status} "
        f"text={own.body['document']['text'][:38]!r}...")

    stolen = call("GET", f"/v1/documents/{acme_doc_id}", "globex")
    fictional = call("GET", f"/v1/documents/{_FICTIONAL_DOC_ID}", "globex")
    add(f"  globex  GET  acme's real document id       -> {stolen.status} "
        f"{stolen.body['error']}")
    add(f"  globex  GET  an id that never existed      -> {fictional.status} "
        f"{fictional.body['error']}")
    add("")
    a = _canonical(_body_without_request_id(stolen), acme_doc_id)
    b = _canonical(_body_without_request_id(fictional), _FICTIONAL_DOC_ID)
    add("  the two reply bodies, with the id the caller itself supplied masked out")
    add(f"    real id      {a.decode('utf-8')}")
    add(f"    fake id      {b.decode('utf-8')}")
    add(f"    byte identical: {a == b}   ({len(a)} bytes each), "
        f"status {stolen.status} == {fictional.status}")
    add("  so the endpoint cannot be used to test whether a document exists. A 403")
    add("  here would confirm the id is real and let globex map acme's id space.")
    listed = call("GET", "/v1/documents", "globex")
    add(f"  globex  GET  /v1/documents                 -> {listed.status} "
        f"count={listed.body['count']} (its own row only, not acme's two)")
    add("          the store is keyed on (tenant, document), so there is no accessor")
    add("          a developer could call that forgets to name the owner")

    # -- 3. scopes ---------------------------------------------------------
    add("")
    add("-" * 78)
    add("3. PERMISSIONS AND KEYS")
    add("-" * 78)
    no_key = gateway.handle("GET", "/v1/documents", {}, None)
    add(f"  no key at all                        -> {no_key.status} {no_key.body['error']}")
    forged = gateway.handle("GET", "/v1/documents",
                            {"Authorization": "Bearer mtk_acme_deadbeef_" + "0" * 32}, None)
    add(f"  forged key                           -> {forged.status} {forged.body['error']}")
    denied = call("POST", "/v1/documents", "initech", {"text": "attempt"})
    add(f"  initech writes with a read-only key  -> {denied.status} {denied.body['error']}: "
        f"needs {denied.body['required_scope']}")
    add(f"  initech holds                        {denied.body['granted_scopes']}")
    missing = call("GET", "/v1/nope", "acme")
    add(f"  a route that does not exist          -> {missing.status} {missing.body['error']}")

    # -- 4. rate limit -----------------------------------------------------
    add("")
    add("-" * 78)
    add("4. RATE LIMIT  initech's bucket holds 3 and refills at 0.5 per second")
    add("-" * 78)
    codes, retry_after = [], None
    for _ in range(6):
        r = call("GET", "/v1/documents", "initech")
        codes.append(r.status)
        if r.status == 429 and retry_after is None:
            retry_after = r.headers.get("Retry-After")
    first_429 = codes.index(429) + 1 if 429 in codes else None
    add(f"  six requests back to back            {codes}")
    add(f"  burst absorbed                       {codes.count(200)} requests, "
        f"first 429 at request {first_429}")
    add(f"  Retry-After header                   {retry_after} s "
        f"(a 429 with no Retry-After produces a hot retry loop)")
    add("  the clock is moved forward 2.2 s instead of sleeping, which is the same")
    add("  refill arithmetic the bucket does against a real clock")
    clock.advance(2.2)
    after_refill = call("GET", "/v1/documents", "initech")
    add(f"  after the refill                     {after_refill.status}, bucket now holds "
        f"{gateway.limiter.remaining('initech'):.2f} tokens")

    # -- 5. budget ---------------------------------------------------------
    add("")
    add("-" * 78)
    add("5. TOKEN BUDGET  globex has a 260 token monthly allowance")
    add("-" * 78)
    long_prompt = ("Summarise the onboarding checklist for a new workspace, including "
                   "the steps an administrator must complete before inviting members.")
    add(f"  {'call':>5}{'status':>8}{'reserved':>11}{'charged':>10}"
        f"{'committed':>12}{'available':>11}")
    add("  " + "-" * 57)
    for n in range(1, 8):
        r = call("POST", "/v1/complete", "globex", {"prompt": long_prompt, "max_tokens": 64})
        if r.status == 200:
            u, bud = r.body["usage"], r.body["budget"]
            add(f"  {n:>5}{r.status:>8}{u['estimated_tokens']:>11}{u['actual_tokens']:>10}"
                f"{bud['committed']:>12}{bud['available']:>11}")
        else:
            bud = r.body["budget"]
            add(f"  {n:>5}{r.status:>8}{r.body.get('estimated_tokens', 0):>11}{'-':>10}"
                f"{bud['committed']:>12}{bud['available']:>11}   {r.body['error']}")
            break
    add("  the reservation is taken before the model is called, so a request that")
    add("  would blow the cap is refused rather than billed. 402 is not 429: a 429")
    add("  clears in seconds, a 402 clears when the quota changes or the month rolls.")
    add(f"  reservations left open after settlement: {gateway.budgets.open_reservations()} "
        f"(any other number means quota leaked)")

    # -- 6. revocation and billing ----------------------------------------
    add("")
    add("-" * 78)
    add("6. REVOCATION AND THE BILL")
    add("-" * 78)
    before_revoke = call("GET", "/v1/documents", "acme")
    gateway.keys.revoke(records["acme"].key_id)
    after_revoke = call("GET", "/v1/documents", "acme")
    add(f"  acme before revoke -> {before_revoke.status}    after revoke -> "
        f"{after_revoke.status} {after_revoke.body['message']}")
    add("")
    add(f"  {'tenant':<9}{'requests':>9}{'billable':>10}{'rejected':>10}"
        f"{'tokens':>9}{'cost usd':>11}   statuses")
    add("  " + "-" * 76)
    for name in _TENANTS:
        rep = gateway.meter.billing_report(name)
        add(f"  {name:<9}{rep['requests']:>9}{rep['billable_requests']:>10}"
            f"{rep['rejected_requests']:>10}{rep['total_tokens']:>9}"
            f"{rep['cost_usd']:>11.6f}   {rep['by_status']}")
    add("  rejected requests are metered too, at zero cost. A meter that counts only")
    add("  successes cannot answer the question support actually gets asked.")
    add("")
    add("  the last two structured log lines, keys redacted:")
    for line in gateway.logs[-2:]:
        add("    " + json.dumps(line, separators=(",", ":"))[:150])

    add("")
    add(f"  {len(gateway.logs)} requests handled in this page load, all in process.")
    add("  Nothing here bound a port: the same Gateway.handle that server.py calls was")
    add("  called directly, and the rate-limit clock was advanced instead of slept.")
    return "\n".join(out)
