# 07. Multi-Tenant LLM API

A real HTTP gateway that puts per-tenant keys, rate limits, token budgets and hard
data isolation in front of an LLM, and can produce a billing report afterwards.

## The problem

The first version of an internal LLM service is a single endpoint with a shared
key. It survives contact with a second team and dies on the third. The specific
failures, in the order they usually arrive:

1. One team writes a retry loop with no backoff and consumes the whole rate
   budget for everyone else.
2. A batch job sends a 200,000 token context and the monthly spend is gone in an
   afternoon, discovered when the invoice arrives.
3. Someone builds a document endpoint, forgets a `WHERE tenant_id = ?`, and one
   customer can read another customer's data by guessing an id.
4. A customer asks why they were charged, and there is no per-tenant meter and no
   request id in the logs, so nobody can answer.

Failure 3 is the one that ends a contract. This project treats it as the primary
requirement and the rest as table stakes.

## What this builds

- `keys.py` API keys shown once, stored as a salted hash, prefixed so they are
  identifiable in logs, scoped, and revocable.
- `limits.py` per-tenant token bucket with separate burst and sustained settings,
  returning an accurate `Retry-After`.
- `budget.py` monthly token quota with two-phase accounting: reserve an upper
  bound before the call, reconcile against actual usage after it.
- `isolation.py` tenant-namespaced document storage with no unscoped accessor.
- `metering.py` a usage record per request, including rejected ones, and a
  per-tenant billing report priced through `llmkit.estimate_cost`.
- `gateway.py` the pipeline: request id, authenticate, authorise, throttle,
  reserve budget, dispatch, meter, log.
- `server.py` a thin `http.server` adapter, so the policy stack is testable
  without a socket and still runs over one.

## Architecture

```
  HTTP request
       |
       v
  [ request id ]  req_xxxxxxxxxxxx, attached to the response header and every log line
       |
       v
  [ authenticate ] --- 401 --> unknown, malformed or revoked key
       |
       v
  [ authorise ]    --- 403 --> valid key, missing scope
       |
       v
  [ rate limit ]   --- 429 --> token bucket empty, Retry-After: N
       |
       v
  [ budget reserve ] - 402 --> monthly token quota exhausted
       |
       v
  [ handler ] --> TenantStore keyed by (tenant_id, doc_id)  --- 404 --> not yours
       |            EchoLLM completion
       v
  [ commit actual tokens ] --> UsageMeter --> billing report
```

## Design decisions

**A key embeds its own record id, and verification is one hash comparison.**
The obvious alternative, hashing the presented secret against every stored key
until one matches, makes authentication a table scan that gets slower with every
customer. Embedding a non-secret key id makes it O(1). It also gives the prefix
that makes a leaked key actionable: `mtk_globex_49b5969f_********` in a log line
names the tenant and the exact key to revoke without being a credential itself.

**One fast SHA-256 over salt plus secret, not PBKDF2 or scrypt.** Slow KDFs
exist to make low-entropy human passwords expensive to brute force. These
secrets are 128 bits from `secrets.token_hex`, so a fast hash costs an attacker
nothing they did not already face, and the gateway avoids paying a KDF on every
request. Comparison is still constant time.

**Token bucket, not a fixed window.** A fixed window of 60 per minute permits 60
requests at 11:59:59 and 60 more at 12:00:00, so the real worst case is double
the configured limit inside one second. A bucket separates sustained rate from
burst tolerance, which are the two things an operator actually wants to set. A
sliding window log behaves as well but stores a timestamp per request, which is
unbounded memory per tenant under exactly the attack it is meant to survive.

**Budget is reserved before the call, not metered after it.** Metering after the
fact makes a cap a report. The reservation takes the upper bound (prompt
estimate plus `max_tokens`) under the same lock as the check, so two concurrent
requests cannot both pass the same remaining balance. Reconciliation at commit
time charges the real number, because `llmkit.count_tokens` is a calibrated
estimator and a hold based on an estimate should never become the invoice.

**A cross-tenant read returns 404, not 403.** Returning 403 confirms the id
exists, which turns the endpoint into an existence oracle: tenant B can map
tenant A's id space one request at a time without ever reading a document. The
storage key is the `(tenant_id, doc_id)` pair, so there is no accessor that can
be called without naming an owner. Isolation is structural, not a filter a
developer has to remember.

**402 and 429 are kept distinct.** Both mean "later" to a naive client, but 429
clears in seconds and 402 clears when a human raises a quota. Collapsing them
produces clients that hot-retry into a wall and support tickets that cannot be
triaged. There is a test asserting the two codes never substitute for each other.

## Running it

```bash
python3 projects/p07_multi_tenant_api/demo.py
python3 -m pytest projects/p07_multi_tenant_api -q
```

The demo starts the gateway on an ephemeral loopback port, then drives it with
`urllib` as three tenants with deliberately different postures: `acme` (all
scopes, 4000 token budget), `globex` (all scopes, 260 token budget) and
`initech` (read-only scopes, burst 3 at 0.5 requests per second). It prints key
issuance, authentication, the isolation probe, a scope rejection, the throttle
and its recovery, budget exhaustion, revocation, the billing report and the
structured log.

## Results

Measured by running `demo.py` on Python 3.13, macOS, all traffic over
`127.0.0.1`. Latency figures vary run to run; the status codes and token counts
do not.

**Data isolation.** `acme` stored a document and read it back with status 200.
`globex` requested that same document id and received 404 with the identical
error body a completely fictional id returns. `globex`'s own document list
returned `count=0`. A grounded completion issued by `globex` with
`use_documents: true` did not contain `acme`'s text.

**Rate limiting.** With `burst=3, per_second=0.5`, `initech`'s first three
requests returned 200 and the next three returned 429 with `Retry-After: 2`.
After sleeping 2.2 seconds the next request returned 200.

**Token budget.** `globex` with a 260 token monthly quota: three completions
succeeded, each reserving 95 tokens (12 prompt tokens estimated plus a 64 token
`max_tokens` ceiling plus message overhead) and committing 56 actual tokens, for
168 committed. The fourth returned 402 because 95 was needed and 92 remained.
Open reservations after settlement: 0.

**A single oversized request cannot overshoot.** A tenant with a 500 token quota
sending `max_tokens: 100000` receives 402 with `estimated_tokens` above the
quota, and the committed total stays at 0. This is a test, not a demo line.

**Billing report, `globex`, one period.** 6 requests, 4 billable, 2 rejected;
93 prompt tokens and 75 completion tokens; $0.000059 at the `small` price tier;
statuses `{200: 4, 404: 1, 402: 1}`. The `initech` report shows
`{200: 4, 403: 1, 429: 3}`, so a throttled customer can see their own throttling.

**Logs.** 21 structured lines across the run, all 21 carrying a request id that
matches the `X-Request-Id` header on the corresponding response. Every line
writes the key as `mtk_<tenant>_<keyid>_********`; the secret appears in no line.

**Tests.** 26 tests, including the isolation proof over a real socket, the
402-is-not-429 assertion, ledger balance across reserve, commit and release, and
a failed upstream call releasing its hold rather than charging for it.

## Limits

- Storage is in-memory dictionaries. The `(tenant_id, doc_id)` key is exactly the
  primary key a row store would use, and the interfaces do not change, but there
  is no persistence, no replication and no per-tenant encryption key here.
- Rate limit state is per process. Behind more than one instance the buckets
  need to move to Redis with an atomic script, otherwise the effective limit is
  the configured limit multiplied by the instance count.
- Budgets are a soft monthly cap on a single node with the same caveat. A commit
  that overshoots its reservation is allowed to push the period slightly over
  quota and the next reserve then fails, which is correct for a metered resource
  but is not a hard spend ceiling.
- Reservations live in memory. A process crash between reserve and commit leaks
  a hold. `open_reservations()` exposes the count; a production version needs a
  timeout that sweeps holds older than the longest possible request.
- `estimate_cost` is priced from `llmkit`'s editable price book. Against the
  default echo provider a real bill would be zero; the `small` tier is used here
  so the arithmetic is exercised.
- Token counts come from `llmkit.count_tokens`, a calibrated estimator rather
  than a real BPE tokenizer. The reserve is therefore approximate and the commit
  is exact, which is why the two-phase design exists rather than being optional.
