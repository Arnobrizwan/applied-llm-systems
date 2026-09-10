"""The request pipeline: authenticate, authorise, throttle, budget, dispatch, meter.

Kept deliberately free of `http.server`. `Gateway.handle` takes a method, a
path, headers and a body and returns a `Response`, so the entire policy stack
is unit-testable at microsecond speed and the HTTP layer in `server.py` is a
forty-line adapter. The alternative, putting this logic inside a
`BaseHTTPRequestHandler`, is the standard way a gateway becomes untestable:
every assertion then needs a socket.

Order of the checks is a design decision, not an accident:

  request id -> authenticate -> authorise (scope) -> rate limit -> budget

Authentication comes before rate limiting so an unauthenticated flood cannot
consume a real tenant's bucket. Rate limiting comes before the budget because
the budget check is the more expensive one and because a throttled request
should not disturb quota accounting at all. The budget reservation is last,
immediately before the model call, so the window between holding tokens and
spending them is as short as possible.

Status codes are distinct on purpose:
  401 no or bad or revoked key
  403 valid key, wrong scope
  404 unknown route, or a document this tenant does not own
  402 monthly token budget exhausted
  429 rate limited, with Retry-After

402 and 429 both mean "come back later" to a naive client but they mean very
different things to a customer: 429 clears in seconds, 402 clears when they
raise the quota or the month rolls over. Collapsing them into one code is a
support-ticket generator.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from llmkit import EchoLLM, LLMProvider, count_message_tokens

from .budget import BudgetExceeded, BudgetLedger
from .isolation import TenantStore
from .keys import ApiKeyRecord, ApiKeyStore, AuthError, redact
from .limits import RateLimitConfig, RateLimiter
from .metering import UsageMeter


@dataclass
class Response:
    status: int
    body: Dict[str, Any]
    headers: Dict[str, str] = field(default_factory=dict)

    def json_bytes(self) -> bytes:
        return json.dumps(self.body, indent=2).encode("utf-8")


@dataclass
class Tenant:
    tenant_id: str
    name: str
    plan: str = "starter"


class Gateway:
    """Everything a multi-tenant LLM API needs before the model is reached."""

    def __init__(
        self,
        llm: Optional[LLMProvider] = None,
        default_rate: Optional[RateLimitConfig] = None,
        price_tier: str = "small",
        log_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.llm = llm or EchoLLM()
        self.tenants: Dict[str, Tenant] = {}
        self.keys = ApiKeyStore()
        self.limiter = RateLimiter(default_rate or RateLimitConfig(burst=10, per_second=5.0), clock=clock)
        self.budgets = BudgetLedger()
        self.store = TenantStore()
        self.meter = UsageMeter(price_tier=price_tier)
        self.logs: List[Dict[str, Any]] = []
        self._log_sink = log_sink

    # -- provisioning ----------------------------------------------------
    def register_tenant(
        self,
        tenant_id: str,
        name: str,
        scopes: List[str],
        monthly_tokens: int,
        rate: Optional[RateLimitConfig] = None,
        plan: str = "starter",
    ) -> Tuple[Tenant, str, ApiKeyRecord]:
        """Create a tenant and its first key. The secret is returned once."""
        tenant = Tenant(tenant_id, name, plan)
        self.tenants[tenant_id] = tenant
        self.budgets.set_budget(tenant_id, monthly_tokens)
        if rate is not None:
            self.limiter.configure(tenant_id, rate)
        secret, record = self.keys.issue(tenant_id, scopes, label=f"{name} primary")
        return tenant, secret, record

    # -- logging ---------------------------------------------------------
    def _log(self, **fields: Any) -> None:
        """One structured line per request. The request id is always present.

        Correlating a customer complaint to a server-side event is impossible
        without a shared identifier, so the id goes in the log and in the
        response header, and the key is written redacted so the log is not
        itself a secret.
        """
        line = {"ts": round(time.time(), 3), **fields}
        self.logs.append(line)
        if self._log_sink:
            self._log_sink(line)

    # -- pipeline --------------------------------------------------------
    def handle(
        self,
        method: str,
        path: str,
        headers: Optional[Dict[str, str]] = None,
        body: Optional[Dict[str, Any]] = None,
    ) -> Response:
        started = time.perf_counter()
        request_id = "req_" + uuid.uuid4().hex[:12]
        headers = {k.lower(): v for k, v in (headers or {}).items()}
        route = f"{method.upper()} {path}"
        presented = self._bearer(headers)

        def finish(resp: Response, record: Optional[ApiKeyRecord] = None,
                   prompt_tokens: int = 0, completion_tokens: int = 0) -> Response:
            latency = (time.perf_counter() - started) * 1000.0
            tenant_id = record.tenant_id if record else "-"
            key_id = record.key_id if record else "-"
            self.meter.record(request_id, tenant_id, key_id, route, resp.status,
                              prompt_tokens, completion_tokens, latency)
            self._log(request_id=request_id, tenant=tenant_id, key=redact(presented or ""),
                      route=route, status=resp.status, tokens=prompt_tokens + completion_tokens,
                      latency_ms=round(latency, 3))
            resp.headers["X-Request-Id"] = request_id
            resp.body.setdefault("request_id", request_id)
            return resp

        # 1. authenticate
        try:
            record = self.keys.verify(presented)
        except AuthError as exc:
            return finish(Response(exc.status, {"error": "unauthorized", "message": str(exc)}))

        # 2. route table. Each entry names the scope the route demands and the
        # handler key, kept as a string rather than a bound method because
        # `self._x is self._x` is False in Python and identity dispatch on
        # bound methods is a subtle way to write a bug that only bites later.
        routes: Dict[Tuple[str, str], Tuple[str, str]] = {
            ("POST", "/v1/documents"): ("documents:write", "create_document"),
            ("GET", "/v1/documents"): ("documents:read", "list_documents"),
            ("POST", "/v1/complete"): ("llm:complete", "complete"),
            ("GET", "/v1/usage"): ("usage:read", "usage"),
        }
        method_u = method.upper()
        entry = routes.get((method_u, path))
        doc_id: Optional[str] = None
        if entry is None and method_u == "GET" and path.startswith("/v1/documents/"):
            doc_id = path.rsplit("/", 1)[-1]
            entry = ("documents:read", "get_document")
        if entry is None:
            return finish(Response(404, {"error": "not_found", "message": f"no route {route}"}), record)

        required_scope, action = entry

        # 3. authorise
        if not record.allows(required_scope):
            return finish(
                Response(403, {"error": "forbidden",
                               "message": f"key {record.public_id} lacks scope {required_scope}",
                               "required_scope": required_scope,
                               "granted_scopes": sorted(record.scopes)}),
                record,
            )

        # 4. rate limit
        allowed, retry_after = self.limiter.check(record.tenant_id)
        if not allowed:
            return finish(
                Response(429,
                         {"error": "rate_limited",
                          "message": "per-tenant rate limit exceeded",
                          "retry_after_seconds": retry_after},
                         {"Retry-After": str(retry_after)}),
                record,
            )

        # 5. dispatch
        if action == "complete":
            return self._complete_with_budget(record, body or {}, finish)
        if action == "get_document":
            return finish(self._get_document(record, doc_id or ""), record)
        if action == "create_document":
            return finish(self._create_document(record, body or {}), record)
        if action == "list_documents":
            return finish(self._list_documents(record), record)
        return finish(self._usage(record), record)

    @staticmethod
    def _bearer(headers: Dict[str, str]) -> Optional[str]:
        auth = headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            return auth[7:].strip()
        return headers.get("x-api-key")

    # -- route handlers --------------------------------------------------
    def _create_document(self, record: ApiKeyRecord, body: Dict[str, Any]) -> Response:
        text = (body or {}).get("text")
        if not isinstance(text, str) or not text.strip():
            return Response(400, {"error": "bad_request", "message": "field 'text' is required"})
        doc = self.store.put(record.tenant_id, text, (body or {}).get("metadata"))
        return Response(201, {"document": doc.to_public()})

    def _get_document(self, record: ApiKeyRecord, doc_id: str) -> Response:
        doc = self.store.get(record.tenant_id, doc_id)
        if doc is None:
            # Same body whether the id is unknown or owned by another tenant.
            return Response(404, {"error": "not_found", "message": f"no document {doc_id}"})
        return Response(200, {"document": doc.to_public()})

    def _list_documents(self, record: ApiKeyRecord) -> Response:
        docs = self.store.list(record.tenant_id)
        return Response(200, {"documents": [d.to_public() for d in docs], "count": len(docs)})

    def _usage(self, record: ApiKeyRecord) -> Response:
        return Response(200, {
            "billing": self.meter.billing_report(record.tenant_id),
            "budget": self.budgets.state(record.tenant_id),
            "rate_limit_tokens_remaining": round(self.limiter.remaining(record.tenant_id), 2),
        })

    def _complete_with_budget(self, record: ApiKeyRecord, body: Dict[str, Any], finish) -> Response:
        prompt = body.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            return finish(Response(400, {"error": "bad_request", "message": "field 'prompt' is required"}), record)
        max_tokens = int(body.get("max_tokens", 128))

        # Optional grounding: the tenant's own documents, never anyone else's.
        messages = []
        if body.get("use_documents"):
            hits = self.store.search(record.tenant_id, prompt, limit=3)
            if hits:
                evidence = "\n".join(f"[S{i + 1}] {d.text}" for i, d in enumerate(hits))
                messages.append({"role": "system", "content": evidence})
        messages.append({"role": "user", "content": prompt})

        # Pre-flight: reserve the upper bound before the model is touched.
        estimate = count_message_tokens(messages) + max_tokens
        try:
            reservation = self.budgets.reserve(record.tenant_id, estimate)
        except BudgetExceeded as exc:
            state = self.budgets.state(record.tenant_id)
            return finish(
                Response(402, {
                    "error": "budget_exhausted",
                    "message": str(exc),
                    "estimated_tokens": estimate,
                    "budget": state,
                }),
                record,
            )

        try:
            resp = self.llm.complete(messages, max_tokens=max_tokens)
        except Exception as exc:  # the hold must not survive a failed call
            self.budgets.release(reservation)
            return finish(Response(502, {"error": "upstream_error", "message": str(exc)}), record)

        actual = self.budgets.commit(reservation, resp.total_tokens)
        return finish(
            Response(200, {
                "completion": resp.text,
                "model": resp.model,
                "usage": {
                    "estimated_tokens": estimate,
                    "actual_tokens": actual,
                    "prompt_tokens": resp.prompt_tokens,
                    "completion_tokens": resp.completion_tokens,
                },
                "budget": self.budgets.state(record.tenant_id),
            }),
            record,
            resp.prompt_tokens,
            resp.completion_tokens,
        )
