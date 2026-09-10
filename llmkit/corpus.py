"""A small, self-contained evaluation corpus.

Twenty short documents describing a fictional SaaS platform ("Meridian"), plus a
hand-written gold set of questions with the document that answers each one. It is
fictional on purpose: a public model cannot have memorised it, so a retrieval
score measured against it is a measure of retrieval, not of pretraining.

Used by projects 01, 04, 05, 09 and 15.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from .types import Chunk, Document

_DOCS: List[Tuple[str, str, str]] = [
    ("auth-overview", "Authentication overview",
     "Meridian authenticates every request with a bearer token in the Authorization header. "
     "Tokens are issued per workspace and never per user. A token carries the scopes granted at "
     "creation time and cannot be widened afterwards; to add a scope you mint a new token and "
     "revoke the old one. Tokens are shown exactly once at creation and stored only as a salted hash."),
    ("auth-rotation", "Token rotation",
     "Meridian tokens expire 90 days after creation. A rotation window of 14 days lets the old and new "
     "token both authenticate, so a deploy can roll forward without downtime. Rotation is triggered from "
     "the workspace settings page or with POST /v1/tokens/rotate. Revoking a token takes effect within "
     "30 seconds across all regions."),
    ("rate-limits", "Rate limits",
     "The default rate limit is 600 requests per minute per workspace, burstable to 1000 for 10 seconds. "
     "Exceeding the limit returns HTTP 429 with a Retry-After header in seconds. Rate limits are counted "
     "per workspace, not per token, so adding tokens does not add capacity. Enterprise plans can raise the "
     "sustained limit to 5000 requests per minute on request."),
    ("errors", "Error model",
     "Every Meridian error response carries a stable machine-readable code, a human message and a request_id. "
     "4xx codes are caller errors and should not be retried unchanged. 429 and 5xx are retryable and should "
     "use exponential backoff with jitter. The request_id is the only value support needs to trace a call end to end."),
    ("regions", "Regions and data residency",
     "Meridian runs in three regions: us-east, eu-west and ap-southeast. A workspace is pinned to one region "
     "at creation and cannot be moved; customer data never leaves its region. Cross-region reads are served "
     "from a read-only replica with a lag budget of 5 seconds."),
    ("billing", "Billing and invoices",
     "Meridian bills monthly in arrears on the first of the month. Usage is metered in compute-seconds and "
     "rounded up to the nearest second per request. Invoices are generated in the workspace currency chosen "
     "at signup. A failed payment retries on days 3, 7 and 14 before the workspace is suspended."),
    ("plans", "Plan tiers",
     "There are three plans. Starter includes 100000 compute-seconds and community support. Growth includes "
     "1 million compute-seconds, a 99.9 percent uptime SLA and email support with a 1 business day response. "
     "Enterprise adds a 99.95 percent SLA, single sign-on, audit log export and a named support contact."),
    ("sla", "Service level agreement",
     "The Growth SLA is 99.9 percent monthly uptime, measured as successful responses over total valid requests, "
     "excluding scheduled maintenance announced 72 hours ahead. Breaching the SLA credits 10 percent of the monthly "
     "fee per 0.1 percent below target, capped at 50 percent. Credits must be claimed within 30 days."),
    ("webhooks", "Webhooks",
     "Meridian delivers webhooks at least once, so consumers must be idempotent on the event id. Each delivery is "
     "signed with an HMAC-SHA256 signature over the raw body using the endpoint secret. Failed deliveries retry for "
     "24 hours with exponential backoff. An endpoint failing for 24 hours straight is disabled and the workspace is emailed."),
    ("sdk", "Official SDKs",
     "Meridian publishes official SDKs for Python, TypeScript and Go. The SDKs retry idempotent requests automatically, "
     "honour Retry-After, and expose the request_id on every error object. The Python SDK requires 3.9 or newer. "
     "Community SDKs exist for Ruby and Java but are not covered by the SLA."),
    ("pagination", "Pagination",
     "List endpoints are cursor paginated. Pass limit up to 200 and follow the next_cursor field until it is null. "
     "Offset pagination is not supported because it produces duplicates and gaps when rows are inserted mid-scan. "
     "Cursors are opaque and expire after 24 hours."),
    ("idempotency", "Idempotency keys",
     "Any POST accepts an Idempotency-Key header. Meridian stores the first response for that key for 24 hours and "
     "replays it for repeat requests, so a network timeout can be retried without double-charging. Keys are scoped to "
     "the workspace and the endpoint. Reusing a key with a different body returns 422."),
    ("audit", "Audit log",
     "Audit events record who did what, when, and from which IP, and are retained for 400 days on Enterprise and "
     "90 days otherwise. Events are immutable and exportable as newline-delimited JSON. Audit export is available "
     "only to workspace owners."),
    ("sso", "Single sign-on",
     "Enterprise workspaces can enable SAML 2.0 single sign-on with any IdP supporting SP-initiated flows. "
     "Just-in-time provisioning creates a member on first login with the default role. Enabling SSO does not "
     "disable existing password logins until enforcement is switched on separately."),
    ("roles", "Roles and permissions",
     "There are four roles: owner, admin, developer and viewer. Only owners can delete a workspace, export the audit "
     "log or change billing. Admins manage members and tokens. Developers can deploy and read logs. Viewers have "
     "read-only access to dashboards and no token access at all."),
    ("logs", "Log retention",
     "Request logs are retained for 30 days and are searchable by request_id, status code and endpoint. "
     "Log lines over 64 KB are truncated with a marker. Logs can be streamed to an external sink over HTTPS; "
     "delivery is best effort and is not part of the SLA."),
    ("maintenance", "Scheduled maintenance",
     "Scheduled maintenance windows are announced at least 72 hours ahead on the status page and by email to owners. "
     "Windows are capped at two hours and never overlap between regions, so a multi-region deployment stays available "
     "throughout. Emergency maintenance skips the notice period and is announced when it starts."),
    ("incidents", "Incident response",
     "Severity 1 means total loss of service in a region and pages the on-call within 5 minutes. Severity 2 is degraded "
     "performance affecting many workspaces. A public post-incident review is published within 5 business days for every "
     "severity 1, including timeline, root cause and the corrective actions with owners and dates."),
    ("data-export", "Data export and deletion",
     "A workspace owner can request a full export as newline-delimited JSON; the export link is valid for 7 days. "
     "Deleting a workspace starts a 30 day grace period during which it can be restored, after which data is purged "
     "from primary storage within 24 hours and from backups within 35 days."),
    ("support", "Support channels",
     "Community support is a public forum with no response guarantee. Growth adds email support with a 1 business day "
     "first response. Enterprise adds a named contact, a shared channel and a 1 hour first response for severity 1. "
     "Support cannot access customer data without an explicit, time-boxed grant from an owner."),
]

# question -> (expected doc id, a phrase the answer must contain)
GOLD: List[Tuple[str, str, str]] = [
    ("How long is a Meridian token valid before it expires?", "auth-rotation", "90 days"),
    ("What happens if I go over the request rate limit?", "rate-limits", "429"),
    ("Can I move a workspace to a different region later?", "regions", "cannot be moved"),
    ("How are webhook deliveries authenticated?", "webhooks", "HMAC-SHA256"),
    ("What is the uptime target on the Growth plan?", "plans", "99.9"),
    ("How do I stop a retried payment request from charging twice?", "idempotency", "Idempotency-Key"),
    ("Which role is allowed to export the audit log?", "roles", "owner"),
    ("How long are request logs kept?", "logs", "30 days"),
    ("How much notice is given before scheduled maintenance?", "maintenance", "72 hours"),
    ("When is a public post-incident review published?", "incidents", "5 business days"),
    ("How long can a deleted workspace be restored?", "data-export", "30 day"),
    ("What is the maximum page size on list endpoints?", "pagination", "200"),
    ("Which Python version does the official SDK need?", "sdk", "3.9"),
    ("How is a failed invoice payment retried?", "billing", "3, 7 and 14"),
    ("What does a severity 1 incident mean?", "incidents", "total loss of service"),
]


def documents() -> List[Document]:
    return [Document(id=d[0], text=d[2], metadata={"title": d[1]}) for d in _DOCS]


def chunks() -> List[Chunk]:
    """One chunk per document: these are already chunk-sized."""
    return [
        Chunk(id=f"{d.id}#0", doc_id=d.id, text=d.text, ordinal=0, metadata=dict(d.metadata))
        for d in documents()
    ]


def gold_questions() -> List[Dict[str, str]]:
    return [{"question": q, "doc_id": d, "must_contain": p} for q, d, p in GOLD]


def by_id() -> Dict[str, Document]:
    return {d.id: d for d in documents()}
