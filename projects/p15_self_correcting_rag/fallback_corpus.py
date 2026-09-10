"""A second, separate corpus that the primary index does not contain.

This exists so the escalation ladder has somewhere real to escalate *to*. Without
it, "fall back to web search" is an untested branch that returns an empty list in
every demo and every test, which is the same as not having built it.

The content is deliberately the kind of material that lives outside a product
documentation set: status-page incident notes, release notes, beta features and
a deprecation notice. Those are exactly the questions that make a documentation
RAG system look broken, because the answer genuinely is not in the docs.

In a real deployment this module is replaced by a search API behind the same
`SearchTool` interface: Tavily, Brave, Exa, SerpAPI, or an internal enterprise
index. Nothing above `SearchTool.search` knows which one it is talking to.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from llmkit.types import Chunk, Document

_DOCS: List[Tuple[str, str, str]] = [
    ("status-2026-03", "Incident report 2026-03-14",
     "On 14 March 2026 the eu-west region served elevated 503 responses for 47 minutes after a "
     "bad configuration push to the edge tier. The trigger was a malformed routing rule that "
     "passed staging validation. Meridian has since added a canary stage that holds a config "
     "change for 10 minutes on one edge node before a fleet-wide rollout."),
    ("release-2026-06", "Release notes June 2026",
     "The June 2026 release adds cursor-based pagination to the audit log endpoint, raises the "
     "webhook payload ceiling from 256 KB to 1 MB, and deprecates the v0 tokens endpoint. "
     "The v0 tokens endpoint stops accepting requests on 1 January 2027."),
    ("beta-graphql", "GraphQL API beta",
     "A GraphQL API is in closed beta for Enterprise workspaces. It exposes read-only queries "
     "over workspaces, members and audit events. Mutations are not implemented and there is no "
     "timeline for them. Beta endpoints are excluded from the uptime SLA."),
    ("cli-tool", "Meridian CLI",
     "The Meridian CLI is distributed through Homebrew and as a static binary. It wraps the same "
     "REST API the SDKs use and stores credentials in the operating system keychain rather than "
     "a dotfile. The CLI requires a token with at least developer scope."),
    ("ip-allowlist", "IP allowlisting",
     "Enterprise workspaces can restrict API access to a list of up to 50 CIDR ranges. "
     "Allowlist changes take effect within 60 seconds. A workspace that allowlists a range it "
     "cannot reach from is locked out and must contact support to recover."),
    ("custom-domains", "Custom domains",
     "Webhook endpoints and hosted pages can use a customer-owned domain. Domain verification is "
     "a TXT record and certificates are issued automatically and renewed 30 days before expiry. "
     "Apex domains are supported through ALIAS records where the DNS provider offers them."),
    ("mobile-sdk", "Mobile SDKs",
     "The iOS and Android SDKs are community maintained and are not covered by the SLA. They "
     "target read-only workloads and do not implement webhook verification, because shipping an "
     "endpoint secret inside a mobile binary is not a safe pattern."),
    ("roadmap-note", "Public roadmap policy",
     "Meridian does not publish dated roadmap commitments. Feature requests are tracked publicly "
     "but carry no delivery date, and a request being accepted is not a commitment to ship it."),
]


def documents() -> List[Document]:
    return [Document(id=d[0], text=d[2], metadata={"title": d[1], "source": "fallback"})
            for d in _DOCS]


def chunks() -> List[Chunk]:
    return [
        Chunk(id=f"fallback:{d.id}#0", doc_id=d.id, text=d.text, ordinal=0,
              metadata=dict(d.metadata))
        for d in documents()
    ]


# Questions the primary corpus genuinely cannot answer but the fallback can.
# These are what prove the ladder reaches its last rung and produces a real
# answer there rather than an empty result.
FALLBACK_GOLD: List[Dict[str, str]] = [
    {"question": "What caused the eu-west outage in March 2026?",
     "doc_id": "status-2026-03", "must_contain": "routing rule"},
    {"question": "When does the v0 tokens endpoint stop working?",
     "doc_id": "release-2026-06", "must_contain": "1 January 2027"},
    {"question": "How many CIDR ranges can an allowlist hold?",
     "doc_id": "ip-allowlist", "must_contain": "50"},
    {"question": "Are the mobile SDKs covered by the SLA?",
     "doc_id": "mobile-sdk", "must_contain": "not covered"},
]
