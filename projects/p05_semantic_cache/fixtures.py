"""Labelled fixtures: paraphrases that should hit, near-misses that must not.

This file is the experiment, so it is worth being explicit about how it was
built and what it can and cannot prove.

`PARAPHRASES` are pairs that mean the same thing and should be served from cache.
They are worded the way support tickets actually vary: word order, a different
verb for the same action, a contraction, an extra polite clause. They are *not*
distant paraphrases with no shared vocabulary, because the default embedder here
is `HashingEmbedder`, which is a bag of word and character n-grams and is lexical
by construction. A pair like "when does my key stop working" against "token
expiry period" is a fair test of a real sentence encoder and an unfair test of
this one, so including it would be measuring the embedder rather than the cache.

`NEAR_MISSES` are the interesting half. Every pair is built to be *lexically
almost identical* to its base query and semantically incompatible with it. That
is deliberate: a near-miss that shares no words is rejected by any threshold and
proves nothing. These pairs sit above the similarity threshold, which is exactly
the position where a cosine-only cache silently serves the wrong answer, and they
are the only thing the salience guard is there to catch.

Each near-miss names the class of difference it turns on, so a failure in the
sweep points at which guard check regressed rather than at "something broke".
"""
from __future__ import annotations

from typing import Dict, List, NamedTuple, Tuple

NAMESPACE = "acme"


class Pair(NamedTuple):
    base: str
    probe: str
    kind: str


# Base queries, each with the answer a model produced for it once. The answers
# are short because what matters is whether the wrong one gets served, not how
# well written it is.
BASE_ANSWERS: Dict[str, str] = {
    "How long is the free trial on Meridian?":
        "The free trial runs for 14 days and needs no card.",
    "What is the default rate limit per workspace?":
        "600 requests per minute per workspace, burstable to 1000 for 10 seconds.",
    "How do I rotate a token after 90 days?":
        "Call POST /v1/tokens/rotate; the old token keeps working for the 14 day window.",
    "Which role is allowed to delete a workspace?":
        "Only the owner role can delete a workspace.",
    "How do I enable single sign-on for my workspace?":
        "Enterprise workspaces enable SAML 2.0 SSO from workspace settings.",
    "What is the maximum page size on list endpoints?":
        "The maximum limit on a list endpoint is 200 records per page.",
    "What happens before scheduled maintenance starts?":
        "Owners are emailed and the status page is updated at least 72 hours ahead.",
    "How do I add a member to my workspace?":
        "Invite them from the members page; admins and owners can invite.",
    "What is the uptime target on the Growth plan?":
        "Growth carries a 99.9 percent monthly uptime target.",
    "How is data handled in the us-east region?":
        "Workspace data stays in us-east and never leaves the region it was created in.",
    "How long are request logs retained?":
        "Request logs are retained for 30 days and are searchable by request_id.",
    "How do I read events from the sandbox environment?":
        "Point the SDK at the sandbox base URL and use a sandbox-scoped token.",
}

# Should hit: same intent, different surface form.
PARAPHRASES: List[Pair] = [
    Pair("How long is the free trial on Meridian?",
         "On Meridian, how long is the free trial?", "word order"),
    Pair("What is the default rate limit per workspace?",
         "What is the default per workspace rate limit?", "word order"),
    Pair("How do I rotate a token after 90 days?",
         "How do I rotate a token once it is 90 days old?", "clause rewrite"),
    Pair("Which role is allowed to delete a workspace?",
         "Which role is allowed to delete a workspace account?", "extra noun"),
    Pair("How do I enable single sign-on for my workspace?",
         "How do I enable single sign-on for our workspace?", "pronoun swap"),
    Pair("What is the maximum page size on list endpoints?",
         "On list endpoints, what is the maximum page size?", "word order"),
    Pair("What happens before scheduled maintenance starts?",
         "What happens before a scheduled maintenance window starts?", "extra noun"),
    Pair("How do I add a member to my workspace?",
         "How do I add a new member to my workspace?", "modifier added"),
    Pair("What is the uptime target on the Growth plan?",
         "What is the uptime target for the Growth plan?", "preposition swap"),
    Pair("How is data handled in the us-east region?",
         "In the us-east region, how is data handled?", "word order"),
    Pair("How long are request logs retained?",
         "How long are the request logs retained for?", "function words"),
    Pair("How do I read events from the sandbox environment?",
         "How do I read events out of the sandbox environment?", "preposition swap"),
]

# Must not hit: near-identical wording, incompatible meaning.
NEAR_MISSES: List[Pair] = [
    Pair("How long is the free trial on Meridian?",
         "How long is the paid trial on Meridian?", "contrastive: free vs paid"),
    Pair("What is the default rate limit per workspace?",
         "What is the enterprise rate limit per workspace?", "contrastive: plan tier"),
    Pair("How do I rotate a token after 90 days?",
         "How do I rotate a token after 30 days?", "number: 90 vs 30"),
    Pair("Which role is allowed to delete a workspace?",
         "Which role is allowed to restore a workspace?", "contrastive: delete vs restore"),
    Pair("How do I enable single sign-on for my workspace?",
         "How do I disable single sign-on for my workspace?", "contrastive: enable vs disable"),
    Pair("What is the maximum page size on list endpoints?",
         "What is the minimum page size on list endpoints?", "contrastive: max vs min"),
    Pair("What happens before scheduled maintenance starts?",
         "What happens after scheduled maintenance starts?", "contrastive: before vs after"),
    Pair("How do I add a member to my workspace?",
         "How do I remove a member from my workspace?", "contrastive: add vs remove"),
    Pair("What is the uptime target on the Growth plan?",
         "What is the uptime target on the Starter plan?", "entity: Growth vs Starter"),
    Pair("How is data handled in the us-east region?",
         "How is data handled in the eu-west region?", "contrastive: region"),
    Pair("How long are request logs retained?",
         "How long are request logs retained without an Enterprise plan?", "entity added"),
    Pair("How do I read events from the sandbox environment?",
         "How do I read events from the production environment?",
         "contrastive: sandbox vs production"),
    Pair("Which role is allowed to delete a workspace?",
         "Which role is not allowed to delete a workspace?", "negation"),
    Pair("How do I add a member to my workspace?",
         "How do I add a member without admin rights?", "negation: without"),
]


def base_queries() -> List[str]:
    return list(BASE_ANSWERS)


def labelled_probes() -> List[Tuple[Pair, bool]]:
    """Every probe with its ground-truth label. True means "should be served"."""
    return [(p, True) for p in PARAPHRASES] + [(p, False) for p in NEAR_MISSES]
