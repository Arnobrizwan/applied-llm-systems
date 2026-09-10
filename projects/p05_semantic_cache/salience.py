"""The safety check that runs after a similarity hit and before it is served.

The failure this prevents
-------------------------
"How long is the free trial" and "how long is the paid trial" are one word apart.
Any embedding model, hashing or transformer, puts them close together, because
they are close together: same topic, same shape, same intent, one flipped
attribute. A cosine threshold cannot separate them without also rejecting genuine
paraphrases, and this repo has the measurement to prove it. On the shipped
fixture the highest-similarity near-miss ("which role is **not** allowed to
delete a workspace", 0.894) scores higher than several genuine paraphrases, the
lowest of which is 0.703. The two distributions overlap across their whole range.
There is no threshold that separates them. That is the entire argument for this
file.

So the guard does not try to be a better similarity function. It looks only at
the tokens where a one-word change flips the answer, and it refuses the hit when
they disagree. Four classes, each chosen because it changes the correct answer
rather than the topic:

1. **Numbers.** "after 90 days" and "after 30 days" have different answers.
2. **Capitalised entities.** "the Growth plan" and "the Starter plan" have
   different answers. The first word of each sentence is skipped, because a
   sentence-initial capital carries no information.
3. **Negation.** "which role is allowed" and "which role is not allowed" have
   opposite answers, and "without admin rights" changes the question entirely.
4. **Contrastive pairs.** free/paid, enable/disable, min/max, before/after,
   add/remove, sandbox/production, and so on. This class exists because the
   canonical example of the whole problem, free trial against paid trial, is
   invisible to the first three checks: no number, no capital, no negation.

Why this is a curated list and not a model
------------------------------------------
A cross-encoder or an NLI model would generalise past this list, and would also
cost a model call per lookup, which defeats the purpose of a cache. The list is
cheap, auditable, and extendable per deployment through `extra_groups`. It will
miss contrasts nobody wrote down. That is a real limit and it is in the README.

The asymmetry is deliberate: a false miss costs one model call, a false hit
serves a confidently wrong answer to a user. The guard is tuned to refuse when
unsure.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_WORD_RE = re.compile(r"[a-z0-9][a-z0-9_-]*")
_ENTITY_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

NEGATIONS: Set[str] = {
    "not", "no", "never", "none", "without", "cannot", "cant", "wont", "dont",
    "doesnt", "isnt", "arent", "except", "excluding", "unless", "neither", "nor",
    "non",
}

# Each group maps a canonical member to the surface forms that mean it. Surface
# forms are listed rather than stemmed so the behaviour is predictable: a
# stemmer that folded "restore" and "restoration" would also fold things nobody
# checked, and a cache guard is the wrong place to be surprised.
CONTRASTIVE_GROUPS: Dict[str, Dict[str, Sequence[str]]] = {
    "cost": {
        "free": ("free", "complimentary", "unpaid"),
        "paid": ("paid", "billed", "chargeable", "charged"),
    },
    "enablement": {
        "enable": ("enable", "enables", "enabled", "enabling", "activate", "turn-on"),
        "disable": ("disable", "disables", "disabled", "disabling", "deactivate", "turn-off"),
    },
    "membership": {
        "add": ("add", "adds", "adding", "invite", "invites", "inviting"),
        "remove": ("remove", "removes", "removing", "revoke", "revokes", "revoking"),
    },
    "lifecycle": {
        "delete": ("delete", "deletes", "deleting", "purge", "destroy"),
        "restore": ("restore", "restores", "restoring", "recover", "undelete"),
    },
    "bound": {
        "maximum": ("maximum", "max", "largest", "highest", "upper", "most"),
        "minimum": ("minimum", "min", "smallest", "lowest", "lower", "least"),
    },
    "sequence": {
        "before": ("before", "prior", "beforehand", "ahead"),
        "after": ("after", "afterwards", "following", "post"),
    },
    "cadence": {
        "daily": ("daily", "day",),
        "monthly": ("monthly", "month"),
        "annual": ("annual", "annually", "yearly", "year"),
    },
    "environment": {
        "sandbox": ("sandbox", "test", "staging", "development"),
        "production": ("production", "prod", "live"),
    },
    "access": {
        "read": ("read", "reads", "reading", "readonly", "read-only"),
        "write": ("write", "writes", "writing", "readwrite", "read-write"),
    },
    "plan": {
        "starter": ("starter",),
        "growth": ("growth",),
        "enterprise": ("enterprise",),
    },
    "role": {
        "owner": ("owner", "owners"),
        "admin": ("admin", "admins", "administrator"),
        "developer": ("developer", "developers"),
        "viewer": ("viewer", "viewers"),
    },
    "region": {
        "us-east": ("us-east", "useast"),
        "eu-west": ("eu-west", "euwest"),
        "ap-southeast": ("ap-southeast", "apsoutheast"),
    },
    "direction": {
        "increase": ("increase", "raise", "raised", "up", "grow"),
        "decrease": ("decrease", "lower", "lowered", "down", "shrink"),
    },
}


def _build_surface_index(groups: Dict[str, Dict[str, Sequence[str]]]) -> Dict[str, List[Tuple[str, str]]]:
    index: Dict[str, List[Tuple[str, str]]] = {}
    for group, members in groups.items():
        for member, surfaces in members.items():
            for surface in surfaces:
                index.setdefault(surface, []).append((group, member))
    return index


@dataclass
class SalienceReport:
    """Outcome of comparing two queries, with enough detail to debug a refusal."""

    ok: bool
    reason: str = "salient tokens agree"
    check: Optional[str] = None
    incoming: List[str] = field(default_factory=list)
    cached: List[str] = field(default_factory=list)

    def describe(self) -> str:
        if self.ok:
            return self.reason
        return f"{self.check}: incoming={sorted(self.incoming)} cached={sorted(self.cached)}"


class SalienceGuard:
    """Compares the tokens where a one-word difference flips the answer."""

    def __init__(
        self,
        check_numbers: bool = True,
        check_entities: bool = True,
        check_negation: bool = True,
        check_contrastive: bool = True,
        extra_groups: Optional[Dict[str, Dict[str, Sequence[str]]]] = None,
        ignore_entities: Iterable[str] = (),
    ):
        self.check_numbers = check_numbers
        self.check_entities = check_entities
        self.check_negation = check_negation
        self.check_contrastive = check_contrastive
        groups = dict(CONTRASTIVE_GROUPS)
        if extra_groups:
            groups.update(extra_groups)
        self.groups = groups
        self._surface_index = _build_surface_index(groups)
        # Product names appear in every query for a single-product deployment and
        # comparing them is pure noise, so they can be ignored by configuration.
        self.ignore_entities = {e.lower() for e in ignore_entities}

    # -- extractors ------------------------------------------------------
    @staticmethod
    def numbers(text: str) -> Set[str]:
        """Bare numeric literals. Units are not parsed on purpose.

        "90 days" against "90 hours" is a unit difference, not a number
        difference, and it is caught as a plain vocabulary difference by the
        similarity score rather than being worth a units ontology here.
        """
        return set(_NUMBER_RE.findall(text))

    def entities(self, text: str) -> Set[str]:
        """Mid-sentence capitalised tokens, lowercased for comparison."""
        found: Set[str] = set()
        for sentence in _SENTENCE_RE.split(text.strip()):
            words = _ENTITY_RE.findall(sentence)
            for position, word in enumerate(words):
                if position == 0:
                    continue  # sentence-initial capitals carry no signal
                if word[0].isupper():
                    lowered = word.lower()
                    if lowered not in self.ignore_entities:
                        found.add(lowered)
        return found

    @staticmethod
    def negations(text: str) -> Set[str]:
        tokens = _WORD_RE.findall(text.lower().replace("'", ""))
        return {t for t in tokens if t in NEGATIONS}

    def contrastive(self, text: str) -> Dict[str, Set[str]]:
        signature: Dict[str, Set[str]] = {}
        for token in _WORD_RE.findall(text.lower()):
            for group, member in self._surface_index.get(token, ()):
                signature.setdefault(group, set()).add(member)
        return signature

    # -- comparison ------------------------------------------------------
    def compare(self, incoming: str, cached: str) -> SalienceReport:
        if self.check_numbers:
            a, b = self.numbers(incoming), self.numbers(cached)
            if a != b:
                return SalienceReport(False, "numeric literals differ", "numbers",
                                      sorted(a), sorted(b))
        if self.check_entities:
            a, b = self.entities(incoming), self.entities(cached)
            if a != b:
                return SalienceReport(False, "named entities differ", "entities",
                                      sorted(a), sorted(b))
        if self.check_negation:
            a, b = self.negations(incoming), self.negations(cached)
            if a != b:
                return SalienceReport(False, "negation does not match", "negation",
                                      sorted(a), sorted(b))
        if self.check_contrastive:
            sig_a, sig_b = self.contrastive(incoming), self.contrastive(cached)
            for group in set(sig_a) | set(sig_b):
                left, right = sig_a.get(group, set()), sig_b.get(group, set())
                if left != right:
                    # A group present on one side and absent on the other counts as
                    # a mismatch: "the rate limit" and "the enterprise rate limit"
                    # are different questions even though one is a strict subset.
                    return SalienceReport(False, f"contrastive group {group!r} differs",
                                          f"contrastive:{group}", sorted(left), sorted(right))
        return SalienceReport(True)
