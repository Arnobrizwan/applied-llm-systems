"""Semantic memory: extracted facts, deduplicated, with contradictions resolved.

The extractor here is a small set of rules, not a model call. That is a real
simplification and it is stated in the README, but the *shape* is the part worth
building: a triple with a (subject, predicate) key, a confidence, provenance back
to a turn, and an explicit supersession chain. Swapping the rule extractor for an
LLM extractor changes one method and nothing else in this file.

Contradiction policy: a newer fact with the same key supersedes the older one,
and the older one is kept with a pointer rather than deleted. Deleting is
tempting and wrong. "You told me Postgres 14 at turn 2 and Postgres 16 at turn
18" is the answer a user actually wants when they ask why the agent changed its
mind, and it is impossible to give from a store that overwrites.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

from llmkit import keyword_overlap, tokenize

from .records import Fact

# (pattern, subject template, predicate, object group). The subject template may
# reference group 1 with {0}; predicates are fixed so the same statement phrased
# two ways still collides on one key.
_RULES: List[Tuple[re.Pattern, str, str, int]] = [
    (re.compile(r"\bour ([\w][\w \-]{2,28}?) (?:is|are) (?:now )?([^.,;!?]{2,60})", re.I), "our {0}", "is", 2),
    (re.compile(r"\bmy ([\w][\w \-]{2,28}?) (?:is|are) ([^.,;!?]{2,60})", re.I), "my {0}", "is", 2),
    (re.compile(r"\bwe (?:run|use|are using|deploy) ([^.,;!?]{2,60})", re.I), "we", "use", 1),
    (re.compile(r"\bwe (?:are|'re) (?:on|based in) ([^.,;!?]{2,60})", re.I), "we", "are on", 1),
    (re.compile(r"\bi (?:work|am working) (?:at|for) ([^.,;!?]{2,40})", re.I), "i", "work at", 1),
    (re.compile(r"\bi (?:prefer|want|need) ([^.,;!?]{2,60})", re.I), "i", "prefer", 1),
    (re.compile(r"\bcall me ([A-Za-z][\w ]{1,24})", re.I), "i", "am called", 1),
]

_TRAILING = re.compile(r"\s+(?:and|but|so|because|which|that)\b.*$", re.I)
# Split a sentence where a conjunction introduces a new subject, so "our
# workspace is pinned to eu-west and our production database is Postgres 14"
# yields two facts rather than one fact with the second half swallowed into its
# object. Splitting on every "and" would cut clauses that are genuinely one
# claim, so the lookahead requires a subject pronoun after the conjunction.
_CLAUSE_SPLIT = re.compile(r"\s+(?:and|but)\s+(?=(?:our|my|we|i)\b)", re.I)


def _clean(value: str) -> str:
    value = _TRAILING.sub("", value.strip())
    return value.strip(" .,:;\"'").lower()


class FactExtractor:
    """Rule-based (subject, predicate, object) extraction from a turn.

    Only user turns are mined by default. An agent's own reply is a restatement
    of what it already believes, so extracting from it feeds the store its own
    output and quietly promotes a guess into a remembered fact.
    """

    def __init__(self, rules: Optional[Sequence[Tuple[re.Pattern, str, str, int]]] = None):
        self.rules = list(rules or _RULES)

    @staticmethod
    def _matches(pattern: re.Pattern, clauses: Sequence[str]):
        for clause in clauses:
            for match in pattern.finditer(clause):
                yield match

    def extract(self, text: str, turn_index: int) -> List[Fact]:
        found: List[Fact] = []
        seen: set = set()
        clauses = [c for c in _CLAUSE_SPLIT.split(text) if c.strip()]
        for pattern, subject_tpl, predicate, obj_group in self.rules:
            for match in self._matches(pattern, clauses):
                obj = _clean(match.group(obj_group))
                if len(obj.split()) > 12 or len(obj) < 2:
                    continue
                subject = subject_tpl.format(_clean(match.group(1))) if "{0}" in subject_tpl else subject_tpl
                key = (subject, predicate, obj)
                if key in seen:
                    continue
                seen.add(key)
                found.append(
                    Fact(
                        id=f"f{turn_index}.{len(found)}",
                        subject=subject,
                        predicate=predicate,
                        object=obj,
                        turn_index=turn_index,
                        confidence=0.75,
                        evidence=text.strip(),
                    )
                )
        return found


class SemanticMemory:
    """Fact store with deduplication, supersession and provenance."""

    def __init__(self, extractor: Optional[FactExtractor] = None, max_history: int = 5):
        if max_history < 1:
            raise ValueError("max_history must be at least 1")
        self.extractor = extractor or FactExtractor()
        self.max_history = max_history
        self.facts: List[Fact] = []
        self._by_id: Dict[str, Fact] = {}

    def __len__(self) -> int:
        return len(self.facts)

    @property
    def tokens(self) -> int:
        """Every stored fact, including superseded versions."""
        return sum(f.tokens for f in self.facts)

    @property
    def active_tokens(self) -> int:
        """Only the facts that can reach a prompt. This is the budgeted number."""
        return sum(f.tokens for f in self.active)

    @property
    def provenance_tokens(self) -> int:
        return self.tokens - self.active_tokens

    @property
    def active(self) -> List[Fact]:
        return [f for f in self.facts if f.active]

    def get(self, fact_id: str) -> Optional[Fact]:
        return self._by_id.get(fact_id)

    def observe(self, text: str, turn_index: int) -> Tuple[List[Fact], List[Tuple[Fact, Fact]]]:
        """Extract from a turn. Returns (new facts, (old, new) supersessions)."""
        added: List[Fact] = []
        superseded: List[Tuple[Fact, Fact]] = []
        for candidate in self.extractor.extract(text, turn_index):
            existing = self._find_active(candidate.key)
            if existing is None:
                self._insert(candidate)
                added.append(candidate)
                continue
            if self._same_claim(existing, candidate):
                # Deduplication: the same fact restated raises confidence and
                # refreshes provenance, but does not cost another record.
                existing.confidence = min(0.99, existing.confidence + 0.1)
                existing.touch(turn_index)
                continue
            if existing.pinned:
                # A pinned fact is an operator assertion. A user sentence does
                # not get to overwrite it silently; the conflict is surfaced by
                # leaving the pinned fact in place and recording nothing new.
                continue
            candidate.supersedes = existing.id
            existing.superseded_by = candidate.id
            self._insert(candidate)
            self._trim_history(candidate.key)
            added.append(candidate)
            superseded.append((existing, candidate))
        return added, superseded

    def pin(self, fact_id: str) -> Fact:
        fact = self._by_id[fact_id]
        fact.pinned = True
        return fact

    def add_fact(self, fact: Fact) -> Fact:
        self._insert(fact)
        return fact

    def remove(self, fact_id: str) -> bool:
        fact = self._by_id.pop(fact_id, None)
        if fact is None:
            return False
        self.facts = [f for f in self.facts if f.id != fact_id]
        return True

    def history(self, subject: str, predicate: str) -> List[Fact]:
        """Every version of one claim, oldest first. The provenance trail."""
        chain = [f for f in self.facts if f.key == (subject, predicate)]
        chain.sort(key=lambda f: f.turn_index)
        return chain

    def search(self, query: str, k: int = 5, now_turn: int = 0) -> List[Fact]:
        """Lexical match over active facts, scored by overlap and confidence.

        Facts are short strings of the user's own words, so lexical overlap is a
        good match signal here in a way it would not be for long documents. The
        subject and object are weighted separately because a query usually names
        the subject ("which region") and the answer is the object.
        """
        query_terms = set(tokenize(query))
        scored: List[Tuple[float, Fact]] = []
        for fact in self.active:
            subject_hit = keyword_overlap(query, f"{fact.subject} {fact.predicate}")
            object_hit = keyword_overlap(query, fact.object)
            term_hit = len(query_terms & set(tokenize(fact.text))) * 0.1
            score = 1.6 * subject_hit + 0.8 * object_hit + term_hit + 0.1 * fact.confidence
            if score > 0.05:
                scored.append((score, fact))
        scored.sort(key=lambda s: (-s[0], s[1].turn_index))
        hits = [f for _, f in scored[:k]]
        for fact in hits:
            fact.touch(now_turn)
        return hits

    # -- internals -------------------------------------------------------
    def _trim_history(self, key: Tuple[str, str]) -> List[Fact]:
        """Bound one claim's version chain. Oldest superseded versions go first.

        Provenance is deliberately outside the prompt-token budget, so something
        else has to stop it growing without limit across a long-lived session.
        A fixed depth per claim is the simplest bound that still answers the
        question people actually ask, which is "what did I tell you last time",
        not "what did I tell you nine revisions ago".
        """
        chain = [f for f in self.facts if f.key == key and not f.active]
        chain.sort(key=lambda f: f.turn_index)
        dropped: List[Fact] = []
        while len(chain) > self.max_history - 1:
            oldest = chain.pop(0)
            self.remove(oldest.id)
            dropped.append(oldest)
        return dropped

    def _insert(self, fact: Fact) -> None:
        if fact.id in self._by_id:
            fact.id = f"{fact.id}.{len(self.facts)}"
        self.facts.append(fact)
        self._by_id[fact.id] = fact

    def _find_active(self, key: Tuple[str, str]) -> Optional[Fact]:
        for fact in reversed(self.facts):
            if fact.key == key and fact.active:
                return fact
        return None

    @staticmethod
    def _same_claim(a: Fact, b: Fact) -> bool:
        """Restatement, not contradiction.

        Exact match, or one object contained in the other ("eu-west" against
        "eu-west region"). Anything else with the same key is treated as a
        contradiction, which is the safe direction to err in: a false
        contradiction keeps both versions and both are retrievable, while a false
        restatement silently discards the new information.
        """
        if a.object == b.object:
            return True
        short, long_ = sorted([a.object, b.object], key=len)
        return short in long_ and len(short) >= 4
