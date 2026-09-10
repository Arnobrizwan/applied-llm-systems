"""Eviction: a decay score over recency, access frequency and importance.

A hard token budget on long-lived memory is not optional. A memory store with no
ceiling is a memory leak with a nicer name: it grows with session length, and the
first symptom is a context assembly step quietly dropping the documents it needed
because memory ate the budget.

Score, all three terms normalised to roughly [0, 1]:

    score = 0.45 * recency + 0.25 * frequency + 0.30 * importance

Recency is an exponential decay over turns since last use, not since creation.
That distinction is the whole point: a fact from turn 2 that the agent looked up
at turn 29 is recent, and a fact from turn 28 nobody has touched is not.
Frequency is log-scaled so a record accessed twenty times does not become
permanently unevictable. Importance is set by the store that owns the record.

Pinned facts are never evicted and are not scored at all. A pin is a caller
saying "this is a correctness requirement, not a preference", and an eviction
policy that can override that is an eviction policy nobody will pin against.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple

RECENCY_WEIGHT = 0.45
FREQUENCY_WEIGHT = 0.25
IMPORTANCE_WEIGHT = 0.30

# Facts outrank episodes at equal decay. A fact is the distilled output of the
# whole pipeline: roughly fifteen tokens carrying one complete claim, already
# deduplicated and with its contradictions resolved. An episode is roughly
# seventy tokens of compressed conversation that may or may not contain a claim
# at all. Evicting the fact to keep the episode spends more budget to retain less
# information, so the multiplier makes that trade explicit rather than leaving it
# to whichever record happened to be touched more recently.
KIND_WEIGHT = {"fact": 1.4, "episode": 1.0}


@dataclass
class EvictionCandidate:
    record: Any
    kind: str  # "fact" | "episode"
    score: float
    tokens: int

    @property
    def id(self) -> str:
        return self.record.id


def decay_score(record: Any, now_turn: int, half_life: float = 15.0, importance: float = 0.5) -> float:
    """Blended keep-value for one record. Higher means keep."""
    if half_life <= 0:
        raise ValueError("half_life must be positive")
    last_used = max(getattr(record, "last_used_turn", 0), getattr(record, "turn_index", 0),
                    getattr(record, "last_turn", 0))
    age = max(0, now_turn - last_used)
    recency = math.exp(-age / half_life)
    frequency = math.log1p(getattr(record, "access_count", 0)) / math.log(11.0)  # 10 hits saturates
    frequency = min(1.0, frequency)
    return (
        RECENCY_WEIGHT * recency
        + FREQUENCY_WEIGHT * frequency
        + IMPORTANCE_WEIGHT * max(0.0, min(1.0, importance))
    )


def keep_score(record: Any, kind: str, now_turn: int, half_life: float, importance: float) -> float:
    """Decay score adjusted for what kind of record this is."""
    return decay_score(record, now_turn, half_life, importance) * KIND_WEIGHT.get(kind, 1.0)


def fact_importance(fact: Any) -> float:
    """Importance of an active fact. Pinned facts sit at the ceiling."""
    if fact.pinned:
        return 1.0
    return min(1.0, 0.45 + 0.4 * fact.confidence)


def plan_evictions(
    facts: Sequence[Any],
    episodes: Sequence[Any],
    now_turn: int,
    budget_tokens: int,
    half_life: float = 15.0,
) -> Tuple[List[EvictionCandidate], int]:
    """Choose the cheapest-to-lose records that bring the store under budget.

    Returns the eviction list and the token total that would remain. Lowest score
    first; ties broken by larger token size, because when two records are equally
    worthless the one that frees more budget is the better choice.
    """
    if budget_tokens < 0:
        raise ValueError("budget_tokens cannot be negative")

    candidates: List[EvictionCandidate] = []
    protected_tokens = 0
    for fact in facts:
        if not fact.active:
            # Superseded facts are provenance, not answers: `search` never
            # returns them, so they can never enter a prompt and they are not
            # part of the prompt-bearing budget this function defends. Their
            # growth is bounded separately, by a cap on chain length.
            continue
        if fact.pinned:
            protected_tokens += fact.tokens
            continue
        candidates.append(
            EvictionCandidate(
                record=fact,
                kind="fact",
                score=keep_score(fact, "fact", now_turn, half_life, fact_importance(fact)),
                tokens=fact.tokens,
            )
        )
    for episode in episodes:
        candidates.append(
            EvictionCandidate(
                record=episode,
                kind="episode",
                score=keep_score(episode, "episode", now_turn, half_life, episode.importance),
                tokens=episode.tokens,
            )
        )

    total = protected_tokens + sum(c.tokens for c in candidates)
    if total <= budget_tokens:
        return [], total

    candidates.sort(key=lambda c: (c.score, -c.tokens, c.id))
    evicted: List[EvictionCandidate] = []
    for candidate in candidates:
        if total <= budget_tokens:
            break
        evicted.append(candidate)
        total -= candidate.tokens
    return evicted, total
