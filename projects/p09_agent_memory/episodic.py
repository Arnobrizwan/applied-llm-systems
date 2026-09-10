"""Episodic memory: an append-only log of what happened, retrieved by blended
recency and relevance.

Append-only is a deliberate constraint. An episodic log that can be edited in
place cannot answer "what did the agent believe at turn 12", which is the
question every conversational bug report turns into. Episodes are created,
scored, retrieved and eventually evicted, but never rewritten.

Retrieval blends relevance and recency because either one alone fails in a way
users notice. Pure relevance re-surfaces a topic the conversation moved on from
twenty turns ago. Pure recency makes the agent unable to answer anything about
the start of a long session, which is the exact complaint this project exists to
fix.
"""
from __future__ import annotations

import math
import re
from typing import List, Optional, Sequence, Tuple

from llmkit import Embedder, cosine, get_embedder, sentence_split, truncate_to_tokens

from .records import Episode, Turn

# A sentence with a number, a capitalised name, a code identifier or a version
# string is far more likely to carry a fact worth remembering than one without.
_FACT_MARKERS = re.compile(r"\d|[A-Z]{2,}|[a-z]+[-_][a-z]+|\bv\d")
_ACK = re.compile(
    r"^(ok|okay|thanks|thank you|got it|sure|understood|noted|yes|no|right|sounds good)\b", re.I
)


def extractive_summary(turns: Sequence[Turn], max_tokens: int) -> str:
    """Keep the fact-bearing sentences, drop the acknowledgements.

    This is the default summariser and it is deliberately not a model call.
    Compressing working memory happens on the write path of a conversation, so a
    model round trip here adds latency to every turn that overflows the buffer,
    and a model that paraphrases can quietly drop the one identifier the user
    will ask about later. Rule-based extraction is lossy in a predictable
    direction: it keeps numbers, names and identifiers.
    """
    if max_tokens <= 0:
        return ""
    scored: List[Tuple[float, int, str]] = []
    for order, turn in enumerate(turns):
        for sentence in sentence_split(turn.text):
            if _ACK.match(sentence.strip()) and len(sentence.split()) < 8:
                continue
            weight = 1.0 if _FACT_MARKERS.search(sentence) else 0.4
            if turn.role == "user":
                weight += 0.3  # what the user said outranks what the agent replied
            scored.append((weight, order, sentence.strip()))
    if not scored:
        return ""
    scored.sort(key=lambda s: (-s[0], s[1]))

    chosen: List[Tuple[int, str]] = []
    used = 0
    for weight, order, sentence in scored:
        cost = len(sentence.split()) + 2
        if used + cost > max_tokens * 0.9:
            continue
        chosen.append((order, sentence))
        used += cost
    if not chosen:
        chosen = [(scored[0][1], scored[0][2])]
    chosen.sort()
    first, last = turns[0].index, turns[-1].index
    body = " ".join(s for _, s in chosen)
    return truncate_to_tokens(f"turns {first}-{last}: {body}", max_tokens)


def llm_summary(turns: Sequence[Turn], max_tokens: int, llm) -> str:
    """Same job, handed to the model. Opt-in, and always hard-capped.

    Kept as an alternative rather than the default: it is better at prose and
    worse at guarantees, and its output length is a suggestion rather than a
    limit, so the truncation below is load-bearing.
    """
    if llm is None:
        return extractive_summary(turns, max_tokens)
    blocks = "\n".join(f"[S{i + 1}] {t.rendered}" for i, t in enumerate(turns))
    resp = llm.complete(
        [
            {"role": "system", "content": f"Summarise this conversation segment. Keep names, numbers and decisions.\n{blocks}"},
            {"role": "user", "content": "What happened in this segment?"},
        ]
    )
    first, last = turns[0].index, turns[-1].index
    return truncate_to_tokens(f"turns {first}-{last}: {resp.text.strip()}", max_tokens)


class EpisodicMemory:
    """Append-only episode log with blended recency-plus-relevance retrieval."""

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        relevance_weight: float = 0.6,
        recency_half_life: float = 12.0,
        importance_weight: float = 0.15,
    ):
        if recency_half_life <= 0:
            raise ValueError("recency_half_life must be positive")
        self.embedder = embedder or get_embedder()
        self.relevance_weight = relevance_weight
        self.recency_half_life = recency_half_life
        self.importance_weight = importance_weight
        self.episodes: List[Episode] = []
        self._vectors: List[List[float]] = []
        # Monotonic, never reused. Reusing an id after an eviction would make two
        # different episodes indistinguishable in a log, which is precisely the
        # thing an append-only store is supposed to prevent.
        self.sequence = 0

    def __len__(self) -> int:
        return len(self.episodes)

    @property
    def tokens(self) -> int:
        return sum(e.tokens for e in self.episodes)

    def next_id(self) -> str:
        self.sequence += 1
        return f"ep{self.sequence}"

    def append(self, episode: Episode) -> Episode:
        self.episodes.append(episode)
        self._vectors.append(self.embedder.embed_one(episode.text))
        return episode

    def remove(self, episode_id: str) -> bool:
        for idx, ep in enumerate(self.episodes):
            if ep.id == episode_id:
                del self.episodes[idx]
                del self._vectors[idx]
                return True
        return False

    def score(self, episode: Episode, vector: Sequence[float], query_vec: Sequence[float], now_turn: int) -> float:
        """Blended score in roughly [0, 1]. Higher is more worth recalling."""
        relevance = max(0.0, cosine(query_vec, vector))
        age = max(0, now_turn - episode.last_turn)
        recency = math.exp(-age / self.recency_half_life)
        recency_weight = max(0.0, 1.0 - self.relevance_weight - self.importance_weight)
        return (
            self.relevance_weight * relevance
            + recency_weight * recency
            + self.importance_weight * episode.importance
        )

    def search(self, query: str, now_turn: int, k: int = 3) -> List[Tuple[Episode, float]]:
        if not self.episodes:
            return []
        qvec = self.embedder.embed_one(query)
        scored = [
            (ep, self.score(ep, vec, qvec, now_turn))
            for ep, vec in zip(self.episodes, self._vectors)
        ]
        scored.sort(key=lambda s: (-s[1], s[0].id))
        hits = scored[:k]
        for ep, _ in hits:
            ep.touch(now_turn)
        return hits
