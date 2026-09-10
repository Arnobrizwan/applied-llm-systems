"""The manager: one object that owns all three stores and the budget.

Write path, per turn:

    add to working memory
      -> if it overflowed, summarise the evicted turns into an episode
      -> mine the turn for facts, dedupe them, resolve contradictions
      -> if long-lived memory is over budget, evict by decay score

Read path, per query:

    facts first, then episodes, then the recent turn buffer, packed into a
    caller-supplied token budget.

Facts go first on purpose. They are the smallest, densest and most likely to be
the literal answer, and if the recall budget runs out, the thing that survives
should be "our workspace is pinned to eu-west", not two paragraphs of ambient
conversation from which the model might infer it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from llmkit import Embedder, count_tokens

from .episodic import EpisodicMemory, extractive_summary, llm_summary
from .eviction import fact_importance, plan_evictions
from .records import Episode, Fact, MemoryEvent, RecallResult, Turn
from .working import WorkingMemory


@dataclass
class MemoryStats:
    turns: int
    working_tokens: int
    episode_tokens: int
    fact_tokens: int
    provenance_tokens: int
    long_lived_tokens: int
    budget_tokens: int
    episodes: int
    facts_active: int
    facts_superseded: int
    evictions: int

    @property
    def within_budget(self) -> bool:
        return self.long_lived_tokens <= self.budget_tokens

    def row(self) -> str:
        return (
            f"  turns={self.turns:<4} working={self.working_tokens:<5} episodes={self.episode_tokens:<5} "
            f"facts={self.fact_tokens:<5} long-lived={self.long_lived_tokens}/{self.budget_tokens} "
            f"within_budget={self.within_budget}"
        )


@dataclass
class MemoryManager:
    """Working plus episodic plus semantic memory under one token budget."""

    working_max_tokens: int = 220
    budget_tokens: int = 600
    episode_max_tokens: int = 70
    half_life: float = 15.0
    llm: Any = None
    use_llm_summaries: bool = False
    embedder: Optional[Embedder] = None

    working: WorkingMemory = field(init=False)
    episodic: EpisodicMemory = field(init=False)
    semantic: Any = field(init=False)
    events: List[MemoryEvent] = field(default_factory=list, init=False)
    turn_index: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        from .semantic import SemanticMemory  # local import keeps the module graph flat

        self.working = WorkingMemory(max_tokens=self.working_max_tokens)
        self.episodic = EpisodicMemory(embedder=self.embedder)
        self.semantic = SemanticMemory()

    # -- write path ------------------------------------------------------
    def observe(self, role: str, text: str) -> List[MemoryEvent]:
        """Record one turn and run compression and eviction if needed."""
        self.turn_index += 1
        turn = Turn(index=self.turn_index, role=role, text=text)
        events: List[MemoryEvent] = []

        overflowed = self.working.add(turn)
        if overflowed:
            episode = self._compress(overflowed)
            events.append(
                MemoryEvent(
                    turn=self.turn_index,
                    kind="episode",
                    detail=(
                        f"turns {episode.first_turn}-{episode.last_turn} compressed into {episode.id} "
                        f"({episode.original_tokens} -> {episode.tokens} tokens, "
                        f"{episode.compression_ratio * 100:.0f}% kept)"
                    ),
                    tokens=episode.tokens,
                )
            )

        if role == "user":
            added, superseded = self.semantic.observe(text, self.turn_index)
            for old, new in superseded:
                events.append(
                    MemoryEvent(
                        turn=self.turn_index,
                        kind="supersede",
                        detail=(
                            f"'{old.text}' (turn {old.turn_index}) superseded by "
                            f"'{new.text}' (turn {new.turn_index}); old version kept as {old.id}"
                        ),
                    )
                )
            superseded_ids = {old.id for old, _ in superseded}
            for fact in added:
                if fact.supersedes in superseded_ids:
                    continue
                events.append(
                    MemoryEvent(
                        turn=self.turn_index, kind="fact", detail=f"learned '{fact.text}'", tokens=fact.tokens
                    )
                )

        events.extend(self._evict())
        self.events.extend(events)
        return events

    def observe_all(self, turns: Sequence[Tuple[str, str]]) -> List[MemoryEvent]:
        out: List[MemoryEvent] = []
        for role, text in turns:
            out.extend(self.observe(role, text))
        return out

    def pin(self, fact_id: str) -> Fact:
        """Mark a fact as never-evictable. Used for operator-set constraints."""
        return self.semantic.pin(fact_id)

    # -- read path -------------------------------------------------------
    def recall(self, query: str, token_budget: int = 300, facts_k: int = 4, episodes_k: int = 2) -> RecallResult:
        """Assemble what the agent should be told about this query."""
        facts = self.semantic.search(query, k=facts_k, now_turn=self.turn_index)
        episodes = [ep for ep, _ in self.episodic.search(query, now_turn=self.turn_index, k=episodes_k)]

        blocks: List[str] = []
        used = 0
        included_facts: List[Fact] = []
        if facts:
            header = "Known facts:"
            used += count_tokens(header)
            lines = [header]
            for fact in facts:
                if used + fact.tokens > token_budget * 0.5:
                    break
                lines.append(fact.rendered)
                used += fact.tokens
                included_facts.append(fact)
            if len(lines) > 1:
                blocks.append("\n".join(lines))

        included_episodes: List[Episode] = []
        if episodes:
            lines = ["Earlier in this conversation:"]
            used += count_tokens(lines[0])
            for episode in episodes:
                if used + episode.tokens > token_budget * 0.85:
                    break
                lines.append(f"- {episode.text}")
                used += episode.tokens
                included_episodes.append(episode)
            if len(lines) > 1:
                blocks.append("\n".join(lines))

        recent_budget = max(0, token_budget - used - 4)
        recent = self.working.render(max_tokens=recent_budget)
        if recent:
            blocks.append("Recent turns:\n" + recent)

        text = "\n\n".join(blocks)
        return RecallResult(
            text=text,
            facts=included_facts,
            episodes=included_episodes,
            turns=self.working.turns,
        )

    # -- accounting ------------------------------------------------------
    def stats(self) -> MemoryStats:
        facts = self.semantic.facts
        return MemoryStats(
            turns=self.turn_index,
            working_tokens=self.working.tokens,
            episode_tokens=self.episodic.tokens,
            fact_tokens=self.semantic.active_tokens,
            provenance_tokens=self.semantic.provenance_tokens,
            long_lived_tokens=self.episodic.tokens + self.semantic.active_tokens,
            budget_tokens=self.budget_tokens,
            episodes=len(self.episodic),
            facts_active=sum(1 for f in facts if f.active),
            facts_superseded=sum(1 for f in facts if not f.active),
            evictions=sum(1 for e in self.events if e.kind == "evict"),
        )

    def timeline(self, kinds: Sequence[str] = ()) -> List[MemoryEvent]:
        if not kinds:
            return list(self.events)
        return [e for e in self.events if e.kind in kinds]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "stats": vars(self.stats()),
            "facts": [f.to_dict() for f in self.semantic.facts],
            "episodes": [e.to_dict() for e in self.episodic.episodes],
        }

    # -- internals -------------------------------------------------------
    def _compress(self, turns: Sequence[Turn]) -> Episode:
        original = sum(t.tokens for t in turns)
        if self.use_llm_summaries and self.llm is not None:
            text = llm_summary(turns, self.episode_max_tokens, self.llm)
        else:
            text = extractive_summary(turns, self.episode_max_tokens)
        if not text:  # pragma: no cover - only when every turn was an acknowledgement
            text = f"turns {turns[0].index}-{turns[-1].index}: (no fact-bearing content)"
        episode = Episode(
            id=self.episodic.next_id(),
            text=text,
            first_turn=turns[0].index,
            last_turn=turns[-1].index,
            source_turns=[t.index for t in turns],
            original_tokens=original,
            # Episodes that compressed a lot of user speech are worth more than
            # episodes that compressed a run of agent acknowledgements.
            importance=min(1.0, 0.35 + 0.1 * sum(1 for t in turns if t.role == "user")),
            last_used_turn=turns[-1].index,
        )
        return self.episodic.append(episode)

    def _evict(self) -> List[MemoryEvent]:
        evicted, _ = plan_evictions(
            self.semantic.facts,
            self.episodic.episodes,
            now_turn=self.turn_index,
            budget_tokens=self.budget_tokens,
            half_life=self.half_life,
        )
        events: List[MemoryEvent] = []
        for candidate in evicted:
            if candidate.kind == "fact":
                self.semantic.remove(candidate.id)
            else:
                self.episodic.remove(candidate.id)
            events.append(
                MemoryEvent(
                    turn=self.turn_index,
                    kind="evict",
                    detail=(
                        f"evicted {candidate.kind} {candidate.id} "
                        f"(score {candidate.score:.3f}, {candidate.tokens} tokens)"
                    ),
                    tokens=candidate.tokens,
                )
            )
        return events


class RecentWindowBaseline:
    """The no-memory baseline: keep the last turns that fit, forget the rest.

    This is what a chat feature does before anyone builds memory, and it is a
    reasonable default. It fails in exactly one way, and the demo measures it: a
    fact stated early in a long session is gone, because the only thing that
    decides what survives is how recently it was said.
    """

    def __init__(self, token_budget: int = 300):
        self.token_budget = token_budget
        self.turns: List[Turn] = []
        self.turn_index = 0

    def observe(self, role: str, text: str) -> None:
        self.turn_index += 1
        self.turns.append(Turn(index=self.turn_index, role=role, text=text))

    def observe_all(self, turns: Sequence[Tuple[str, str]]) -> None:
        for role, text in turns:
            self.observe(role, text)

    def recall(self, query: str, token_budget: Optional[int] = None) -> RecallResult:
        budget = self.token_budget if token_budget is None else token_budget
        selected: List[Turn] = []
        used = 0
        for turn in reversed(self.turns):
            if used + turn.tokens > budget:
                break
            selected.append(turn)
            used += turn.tokens
        selected.reverse()
        text = "Recent turns:\n" + "\n".join(t.rendered for t in selected)
        return RecallResult(text=text, turns=selected)
