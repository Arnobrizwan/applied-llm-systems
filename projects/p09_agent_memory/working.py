"""Working memory: the recent turn buffer, under a hard token cap.

This is the only store with a cheap, obvious eviction rule (oldest first) and it
is also the only one whose eviction is not allowed to lose information. Turns
that leave here are summarised into an episode by the caller, which is why
`overflow` returns the turns rather than discarding them.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional

from .records import Turn


class WorkingMemory:
    """A FIFO turn buffer with a token cap and a minimum turn floor.

    The floor exists because a cap alone can evict the turn the user just sent.
    A single long paste can exceed the whole cap by itself, and a buffer that
    responds by emptying is worse than useless: the agent loses the question it
    is currently answering. Keeping the last `min_turns` regardless is the
    smaller wrong answer.
    """

    def __init__(self, max_tokens: int = 220, min_turns: int = 2, low_water: float = 0.6):
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if min_turns < 1:
            raise ValueError("min_turns must be at least 1")
        if not 0.0 < low_water <= 1.0:
            raise ValueError("low_water must be in (0, 1]")
        self.max_tokens = max_tokens
        self.min_turns = min_turns
        self.low_water = low_water
        self._turns: Deque[Turn] = deque()

    def __len__(self) -> int:
        return len(self._turns)

    @property
    def turns(self) -> List[Turn]:
        return list(self._turns)

    @property
    def tokens(self) -> int:
        return sum(t.tokens for t in self._turns)

    def add(self, turn: Turn) -> List[Turn]:
        """Append a turn and return the run of turns that had to leave.

        Overflow drains down to a low-water mark rather than to the cap. Evicting
        one turn per overflowing turn would be tidier and would produce a stream
        of single-turn episodes that compress nothing, because a summary of one
        sentence is that sentence. Draining a batch gives the summariser a run of
        related turns to work with, and it means compression runs once every few
        turns instead of on every turn past the cap.
        """
        self._turns.append(turn)
        if self.tokens <= self.max_tokens:
            return []
        target = int(self.max_tokens * self.low_water)
        evicted: List[Turn] = []
        while self.tokens > target and len(self._turns) > self.min_turns:
            evicted.append(self._turns.popleft())
        return evicted

    def render(self, max_tokens: Optional[int] = None) -> str:
        """Most recent turns first-to-last, trimmed from the front if needed."""
        selected: List[Turn] = []
        used = 0
        for turn in reversed(self._turns):
            if max_tokens is not None and used + turn.tokens > max_tokens:
                break
            selected.append(turn)
            used += turn.tokens
        selected.reverse()
        return "\n".join(t.rendered for t in selected)

    def clear(self) -> None:
        self._turns.clear()
