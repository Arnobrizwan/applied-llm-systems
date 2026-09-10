"""The escalation tool interface and its offline implementation.

`SearchTool` is one method. That is on purpose: the last rung of the escalation
ladder has to be swappable without the agent knowing, and the smallest possible
surface is the one most likely to survive being pointed at Tavily, Brave, Exa,
an internal enterprise index or a colleague's endpoint.

`FallbackSearch` is a real retriever over a real second corpus, not a stub that
returns an empty list. An untested fallback branch is a fallback branch that does
not work, and the agent's whole design rests on being able to escalate.

Results come back as `ScoredChunk`, the same type the primary retriever returns,
so escalated evidence flows into critique, answering and citation validation
through exactly the same code path. Giving external results their own type would
mean two answer paths, and the second one would be the one that skips validation.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

from llmkit import Chunk, ScoredChunk

from projects.p01_rag_pipeline.retrieval import HybridRetriever

from . import fallback_corpus


class SearchTool(ABC):
    """A source of evidence outside the primary index."""

    name: str = "search"

    @abstractmethod
    def search(self, query: str, k: int = 5) -> List[ScoredChunk]:
        ...


class FallbackSearch(SearchTool):
    """Offline hybrid search over a second corpus.

    Uses the same `HybridRetriever` as the primary index. Reusing it rather than
    writing a simpler lookup keeps one honest property: escalated results are
    ranked by the same method as primary results, so a confidence produced from
    them means the same thing.
    """

    name = "offline_fallback"

    def __init__(self, chunks: Optional[Sequence[Chunk]] = None):
        self.chunks = list(chunks or fallback_corpus.chunks())
        self.retriever = HybridRetriever(self.chunks)
        self.calls = 0

    def search(self, query: str, k: int = 5) -> List[ScoredChunk]:
        self.calls += 1
        hits = self.retriever.retrieve(query, k=k, mode="hybrid")
        for hit in hits:
            # Marked so a reviewer reading a trace or a citation can tell at a
            # glance that this evidence did not come from the primary index.
            hit.source = f"{hit.source}:{self.name}"
        return hits


class NullSearch(SearchTool):
    """A tool that finds nothing. Used to test that abstention still happens.

    This is the control: if the agent only abstains because the fallback happens
    to be weak, that is not abstention logic, it is luck.
    """

    name = "null"

    def __init__(self) -> None:
        self.calls = 0

    def search(self, query: str, k: int = 5) -> List[ScoredChunk]:
        self.calls += 1
        return []
