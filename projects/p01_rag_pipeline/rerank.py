"""Reranking: a feature-based scorer plus MMR diversity.

What this stands in for, said plainly
-------------------------------------
In a funded system this stage is a cross-encoder (a bge-reranker, Cohere Rerank,
a fine-tuned MiniLM) that reads the query and the chunk together and emits one
relevance number. That is strictly better than what is implemented here, and it
is also a model download or a paid API call, neither of which this repo is
allowed to require.

So this is a transparent feature-based reranker: idf-weighted query term
coverage, exact phrase hits, a position prior and a length penalty, followed by
MMR selection for diversity. It is not pretending to be a cross-encoder. It is
occupying the same slot in the pipeline with the same interface, so swapping a
real one in is a subclass of `Reranker` and a one-line change in the pipeline.

Why a feature scorer helps at all when the retrievers already scored
--------------------------------------------------------------------
The retrievers score at index granularity: BM25 sees a bag of terms, the dense
index sees a single averaged vector. Neither can see that a chunk contains the
query's exact phrase, or that it covers four of the query's five rare terms
rather than one rare term four times. Those are cheap signals computed over a
20-item candidate list, which is a completely different cost regime from
computing them over the whole corpus.

Why MMR and not just top-k by relevance
---------------------------------------
Top-k by relevance on a chunked corpus reliably returns five near-identical
chunks, because a document that mentions the answer once usually mentions it in
three adjacent overlapping chunks. That wastes the context budget and, worse, it
makes an answer look well-supported by five sources when it has one source cited
five times. MMR trades a little relevance for coverage of distinct material.
"""
from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

from llmkit import Chunk, ScoredChunk, keyword_overlap

from .terms import content_terms, folded_sequence, term_set


class Reranker(ABC):
    """The swappable slot. A cross-encoder implements exactly this."""

    name: str = "base"

    @abstractmethod
    def rerank(self, query: str, candidates: Sequence[ScoredChunk], k: int) -> List[ScoredChunk]:
        ...


class IdentityReranker(Reranker):
    """Pass-through. This is the control arm in the evaluation, not dead code."""

    name = "identity"

    def rerank(self, query: str, candidates: Sequence[ScoredChunk], k: int) -> List[ScoredChunk]:
        return list(candidates[:k])


@dataclass
class RerankWeights:
    """Feature weights, summing to 1.0 over the relevance features.

    Coverage carries most of the weight because "does this chunk actually contain
    the rare words the user asked about" is the signal that correlates with a
    human calling the result relevant. Phrase match is a strong but sparse signal,
    so it earns a real share without being able to dominate. Position and length
    are priors, not evidence, and are weighted accordingly.
    """

    coverage: float = 0.55
    phrase: float = 0.25
    position: float = 0.10
    length: float = 0.10

    def total(self) -> float:
        return self.coverage + self.phrase + self.position + self.length


class FeatureReranker(Reranker):
    """Interpretable relevance features plus MMR selection.

    Every component score is written back onto `ScoredChunk.components`, so a bad
    ranking can be explained after the fact from a log line instead of by
    re-running the query and squinting at it. That is the practical advantage of
    this class over a cross-encoder, and roughly the only one.
    """

    name = "feature+mmr"

    def __init__(
        self,
        document_frequencies: Optional[Dict[str, int]] = None,
        corpus_size: int = 0,
        weights: Optional[RerankWeights] = None,
        mmr_lambda: float = 0.75,
        target_chars: int = 900,
        similarity_fn: Optional[Callable[[str, str], float]] = None,
    ):
        if not 0.0 <= mmr_lambda <= 1.0:
            raise ValueError("mmr_lambda must be in [0, 1]")
        self.df = document_frequencies or {}
        self.corpus_size = max(corpus_size, 1)
        self.weights = weights or RerankWeights()
        self.mmr_lambda = mmr_lambda
        self.target_chars = target_chars
        # Default redundancy measure is lexical overlap so the reranker works with
        # no embedder at all; the pipeline injects a cosine-based one because it
        # already has the vectors in memory.
        self.similarity_fn = similarity_fn

    # -- features --------------------------------------------------------
    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1.0 + (self.corpus_size - df + 0.5) / (df + 0.5))

    def coverage(self, query_terms: Sequence[str], chunk_text: str) -> float:
        """idf-weighted fraction of the query's content terms present in the chunk.

        Weighted rather than raw because covering `idempotency` matters more than
        covering `request`, and an unweighted count lets a chunk stuffed with
        common query words outrank the one chunk that has the rare one.
        """
        if not query_terms:
            return 0.0
        present = term_set(chunk_text)
        num = sum(self._idf(t) for t in query_terms if t in present)
        den = sum(self._idf(t) for t in query_terms)
        return num / den if den else 0.0

    @staticmethod
    def phrase_hit(query_terms: Sequence[str], chunk_text: str) -> float:
        """Longest contiguous query n-gram (n >= 2) found in the chunk, normalised.

        A user typing "token rotation window" is expressing an ordering that a bag
        of words discards. Contiguous multi-word hits are rare enough to be a
        strong positive signal when they do fire.
        """
        if len(query_terms) < 2:
            return 0.0
        body = " ".join(folded_sequence(chunk_text))
        best = 0
        for size in range(len(query_terms), 1, -1):
            if size <= best:
                break
            for start in range(0, len(query_terms) - size + 1):
                gram = " ".join(query_terms[start : start + size])
                if gram in body:
                    best = max(best, size)
                    break
        return best / len(query_terms)

    @staticmethod
    def position_prior(chunk: Chunk) -> float:
        """Earlier chunks of a document score slightly higher.

        Technical documents front-load definitions and put edge cases at the
        bottom, so chunk 0 answers "what is X" more often than chunk 7 does. It is
        a weak prior and is weighted as one; it must never outvote coverage.
        """
        return 1.0 / (1.0 + float(max(chunk.ordinal, 0)))

    def length_score(self, chunk_text: str) -> float:
        """Penalise chunks well over the target length.

        A long chunk wins term-coverage contests by accident: it contains more
        words, so it contains more query words. This is the correction for that,
        and it is why coverage is not used alone.
        """
        length = max(len(chunk_text), 1)
        if length <= self.target_chars:
            return 1.0
        return self.target_chars / length

    def relevance(self, query: str, chunk: Chunk) -> Dict[str, float]:
        terms = content_terms(query)
        features = {
            "coverage": self.coverage(terms, chunk.text),
            "phrase": self.phrase_hit(folded_sequence(query), chunk.text),
            "position": self.position_prior(chunk),
            "length": self.length_score(chunk.text),
        }
        weights = self.weights
        features["relevance"] = (
            weights.coverage * features["coverage"]
            + weights.phrase * features["phrase"]
            + weights.position * features["position"]
            + weights.length * features["length"]
        ) / (weights.total() or 1.0)
        return features

    # -- selection -------------------------------------------------------
    def _redundancy(self, candidate: Chunk, selected: Sequence[Chunk]) -> float:
        if not selected:
            return 0.0
        if self.similarity_fn is not None:
            return max(self.similarity_fn(candidate.id, s.id) for s in selected)
        return max(keyword_overlap(candidate.text, s.text) for s in selected)

    def rerank(self, query: str, candidates: Sequence[ScoredChunk], k: int) -> List[ScoredChunk]:
        if not candidates:
            return []
        scored: List[ScoredChunk] = []
        for cand in candidates:
            features = self.relevance(query, cand.chunk)
            components = dict(cand.components)
            components.update({key: round(val, 4) for key, val in features.items()})
            components["retriever_score"] = round(cand.score, 4)
            scored.append(
                ScoredChunk(
                    chunk=cand.chunk,
                    score=features["relevance"],
                    source=f"{cand.source}+rerank",
                    components=components,
                )
            )

        # MMR: greedily take the candidate with the best relevance-minus-redundancy
        # trade-off, so the final set covers distinct material rather than
        # restating the best chunk from four angles.
        pool = sorted(scored, key=lambda s: (-s.score, s.chunk.id))
        selected: List[ScoredChunk] = []
        while pool and len(selected) < k:
            best_idx = 0
            best_value = -math.inf
            for idx, cand in enumerate(pool):
                redundancy = self._redundancy(cand.chunk, [s.chunk for s in selected])
                value = self.mmr_lambda * cand.score - (1.0 - self.mmr_lambda) * redundancy
                if value > best_value:
                    best_value, best_idx = value, idx
            chosen = pool.pop(best_idx)
            chosen.components["mmr"] = round(best_value, 4)
            selected.append(chosen)
        return selected
