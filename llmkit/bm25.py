"""BM25 Okapi, pure Python.

Lexical retrieval is half of hybrid search and it is the half that saves you when
an embedding model has never seen your product names, error codes or SKUs.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "was", "were",
    "for", "on", "with", "as", "by", "at", "it", "this", "that", "be", "from",
}


def tokenize(text: str, drop_stopwords: bool = True) -> List[str]:
    toks = _TOKEN_RE.findall((text or "").lower())
    if drop_stopwords:
        return [t for t in toks if t not in _STOP]
    return toks


class BM25:
    """Standard BM25 Okapi with k1/b tuning and incremental `add`."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids: List[str] = []
        self.doc_tokens: List[List[str]] = []
        self.doc_freqs: List[Counter] = []
        self.df: Counter = Counter()
        self.avgdl: float = 0.0

    def add(self, doc_id: str, text: str) -> None:
        toks = tokenize(text)
        self.doc_ids.append(doc_id)
        self.doc_tokens.append(toks)
        tf = Counter(toks)
        self.doc_freqs.append(tf)
        for term in tf:
            self.df[term] += 1
        total = sum(len(t) for t in self.doc_tokens)
        self.avgdl = total / len(self.doc_tokens) if self.doc_tokens else 0.0

    def add_many(self, docs: Iterable[Tuple[str, str]]) -> None:
        for doc_id, text in docs:
            self.add(doc_id, text)

    def _idf(self, term: str) -> float:
        n = len(self.doc_ids)
        df = self.df.get(term, 0)
        # +1 keeps the idf of a term present in every document positive
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def score(self, query: str, doc_index: int) -> float:
        tf = self.doc_freqs[doc_index]
        dl = len(self.doc_tokens[doc_index])
        if dl == 0:
            return 0.0
        total = 0.0
        for term in tokenize(query):
            f = tf.get(term, 0)
            if not f:
                continue
            denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
            total += self._idf(term) * (f * (self.k1 + 1)) / denom
        return total

    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        scored = [(self.doc_ids[i], self.score(query, i)) for i in range(len(self.doc_ids))]
        scored = [s for s in scored if s[1] > 0.0]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:k]


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int = 60) -> List[Tuple[str, float]]:
    """RRF: fuse ranked id lists without needing comparable score scales.

    score(d) = sum over lists of 1/(k + rank). Chosen over score normalisation
    because BM25 scores are unbounded and cosine scores are not, so any linear
    blend of the two silently re-weights itself as the corpus grows.
    """
    fused: Dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: (-x[1], x[0]))
