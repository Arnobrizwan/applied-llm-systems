"""Hybrid retrieval: dense vectors, BM25, and reciprocal rank fusion.

Why both indexes exist
----------------------
Dense retrieval matches meaning and degrades gracefully on paraphrase. It also
fails hard and quietly on tokens the encoder has never seen: product names,
error codes, SKUs, version strings, internal acronyms. BM25 is the opposite. It
cannot paraphrase at all, but a workspace admin searching for `429` or
`Idempotency-Key` gets an exact hit every time.

Running only one of them means accepting one of those failure modes as policy.

Why RRF and not a weighted blend of scores
------------------------------------------
BM25 scores are unbounded and corpus-dependent: they grow with idf, so the same
query scores differently once you add documents. Cosine similarity is bounded in
[-1, 1]. Any `alpha * cosine + (1 - alpha) * bm25` blend therefore needs a
normaliser, and every normaliser is a lie of some kind: min-max normalisation is
computed over the candidate list, so a document's fused score changes depending
on which other documents happened to be retrieved with it, and z-score
normalisation assumes a distribution that a 3-hit candidate list does not have.

Reciprocal rank fusion throws the scores away and keeps only the ordering, which
is the part both retrievers agree on the meaning of. score(d) = sum 1/(k + rank).
It has one tuning constant, it cannot be destabilised by a scale change in either
retriever, and a document ranked well by both systems beats a document ranked
first by one and missing from the other. That last property is the whole point:
agreement between two independent retrievers is the strongest cheap signal
available before a reranker runs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from llmkit import (
    BM25,
    Chunk,
    Embedder,
    InMemoryVectorStore,
    ScoredChunk,
    cosine,
    get_embedder,
    reciprocal_rank_fusion,
)

from .terms import document_frequencies as folded_document_frequencies

VALID_MODES = ("vector", "bm25", "hybrid")


@dataclass
class RetrievalConfig:
    """Retrieval knobs.

    `candidate_k` is deliberately much larger than `k`. Fusion and reranking can
    only reorder what retrieval handed them, so the depth of the candidate pool
    is the ceiling on how much either stage can help. Fetching 20 and returning 5
    costs nothing at this corpus size and is what lets the reranker rescue a
    chunk that BM25 ranked eleventh.
    """

    mode: str = "hybrid"
    k: int = 5
    candidate_k: int = 20
    rrf_k: int = 60

    def validated(self) -> "RetrievalConfig":
        if self.mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {self.mode!r}")
        if self.k <= 0 or self.candidate_k <= 0:
            raise ValueError("k and candidate_k must be positive")
        return self


class HybridRetriever:
    """Owns both indexes and the fusion step.

    The embedding vectors are computed once and handed to the vector store rather
    than letting the store embed internally, so the retriever can reuse the exact
    same vectors for the reranker's diversity calculation. Re-embedding a chunk to
    measure how similar it is to another chunk it was just retrieved beside is
    pure waste, and with a real encoder it is the expensive kind.
    """

    def __init__(
        self,
        chunks: Sequence[Chunk],
        embedder: Optional[Embedder] = None,
        config: Optional[RetrievalConfig] = None,
    ):
        if not chunks:
            raise ValueError("cannot build a retriever over zero chunks")
        self.config = (config or RetrievalConfig()).validated()
        self.embedder = embedder or get_embedder()
        self.chunks_by_id: Dict[str, Chunk] = {c.id: c for c in chunks}

        vectors = self.embedder.embed([c.text for c in chunks])
        self.vectors: Dict[str, List[float]] = {c.id: v for c, v in zip(chunks, vectors)}

        self.store = InMemoryVectorStore(embedder=self.embedder)
        self.store.add(list(chunks), vectors)

        self.bm25 = BM25()
        for chunk in chunks:
            self.bm25.add(chunk.id, chunk.text)

    def __len__(self) -> int:
        return len(self.chunks_by_id)

    # -- single-retriever paths ------------------------------------------
    def vector_search(self, query: str, k: int) -> List[ScoredChunk]:
        return self.store.search(query, k=k)

    def bm25_search(self, query: str, k: int) -> List[ScoredChunk]:
        return [
            ScoredChunk(chunk=self.chunks_by_id[cid], score=score, source="bm25")
            for cid, score in self.bm25.search(query, k=k)
        ]

    # -- fusion ----------------------------------------------------------
    def hybrid_search(self, query: str, k: int, candidate_k: int) -> List[ScoredChunk]:
        dense = self.vector_search(query, candidate_k)
        lexical = self.bm25_search(query, candidate_k)

        dense_rank = {sc.chunk.id: i + 1 for i, sc in enumerate(dense)}
        lexical_rank = {sc.chunk.id: i + 1 for i, sc in enumerate(lexical)}
        dense_score = {sc.chunk.id: sc.score for sc in dense}
        lexical_score = {sc.chunk.id: sc.score for sc in lexical}

        fused = reciprocal_rank_fusion(
            [[sc.chunk.id for sc in dense], [sc.chunk.id for sc in lexical]],
            k=self.config.rrf_k,
        )

        out: List[ScoredChunk] = []
        for chunk_id, score in fused[:k]:
            out.append(
                ScoredChunk(
                    chunk=self.chunks_by_id[chunk_id],
                    score=score,
                    source="hybrid",
                    components={
                        # Ranks are reported as 0.0 when a retriever did not return
                        # the chunk at all, which is a materially different event
                        # from "ranked last" and worth being able to see in a log.
                        "vector_rank": float(dense_rank.get(chunk_id, 0)),
                        "bm25_rank": float(lexical_rank.get(chunk_id, 0)),
                        "vector_score": round(dense_score.get(chunk_id, 0.0), 4),
                        "bm25_score": round(lexical_score.get(chunk_id, 0.0), 4),
                        "agreement": 1.0 if chunk_id in dense_rank and chunk_id in lexical_rank else 0.0,
                    },
                )
            )
        return out

    # -- public API ------------------------------------------------------
    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        mode: Optional[str] = None,
        candidate_k: Optional[int] = None,
    ) -> List[ScoredChunk]:
        query = (query or "").strip()
        if not query:
            return []
        mode = mode or self.config.mode
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
        k = k or self.config.k
        candidate_k = max(candidate_k or self.config.candidate_k, k)

        if mode == "vector":
            return self.vector_search(query, k)
        if mode == "bm25":
            return self.bm25_search(query, k)
        return self.hybrid_search(query, k, candidate_k)

    def candidates(
        self,
        query: str,
        mode: Optional[str] = None,
        candidate_k: Optional[int] = None,
    ) -> List[ScoredChunk]:
        """The deep candidate pool a reranker should be given."""
        depth = candidate_k or self.config.candidate_k
        return self.retrieve(query, k=depth, mode=mode, candidate_k=depth)

    def similarity(self, chunk_id_a: str, chunk_id_b: str) -> float:
        """Cosine between two indexed chunks. Used for MMR diversity."""
        a = self.vectors.get(chunk_id_a)
        b = self.vectors.get(chunk_id_b)
        if a is None or b is None:
            return 0.0
        return cosine(a, b)

    def document_frequencies(self) -> Tuple[Dict[str, int], int]:
        """(folded term -> document frequency, corpus size), for idf weighting.

        Computed over folded terms rather than reused from BM25's raw counts,
        because the reranker and the answerer compare folded vocabulary. Looking
        up the idf of "expir" in a table keyed on "expires" returns a document
        frequency of zero, which makes the corpus's most common terms look like
        its rarest ones and inverts the weighting.
        """
        return folded_document_frequencies(c.text for c in self.chunks_by_id.values())
