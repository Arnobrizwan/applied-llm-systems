"""A small, honest in-memory vector store.

Exact brute-force cosine search. No ANN index, because at the corpus sizes these
demos run (thousands of chunks) exact search is both faster to reason about and
fast enough, and pretending otherwise would be theatre. The interface mirrors
what FAISS/pgvector expose so swapping one in is a class change, not a rewrite.
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .embeddings import Embedder, cosine, get_embedder
from .types import Chunk, ScoredChunk


class InMemoryVectorStore:
    def __init__(self, embedder: Optional[Embedder] = None):
        self.embedder = embedder or get_embedder()
        self.chunks: List[Chunk] = []
        self.vectors: List[List[float]] = []
        self._by_id: Dict[str, int] = {}

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: Sequence[Chunk], vectors: Optional[Sequence[Sequence[float]]] = None) -> int:
        if vectors is None:
            vectors = self.embedder.embed([c.text for c in chunks])
        if len(vectors) != len(chunks):
            raise ValueError("vectors and chunks must be the same length")
        added = 0
        for chunk, vec in zip(chunks, vectors):
            if chunk.id in self._by_id:
                idx = self._by_id[chunk.id]
                self.chunks[idx] = chunk
                self.vectors[idx] = list(vec)
                continue
            self._by_id[chunk.id] = len(self.chunks)
            self.chunks.append(chunk)
            self.vectors.append(list(vec))
            added += 1
        return added

    def get(self, chunk_id: str) -> Optional[Chunk]:
        idx = self._by_id.get(chunk_id)
        return self.chunks[idx] if idx is not None else None

    def search(self, query: str, k: int = 5,
               where: Optional[Callable[[Chunk], bool]] = None) -> List[ScoredChunk]:
        if not self.chunks:
            return []
        qvec = self.embedder.embed_one(query)
        return self.search_by_vector(qvec, k=k, where=where)

    def search_by_vector(self, qvec: Sequence[float], k: int = 5,
                         where: Optional[Callable[[Chunk], bool]] = None) -> List[ScoredChunk]:
        scored: List[ScoredChunk] = []
        for chunk, vec in zip(self.chunks, self.vectors):
            if where and not where(chunk):
                continue
            scored.append(ScoredChunk(chunk=chunk, score=cosine(qvec, vec), source="vector"))
        scored.sort(key=lambda s: (-s.score, s.chunk.id))
        return scored[:k]

    # -- persistence -----------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        payload = {
            "embedder": self.embedder.name,
            "dim": self.embedder.dim,
            "items": [
                {
                    "chunk": {
                        "id": c.id, "doc_id": c.doc_id, "text": c.text,
                        "ordinal": c.ordinal, "metadata": c.metadata,
                    },
                    "vector": v,
                }
                for c, v in zip(self.chunks, self.vectors)
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, path: str, embedder: Optional[Embedder] = None) -> "InMemoryVectorStore":
        with open(path, encoding="utf-8") as f:
            payload: Dict[str, Any] = json.load(f)
        store = cls(embedder=embedder)
        if payload.get("embedder") != store.embedder.name:
            raise ValueError(
                f"index was built with embedder {payload.get('embedder')!r} but "
                f"{store.embedder.name!r} is active - rebuild the index"
            )
        chunks = [Chunk(**item["chunk"]) for item in payload["items"]]
        vectors = [item["vector"] for item in payload["items"]]
        store.add(chunks, vectors)
        return store
