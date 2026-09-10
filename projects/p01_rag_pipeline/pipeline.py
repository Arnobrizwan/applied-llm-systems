"""The pipeline facade: ingest, index, retrieve, rerank, answer.

One object owns the wiring so that the evaluation can flip a single field and
compare four honest configurations of the same system, rather than four systems
that differ in ways nobody wrote down.

Every stage runs inside a `llmkit.tracer` span. Retrieval quality problems are
almost never visible in the final answer text, which is exactly why they survive
into production: the answer reads fine, it is just answering from the wrong
chunk. Per-stage spans with the candidate ids attached are what turns "the bot
said something odd" into "BM25 returned nothing for this query because the user
spelled the error code with a hyphen".
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional, Sequence

from llmkit import Chunk, LLMProvider, ScoredChunk, Tracer, get_llm, tracer as default_tracer

from .answer import Answer, CitedAnswerer
from .ingest import Ingestor, IngestConfig, IngestStats
from .rerank import FeatureReranker, IdentityReranker, Reranker
from .retrieval import HybridRetriever, RetrievalConfig


@dataclass
class PipelineConfig:
    """A named point in the design space, so configurations can be compared.

    `label` exists because the evaluation prints a comparison table and a row
    called `mode=hybrid rerank=True` is harder to read than one called
    `hybrid+rerank`.
    """

    label: str = "hybrid+rerank"
    mode: str = "hybrid"
    k: int = 5
    candidate_k: int = 20
    rerank: bool = True
    grounding_floor: float = 0.30
    max_evidence: int = 5


PRESETS: Dict[str, PipelineConfig] = {
    "vector": PipelineConfig(label="vector", mode="vector", rerank=False),
    "bm25": PipelineConfig(label="bm25", mode="bm25", rerank=False),
    "hybrid": PipelineConfig(label="hybrid", mode="hybrid", rerank=False),
    "hybrid+rerank": PipelineConfig(label="hybrid+rerank", mode="hybrid", rerank=True),
}


class RagPipeline:
    def __init__(
        self,
        chunks: Sequence[Chunk],
        llm: Optional[LLMProvider] = None,
        config: Optional[PipelineConfig] = None,
        tracer: Optional[Tracer] = None,
        ingest_stats: Optional[IngestStats] = None,
    ):
        self.config = config or PipelineConfig()
        self.chunks = list(chunks)
        self.tracer = tracer or default_tracer
        self.ingest_stats = ingest_stats
        self.retriever = HybridRetriever(
            self.chunks,
            config=RetrievalConfig(
                mode=self.config.mode, k=self.config.k, candidate_k=self.config.candidate_k
            ),
        )
        df, corpus_size = self.retriever.document_frequencies()
        self.reranker: Reranker = FeatureReranker(
            document_frequencies=df,
            corpus_size=corpus_size,
            # The retriever already holds the vectors, so MMR redundancy is a
            # dictionary lookup and a dot product rather than a re-embed.
            similarity_fn=self.retriever.similarity,
        )
        self.identity_reranker: Reranker = IdentityReranker()
        self.answerer = CitedAnswerer(
            llm=llm or get_llm(),
            grounding_floor=self.config.grounding_floor,
            max_evidence=self.config.max_evidence,
            document_frequencies=df,
            corpus_size=corpus_size,
        )

    # -- construction ----------------------------------------------------
    @classmethod
    def build(
        cls,
        paths: Iterable[str] = (),
        include_builtin: bool = True,
        llm: Optional[LLMProvider] = None,
        config: Optional[PipelineConfig] = None,
        ingest_config: Optional[IngestConfig] = None,
        tracer: Optional[Tracer] = None,
    ) -> "RagPipeline":
        ingestor = Ingestor(ingest_config)
        chunks = ingestor.ingest(paths=paths, include_builtin=include_builtin)
        return cls(chunks, llm=llm, config=config, tracer=tracer, ingest_stats=ingestor.stats)

    def with_config(self, config: PipelineConfig) -> "RagPipeline":
        """Reuse the built indexes under a different configuration.

        Rebuilding the vector store per configuration would make the four-way
        comparison measure index construction noise alongside retrieval quality.
        Same chunks, same vectors, same BM25 counts; only the routing changes.
        """
        clone = object.__new__(RagPipeline)
        clone.config = config
        clone.chunks = self.chunks
        clone.tracer = self.tracer
        clone.ingest_stats = self.ingest_stats
        clone.retriever = self.retriever
        clone.reranker = self.reranker
        clone.identity_reranker = self.identity_reranker
        clone.answerer = replace_answerer(self.answerer, config)
        return clone

    # -- stages ----------------------------------------------------------
    def retrieve(
        self,
        question: str,
        k: Optional[int] = None,
        mode: Optional[str] = None,
        rerank: Optional[bool] = None,
    ) -> List[ScoredChunk]:
        cfg = self.config
        k = k or cfg.k
        mode = mode or cfg.mode
        use_rerank = cfg.rerank if rerank is None else rerank

        with self.tracer.span("rag.retrieve", mode=mode, k=k, rerank=use_rerank) as span:
            if not use_rerank:
                hits = self.retriever.retrieve(question, k=k, mode=mode)
                span.attributes["candidates"] = len(hits)
                span.attributes["hit_ids"] = [h.chunk.id for h in hits]
                return hits

            with self.tracer.span("rag.candidates", mode=mode, depth=cfg.candidate_k) as cand_span:
                candidates = self.retriever.candidates(
                    question, mode=mode, candidate_k=cfg.candidate_k
                )
                cand_span.attributes["candidates"] = len(candidates)
            with self.tracer.span("rag.rerank", reranker=self.reranker.name) as rr_span:
                hits = self.reranker.rerank(question, candidates, k=k)
                rr_span.attributes["hit_ids"] = [h.chunk.id for h in hits]
            span.attributes["candidates"] = len(candidates)
            span.attributes["hit_ids"] = [h.chunk.id for h in hits]
            return hits

    def ask(
        self,
        question: str,
        k: Optional[int] = None,
        mode: Optional[str] = None,
        rerank: Optional[bool] = None,
    ) -> Answer:
        with self.tracer.span("rag.ask", question=question, config=self.config.label) as span:
            evidence = self.retrieve(question, k=k, mode=mode, rerank=rerank)
            with self.tracer.span("rag.answer") as ans_span:
                answer = self.answerer.answer(question, evidence)
                ans_span.attributes.update(
                    {
                        "refused": answer.refused,
                        "reason": answer.reason,
                        "grounding": round(answer.grounding, 4),
                        "valid_citations": len(answer.valid_citations),
                        "invalid_citations": len(answer.invalid_citations),
                        # Named to match the tracer's LLM attribute convention so
                        # Tracer.summary() rolls token spend up without a custom
                        # aggregator.
                        "llm.total_tokens": answer.total_tokens,
                    }
                )
            span.attributes["refused"] = answer.refused
            span.attributes["cited_docs"] = answer.cited_doc_ids
            return answer


def replace_answerer(answerer: CitedAnswerer, config: PipelineConfig) -> CitedAnswerer:
    """Clone an answerer with configuration-specific limits, reusing its idf table."""
    return CitedAnswerer(
        llm=answerer.llm,
        grounding_floor=config.grounding_floor,
        max_evidence=config.max_evidence,
        max_evidence_tokens=answerer.max_evidence_tokens,
        document_frequencies=answerer.df,
        corpus_size=answerer.corpus_size,
        extractive_fallback=answerer.extractive_fallback,
    )


def preset(name: str) -> PipelineConfig:
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    return replace(PRESETS[name])
