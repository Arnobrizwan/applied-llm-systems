"""Answer synthesis with enforced, verified citations.

The failure this file exists to prevent
---------------------------------------
A RAG system that asks a model to cite its sources and then ships whatever comes
back has not built a citation feature. It has built a citation-shaped feature. A
model will happily emit `[S4]` when four sources were supplied and the claim came
from none of them, and it will emit `[S7]` when only five were supplied at all.
Both look correct to a user and to a reviewer skimming the output.

So there are three separate gates here and each one can fail independently:

1. A refusal gate, evaluated before the model is called at all. If nothing
   retrieved is a plausible answer, the cheapest correct behaviour is to say so
   and never spend the tokens.
2. A citation contract in the prompt. Evidence is numbered `[S1]`, `[S2]`, and
   the model is told every sentence must carry the marker it came from.
3. A validator over the output. Every marker the model emitted is resolved
   against the evidence set that was actually sent. Markers that do not resolve
   are stripped from the answer and recorded, because leaving an unresolvable
   citation in the text is worse than having no citation: it manufactures
   confidence that nothing supports.

Why the refusal gate does not threshold on the retriever's score
---------------------------------------------------------------
It is tempting to write `if top_score < 0.4: refuse`. That number means three
different things in this pipeline. A cosine score is bounded in [-1, 1]. A BM25
score is unbounded and grows with corpus idf. An RRF score depends only on rank,
so the top hit scores about 1/61 whether it is a perfect answer or unrelated
noise, and thresholding it would make hybrid mode refuse everything or nothing.

The gate therefore uses a retriever-independent signal computed here: how much of
the question's content vocabulary the retrieved evidence actually covers,
weighted so that rare terms count for more. It is comparable across every
retrieval mode, which is the property that makes the four-way comparison in the
evaluation meaningful.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from llmkit import (
    Chunk,
    LLMProvider,
    ScoredChunk,
    count_message_tokens,
    get_llm,
    system,
    truncate_to_tokens,
    user,
)

from .terms import content_terms, term_set

CITATION_RE = re.compile(r"\[S(\d+)\]")
REFUSAL_TEXT = "insufficient evidence"

# The words "judge", "grade" and "score" are deliberately absent from this
# instruction block. Some providers route on prompt keywords, and a synthesis
# prompt that reads like an evaluation prompt gets evaluation-shaped output back.
SYSTEM_TEMPLATE = """You answer questions using only the numbered evidence below.

Rules:
- Use only what the evidence states. Do not add outside knowledge.
- End every sentence with the marker of the evidence it came from, like [S1].
- If the evidence does not contain the answer, reply exactly: {refusal}

Evidence:
{evidence}"""


@dataclass
class Citation:
    marker: str
    chunk_id: Optional[str]
    doc_id: Optional[str]
    title: Optional[str]
    valid: bool


@dataclass
class Answer:
    """Everything a caller or an auditor needs about one answered question."""

    question: str
    text: str
    citations: List[Citation] = field(default_factory=list)
    evidence: List[ScoredChunk] = field(default_factory=list)
    refused: bool = False
    reason: str = "answered"
    grounding: float = 0.0
    fallback_used: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0

    @property
    def valid_citations(self) -> List[Citation]:
        return [c for c in self.citations if c.valid]

    @property
    def invalid_citations(self) -> List[Citation]:
        return [c for c in self.citations if not c.valid]

    @property
    def cited_doc_ids(self) -> List[str]:
        seen: List[str] = []
        for c in self.valid_citations:
            if c.doc_id and c.doc_id not in seen:
                seen.append(c.doc_id)
        return seen

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class CitedAnswerer:
    """Builds the evidence prompt, calls the model, verifies what comes back."""

    def __init__(
        self,
        llm: Optional[LLMProvider] = None,
        grounding_floor: float = 0.30,
        max_evidence: int = 5,
        max_evidence_tokens: int = 200,
        document_frequencies: Optional[Dict[str, int]] = None,
        corpus_size: int = 0,
        extractive_fallback: bool = True,
    ):
        self.llm = llm or get_llm()
        self.grounding_floor = grounding_floor
        self.max_evidence = max_evidence
        self.max_evidence_tokens = max_evidence_tokens
        self.df = document_frequencies or {}
        self.corpus_size = max(corpus_size, 1)
        self.extractive_fallback = extractive_fallback

    # -- grounding -------------------------------------------------------
    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1.0 + (self.corpus_size - df + 0.5) / (df + 0.5))

    def grounding_strength(self, question: str, evidence: Sequence[ScoredChunk]) -> float:
        """Best idf-weighted question-term coverage across the evidence set.

        Max rather than mean: one chunk that fully answers the question is a good
        outcome even when the other four are noise, and averaging would punish it
        for the company it keeps.
        """
        terms = content_terms(question)
        if not terms or not evidence:
            return 0.0
        denominator = sum(self._idf(t) for t in terms)
        if denominator <= 0:
            return 0.0
        best = 0.0
        for item in evidence:
            present = term_set(item.chunk.text)
            covered = sum(self._idf(t) for t in terms if t in present)
            best = max(best, covered / denominator)
        return best

    # -- prompt ----------------------------------------------------------
    def build_messages(self, question: str, evidence: Sequence[ScoredChunk]) -> Tuple[List, Dict[str, Chunk]]:
        """Number the evidence and return the marker -> chunk map used to verify.

        The map is built here, from the exact list that goes into the prompt.
        Rebuilding it later from the retrieval results would let the two drift
        apart, and the whole validation step is only as trustworthy as the claim
        that these are the chunks the model was actually shown.
        """
        blocks: List[str] = []
        marker_map: Dict[str, Chunk] = {}
        for idx, item in enumerate(evidence[: self.max_evidence], start=1):
            marker = f"S{idx}"
            marker_map[marker] = item.chunk
            body = truncate_to_tokens(item.chunk.text, self.max_evidence_tokens)
            title = item.chunk.metadata.get("doc_title") or item.chunk.doc_id
            blocks.append(f"[{marker}] ({title}) {body}")
        prompt = SYSTEM_TEMPLATE.format(refusal=REFUSAL_TEXT, evidence="\n".join(blocks))
        return [system(prompt), user(question)], marker_map

    # -- validation ------------------------------------------------------
    @staticmethod
    def _strip_marker(text: str, marker: str) -> str:
        cleaned = text.replace(f" [{marker}]", "").replace(f"[{marker}]", "")
        return re.sub(r"\s{2,}", " ", cleaned).strip()

    def validate(self, text: str, marker_map: Dict[str, Chunk]) -> Tuple[str, List[Citation]]:
        citations: List[Citation] = []
        cleaned = text
        for raw in dict.fromkeys(CITATION_RE.findall(text)):
            marker = f"S{raw}"
            chunk = marker_map.get(marker)
            if chunk is None:
                citations.append(Citation(marker, None, None, None, False))
                cleaned = self._strip_marker(cleaned, marker)
                continue
            citations.append(
                Citation(
                    marker=marker,
                    chunk_id=chunk.id,
                    doc_id=chunk.doc_id,
                    title=chunk.metadata.get("doc_title") or chunk.doc_id,
                    valid=True,
                )
            )
        return cleaned, citations

    # -- fallback --------------------------------------------------------
    @staticmethod
    def _best_sentence(question: str, chunk: Chunk) -> str:
        terms = set(content_terms(question))
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", chunk.text) if s.strip()]
        if not sentences:
            return chunk.text.strip()
        return max(sentences, key=lambda s: len(terms & term_set(s)))

    def _extractive_answer(self, question: str, marker_map: Dict[str, Chunk]) -> Tuple[str, List[Citation]]:
        """Deterministic cited answer used when the model breaks the contract.

        Shipping an uncited answer from a system that advertises citations is a
        product bug, not a model quirk. When the model returns something with no
        resolvable marker, this quotes the most on-topic sentence from the top
        piece of evidence and attaches the marker by construction. It is worse
        prose than the model would have written and it is always attributable.
        """
        marker, chunk = next(iter(marker_map.items()))
        sentence = self._best_sentence(question, chunk).rstrip(".")
        text = f"{sentence} [{marker}]."
        citation = Citation(
            marker=marker,
            chunk_id=chunk.id,
            doc_id=chunk.doc_id,
            title=chunk.metadata.get("doc_title") or chunk.doc_id,
            valid=True,
        )
        return text, [citation]

    # -- main ------------------------------------------------------------
    def answer(self, question: str, evidence: Sequence[ScoredChunk]) -> Answer:
        grounding = self.grounding_strength(question, evidence)

        if not evidence:
            return Answer(question=question, text=REFUSAL_TEXT, refused=True,
                          reason="no_evidence_retrieved", grounding=0.0)
        if grounding < self.grounding_floor:
            # Refuse before spending a call. The tokens for a prompt that is going
            # to produce a hedge are tokens spent producing a hedge.
            return Answer(question=question, text=REFUSAL_TEXT, refused=True,
                          reason="below_grounding_floor", grounding=grounding,
                          evidence=list(evidence[: self.max_evidence]))

        messages, marker_map = self.build_messages(question, evidence)
        response = self.llm.complete(messages)
        text, citations = self.validate(response.text.strip(), marker_map)

        fallback_used = False
        reason = "answered"
        if not any(c.valid for c in citations):
            if not self.extractive_fallback:
                return Answer(
                    question=question, text=REFUSAL_TEXT, refused=True,
                    reason="citation_contract_broken", grounding=grounding,
                    evidence=list(evidence[: self.max_evidence]),
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                    latency_ms=response.latency_ms,
                )
            text, citations = self._extractive_answer(question, marker_map)
            fallback_used = True
            reason = "extractive_fallback"
        elif any(not c.valid for c in citations):
            reason = "answered_with_stripped_citations"

        if text.strip().lower().startswith(REFUSAL_TEXT):
            return Answer(question=question, text=REFUSAL_TEXT, refused=True,
                          reason="model_declined", grounding=grounding,
                          evidence=list(evidence[: self.max_evidence]),
                          prompt_tokens=response.prompt_tokens,
                          completion_tokens=response.completion_tokens,
                          latency_ms=response.latency_ms)

        return Answer(
            question=question,
            text=text,
            citations=citations,
            evidence=list(evidence[: self.max_evidence]),
            refused=False,
            reason=reason,
            grounding=grounding,
            fallback_used=fallback_used,
            prompt_tokens=response.prompt_tokens or count_message_tokens(messages),
            completion_tokens=response.completion_tokens,
            latency_ms=response.latency_ms,
        )
