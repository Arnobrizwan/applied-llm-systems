"""Retrieval critique: does this evidence actually answer the question.

The single most useful thing a RAG system can know about itself is whether what
it just retrieved is good enough to answer from. Almost none of them ask. They
retrieve, they stuff, they generate, and the first entity that discovers the
evidence was wrong is the user.

Three independent signals are combined here, chosen because they fail in
different ways and so are worth combining rather than picking one:

**Coverage.** How much of the question's content vocabulary appears anywhere in
the evidence, weighted so that rare terms count for more. Cheap, deterministic,
and it is the signal that catches "nothing here is about what was asked". Its
weakness is that a document can contain every query term and still state the
opposite of the answer.

**Agreement.** Mean pairwise lexical overlap between the retrieved chunks. High
agreement means retrieval converged on one topic; low agreement means it returned
a spread of unrelated things, which is what a retriever does when it has no good
answer and is ranking noise. Its weakness is that five copies of the same wrong
chunk also agree, which is why the retriever above it uses MMR.

**Judge verdict.** An LLM asked whether the evidence answers the question. It is
the only signal here with any semantic understanding, and it is also the only one
that costs a model call and can be wrong in ways the other two cannot. Weighted
accordingly rather than trusted alone.

A fourth signal was built, measured and removed
-----------------------------------------------
A specificity penalty was added first: multiply the confidence down when the
highest-idf term in the question appears nowhere in the evidence, on the theory
that "which new regions will Meridian launch in 2028" is unanswerable precisely
because "2028" is missing. It reads well and it did not survive measurement. Swept
at penalties of 0.0, 0.2, 0.35, 0.5 and 0.6 with the abstention threshold
re-calibrated for each, every setting produced the same outcome on all 27
evaluation questions: the same three answerable questions handled the same way,
the same three adversarial questions leaked. The penalty rescaled the confidence
axis and moved nothing across a decision boundary. It was removed rather than
kept as decoration. The residual failures it was supposed to catch are listed in
the README, and they need semantic understanding rather than another lexical
heuristic.

`missing_terms` survives on the `Critique` because it is the single most useful
field when reading a failed run, even though it no longer feeds the score.

Why a weighted blend and not a classifier
-----------------------------------------
A learned confidence model is better and needs labelled data from production
traffic that a new system does not have yet. A weighted blend of three
interpretable signals can be tuned by looking at the components on the failures,
which is the right tool for the stage where you have no labels. The weights are
constructor arguments so this is not a fixed decision.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from llmkit import LLMProvider, ScoredChunk, get_llm, keyword_overlap, system, user

from projects.p01_rag_pipeline.terms import content_terms, term_set

_JSON_RE = re.compile(r"\{[\s\S]*\}")


@dataclass
class Critique:
    """A verdict on one evidence set, with every component kept for debugging."""

    confidence: float
    coverage: float
    agreement: float
    judge_score: float
    judge_verdict: str
    missing_terms: List[str] = field(default_factory=list)
    evidence_count: int = 0
    notes: str = ""

    def as_line(self) -> str:
        return (
            f"conf={self.confidence:.3f} cover={self.coverage:.3f} "
            f"agree={self.agreement:.3f} judge={self.judge_verdict}"
            f"({self.judge_score:.2f}) missing={self.missing_terms[:3]}"
        )


class RetrievalCritic:
    def __init__(
        self,
        llm: Optional[LLMProvider] = None,
        document_frequencies: Optional[Dict[str, int]] = None,
        corpus_size: int = 0,
        weights: Optional[Dict[str, float]] = None,
        use_judge: bool = True,
    ):
        self.llm = llm or get_llm()
        self.df = document_frequencies or {}
        self.corpus_size = max(corpus_size, 1)
        self.weights = weights or {"coverage": 0.55, "agreement": 0.15, "judge": 0.30}
        self.use_judge = use_judge
        self.judge_calls = 0

    # -- components ------------------------------------------------------
    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1.0 + (self.corpus_size - df + 0.5) / (df + 0.5))

    def coverage(self, question: str, evidence: Sequence[ScoredChunk]) -> tuple:
        """idf-weighted coverage over the union of the evidence, plus what is missing.

        Union rather than best-single-chunk, unlike the answerer's grounding
        gate. The critic is asking whether the evidence *set* is sufficient, and a
        question whose answer is split across two chunks is answerable; the
        answerer's gate asks whether any single chunk is a plausible source, which
        is a different question with a different right answer.
        """
        terms = content_terms(question)
        if not terms:
            return 0.0, []
        present: set = set()
        for item in evidence:
            present |= term_set(item.chunk.text)
        denominator = sum(self._idf(t) for t in terms)
        if denominator <= 0:
            return 0.0, []
        covered = sum(self._idf(t) for t in terms if t in present)
        missing = [t for t in terms if t not in present]
        return covered / denominator, missing

    @staticmethod
    def agreement(evidence: Sequence[ScoredChunk]) -> float:
        """Mean pairwise lexical overlap. One chunk is treated as no evidence of agreement."""
        texts = [item.chunk.text for item in evidence]
        if len(texts) < 2:
            return 0.0
        pairs = [
            keyword_overlap(texts[i], texts[j])
            for i in range(len(texts))
            for j in range(i + 1, len(texts))
        ]
        return sum(pairs) / len(pairs)

    def judge(self, question: str, evidence: Sequence[ScoredChunk]) -> tuple:
        """Ask the model whether the context answers the question.

        Returns (score, verdict). A provider that cannot produce a parseable
        verdict is treated as an abstention worth 0.5 rather than as a failure:
        an unparseable judge should not be able to veto an otherwise well
        covered evidence set, and it should not be able to rubber-stamp one
        either.
        """
        if not self.use_judge or not evidence:
            return 0.5, "skipped"
        self.judge_calls += 1
        blocks = "\n".join(f"[S{i}] {e.chunk.text}" for i, e in enumerate(evidence, start=1))
        prompt = (
            "You are a strict judge. Decide whether the context below is enough to "
            "answer the question. Reply with JSON containing score, verdict "
            "(pass or fail) and reason.\n\nContext:\n" + blocks
        )
        response = self.llm.complete([system(prompt), user(question)])
        match = _JSON_RE.search(response.text or "")
        if not match:
            return 0.5, "unparseable"
        try:
            payload = json.loads(match.group(0))
        except (ValueError, TypeError):
            return 0.5, "unparseable"
        verdict = str(payload.get("verdict", "unparseable")).lower()
        try:
            score = float(payload.get("score", 0.5))
        except (TypeError, ValueError):
            score = 0.5
        score = min(max(score, 0.0), 1.0)
        if verdict == "fail":
            # A failing verdict contributes nothing rather than its own score. A
            # judge that says "fail, 0.9 confidence" is 0.9 confident of failure.
            score = 0.0
        return score, verdict

    # -- driver ----------------------------------------------------------
    def critique(self, question: str, evidence: Sequence[ScoredChunk]) -> Critique:
        if not evidence:
            return Critique(0.0, 0.0, 0.0, 0.0, "no_evidence",
                            missing_terms=content_terms(question), evidence_count=0,
                            notes="retrieval returned nothing")
        coverage, missing = self.coverage(question, evidence)
        agreement = self.agreement(evidence)
        judge_score, verdict = self.judge(question, evidence)
        weights = self.weights
        total = sum(weights.values()) or 1.0
        confidence = (
            weights["coverage"] * coverage
            + weights["agreement"] * agreement
            + weights["judge"] * judge_score
        ) / total
        return Critique(
            confidence=confidence,
            coverage=coverage,
            agreement=agreement,
            judge_score=judge_score,
            judge_verdict=verdict,
            missing_terms=missing,
            evidence_count=len(evidence),
        )
