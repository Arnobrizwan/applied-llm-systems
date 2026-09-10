"""Query rewriting: three reformulations, fused.

A first-pass retrieval miss usually has one of three causes, and each of the
strategies here targets one of them.

**The question asks two things at once.** "Which role can export the audit log
and how long is it retained" has its answer split across two documents, and a
single embedding of the whole sentence lands between them. `decompose` splits on
coordinating conjunctions and retrieves each part.

**The question is mostly function words.** "How do I go about making sure that a
retried request does not charge twice" is 15 tokens of which 3 carry signal. Both
the dense and the lexical retriever dilute those 3 across the rest. `keywords`
strips it to the content terms.

**The question and the document do not share vocabulary.** A user asks "when does
my key stop working", the document says "tokens expire 90 days after creation".
HyDE addresses this by generating a hypothetical answer and retrieving with that
instead, on the basis that an answer looks more like the document than the
question does.

Honest note on the HyDE implementation
--------------------------------------
True HyDE generates a plausible answer from the model's own knowledge, with no
retrieval involved. `EchoLLM` is a deterministic rule engine with no knowledge to
draw on, so asking it to invent documentation text returns the prompt back. What
is implemented instead is grounded: retrieve once, hand the top hit to the model,
and ask it for the answer sentence. That makes it pseudo-relevance feedback
rather than HyDE, and it works offline. With a real model, drop the evidence from
`_hyde_messages` and it becomes HyDE proper. The interface, the fusion and
everything downstream are unchanged either way, which is the point of writing it
behind a strategy interface.

The results of all reformulations are fused with reciprocal rank fusion for the
same reason the base retriever uses it: the reformulations produce rankings on
incomparable scales, and agreement across reformulations is the signal worth
keeping.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from llmkit import (
    LLMProvider,
    ScoredChunk,
    get_llm,
    reciprocal_rank_fusion,
    system,
    tokenize,
    user,
)

from projects.p01_rag_pipeline.terms import QUESTION_WORDS

_SPLIT_RE = re.compile(r"\s+(?:and|or|as well as|plus|also)\s+", re.I)
_TRAILING_PUNCT = re.compile(r"[?.!,;:]+$")


@dataclass
class Reformulation:
    strategy: str
    query: str


class QueryRewriter:
    """Produces up to N reformulations of a question, deduplicated."""

    def __init__(self, llm: Optional[LLMProvider] = None, max_variants: int = 3):
        self.llm = llm or get_llm()
        self.max_variants = max_variants

    # -- strategies ------------------------------------------------------
    @staticmethod
    def decompose(question: str) -> List[str]:
        """Split a compound question into independently retrievable parts.

        Only splits where both halves still contain a content term. Splitting
        "rate limits and quotas" into "rate limits" and "quotas" is useful;
        splitting "before and after" into two fragments is not.
        """
        parts = [p.strip() for p in _SPLIT_RE.split(question) if p.strip()]
        if len(parts) < 2:
            return []
        useful = [p for p in parts if len([t for t in tokenize(p) if t not in QUESTION_WORDS]) >= 2]
        return useful if len(useful) >= 2 else []

    @staticmethod
    def keywords(question: str) -> str:
        """Content terms only, in their original surface form.

        Surface form, not folded: BM25 indexes raw tokens, so handing it a folded
        stem would match nothing. The folding in project 01 is used for scoring
        comparisons, not for building queries.
        """
        kept = [t for t in tokenize(question) if t not in QUESTION_WORDS]
        return " ".join(kept) if kept else _TRAILING_PUNCT.sub("", question)

    def _hyde_messages(self, question: str, evidence: Sequence[ScoredChunk]) -> List:
        blocks = []
        for idx, item in enumerate(evidence[:2], start=1):
            blocks.append(f"[S{idx}] {item.chunk.text}")
        instruction = (
            "Write the single sentence of product documentation that answers the "
            "user's question. Reply with that sentence only.\n\n" + "\n".join(blocks)
        )
        return [system(instruction), user(question)]

    def hypothetical_answer(self, question: str, evidence: Sequence[ScoredChunk]) -> Optional[str]:
        if not evidence:
            return None
        response = self.llm.complete(self._hyde_messages(question, evidence))
        text = (response.text or "").strip()
        # Providers that route on prompt keywords can return a JSON verdict here
        # instead of prose. Detect it and discard rather than retrieving with a
        # JSON blob as the query.
        if not text or text.startswith("{") or text.startswith("["):
            return None
        try:
            json.loads(text)
            return None
        except (ValueError, TypeError):
            pass
        # Strip evidence markers and repair the whitespace they leave behind, so
        # the pseudo-document reads as a sentence rather than as prompt residue.
        cleaned = re.sub(r"\[S\d+\]", "", text)
        cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned).strip()
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        if not cleaned or cleaned.lower() == question.strip().lower():
            return None
        return cleaned

    # -- driver ----------------------------------------------------------
    def rewrite(self, question: str, evidence: Sequence[ScoredChunk] = ()) -> List[Reformulation]:
        variants: List[Reformulation] = []
        seen = {question.strip().lower()}

        def add(strategy: str, text: Optional[str]) -> None:
            if not text:
                return
            key = text.strip().lower()
            if key in seen:
                return
            seen.add(key)
            variants.append(Reformulation(strategy, text.strip()))

        for part in self.decompose(question):
            add("decompose", part)
        add("keywords", self.keywords(question))
        add("hypothetical", self.hypothetical_answer(question, evidence))
        return variants[: self.max_variants]


def fuse(
    result_sets: Sequence[Sequence[ScoredChunk]],
    k: int = 5,
    rrf_k: int = 60,
) -> List[ScoredChunk]:
    """Fuse several reformulations' results by rank.

    A chunk that several independent reformulations of the same question all
    retrieve is far more likely to be the answer than one that only the luckiest
    phrasing found. That is the same agreement argument the base retriever makes
    between two retrievers, applied between several queries.
    """
    by_id: Dict[str, ScoredChunk] = {}
    rankings: List[List[str]] = []
    for results in result_sets:
        ranking: List[str] = []
        for item in results:
            by_id.setdefault(item.chunk.id, item)
            ranking.append(item.chunk.id)
        if ranking:
            rankings.append(ranking)
    if not rankings:
        return []

    fused: List[ScoredChunk] = []
    for chunk_id, score in reciprocal_rank_fusion(rankings, k=rrf_k)[:k]:
        original = by_id[chunk_id]
        fused.append(
            ScoredChunk(
                chunk=original.chunk,
                score=score,
                source=f"{original.source}+rewrite",
                components={**original.components, "reformulations": float(len(rankings))},
            )
        )
    return fused
