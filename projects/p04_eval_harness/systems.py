"""The systems the harness evaluates.

A harness with nothing real to measure proves nothing, so this module ships a
genuine retrieval-augmented answerer over `llmkit.corpus` (hybrid BM25 plus
embedding retrieval, fused with reciprocal rank fusion, answered extractively)
and two deliberately worse variants of it. The demo grades all three.

The variants are not random noise. They are the two regression shapes a quality
gate has to tell apart:

  `noisy`   - one retrieval knob moved. Real behaviour changes on a case or two.
              A gate must NOT block this on a 23 case set, because the evidence
              does not support calling it a regression.
  `degraded`- abstention switched off and answers truncated. This is a real
              regression and the gate must block it.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Set

from llmkit import (
    BM25, EchoLLM, InMemoryVectorStore, LLMProvider, Chunk, reciprocal_rank_fusion,
    sentence_split, system, tokenize, user,
)
from llmkit.corpus import documents

from .dataset import EvalCase
from .scorers import extract_json

ANSWER_INSTRUCTIONS = (
    "Answer the question using only the numbered evidence blocks. Cite the block "
    "you used. If the evidence does not contain the answer, say you do not know."
)

# The answer comes back through the schema path rather than as free text. Two
# reasons, and the second one is the honest one. First, a caller that has to
# parse an answer out of prose is a caller that breaks on a phrasing change.
# Second, llmkit.EchoLLM routes any prompt containing the words "rate", "score"
# or "grade" into its judge branch, and the Meridian corpus is full of the phrase
# "rate limit", so the free-text path returned judge verdicts instead of answers.
# The schema path is checked first inside EchoLLM and is not affected.
ANSWER_SCHEMA: Dict[str, object] = {
    "type": "object",
    "required": ["answer", "grounded"],
    "properties": {
        "answer": {"type": "string"},
        "grounded": {"type": "boolean"},
    },
}

# Interrogatives and generic verbs carry no corpus signal, and leaving them in
# the vocabulary check makes every "how long ..." question look unanswerable.
# llmkit.tokenize drops ordinary stopwords but keeps these on purpose, because
# retrieval scoring wants them; the abstention check does not.
_QUERY_STOPWORDS: Set[str] = {
    "how", "what", "when", "where", "which", "who", "why", "does", "do", "did",
    "can", "could", "should", "would", "will", "i", "my", "me", "you", "your",
    "there", "they", "them", "please", "tell", "give", "get", "let", "more",
    "many", "much", "long", "any", "some", "after", "before", "then", "than",
}


def _stem(token: str) -> str:
    """Crude plural stripper.

    A real stemmer (Porter, Snowball) is a dependency this repo does not take.
    This handles the only morphology the corpus actually needs: token/tokens,
    delivery/deliveries. It is stated as the simplification it is.
    """
    for suffix in ("ies", "es", "s"):
        if len(token) > 4 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def _content_terms(text: str) -> List[str]:
    terms = [_stem(t) for t in tokenize(text)]
    return [t for t in terms if len(t) > 2 and t not in _QUERY_STOPWORDS]


class GroundedQA:
    """Hybrid retrieval plus an extractive answer, with an abstention rule.

    Retrieval is BM25 fused with embedding search rather than either alone:
    BM25 alone misses paraphrases ("stop a retried payment charging twice" for
    an idempotency document) and embeddings alone miss exact identifiers (429,
    HMAC-SHA256). Fusion is reciprocal rank fusion because it needs no score
    calibration between two retrievers whose scores are on different scales.

    The abstention rule is the interesting part for evaluation. It abstains when
    most of the question's content terms appear nowhere in the corpus vocabulary,
    which is a crude proxy for "this question is about something we have no
    documents on". It is crude, the margin on this corpus is thin, and the whole
    reason it is measured on an adversarial slice is that a heuristic like this
    degrades silently when the corpus changes.
    """

    def __init__(self, k: int = 2, abstain_oov_ratio: Optional[float] = 0.6,
                 answer_word_limit: Optional[int] = None,
                 llm: Optional[LLMProvider] = None):
        self.k = k
        self.abstain_oov_ratio = abstain_oov_ratio
        self.answer_word_limit = answer_word_limit
        self.llm = llm or EchoLLM()
        # Sentence-level chunks, not document-level. Measured on the gold set,
        # document-level evidence caps the pass rate because the answer-bearing
        # sentence is often the second or third in the document and never makes
        # it into the generated answer. Sentence granularity costs recall at the
        # same k and buys precision; the demo reports the k sweep that shows it.
        self._chunks: List[Chunk] = []
        for doc in documents():
            for i, sent in enumerate(sentence_split(doc.text)):
                self._chunks.append(Chunk(id=f"{doc.id}#{i}", doc_id=doc.id, text=sent,
                                          ordinal=i, metadata=dict(doc.metadata)))
        self._by_id = {c.id: c for c in self._chunks}
        self._bm25 = BM25()
        for c in self._chunks:
            self._bm25.add(c.id, c.text)
        self._vectors = InMemoryVectorStore()
        self._vectors.add(self._chunks)
        self._vocabulary: Set[str] = set()
        for c in self._chunks:
            self._vocabulary.update(_stem(t) for t in tokenize(c.text))

    # -- retrieval -------------------------------------------------------
    def retrieve(self, question: str) -> List[str]:
        """Fuse a lexical and a semantic ranking over a deeper pool than k.

        The pool is k+2 on each side so fusion has something to disagree about.
        Fusing two top-k lists at the same k mostly reproduces whichever list was
        already right, which defeats the point of fusing.
        """
        lexical = [cid for cid, _ in self._bm25.search(question, k=self.k + 2)]
        semantic = [h.chunk.id for h in self._vectors.search(question, k=self.k + 2)]
        fused = reciprocal_rank_fusion([lexical, semantic])
        return [cid for cid, _ in fused[: self.k]]

    def unknown_term_ratio(self, question: str) -> float:
        terms = _content_terms(question)
        if not terms:
            return 1.0
        missing = sum(1 for t in terms if t not in self._vocabulary)
        return missing / len(terms)

    # -- answering -------------------------------------------------------
    def __call__(self, case: EvalCase) -> str:
        question = (case.input or "").strip()
        if not question:
            # Empty input reaches production through a trimmed template variable.
            # Answering it means answering a question nobody asked.
            return "The question was empty. Could you rephrase what you would like to know?"

        if self.abstain_oov_ratio is not None:
            ratio = self.unknown_term_ratio(question)
            if ratio > self.abstain_oov_ratio:
                return ("I do not know. That is not covered in the Meridian "
                        "documentation I have access to.")

        chunk_ids = self.retrieve(question)
        evidence = "\n".join(
            f"[S{i}] {self._by_id[cid].text}" for i, cid in enumerate(chunk_ids, 1)
        )
        messages = [system(ANSWER_INSTRUCTIONS + "\n" + evidence), user(question)]

        # A case that asks for a specific schema gets that schema verbatim, so the
        # json_schema scorer is grading the system's structured output and not the
        # harness's own wrapper object.
        case_schema = case.metadata.get("schema")
        if case_schema:
            return self.llm.complete(messages, json_schema=case_schema).text

        resp = self.llm.complete(messages, json_schema=ANSWER_SCHEMA)
        payload = extract_json(resp.text)
        text = payload.get("answer", "") if isinstance(payload, dict) else resp.text
        if self.answer_word_limit:
            words = text.split()
            if len(words) > self.answer_word_limit:
                text = " ".join(words[: self.answer_word_limit]) + " ..."
        return text


def baseline_system(**overrides) -> GroundedQA:
    """The reference configuration. This is what the committed baseline measures."""
    params: Dict[str, object] = {"k": 2, "abstain_oov_ratio": 0.6}
    params.update(overrides)
    return GroundedQA(**params)  # type: ignore[arg-type]


def noisy_system() -> GroundedQA:
    """One retrieval knob moved: k=2 becomes k=3. A small, real behaviour change.

    This is the case the gate has to get right. Something did change, the number
    does move, and on 23 cases the move is not evidence of a regression.
    """
    return GroundedQA(k=3, abstain_oov_ratio=0.6)


def degraded_system() -> GroundedQA:
    """Abstention removed and answers clipped. The regression the gate must catch."""
    return GroundedQA(k=2, abstain_oov_ratio=None, answer_word_limit=6)


def answers_for(system_fn, cases: Sequence[EvalCase]) -> List[str]:
    """Collect one answer per case. Used for pairwise comparison between systems."""
    return [system_fn(c) for c in cases]


_WS = re.compile(r"\s+")


def normalise_answer(text: str) -> str:
    """Shared whitespace normalisation for display in the demo report."""
    return _WS.sub(" ", text or "").strip()
