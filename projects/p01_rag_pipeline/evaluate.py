"""Evaluation: four configurations, one table, no invented numbers.

What is measured and why each metric earns its place
----------------------------------------------------
recall@1 / @3 / @5
    Did the document that actually answers the question appear in the top k.
    recall@1 is the number that matters when the answer is generated from one
    chunk; recall@5 is the ceiling on what any reranker downstream could ever
    achieve, because a reranker can only reorder what retrieval already found.

MRR
    Mean reciprocal rank of the first correct document. It separates two systems
    that have identical recall@5 but put the right answer first versus fifth,
    which is a real quality difference that recall@5 hides completely.

citation validity
    Of every citation marker the model emitted, the fraction that resolves to a
    chunk that was genuinely in the prompt. This is the metric that catches a
    model inventing `[S7]` when five sources were supplied.

contract break rate
    The fraction of answered questions where the model produced no resolvable
    citation at all and the deterministic extractive fallback had to run. This is
    reported separately from citation validity on purpose: folding them together
    would let a system with a good fallback hide a model that never complies.

refusal rate
    The fraction of questions where the pipeline declined to answer. Read it
    against answer accuracy: a system that refuses everything has perfect
    citation validity and is useless.

grounded answer rate
    The fraction of questions where the final answer text contains the gold
    phrase from the corpus. It is a substring check, not a semantic one, so it
    under-counts correct answers that paraphrase. It is reported as a floor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .pipeline import PipelineConfig, RagPipeline, preset


@dataclass
class RetrievalMetrics:
    n: int = 0
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    mrr: float = 0.0


@dataclass
class AnswerMetrics:
    n: int = 0
    refusal_rate: float = 0.0
    citation_validity: float = 0.0
    citations_emitted: int = 0
    citations_invalid: int = 0
    contract_break_rate: float = 0.0
    grounded_answer_rate: float = 0.0
    cited_doc_precision: float = 0.0
    avg_prompt_tokens: float = 0.0


@dataclass
class ConfigReport:
    label: str
    retrieval: RetrievalMetrics
    answer: AnswerMetrics
    per_question: List[Dict] = field(default_factory=list)


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def evaluate_retrieval(
    pipeline: RagPipeline,
    questions: Sequence[Dict[str, str]],
    depth: int = 5,
) -> RetrievalMetrics:
    """Rank the gold document and roll up recall/MRR.

    Retrieval is scored at document granularity, not chunk granularity. The gold
    set names the document that answers each question, and a chunk-level score
    would reward or punish an arbitrary chunking decision rather than the
    retriever.
    """
    hits_at = {1: 0, 3: 0, 5: 0}
    reciprocal = 0.0
    for gold in questions:
        results = pipeline.retrieve(gold["question"], k=depth)
        doc_ids: List[str] = []
        for item in results:
            if item.chunk.doc_id not in doc_ids:
                doc_ids.append(item.chunk.doc_id)
        rank = doc_ids.index(gold["doc_id"]) + 1 if gold["doc_id"] in doc_ids else 0
        if rank:
            reciprocal += 1.0 / rank
            for cutoff in hits_at:
                if rank <= cutoff:
                    hits_at[cutoff] += 1
    total = len(questions)
    return RetrievalMetrics(
        n=total,
        recall_at_1=_safe_div(hits_at[1], total),
        recall_at_3=_safe_div(hits_at[3], total),
        recall_at_5=_safe_div(hits_at[5], total),
        mrr=_safe_div(reciprocal, total),
    )


def evaluate_answers(
    pipeline: RagPipeline,
    questions: Sequence[Dict[str, str]],
) -> tuple:
    refusals = 0
    emitted = 0
    invalid = 0
    breaks = 0
    grounded = 0
    correct_doc_cited = 0
    answered = 0
    prompt_tokens = 0
    rows: List[Dict] = []

    for gold in questions:
        answer = pipeline.ask(gold["question"])
        prompt_tokens += answer.prompt_tokens
        emitted += len(answer.citations)
        invalid += len(answer.invalid_citations)
        if answer.refused:
            refusals += 1
        else:
            answered += 1
            if answer.fallback_used:
                breaks += 1
            if gold["must_contain"].lower() in answer.text.lower():
                grounded += 1
            if gold["doc_id"] in answer.cited_doc_ids:
                correct_doc_cited += 1
        rows.append(
            {
                "question": gold["question"],
                "gold_doc": gold["doc_id"],
                "refused": answer.refused,
                "reason": answer.reason,
                "cited": answer.cited_doc_ids,
                "grounding": round(answer.grounding, 3),
                "text": answer.text,
            }
        )

    total = len(questions)
    metrics = AnswerMetrics(
        n=total,
        refusal_rate=_safe_div(refusals, total),
        # Denominator is markers emitted, not questions asked: a system that emits
        # ten markers and gets one wrong is not the same as one that emits one.
        citation_validity=_safe_div(emitted - invalid, emitted) if emitted else 0.0,
        citations_emitted=emitted,
        citations_invalid=invalid,
        contract_break_rate=_safe_div(breaks, answered),
        grounded_answer_rate=_safe_div(grounded, total),
        cited_doc_precision=_safe_div(correct_doc_cited, answered),
        avg_prompt_tokens=_safe_div(prompt_tokens, total),
    )
    return metrics, rows


def evaluate_config(
    base: RagPipeline,
    questions: Sequence[Dict[str, str]],
    config: PipelineConfig,
) -> ConfigReport:
    pipeline = base.with_config(config)
    retrieval = evaluate_retrieval(pipeline, questions)
    answer, rows = evaluate_answers(pipeline, questions)
    return ConfigReport(label=config.label, retrieval=retrieval, answer=answer, per_question=rows)


def evaluate_all(
    base: RagPipeline,
    questions: Sequence[Dict[str, str]],
    labels: Optional[Sequence[str]] = None,
) -> List[ConfigReport]:
    labels = labels or ["vector", "bm25", "hybrid", "hybrid+rerank"]
    return [evaluate_config(base, questions, preset(label)) for label in labels]


def format_comparison(reports: Sequence[ConfigReport]) -> str:
    header = (
        f"{'configuration':<16}{'R@1':>7}{'R@3':>7}{'R@5':>7}{'MRR':>7}"
        f"{'cite-val':>10}{'break':>8}{'refuse':>8}{'grounded':>10}"
    )
    lines = [header, "-" * len(header)]
    for report in reports:
        r, a = report.retrieval, report.answer
        lines.append(
            f"{report.label:<16}"
            f"{r.recall_at_1:>7.2f}{r.recall_at_3:>7.2f}{r.recall_at_5:>7.2f}{r.mrr:>7.3f}"
            f"{a.citation_validity:>10.2f}{a.contract_break_rate:>8.2f}"
            f"{a.refusal_rate:>8.2f}{a.grounded_answer_rate:>10.2f}"
        )
    return "\n".join(lines)
