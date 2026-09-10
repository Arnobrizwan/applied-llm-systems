"""Citation enforcement, refusal, and the end-to-end evaluation."""
import pytest

from llmkit import ScriptedLLM
from llmkit.corpus import gold_questions

from projects.p01_rag_pipeline.answer import REFUSAL_TEXT, CitedAnswerer
from projects.p01_rag_pipeline.evaluate import evaluate_all, format_comparison
from projects.p01_rag_pipeline.pipeline import RagPipeline, preset


@pytest.fixture(scope="module")
def pipeline():
    return RagPipeline.build(config=preset("hybrid+rerank"))


@pytest.fixture(scope="module")
def evidence(pipeline):
    return pipeline.retrieve("How long is a Meridian token valid before it expires?", k=3)


def test_answer_cites_a_chunk_that_was_actually_retrieved(pipeline):
    answer = pipeline.ask("How long is a Meridian token valid before it expires?")
    assert not answer.refused
    assert answer.valid_citations
    retrieved_ids = {item.chunk.id for item in answer.evidence}
    assert all(c.chunk_id in retrieved_ids for c in answer.valid_citations)
    assert "auth-rotation" in answer.cited_doc_ids


def test_hallucinated_citation_is_stripped_not_shipped(pipeline, evidence):
    """A marker pointing at evidence that was never supplied must not survive."""
    liar = ScriptedLLM(["Tokens expire after 90 days [S1]. They are unlimited on Enterprise [S9]."])
    answerer = CitedAnswerer(llm=liar, document_frequencies=pipeline.answerer.df,
                             corpus_size=pipeline.answerer.corpus_size)
    answer = answerer.answer("How long is a token valid before it expires?", evidence)

    assert [c.marker for c in answer.invalid_citations] == ["S9"]
    assert "[S9]" not in answer.text, "an unresolvable citation is worse than no citation"
    assert "[S1]" in answer.text
    assert answer.reason == "answered_with_stripped_citations"


def test_uncited_answer_falls_back_to_a_cited_extract(pipeline, evidence):
    """When the model ignores the contract the system must not ship bare prose."""
    chatty = ScriptedLLM(["Sure, tokens last about three months and you can renew them."])
    answerer = CitedAnswerer(llm=chatty, document_frequencies=pipeline.answerer.df,
                             corpus_size=pipeline.answerer.corpus_size)
    answer = answerer.answer("How long is a token valid before it expires?", evidence)

    assert answer.fallback_used
    assert answer.reason == "extractive_fallback"
    assert answer.valid_citations
    assert "three months" not in answer.text
    quoted = answer.valid_citations[0].chunk_id
    source = [e.chunk for e in evidence if e.chunk.id == quoted][0]
    stripped = answer.text.split(" [")[0]
    assert stripped in source.text, "the fallback must quote the evidence verbatim"


def test_uncited_answer_can_be_configured_to_refuse_instead(pipeline, evidence):
    chatty = ScriptedLLM(["Tokens last about three months."])
    answerer = CitedAnswerer(llm=chatty, extractive_fallback=False,
                             document_frequencies=pipeline.answerer.df,
                             corpus_size=pipeline.answerer.corpus_size)
    answer = answerer.answer("How long is a token valid before it expires?", evidence)
    assert answer.refused and answer.reason == "citation_contract_broken"


def test_off_corpus_question_is_refused_without_calling_the_model(pipeline):
    counter = ScriptedLLM(["this should never be reached"])
    answerer = CitedAnswerer(llm=counter, document_frequencies=pipeline.answerer.df,
                             corpus_size=pipeline.answerer.corpus_size)
    hits = pipeline.retrieve("What is the capital city of Iceland?", k=5)
    answer = answerer.answer("What is the capital city of Iceland?", hits)

    assert answer.refused and answer.reason == "below_grounding_floor"
    assert answer.text == REFUSAL_TEXT
    assert counter.calls == [], "refusing after paying for the call defeats the point"


def test_empty_retrieval_refuses_rather_than_answering_from_memory(pipeline):
    answer = pipeline.answerer.answer("anything at all", [])
    assert answer.refused and answer.reason == "no_evidence_retrieved"


def test_grounding_is_comparable_across_retrieval_modes(pipeline):
    """The refusal gate must not be a threshold on an incomparable raw score.

    BM25 scores are unbounded, cosine is bounded and RRF depends only on rank, so
    the same chunk arrives with wildly different scores per mode. The grounding
    signal is computed from the question and the chunk text, so it does not move.
    """
    question = "How long are request logs kept?"
    values = []
    for mode in ("vector", "bm25", "hybrid"):
        hits = pipeline.retrieve(question, k=5, mode=mode, rerank=False)
        top_scores = [h.score for h in hits]
        values.append((pipeline.answerer.grounding_strength(question, hits), top_scores[0]))

    groundings = [v[0] for v in values]
    raw_scores = [v[1] for v in values]
    assert max(groundings) - min(groundings) < 0.05
    assert max(raw_scores) / max(min(raw_scores), 1e-9) > 10.0


def test_evidence_markers_map_one_to_one_onto_the_prompt(pipeline, evidence):
    messages, marker_map = pipeline.answerer.build_messages("when do tokens expire", evidence)
    body = messages[0].content
    assert len(marker_map) == len(evidence)
    for marker, chunk in marker_map.items():
        assert f"[{marker}]" in body
        assert chunk.text[:40] in body


def test_hybrid_plus_rerank_is_the_best_configuration_measured(pipeline):
    reports = {r.label: r for r in evaluate_all(pipeline, gold_questions())}
    best = reports["hybrid+rerank"]

    assert best.retrieval.recall_at_1 > reports["vector"].retrieval.recall_at_1
    assert best.retrieval.mrr >= max(r.retrieval.mrr for r in reports.values())
    # Every emitted citation resolves in every configuration; that is the floor,
    # not a nice-to-have.
    assert all(r.answer.citation_validity == 1.0 for r in reports.values())
    assert "hybrid+rerank" in format_comparison(list(reports.values()))


def test_evaluation_counts_citations_not_questions(pipeline):
    report = evaluate_all(pipeline, gold_questions(), labels=["hybrid+rerank"])[0]
    assert report.answer.citations_emitted > report.answer.n, \
        "answers cite more than one source, so the denominator cannot be question count"
    assert report.answer.citations_invalid == 0
    assert len(report.per_question) == len(gold_questions())
