"""Query rewriting, retrieval critique, and the four-outcome evaluation."""
import pytest

from llmkit import ScoredChunk, ScriptedLLM
from llmkit.corpus import chunks, gold_questions

from projects.p01_rag_pipeline.pipeline import RagPipeline, preset
from projects.p15_self_correcting_rag.adversarial import adversarial_questions
from projects.p15_self_correcting_rag.agent import SelfCorrectingRAG
from projects.p15_self_correcting_rag.critique import RetrievalCritic
from projects.p15_self_correcting_rag.evaluate import (
    ABSTAINED, CORRECT, CORRECTED, WRONG, calibrate_abstention, escalation_histogram,
    evaluate_agent, evaluate_single_shot, format_confusion,
)
from projects.p15_self_correcting_rag.fallback_corpus import FALLBACK_GOLD
from projects.p15_self_correcting_rag.rewrite import QueryRewriter, fuse
from projects.p15_self_correcting_rag.tools import FallbackSearch


@pytest.fixture(scope="module")
def pipeline():
    return RagPipeline.build(config=preset("hybrid+rerank"))


@pytest.fixture(scope="module")
def critic(pipeline):
    df, size = pipeline.retriever.document_frequencies()
    return RetrievalCritic(document_frequencies=df, corpus_size=size)


@pytest.fixture(scope="module")
def agent(pipeline):
    return SelfCorrectingRAG(pipeline, search_tool=FallbackSearch())


def by_doc(doc_id):
    return [c for c in chunks() if c.doc_id == doc_id][0]


def scored(doc_id, score=1.0):
    return ScoredChunk(chunk=by_doc(doc_id), score=score)


# -- rewriting ---------------------------------------------------------------
def test_decomposition_splits_only_where_both_halves_are_retrievable():
    rewriter = QueryRewriter()
    parts = rewriter.decompose("Which role can export the audit log and how long are logs kept?")
    assert len(parts) == 2
    assert "audit log" in parts[0]
    # "before and after" is a single concept, not two retrievable questions.
    assert rewriter.decompose("What happens before and after maintenance?") == []


def test_keyword_rewrite_keeps_surface_forms_not_stems():
    """BM25 indexes raw tokens, so a folded stem would match nothing."""
    rewriter = QueryRewriter()
    keywords = rewriter.keywords("How long is a Meridian token valid before it expires?")
    assert "expires" in keywords, "the folded form 'expir' is not in any index"
    assert "meridian" in keywords and "token" in keywords
    assert "how" not in keywords.split() and "before" not in keywords.split()


def test_hypothetical_rewrite_discards_non_prose_output():
    """A provider that returns JSON must not become the retrieval query."""
    rewriter = QueryRewriter(llm=ScriptedLLM(['{"score": 0.8, "verdict": "pass"}']))
    assert rewriter.hypothetical_answer("when do tokens expire", [scored("auth-rotation")]) is None

    prose = QueryRewriter(llm=ScriptedLLM(["Meridian tokens expire 90 days after creation [S1]."]))
    text = prose.hypothetical_answer("when do tokens expire", [scored("auth-rotation")])
    assert text == "Meridian tokens expire 90 days after creation."


def test_rewrites_are_deduplicated_against_the_original():
    rewriter = QueryRewriter(llm=ScriptedLLM(["when do tokens expire"]))
    variants = rewriter.rewrite("when do tokens expire", [scored("auth-rotation")])
    queries = [v.query.lower() for v in variants]
    assert "when do tokens expire" not in queries
    assert len(queries) == len(set(queries))


def test_fusion_rewards_chunks_several_reformulations_agree_on():
    agreed = [scored("auth-rotation"), scored("errors")]
    other = [scored("billing"), scored("auth-rotation")]
    fused = fuse([agreed, other], k=3)
    assert fused[0].chunk.doc_id == "auth-rotation"
    assert fused[0].components["reformulations"] == 2.0
    assert fuse([], k=3) == []


# -- critique ----------------------------------------------------------------
def test_critique_scores_matching_evidence_above_unrelated_evidence(critic):
    question = "How are webhook deliveries authenticated?"
    good = critic.critique(question, [scored("webhooks")])
    bad = critic.critique(question, [scored("billing")])

    assert good.confidence > bad.confidence
    assert good.coverage > bad.coverage
    # The billing document does not contain the subject of the question at all,
    # which is exactly what `missing_terms` is for when reading a failed run.
    assert "webhook" in bad.missing_terms
    assert "webhook" not in good.missing_terms
    assert len(good.missing_terms) < len(bad.missing_terms)


def test_critique_reports_zero_confidence_when_nothing_was_retrieved(critic):
    verdict = critic.critique("anything", [])
    assert verdict.confidence == 0.0
    assert verdict.judge_verdict == "no_evidence"
    assert verdict.evidence_count == 0


def test_agreement_needs_at_least_two_chunks(critic):
    assert critic.agreement([scored("webhooks")]) == 0.0
    spread = critic.agreement([scored("webhooks"), scored("billing"), scored("roles")])
    focused = critic.agreement([scored("plans"), scored("sla")])
    assert 0.0 <= spread <= 1.0
    assert focused > spread, "documents on one topic must agree more than a random spread"


def test_a_failing_judge_verdict_contributes_nothing(pipeline):
    """"Fail with 0.9 confidence" is 0.9 confident of failure, not 0.9 good."""
    df, size = pipeline.retriever.document_frequencies()
    failing = RetrievalCritic(
        llm=ScriptedLLM(['{"score": 0.9, "verdict": "fail", "reason": "unsupported"}']),
        document_frequencies=df, corpus_size=size,
    )
    passing = RetrievalCritic(
        llm=ScriptedLLM(['{"score": 0.9, "verdict": "pass", "reason": "supported"}']),
        document_frequencies=df, corpus_size=size,
    )
    question = "How are webhook deliveries authenticated?"
    evidence = [scored("webhooks")]
    assert failing.critique(question, evidence).judge_score == 0.0
    assert passing.critique(question, evidence).judge_score == pytest.approx(0.9)
    assert failing.critique(question, evidence).confidence < \
        passing.critique(question, evidence).confidence


def test_an_unparseable_judge_neither_vetoes_nor_rubber_stamps(pipeline):
    df, size = pipeline.retriever.document_frequencies()
    garbled = RetrievalCritic(llm=ScriptedLLM(["I think it is probably fine, honestly"]),
                              document_frequencies=df, corpus_size=size)
    verdict = garbled.critique("How are webhook deliveries authenticated?", [scored("webhooks")])
    assert verdict.judge_verdict == "unparseable"
    assert verdict.judge_score == 0.5


# -- evaluation --------------------------------------------------------------
def test_agent_beats_single_shot_on_the_combined_question_set(agent, pipeline):
    answerable = list(gold_questions()) + list(FALLBACK_GOLD)
    agent_confusion, agent_rows = evaluate_agent(agent, answerable)
    single_confusion, _ = evaluate_single_shot(pipeline, answerable)

    agent_right = agent_confusion.counts[CORRECT] + agent_confusion.counts[CORRECTED]
    single_right = single_confusion.counts[CORRECT] + single_confusion.counts[CORRECTED]
    assert agent_right > single_right
    assert agent_confusion.counts[WRONG] <= single_confusion.counts[WRONG]
    # The corrected column is what the loop bought; an empty one means the loop
    # never repaired anything and the whole project is overhead.
    assert agent_confusion.counts[CORRECTED] > 0
    assert single_confusion.counts[CORRECTED] == 0, "single-shot cannot self-correct by definition"
    assert "self-correcting" in format_confusion([agent_confusion])

    histogram = escalation_histogram(agent_rows)
    assert len(histogram) > 1, "if everything wins at step 1 the ladder is dead weight"
    assert max(histogram) > 1


def test_fallback_only_questions_are_unreachable_without_the_tool(pipeline):
    """Proves the fallback questions really are outside the primary corpus."""
    for gold in FALLBACK_GOLD:
        answer = pipeline.ask(gold["question"])
        assert gold["doc_id"] not in answer.cited_doc_ids


def test_calibration_sweep_picks_a_threshold_from_data(agent):
    rows = calibrate_abstention(
        agent, list(gold_questions()) + list(FALLBACK_GOLD), adversarial_questions()
    )
    assert len(rows) == 8
    # Raising the floor must trade answers for refusals monotonically, or the
    # sweep is measuring noise rather than a threshold.
    assert rows[0].adversarial_abstained <= rows[-1].adversarial_abstained
    assert rows[0].answerable_abstained <= rows[-1].answerable_abstained
    best = max(rows, key=lambda r: r.score)
    assert best.threshold == pytest.approx(agent.abstain_below), \
        "the shipped default has to be the value the sweep chose"


def test_outcome_labels_are_mutually_exclusive(agent):
    confusion, rows = evaluate_agent(agent, adversarial_questions())
    assert confusion.n == len(rows)
    assert sum(confusion.counts.values()) == confusion.n
    assert confusion.counts[CORRECT] == 0, "an unanswerable question cannot be answered correctly"
    assert confusion.counts[ABSTAINED] + confusion.counts[WRONG] == confusion.n
