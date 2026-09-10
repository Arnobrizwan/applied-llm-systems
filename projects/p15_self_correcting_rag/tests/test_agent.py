"""The control loop: bounds, escalation, tracing and abstention."""
import pytest

from llmkit import Tracer
from llmkit.corpus import gold_questions

from projects.p01_rag_pipeline.pipeline import RagPipeline, preset
from projects.p15_self_correcting_rag.adversarial import adversarial_questions
from projects.p15_self_correcting_rag.agent import ABSTAIN_TEXT, SelfCorrectingRAG
from projects.p15_self_correcting_rag.fallback_corpus import FALLBACK_GOLD
from projects.p15_self_correcting_rag.tools import FallbackSearch, NullSearch


@pytest.fixture(scope="module")
def pipeline():
    return RagPipeline.build(config=preset("hybrid+rerank"))


@pytest.fixture(scope="module")
def agent(pipeline):
    return SelfCorrectingRAG(pipeline, search_tool=FallbackSearch())


def test_max_steps_is_a_hard_bound_not_a_suggestion(pipeline):
    """An unbounded repair loop is an unbounded bill."""
    question = "What is the exact discount on a three year prepaid Enterprise contract?"
    for limit in (1, 2, 3, 4):
        capped = SelfCorrectingRAG(pipeline, search_tool=FallbackSearch(), max_steps=limit)
        result = capped.answer(question)
        assert result.step_count <= limit
        assert [s.action for s in result.steps] == \
            ["retrieve", "rewrite", "widen", "fallback"][:limit]


def test_a_confident_first_retrieval_stops_the_ladder_immediately(agent):
    result = agent.answer("How are webhook deliveries authenticated?")
    assert result.step_count == 1
    assert result.steps[0].action == "retrieve"
    assert result.steps[0].accepted
    assert not result.abstained
    assert "webhooks" in result.cited_doc_ids


def test_escalation_reaches_the_search_tool_and_cites_what_it_finds(pipeline):
    """The last rung has to produce usable evidence, not just get called."""
    search = FallbackSearch()
    agent = SelfCorrectingRAG(pipeline, search_tool=search)
    gold = FALLBACK_GOLD[2]  # answerable only from the second corpus
    result = agent.answer(gold["question"])

    assert search.calls >= 1
    assert result.winning_step == len(result.steps), "the fallback rung produced the best evidence"
    assert gold["doc_id"] in result.cited_doc_ids
    assert any(c.chunk_id.startswith("fallback:") for c in result.answer.valid_citations)


def test_escalation_can_never_make_the_answer_worse(agent):
    """Best-so-far tracking, asserted rather than assumed.

    A later rung that retrieves confidently wrong material must not overwrite an
    earlier, better result. The final confidence is therefore the maximum across
    every rung, not the last one.
    """
    for gold in list(gold_questions())[:6]:
        result = agent.answer(gold["question"])
        best = max(step.critique.confidence for step in result.steps)
        assert result.confidence == pytest.approx(best)
        assert result.steps[result.winning_step - 1].critique.confidence == \
            pytest.approx(best)


def test_abstains_below_the_floor_rather_than_guessing(pipeline):
    agent = SelfCorrectingRAG(pipeline, search_tool=FallbackSearch(),
                              accept_above=0.99, abstain_below=0.95)
    result = agent.answer("What is the capital city of Iceland?")
    assert result.abstained
    assert result.answer.text == ABSTAIN_TEXT
    assert result.answer.refused
    assert result.cited_doc_ids == []


def test_abstention_is_not_an_accident_of_a_weak_fallback(pipeline):
    """With a tool that finds nothing, abstention must still come from the floor."""
    null = NullSearch()
    agent = SelfCorrectingRAG(pipeline, search_tool=null)
    result = agent.answer("Which new regions will Meridian launch in 2028?")

    assert null.calls == 1
    assert result.abstained
    assert result.steps[-1].action == "fallback"
    assert result.steps[-1].critique.evidence_count == 0
    # The abstention is driven by the best confidence across the ladder being
    # under the floor, not by the last rung returning nothing.
    assert result.confidence < agent.abstain_below
    assert result.winning_step < len(result.steps)


def test_the_agent_abstains_more_often_than_single_shot_on_unanswerable_questions(agent, pipeline):
    questions = adversarial_questions()
    agent_abstentions = sum(1 for g in questions if agent.answer(g["question"]).abstained)
    single_abstentions = sum(1 for g in questions if pipeline.ask(g["question"]).refused)
    assert agent_abstentions >= single_abstentions
    assert agent_abstentions >= len(questions) // 2


def test_every_step_is_traced_with_its_confidence(pipeline):
    tracer = Tracer("test")
    agent = SelfCorrectingRAG(pipeline, search_tool=FallbackSearch(), tracer=tracer)
    result = agent.answer("How long can a deleted workspace be restored?")

    names = [span.name for span in tracer.spans]
    assert "agent.run" in names
    assert names.count("agent.critique") == result.step_count
    root = [s for s in tracer.spans if s.name == "agent.run"][0]
    assert root.attributes["steps"] == result.step_count
    assert root.attributes["outcome"] in ("answered", "abstained")
    # Every span in the run shares one trace id, which is what makes a single
    # question reconstructable from a log.
    assert {s.trace_id for s in tracer.spans} == {root.trace_id}
    critiques = [s for s in tracer.spans if s.name == "agent.critique"]
    assert all("confidence" in s.attributes for s in critiques)


def test_invalid_thresholds_are_rejected(pipeline):
    with pytest.raises(ValueError):
        SelfCorrectingRAG(pipeline, max_steps=0)
    with pytest.raises(ValueError):
        SelfCorrectingRAG(pipeline, accept_above=0.2, abstain_below=0.8)


def test_explain_reconstructs_the_whole_run(agent):
    result = agent.answer("How long are request logs kept?")
    text = agent.explain(result)
    for step in result.steps:
        assert step.action in text
    assert "final:" in text and "answer:" in text
