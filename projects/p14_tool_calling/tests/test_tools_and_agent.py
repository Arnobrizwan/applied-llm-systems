"""The four shipped tools, the adversarial calculator suite, and the agent loop."""
import pytest
from llmkit import EchoLLM

from projects.p14_tool_calling.agent import ToolAgent
from projects.p14_tool_calling.registry import ToolRegistry
from projects.p14_tool_calling.safe_math import UnsafeExpression, safe_eval
from projects.p14_tool_calling.sandbox import ToolSandbox
from projects.p14_tool_calling.tools import ALL_TOOLS, calculate, convert_units, current_time, search_docs

ATTACKS = [
    "__import__('os').system('id')",       # import via dunder
    "().__class__.__base__.__subclasses__()",  # the classic subclass walk
    "(9).__class__",                        # dunder on a literal
    "(9).real",                             # attribute access with no dunder
    "open('/etc/passwd').read()",           # call on an attribute
    "eval('1+1')",                          # a builtin that is not allowlisted
    "globals()",
    "2 ** 10000000",                        # arithmetic denial of service
    "[x for x in range(10)]",               # comprehension
    "(lambda: 1)()",                        # lambda
    "1 if 1 else 2",                        # conditional expression
    "'abc' * 100",                          # string literal
    "x = 1",                                # statement, not an expression
]


@pytest.mark.parametrize("expression", ATTACKS)
def test_the_calculator_rejects_every_adversarial_expression(expression):
    with pytest.raises(UnsafeExpression):
        safe_eval(expression)


@pytest.mark.parametrize(
    "expression,expected",
    [("2 * (3 + 4)", 14.0), ("sqrt(16)", 4.0), ("10 % 3", 1.0), ("7 // 2", 3.0),
     ("-2 ** 2", -4.0), ("round(pi, 4)", 3.1416), ("max(1, 9, 3)", 9.0), ("2 ** 64", 2.0 ** 64)],
)
def test_the_calculator_still_does_arithmetic_correctly(expression, expected):
    assert safe_eval(expression) == pytest.approx(expected)


def test_the_calculator_tool_wraps_rejection_as_a_permanent_error():
    with pytest.raises(ValueError):
        calculate("__import__('os')")
    assert calculate("600 * 60 / 1000")["result"] == 36.0


def test_unit_conversion_is_affine_for_temperature_and_linear_otherwise():
    assert convert_units(5, "mi", "km")["result"] == pytest.approx(8.04672)
    assert convert_units(37, "c", "f")["result"] == pytest.approx(98.6)
    assert convert_units(0, "c", "k")["result"] == pytest.approx(273.15)
    assert convert_units(1000, "g", "kg")["result"] == pytest.approx(1.0)


def test_converting_across_families_is_a_semantic_error_the_schema_cannot_catch():
    with pytest.raises(ValueError) as excinfo:
        convert_units(5, "km", "kg")
    assert "length" in str(excinfo.value) and "mass" in str(excinfo.value)


def test_corpus_search_returns_the_document_that_answers_the_question():
    hits = search_docs("how long before a token expires", k=3)
    assert "auth-rotation" in [h["doc_id"] for h in hits]
    assert search_docs("token expiry and rotation window", k=1)[0]["doc_id"] == "auth-rotation"
    assert all(set(h) == {"doc_id", "title", "score", "excerpt"} for h in hits)
    assert [h["score"] for h in hits] == sorted((h["score"] for h in hits), reverse=True)


def test_corpus_search_honours_k_and_caps_excerpt_length():
    assert len(search_docs("rate limit", k=1)) == 1
    assert all(len(h["excerpt"]) <= 224 for h in search_docs("rate limit", k=5))


def test_the_clock_is_pinned_so_runs_can_be_replayed():
    first = current_time(6)
    assert first == current_time(6)
    assert first["timestamp"].endswith("+06:00")
    assert current_time(-5)["timestamp"].endswith("-05:00")


def build_agent(**kwargs):
    registry = ToolRegistry()
    registry.register_all(ALL_TOOLS)
    sandbox = ToolSandbox(registry)
    return ToolAgent(EchoLLM(), sandbox, **kwargs), sandbox


def test_the_agent_always_terminates_inside_its_step_budget():
    agent, _sandbox = build_agent(max_steps=3)
    for question in ["What is the rate limit?", "Convert 5 miles to km.", "What time is it?"]:
        run = agent.run(question)
        assert len(run.steps) <= 3
        assert run.stopped_because in ("final_answer", "step_budget")
        assert run.answer


def test_every_agent_tool_call_is_audited_and_accounted_for():
    agent, sandbox = build_agent(max_steps=4)
    run = agent.run("What is the default rate limit per workspace?")
    assert run.llm_calls >= 2  # at least one decision and the final answer
    assert run.tokens > 0
    assert len(sandbox.audit_log) == run.tool_calls


def test_tag_filtering_restricts_which_tools_the_agent_can_choose():
    registry = ToolRegistry()
    registry.register_all(ALL_TOOLS)
    agent = ToolAgent(EchoLLM(), ToolSandbox(registry), tags=("search",), max_steps=4)
    assert agent._action_names() == ["search_docs", "final_answer"]
    run = agent.run("How are failed webhook deliveries retried?")
    assert {s.action for s in run.steps} <= {"search_docs"}


def test_a_failing_tool_produces_an_observation_instead_of_stopping_the_run():
    agent, _sandbox = build_agent(max_steps=5)
    run = agent.run("How many kilometres is 5 miles?")
    assert run.answer
    if run.failed_tool_calls:
        failed = [s for s in run.steps if not s.ok][0]
        assert failed.observation.startswith("ERROR ")


def test_observations_are_cited_back_into_the_prompt_as_evidence_blocks():
    agent, _sandbox = build_agent(max_steps=3)
    run = agent.run("What is the default rate limit per workspace?")
    context = agent._context(run.steps)
    for i, _step in enumerate(run.steps, start=1):
        assert f"[S{i}]" in context
