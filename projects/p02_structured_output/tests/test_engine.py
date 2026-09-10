"""The retry loop: error feedback, attempt bounding, and the fallback contract."""
from llmkit import EchoLLM, ScriptedLLM

from projects.p02_structured_output.engine import StructuredOutputEngine

SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string"}, "score": {"type": "number", "minimum": 0, "maximum": 1}},
    "required": ["label", "score"],
    "additionalProperties": False,
}


def test_a_clean_provider_needs_exactly_one_call():
    engine = StructuredOutputEngine(EchoLLM(), max_attempts=3)
    result = engine.generate("label this ticket", SCHEMA)
    assert result.ok and result.source == "clean" and result.n_attempts == 1


def test_a_repairable_response_costs_no_extra_model_call():
    engine = StructuredOutputEngine(ScriptedLLM(['```json\n{"label": "a", "score": 0.5}\n```']), max_attempts=3)
    result = engine.generate("label this", SCHEMA)
    assert result.ok and result.source == "repaired" and result.n_attempts == 1
    assert result.repairs_used == ["unfence"]


def test_validation_errors_are_fed_back_verbatim_into_the_next_prompt():
    llm = ScriptedLLM(['{"label": "a", "score": 4}', '{"label": "a", "score": 0.5}'])
    result = StructuredOutputEngine(llm, max_attempts=3).generate("label this", SCHEMA)
    assert result.ok and result.source == "reprompted" and result.n_attempts == 2
    second_prompt = llm.calls[1][-1].content
    assert "$.score: must be <= 1, got 4" in second_prompt


def test_a_parse_failure_is_reported_to_the_model_as_a_root_level_error():
    llm = ScriptedLLM(["I cannot do that.", '{"label": "a", "score": 0.5}'])
    result = StructuredOutputEngine(llm, max_attempts=2).generate("label this", SCHEMA)
    assert result.ok
    assert "$: response could not be parsed as JSON" in llm.calls[1][-1].content


def test_the_attempt_budget_is_bounded_and_the_caller_gets_a_fallback():
    llm = ScriptedLLM(["still not json"])
    fallback = {"label": "unknown", "score": 0.0}
    result = StructuredOutputEngine(llm, max_attempts=4).generate("label this", SCHEMA, fallback=fallback)
    assert not result.ok
    assert result.source == "fallback" and result.n_attempts == 4
    assert len(llm.calls) == 4
    assert result.value == fallback


def test_the_fallback_is_copied_so_callers_cannot_poison_the_default():
    fallback = {"label": "unknown", "score": 0.0, "tags": []}
    engine = StructuredOutputEngine(ScriptedLLM(["nope"]), max_attempts=1)
    first = engine.generate("a", SCHEMA, fallback=fallback)
    first.value["tags"].append("mutated")
    second = engine.generate("b", SCHEMA, fallback=fallback)
    assert second.value["tags"] == []


def test_generate_never_raises_even_when_the_provider_is_fully_broken():
    engine = StructuredOutputEngine(EchoLLM(fault_rate=1.0), max_attempts=2)
    results = [engine.generate(f"ticket {i}", SCHEMA, fallback={"label": "x", "score": 0.0}) for i in range(15)]
    assert all(r.value is not None for r in results)
    assert all(r.ok or r.source == "fallback" for r in results)


def test_higher_fault_rates_need_strictly_more_model_calls():
    def calls(fault_rate):
        engine = StructuredOutputEngine(EchoLLM(fault_rate=fault_rate), max_attempts=3)
        return sum(engine.generate(f"ticket {i}", SCHEMA, fallback={"label": "x", "score": 0.0}).n_attempts
                   for i in range(30))

    assert calls(0.0) == 30
    assert calls(1.0) > calls(0.0)


def test_attempt_records_carry_the_token_accounting():
    engine = StructuredOutputEngine(EchoLLM(), max_attempts=2)
    result = engine.generate("label this ticket", SCHEMA)
    assert result.total_tokens > 0
    assert result.attempts[0].prompt_tokens > 0 and result.attempts[0].completion_tokens > 0
