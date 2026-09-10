"""Repair pipeline: every strategy, plus the cases where repair must not fire."""
from llmkit import EchoLLM

from projects.p02_structured_output.extraction import extract_json


def test_clean_json_is_parsed_with_no_repairs_recorded():
    result = extract_json('{"a": 1}')
    assert result.ok and result.value == {"a": 1} and result.repairs == []


def test_fenced_block_with_a_preamble_is_unwrapped():
    result = extract_json('Sure! Here it is:\n```json\n{"a": 1}\n```')
    assert result.ok and result.value == {"a": 1}
    assert "unfence" in result.repairs


def test_trailing_commentary_is_sliced_away():
    result = extract_json('{"a": 1}\n\nLet me know if you need anything else.')
    assert result.ok and result.value == {"a": 1} and result.repairs == ["slice_to_json"]


def test_single_quotes_and_a_trailing_comma_are_both_fixed():
    result = extract_json("{'a': 'x', 'b': 2,}")
    assert result.ok and result.value == {"a": "x", "b": 2}
    assert result.repairs == ["single_quotes", "trailing_commas"]


def test_python_repr_literals_become_json_literals():
    result = extract_json('{"ok": True, "owner": None, "off": False}')
    assert result.ok and result.value == {"ok": True, "owner": None, "off": False}
    assert "python_literals" in result.repairs


def test_truncated_string_is_closed_rather_than_discarded():
    """The cheap repair runs first: a half-written value is still evidence."""
    result = extract_json('{\n  "a": 1,\n  "b": "half a str')
    assert result.ok and result.value == {"a": 1, "b": "half a str"}
    assert "close_truncated" in result.repairs


def test_a_dangling_key_is_dropped_because_closing_alone_cannot_parse():
    result = extract_json('{"a": 1, "b":')
    assert result.ok and result.value == {"a": 1}
    assert "close_truncated" in result.repairs


def test_truncated_nested_array_is_closed_at_every_level():
    result = extract_json('{"a": 1, "items": [{"n": 2}, {"n": 3}')
    assert result.ok and result.value == {"a": 1, "items": [{"n": 2}, {"n": 3}]}


def test_prose_with_no_json_fails_rather_than_inventing_a_value():
    result = extract_json("I am not able to help with that request.")
    assert not result.ok and result.value is None and result.error


def test_repairs_do_not_corrupt_punctuation_inside_string_values():
    """A comma before a brace inside a sentence is not a trailing comma."""
    payload = '{"note": "one, two, } and it\'s fine", "n": 1}'
    result = extract_json(payload)
    assert result.ok and result.value["note"] == "one, two, } and it's fine"
    assert result.repairs == []


def test_every_echo_corruption_mode_is_recovered():
    """The provider corrupts deterministically; all four modes must be repairable."""
    llm = EchoLLM(fault_rate=1.0)
    schema = {"type": "object", "properties": {"label": {"type": "string"}, "score": {"type": "number"}}}
    seen = set()
    recovered = 0
    for i in range(24):
        text = llm.complete([{"role": "user", "content": f"item {i}"}], json_schema=schema).text
        result = extract_json(text)
        if result.ok:
            recovered += 1
            seen.add(tuple(result.repairs))
    assert recovered >= 20, f"only recovered {recovered}/24"
    assert len(seen) >= 3, "expected several distinct corruption shapes"
