"""Validator behaviour, including the edge cases that decide whether output is usable."""
from dataclasses import dataclass, field
from typing import List, Literal, Optional

import pytest

from projects.p02_structured_output.schema_builder import schema_from_dataclass
from projects.p02_structured_output.validator import describe_errors, is_valid, validate

SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "minLength": 2, "maxLength": 8},
        "grade": {"type": "string", "enum": ["a", "b"]},
        "score": {"type": "number", "minimum": 0, "maximum": 1},
        "count": {"type": "integer"},
        "ref": {"type": "string", "pattern": r"^R-\d+$"},
        "tags": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3},
        "meta": {"type": "object", "properties": {"owner": {"type": "string"}}, "required": ["owner"]},
    },
    "required": ["name", "grade", "score"],
    "additionalProperties": False,
}


def rules_for(instance, schema=SCHEMA):
    return {(e.path, e.rule) for e in validate(instance, schema)}


def test_valid_instance_produces_no_errors():
    assert is_valid({"name": "arnob", "grade": "a", "score": 0.5}, SCHEMA)


def test_every_rule_reports_its_own_structured_error():
    bad = {
        "name": "x",
        "grade": "z",
        "score": 1.4,
        "ref": "R2",
        "tags": [],
        "meta": {},
        "surprise": 1,
    }
    assert rules_for(bad) == {
        ("$.name", "minLength"),
        ("$.grade", "enum"),
        ("$.score", "maximum"),
        ("$.ref", "pattern"),
        ("$.tags", "minItems"),
        ("$.meta.owner", "required"),
        ("$.surprise", "additionalProperties"),
    }


def test_booleans_do_not_satisfy_integer_because_json_is_not_python():
    assert rules_for({"name": "ok", "grade": "a", "score": 0.5, "count": True}) == {("$.count", "type")}


def test_integer_accepts_a_whole_float_but_not_a_fractional_one():
    assert is_valid({"name": "ok", "grade": "a", "score": 0.5, "count": 3.0}, SCHEMA)
    assert rules_for({"name": "ok", "grade": "a", "score": 0.5, "count": 3.5}) == {("$.count", "type")}


def test_a_type_mismatch_short_circuits_instead_of_cascading():
    """A string where an object belongs must not also report every missing key."""
    errors = validate("not an object", SCHEMA)
    assert len(errors) == 1
    assert errors[0].rule == "type" and errors[0].expected == "object"


def test_array_item_errors_carry_their_index_in_the_path():
    schema = {"type": "array", "items": {"type": "object", "properties": {"n": {"type": "integer"}},
                                         "required": ["n"]}}
    paths = {e.path for e in validate([{"n": 1}, {"n": "two"}, {}], schema)}
    assert paths == {"$[1].n", "$[2].n"}


def test_nullable_fields_accept_null_and_reject_wrong_types():
    schema = schema_from_dataclass(_Nullable)
    assert is_valid({"owner": None}, schema)
    assert is_valid({"owner": "arnob"}, schema)
    assert rules_for({"owner": 7}, schema) == {("$.owner", "type")}


def test_feedback_is_capped_so_it_cannot_crowd_out_the_prompt():
    many = {"name": "x", "grade": "z", "score": 9, "ref": "bad", "tags": [], "meta": {}, "a": 1, "b": 2, "c": 3}
    text = describe_errors(validate(many, SCHEMA), limit=3)
    assert text.count("\n") == 3
    assert "and" in text.splitlines()[-1] and "more problems" in text.splitlines()[-1]


def test_additional_properties_as_a_schema_validates_the_extra_values():
    schema = {"type": "object", "properties": {}, "additionalProperties": {"type": "integer"}}
    assert is_valid({"anything": 3}, schema)
    assert {e.rule for e in validate({"anything": "no"}, schema)} == {"type"}


@dataclass
class _Nullable:
    owner: Optional[str] = None


@dataclass
class _Nested:
    label: Literal["x", "y"]
    items: List[_Nullable] = field(default_factory=list, metadata={"minItems": 1})


def test_schema_builder_round_trips_through_the_validator():
    schema = schema_from_dataclass(_Nested)
    assert schema["required"] == ["label"]
    assert schema["properties"]["label"]["enum"] == ["x", "y"]
    assert schema["properties"]["items"]["minItems"] == 1
    assert is_valid({"label": "x", "items": [{"owner": "a"}]}, schema)
    assert not is_valid({"label": "z", "items": [{"owner": "a"}]}, schema)


def test_schema_builder_rejects_a_non_dataclass():
    with pytest.raises(TypeError):
        schema_from_dataclass(dict)
