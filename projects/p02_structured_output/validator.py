"""A JSON Schema subset validator, written from scratch.

Why hand-written rather than `jsonschema` or `pydantic`:

* Both are third-party, and one of them (pydantic v2) is a compiled Rust
  extension. This repo has to run on a stock interpreter with no wheels to
  build, so a native dependency is not an option here.
* More importantly, the interesting requirement is not "is this valid" but
  "explain, machine-readably, exactly what is wrong so the next prompt can say
  it". `jsonschema` can produce that, but the moment you want error paths in a
  shape a model reliably understands you end up writing the projection layer
  anyway. Owning the error type is the point of this module.

The subset covers the keywords that actually decide whether a model response is
usable: `type`, `required`, `enum`, `minimum`/`maximum`, `minLength`/
`maxLength`, `pattern`, `properties` (nested), `array` with `items`/`minItems`/
`maxItems`, and `additionalProperties`. Deliberately not covered: `$ref`,
`allOf`/`anyOf`/`oneOf`, `format`, `patternProperties`. Those are rejected
because a schema you hand to a model should be flat enough for the model to
follow; if you need `$ref` indirection, the prompt is already too complicated.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List

__all__ = ["SchemaError", "validate", "is_valid", "describe_errors", "json_type_of"]


@dataclass(frozen=True)
class SchemaError:
    """One structured validation failure.

    Structured rather than a string because the retry loop feeds these straight
    back into the next prompt, and a model corrects "$.items[1].score: maximum,
    got 1.4, expected <= 1" far more reliably than "validation failed".
    """

    path: str
    rule: str
    got: Any
    expected: Any
    message: str

    def as_feedback(self) -> str:
        """One line, model-facing. Kept short: long error text crowds out the task."""
        return f"{self.path}: {self.message}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "rule": self.rule,
            "got": self.got,
            "expected": self.expected,
            "message": self.message,
        }


def json_type_of(value: Any) -> str:
    """The JSON type name for a Python value.

    `bool` is checked before `int` on purpose: in Python `True` is an `int`, in
    JSON it is emphatically not, and a validator that lets `true` satisfy
    `{"type": "integer"}` will pass output that breaks the caller downstream.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _matches_type(value: Any, expected: str) -> bool:
    actual = json_type_of(value)
    if expected == "number":
        return actual in ("number", "integer")
    if expected == "integer":
        # 3.0 is an acceptable integer: JSON has one numeric type and models
        # emit `3.0` for integer fields constantly. 3.5 is not.
        return actual == "integer" or (actual == "number" and float(value).is_integer())
    return actual == expected


def _err(path: str, rule: str, got: Any, expected: Any, message: str) -> SchemaError:
    return SchemaError(path=path, rule=rule, got=got, expected=expected, message=message)


def _check_type(value: Any, schema: Dict[str, Any], path: str) -> List[SchemaError]:
    declared = schema.get("type")
    if declared is None:
        return []
    options = declared if isinstance(declared, list) else [declared]
    if any(_matches_type(value, opt) for opt in options):
        return []
    expected = " or ".join(options)
    return [_err(path, "type", json_type_of(value), expected,
                 f"expected type {expected}, got {json_type_of(value)}")]


def _check_scalar(value: Any, schema: Dict[str, Any], path: str) -> List[SchemaError]:
    errors: List[SchemaError] = []
    if "enum" in schema and value not in schema["enum"]:
        errors.append(_err(path, "enum", value, schema["enum"],
                           f"value must be one of {schema['enum']}, got {value!r}"))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(_err(path, "minimum", value, schema["minimum"],
                               f"must be >= {schema['minimum']}, got {value}"))
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(_err(path, "maximum", value, schema["maximum"],
                               f"must be <= {schema['maximum']}, got {value}"))
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(_err(path, "minLength", len(value), schema["minLength"],
                               f"must be at least {schema['minLength']} characters, got {len(value)}"))
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(_err(path, "maxLength", len(value), schema["maxLength"],
                               f"must be at most {schema['maxLength']} characters, got {len(value)}"))
        pattern = schema.get("pattern")
        if pattern is not None and re.search(pattern, value) is None:
            # Unanchored search, matching JSON Schema semantics. Anchor it in
            # the pattern itself if you mean a full match.
            errors.append(_err(path, "pattern", value, pattern,
                               f"must match pattern {pattern}"))
    return errors


def _check_object(value: Dict[str, Any], schema: Dict[str, Any], path: str) -> List[SchemaError]:
    errors: List[SchemaError] = []
    properties: Dict[str, Any] = schema.get("properties") or {}
    for key in schema.get("required") or []:
        if key not in value:
            errors.append(_err(f"{path}.{key}", "required", None, key,
                               f"required property {key!r} is missing"))
    for key, sub_schema in properties.items():
        if key in value:
            errors.extend(validate(value[key], sub_schema, f"{path}.{key}"))
    extra = [k for k in value if k not in properties]
    additional = schema.get("additionalProperties", True)
    if additional is False:
        for key in extra:
            # Worth catching rather than ignoring: an invented field is usually
            # a sign the model misread the schema, and silently dropping it
            # hides that from the retry prompt.
            errors.append(_err(f"{path}.{key}", "additionalProperties", key, False,
                               f"unexpected property {key!r} is not allowed"))
    elif isinstance(additional, dict):
        for key in extra:
            errors.extend(validate(value[key], additional, f"{path}.{key}"))
    return errors


def _check_array(value: List[Any], schema: Dict[str, Any], path: str) -> List[SchemaError]:
    errors: List[SchemaError] = []
    if "minItems" in schema and len(value) < schema["minItems"]:
        errors.append(_err(path, "minItems", len(value), schema["minItems"],
                           f"must have at least {schema['minItems']} items, got {len(value)}"))
    if "maxItems" in schema and len(value) > schema["maxItems"]:
        errors.append(_err(path, "maxItems", len(value), schema["maxItems"],
                           f"must have at most {schema['maxItems']} items, got {len(value)}"))
    item_schema = schema.get("items")
    if isinstance(item_schema, dict):
        for i, item in enumerate(value):
            errors.extend(validate(item, item_schema, f"{path}[{i}]"))
    return errors


def validate(instance: Any, schema: Dict[str, Any], path: str = "$") -> List[SchemaError]:
    """Validate `instance` against `schema`, returning every failure found.

    Collects all errors instead of stopping at the first one: a retry that fixes
    one field at a time burns one model call per field, which is the expensive
    resource here. The one exception is a type mismatch, which short-circuits
    that node, because "expected object, got string" plus twelve cascading
    "missing required property" errors is noise the model has to read past.
    """
    type_errors = _check_type(instance, schema, path)
    if type_errors:
        return type_errors

    errors = _check_scalar(instance, schema, path)
    if isinstance(instance, dict):
        errors.extend(_check_object(instance, schema, path))
    elif isinstance(instance, list):
        errors.extend(_check_array(instance, schema, path))
    return errors


def is_valid(instance: Any, schema: Dict[str, Any]) -> bool:
    return not validate(instance, schema)


def describe_errors(errors: List[SchemaError], limit: int = 8) -> str:
    """Render errors as the block that goes back to the model.

    Capped at `limit` lines: a badly broken response can produce dozens of
    errors, and a wall of them pushes the original instructions out of the
    model's attention. The first few are the ones worth fixing.
    """
    if not errors:
        return ""
    lines = [f"- {e.as_feedback()}" for e in errors[:limit]]
    if len(errors) > limit:
        lines.append(f"- ... and {len(errors) - limit} more problems")
    return "\n".join(lines)
