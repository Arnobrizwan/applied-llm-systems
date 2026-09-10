"""The @tool decorator: a JSON schema derived from type hints and the docstring.

Why derive rather than declare: a hand-written tool schema is a second copy of
the function signature, and the two diverge the first time someone adds a
parameter. The divergence is silent, because the model happily sends the old
argument set and the function raises a TypeError inside the agent loop, where it
looks like a model failure rather than a stale schema.

Why the docstring rather than a `description=` keyword per parameter: the
docstring already exists, is already the thing a human reads, and is checked by
every reviewer. A separate description string is one more copy to let rot.
Google-style `Args:` sections are parsed because that is what most Python
codebases already write.

Argument handling is validation *and coercion*, and both return structured
errors rather than raising. A model that sends `{"k": "3"}` instead of
`{"k": 3}` has made a formatting mistake, not a reasoning mistake, and coercing
it is cheaper and more reliable than a correction round trip. A model that sends
`{"k": "three"}` gets an error object it can act on, in the same shape every
time.
"""
from __future__ import annotations

import inspect
import re
import typing
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

__all__ = ["ToolSpec", "ArgError", "tool", "coerce_arguments", "schema_from_signature"]

_PRIMITIVES = {
    str: "string",
    bool: "boolean",
    int: "integer",
    float: "number",
    type(None): "null",
}


@dataclass(frozen=True)
class ArgError:
    """A structured, model-facing argument problem."""

    path: str
    rule: str
    got: Any
    expected: Any
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {"path": self.path, "rule": self.rule, "got": self.got,
                "expected": self.expected, "message": self.message}


@dataclass
class ToolSpec:
    """Everything the registry, the sandbox and the prompt need about one tool."""

    name: str
    version: str
    description: str
    parameters: Dict[str, Any]
    func: Callable[..., Any]
    tags: Tuple[str, ...] = ()
    timeout_s: float = 1.0

    @property
    def qualified_name(self) -> str:
        return f"{self.name}@{self.version}"

    def to_prompt_schema(self) -> Dict[str, Any]:
        """The dict rendered into the prompt. Deliberately not the whole spec:
        version, timeout and tags are operational concerns the model cannot act
        on, and every token spent on them is a token not spent on the task."""
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


# -- docstring parsing -----------------------------------------------------

_ARGS_HEADER = re.compile(r"^\s*(Args|Arguments|Parameters)\s*:\s*$", re.I)
_SECTION_HEADER = re.compile(r"^\s*(Returns|Raises|Yields|Examples?|Note)s?\s*:\s*$", re.I)
_ARG_LINE = re.compile(r"^\s{2,}(\*{0,2}\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$")


def parse_docstring(doc: Optional[str]) -> Tuple[str, Dict[str, str]]:
    """Split a docstring into (summary, {param: description}).

    The summary is the first paragraph, not the first line: a one-line summary
    with a wrapped continuation is common and truncating it mid-sentence gives
    the model a description that stops halfway.
    """
    if not doc:
        return "", {}
    lines = inspect.cleandoc(doc).splitlines()
    summary_lines: List[str] = []
    i = 0
    while i < len(lines) and lines[i].strip() and not _ARGS_HEADER.match(lines[i]):
        summary_lines.append(lines[i].strip())
        i += 1
    summary = " ".join(summary_lines).strip()

    params: Dict[str, str] = {}
    in_args = False
    current: Optional[str] = None
    for line in lines[i:]:
        if _ARGS_HEADER.match(line):
            in_args, current = True, None
            continue
        if _SECTION_HEADER.match(line):
            in_args, current = False, None
            continue
        if not in_args:
            continue
        match = _ARG_LINE.match(line)
        if match:
            current = match.group(1).lstrip("*")
            params[current] = match.group(2).strip()
        elif current and line.strip():
            params[current] = (params[current] + " " + line.strip()).strip()
    return summary, params


# -- type hints to schema --------------------------------------------------

def _json_type(annotation: Any) -> Dict[str, Any]:
    if annotation in _PRIMITIVES:
        return {"type": _PRIMITIVES[annotation]}
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin is typing.Literal:
        return {"type": _PRIMITIVES.get(type(args[0]), "string"), "enum": list(args)}
    if origin is typing.Union or str(origin) == "<class 'types.UnionType'>":
        members = [a for a in args if a is not type(None)]
        base = _json_type(members[0]) if members else {"type": "string"}
        if len(args) != len(members):
            declared = base.get("type", "string")
            base = dict(base)
            base["type"] = [declared, "null"] if isinstance(declared, str) else list(declared) + ["null"]
        return base
    if origin in (list, tuple, set):
        return {"type": "array", "items": _json_type(args[0]) if args else {"type": "string"}}
    if origin is dict:
        return {"type": "object"}
    # Unannotated or exotic parameters become strings. A tool with an exotic
    # parameter is a tool the model will struggle with anyway; the honest move
    # is to make that visible in the rendered schema rather than to fail here.
    return {"type": "string"}


def schema_from_signature(func: Callable[..., Any], constraints: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Build the `parameters` object schema for a callable."""
    signature = inspect.signature(func)
    hints = typing.get_type_hints(func)
    _summary, docs = parse_docstring(func.__doc__)
    constraints = constraints or {}

    properties: Dict[str, Any] = {}
    required: List[str] = []
    for name, parameter in signature.parameters.items():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue  # *args/**kwargs cannot be described to a model usefully
        prop = _json_type(hints.get(name, str))
        if name in docs:
            prop["description"] = docs[name]
        prop.update(constraints.get(name, {}))
        if parameter.default is not inspect.Parameter.empty:
            prop["default"] = parameter.default
        else:
            required.append(name)
        properties[name] = prop

    schema: Dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    schema["additionalProperties"] = False
    return schema


def tool(
    _func: Optional[Callable[..., Any]] = None,
    *,
    name: Optional[str] = None,
    version: str = "1.0.0",
    tags: Tuple[str, ...] = (),
    timeout_s: float = 1.0,
    constraints: Optional[Dict[str, Dict[str, Any]]] = None,
):
    """Attach a generated `ToolSpec` to a function without changing it.

    The function stays directly callable and untouched, which matters: a tool
    should be unit-testable as a plain function, and wrapping it would mean
    every test of the underlying logic goes through the agent machinery.

    `constraints` carries the JSON Schema keywords that have no home in a type
    hint (minimum, maxLength). They live at the decorator rather than in the
    body so the schema shown to the model and the check applied before execution
    are the same declaration.
    """

    def decorate(func: Callable[..., Any]) -> Callable[..., Any]:
        summary, _docs = parse_docstring(func.__doc__)
        spec = ToolSpec(
            name=name or func.__name__,
            version=version,
            description=summary or f"Call {func.__name__}.",
            parameters=schema_from_signature(func, constraints),
            func=func,
            tags=tuple(tags),
            timeout_s=timeout_s,
        )
        func.tool_spec = spec  # type: ignore[attr-defined]
        return func

    return decorate(_func) if _func is not None else decorate


# -- validation and coercion ----------------------------------------------

def _coerce_scalar(value: Any, expected: str) -> Tuple[Any, bool]:
    """Return (value, ok). Coercions are narrow and lossless on purpose."""
    if expected == "string":
        return (value, True) if isinstance(value, str) else (str(value), isinstance(value, (int, float, bool)))
    if expected == "boolean":
        if isinstance(value, bool):
            return value, True
        if isinstance(value, str) and value.strip().lower() in ("true", "false"):
            return value.strip().lower() == "true", True
        return value, False
    if expected in ("integer", "number"):
        if isinstance(value, bool):
            return value, False  # True is an int in Python and never a number in JSON
        if isinstance(value, (int, float)):
            if expected == "integer":
                return (int(value), True) if float(value).is_integer() else (value, False)
            return float(value), True
        if isinstance(value, str):
            try:
                parsed = float(value.strip())
            except ValueError:
                return value, False
            if expected == "integer":
                return (int(parsed), True) if parsed.is_integer() else (value, False)
            return parsed, True
    return value, True


def coerce_arguments(schema: Dict[str, Any], args: Dict[str, Any]) -> Tuple[Dict[str, Any], List[ArgError]]:
    """Validate and coerce a model-supplied argument dict.

    Returns the coerced arguments and a list of errors. Nothing raises: these
    errors are written back into the conversation for the model to fix, and an
    exception at this point would mean the agent loop has to catch and translate
    it anyway.
    """
    errors: List[ArgError] = []
    properties: Dict[str, Any] = schema.get("properties") or {}
    required = schema.get("required") or []
    coerced: Dict[str, Any] = {}

    if not isinstance(args, dict):
        return {}, [ArgError("$", "type", type(args).__name__, "object",
                             "arguments must be a JSON object")]

    for key in args:
        if key not in properties and schema.get("additionalProperties") is False:
            errors.append(ArgError(f"$.{key}", "unknown_argument", key, sorted(properties),
                                   f"unknown argument {key!r}; expected one of {sorted(properties)}"))

    for key in required:
        if key not in args:
            errors.append(ArgError(f"$.{key}", "required", None, key,
                                   f"required argument {key!r} is missing"))

    for key, prop in properties.items():
        if key not in args:
            if "default" in prop:
                coerced[key] = prop["default"]
            continue
        value = args[key]
        expected = prop.get("type", "string")
        expected_types = expected if isinstance(expected, list) else [expected]

        if value is None and "null" in expected_types:
            coerced[key] = None
            continue

        target = next((t for t in expected_types if t != "null"), "string")
        if target == "array":
            if not isinstance(value, list):
                errors.append(ArgError(f"$.{key}", "type", type(value).__name__, "array",
                                       f"expected an array, got {type(value).__name__}"))
                continue
            coerced[key] = value
            continue

        value, ok = _coerce_scalar(value, target)
        if not ok:
            errors.append(ArgError(f"$.{key}", "type", args[key], target,
                                   f"expected {target}, got {args[key]!r}"))
            continue

        if "enum" in prop and value not in prop["enum"]:
            errors.append(ArgError(f"$.{key}", "enum", value, prop["enum"],
                                   f"must be one of {prop['enum']}, got {value!r}"))
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if "minimum" in prop and value < prop["minimum"]:
                errors.append(ArgError(f"$.{key}", "minimum", value, prop["minimum"],
                                       f"must be >= {prop['minimum']}, got {value}"))
                continue
            if "maximum" in prop and value > prop["maximum"]:
                errors.append(ArgError(f"$.{key}", "maximum", value, prop["maximum"],
                                       f"must be <= {prop['maximum']}, got {value}"))
                continue
        if isinstance(value, str):
            if "minLength" in prop and len(value) < prop["minLength"]:
                errors.append(ArgError(f"$.{key}", "minLength", len(value), prop["minLength"],
                                       f"must be at least {prop['minLength']} characters"))
                continue
            if "maxLength" in prop and len(value) > prop["maxLength"]:
                errors.append(ArgError(f"$.{key}", "maxLength", len(value), prop["maxLength"],
                                       f"must be at most {prop['maxLength']} characters"))
                continue
        coerced[key] = value

    return coerced, errors
