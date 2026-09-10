"""Schema derivation, argument coercion and registry behaviour."""
from typing import List, Literal, Optional

import pytest

from projects.p14_tool_calling.registry import ToolNotFound, ToolRegistry
from projects.p14_tool_calling.schema import coerce_arguments, parse_docstring, tool


@tool(tags=("demo",), constraints={"count": {"minimum": 1, "maximum": 9}})
def sample(query: str, mode: Literal["fast", "deep"], count: int = 3,
           tags: Optional[List[str]] = None) -> str:
    """Do a sample thing.

    Args:
        query: The thing to look for,
            wrapped across two lines.
        mode: How hard to look.
        count: How many results.

    Returns:
        Something.
    """
    return f"{query}:{mode}:{count}"


PARAMS = sample.tool_spec.parameters


def test_schema_comes_from_the_signature_not_a_second_declaration():
    assert PARAMS["required"] == ["query", "mode"]
    assert PARAMS["properties"]["count"]["default"] == 3
    assert PARAMS["properties"]["mode"]["enum"] == ["fast", "deep"]
    assert PARAMS["properties"]["count"]["minimum"] == 1
    assert PARAMS["additionalProperties"] is False


def test_optional_list_becomes_a_nullable_array_and_is_not_required():
    assert PARAMS["properties"]["tags"]["type"] == ["array", "null"]
    assert "tags" not in PARAMS["required"]


def test_docstring_supplies_the_descriptions_including_wrapped_lines():
    summary, params = parse_docstring(sample.__doc__)
    assert summary == "Do a sample thing."
    assert params["query"] == "The thing to look for, wrapped across two lines."
    assert "Returns" not in params and "Something." not in params.values()


def test_the_decorated_function_is_still_a_plain_callable():
    assert sample("a", "fast") == "a:fast:3"


def test_numeric_strings_are_coerced_because_that_is_a_formatting_slip():
    coerced, errors = coerce_arguments(PARAMS, {"query": "x", "mode": "fast", "count": "7"})
    assert errors == [] and coerced["count"] == 7 and isinstance(coerced["count"], int)


def test_defaults_are_applied_when_the_model_omits_an_optional_argument():
    coerced, errors = coerce_arguments(PARAMS, {"query": "x", "mode": "deep"})
    assert errors == [] and coerced["count"] == 3


def test_booleans_are_not_integers_even_though_python_says_they_are():
    _coerced, errors = coerce_arguments(PARAMS, {"query": "x", "mode": "fast", "count": True})
    assert [e.rule for e in errors] == ["type"]


def test_every_argument_problem_is_reported_at_once_and_nothing_raises():
    _coerced, errors = coerce_arguments(PARAMS, {"mode": "sideways", "count": 99, "extra": 1})
    assert {(e.path, e.rule) for e in errors} == {
        ("$.extra", "unknown_argument"),
        ("$.query", "required"),
        ("$.mode", "enum"),
        ("$.count", "maximum"),
    }


def test_registry_resolves_the_newest_version_unless_pinned():
    registry = ToolRegistry()
    registry.register(sample)

    @tool(name="sample", version="2.1.0")
    def sample_v2(query: str) -> str:
        """Newer sample."""
        return query

    registry.register(sample_v2)
    assert registry.get("sample").version == "2.1.0"
    assert registry.get("sample", "1.0.0").version == "1.0.0"
    assert len(registry) == 2
    assert [s.qualified_name for s in registry.list()] == ["sample@2.1.0"]
    assert len(registry.list(all_versions=True)) == 2


def test_version_ordering_is_numeric_not_lexicographic():
    registry = ToolRegistry()

    @tool(name="t", version="1.9.0")
    def t_a() -> str:
        """A."""
        return "a"

    @tool(name="t", version="1.10.0")
    def t_b() -> str:
        """B."""
        return "b"

    registry.register_all([t_a, t_b])
    assert registry.get("t").version == "1.10.0"


def test_duplicate_registration_is_refused_and_missing_lookups_report_clearly():
    registry = ToolRegistry()
    registry.register(sample)
    with pytest.raises(ValueError):
        registry.register(sample)
    with pytest.raises(ToolNotFound):
        registry.get("nope")
    assert registry.find("nope") is None


def test_tag_filtering_and_stable_ordering():
    registry = ToolRegistry()
    registry.register(sample)

    @tool(tags=("other",))
    def zebra() -> str:
        """Z."""
        return "z"

    @tool(tags=("demo",))
    def alpha() -> str:
        """A."""
        return "a"

    registry.register_all([zebra, alpha])
    assert registry.names() == ["alpha", "sample", "zebra"]
    assert registry.names(tags=("demo",)) == ["alpha", "sample"]
    assert registry.tags() == ["demo", "other"]
    assert registry.render_prompt() == registry.render_prompt()


def test_registering_an_undecorated_function_is_a_type_error():
    with pytest.raises(TypeError):
        ToolRegistry().register(lambda: None)
