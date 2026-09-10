"""Scorer and statistics behaviour, including the edges that produce wrong numbers."""
import pytest

from llmkit import ScriptedLLM

from projects.p04_eval_harness.dataset import EvalCase, load_jsonl, save_jsonl, build_dataset
from projects.p04_eval_harness.scorers import (
    contains, exact_match, extract_json, json_schema, regex, token_f1, validate_schema,
)
from projects.p04_eval_harness.stats import bootstrap_ci, pearson_r, summarise


def case(**kw):
    base = dict(id="t1", input="q", expected="", tags=[], difficulty="medium",
                scorers=["contains"], metadata={})
    base.update(kw)
    return EvalCase(**base)


def test_contains_ignores_case_and_trailing_punctuation():
    c = case(expected="90 days")
    assert contains("Tokens expire after 90 DAYS.", c).passed
    assert not contains("Tokens expire after three months.", c).passed


def test_exact_match_normalises_but_does_not_paraphrase():
    c = case(expected="429")
    assert exact_match(" 429. ", c).passed
    assert not exact_match("HTTP 429", c).passed


def test_token_f1_cannot_be_inflated_by_padding_or_repetition():
    """Two ways to game an overlap score, both of which must fail."""
    c = case(expected="", metadata={"reference": "Request logs are retained for 30 days"})
    honest = token_f1("Request logs are retained for 30 days", c).score
    padded = token_f1(
        "Request logs are retained for 30 days and also here is a great deal of "
        "additional unrelated commentary about pagination cursors and webhooks", c).score
    repeated = token_f1("logs logs logs logs logs logs logs logs", c).score
    assert honest > padded, "padding must cost precision"
    assert repeated < 0.5, "repeating one matching token must not match it repeatedly"


def test_token_f1_reports_zero_on_no_overlap():
    c = case(metadata={"reference": "webhooks are signed with HMAC"})
    r = token_f1("cursors expire after 24 hours", c)
    assert r.score == 0.0 and not r.passed and r.detail


def test_regex_with_a_broken_pattern_fails_loudly():
    r = regex("anything", case(expected="([unclosed"))
    assert not r.passed and "invalid pattern" in r.detail


def test_schema_validator_catches_each_keyword():
    schema = {"type": "object", "required": ["scope", "limit"],
              "properties": {"scope": {"type": "string", "enum": ["workspace", "token"]},
                             "limit": {"type": "integer", "minimum": 1, "maximum": 100}}}
    assert validate_schema({"scope": "workspace", "limit": 50}, schema) == []
    assert validate_schema({"scope": "workspace"}, schema)          # missing required
    assert validate_schema({"scope": "team", "limit": 5}, schema)   # enum
    assert validate_schema({"scope": "token", "limit": 500}, schema)  # maximum
    assert validate_schema({"scope": "token", "limit": "5"}, schema)  # type
    assert validate_schema({"scope": "token", "limit": True}, schema), "bool is not an integer"


def test_json_scorer_survives_a_chatty_wrapper_but_not_broken_json():
    c = case(metadata={"schema": {"type": "object", "required": ["a"],
                                  "properties": {"a": {"type": "integer"}}}})
    assert json_schema('Sure! Here you go:\n```json\n{"a": 3}\n```', c).passed
    assert not json_schema("{'a': 3,}", c).passed
    assert extract_json("no json here") is None


def test_bootstrap_interval_brackets_the_mean_and_narrows_with_n():
    small = summarise([1.0] * 7 + [0.0] * 3)
    large = summarise([1.0] * 70 + [0.0] * 30)
    assert small["ci_low"] <= small["mean"] <= small["ci_high"]
    assert large["ci_width"] < small["ci_width"], "more evidence must mean a tighter interval"


def test_bootstrap_is_reproducible_and_degenerate_cases_are_safe():
    assert bootstrap_ci([1, 0, 1, 1]) == bootstrap_ci([1, 0, 1, 1])
    assert bootstrap_ci([]) == (0.0, 0.0)
    assert bootstrap_ci([0.5]) == (0.5, 0.5)
    assert bootstrap_ci([1.0] * 20) == (1.0, 1.0), "no variance means no interval"


def test_pearson_handles_a_constant_series():
    assert pearson_r([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert pearson_r([1, 1, 1], [1, 2, 3]) == 0.0


def test_dataset_round_trips_through_jsonl(tmp_path):
    original = build_dataset()
    path = tmp_path / "cases.jsonl"
    assert save_jsonl(original, str(path)) == len(original)
    reloaded = load_jsonl(str(path))
    assert [c.to_dict() for c in reloaded] == [c.to_dict() for c in original]


def test_malformed_case_raises_instead_of_being_skipped(tmp_path):
    """Silently dropping a case shrinks the eval set without shrinking confidence."""
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id": "ok", "input": "q"}\n{"id": "broken"\n', encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        load_jsonl(str(path))
    assert "bad.jsonl:2" in str(exc.value)


def test_scripted_provider_keeps_judge_tests_deterministic():
    llm = ScriptedLLM(['{"winner": "A", "reason": "r"}'])
    assert llm.complete([{"role": "user", "content": "x"}]).text.startswith("{")
