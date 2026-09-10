"""Instrumentation, privacy capture, caching and metric aggregation."""
import json

import pytest

from llmkit import EchoLLM, FailingLLM, LLMError, Tracer

from projects.p13_observability.client import (
    InstrumentedLLM, PrivacyPolicy, ResponseCache, observed_request,
)
from projects.p13_observability.dashboard import (
    export_spans_jsonl, load_spans_jsonl, render_dashboard, render_waterfall,
)
from projects.p13_observability.metrics import MetricsView
from projects.p13_observability.pipeline import ObservedPipeline, QUESTIONS

PROMPT = [{"role": "user", "content": "how long is a token valid"}]


def test_privacy_defaults_to_hashing_and_never_writes_prompt_text():
    tracer = Tracer()
    InstrumentedLLM(EchoLLM(), tracer=tracer).complete(PROMPT)
    attrs = tracer.spans[0].attributes
    assert attrs["privacy.mode"] == "hash"
    assert len(attrs["prompt.sha256"]) == 16
    assert attrs["prompt.chars"] == len(PROMPT[0]["content"])
    assert "prompt.text" not in attrs and "prompt.preview" not in attrs


def test_full_capture_is_opt_in_and_records_that_it_happened():
    tracer = Tracer()
    InstrumentedLLM(EchoLLM(), tracer=tracer,
                    privacy=PrivacyPolicy(mode="full")).complete(PROMPT)
    attrs = tracer.spans[0].attributes
    assert attrs["prompt.text"] == PROMPT[0]["content"]
    assert attrs["privacy.mode"] == "full", "an audit must be able to find these spans"


def test_salt_changes_the_fingerprint_so_short_prompts_are_not_guessable():
    plain = PrivacyPolicy().capture("hello")["prompt.sha256"]
    salted = PrivacyPolicy(salt="prod-2026").capture("hello")["prompt.sha256"]
    assert plain != salted
    assert PrivacyPolicy(salt="prod-2026").capture("hello")["prompt.sha256"] == salted
    with pytest.raises(ValueError):
        PrivacyPolicy(mode="everything")


def test_a_call_records_tokens_cost_and_tier_on_the_span():
    tracer = Tracer()
    client = InstrumentedLLM(EchoLLM(), tier="large", tracer=tracer)
    resp = client.complete(PROMPT)
    attrs = tracer.spans[0].attributes
    assert attrs["llm.tier"] == "large"
    assert attrs["llm.total_tokens"] == resp.total_tokens > 0
    assert attrs["llm.cost_usd"] > 0, "the large tier is not free"
    assert attrs["llm.latency_ms"] >= 0


def test_a_failing_upstream_is_recorded_and_re_raised_unchanged():
    """Instrumentation that swallows an exception is instrumentation people turn off."""
    tracer = Tracer()
    client = InstrumentedLLM(FailingLLM(fail_times=99, status=503), tracer=tracer)
    with pytest.raises(LLMError):
        client.complete(PROMPT)
    span = tracer.spans[0]
    assert span.status == "error"
    assert span.attributes["error.type"] == "LLMError"
    assert span.attributes["error.status"] == 503
    assert span.attributes["error.retryable"] is True


def test_cache_hits_get_a_span_a_zero_cost_and_a_hit_rate():
    tracer, cache = Tracer(), ResponseCache()
    client = InstrumentedLLM(EchoLLM(), tier="large", tracer=tracer, cache=cache)
    first = client.complete(PROMPT)
    second = client.complete(PROMPT)
    assert second.cached and not first.cached
    assert second.text == first.text
    assert tracer.spans[1].attributes["llm.cache"] == "hit"
    assert tracer.spans[1].attributes["llm.cost_usd"] == 0.0
    assert MetricsView(tracer.spans).cache_hit_rate == pytest.approx(0.5)


def test_a_cache_hit_cannot_be_mutated_into_the_stored_response():
    cache = ResponseCache()
    client = InstrumentedLLM(EchoLLM(), tracer=Tracer(), cache=cache)
    client.complete(PROMPT)
    hit = client.complete(PROMPT)
    hit.text = "corrupted"
    assert client.complete(PROMPT).text != "corrupted"


def test_request_id_and_trace_id_link_a_multi_step_pipeline():
    tracer = Tracer()
    client = InstrumentedLLM(EchoLLM(), tracer=tracer)
    with observed_request(tracer, "request", request_id="req-1") as root:
        client.complete(PROMPT, step="rerank")
        client.complete(PROMPT, step="generate")
    children = [s for s in tracer.spans if s.parent_id == root.span_id]
    assert len(children) == 2
    assert {s.attributes["request.id"] for s in tracer.spans} == {"req-1"}
    assert {s.trace_id for s in tracer.spans} == {root.trace_id}


def test_error_rate_counts_requests_not_spans():
    """Span level error rate is diluted by healthy siblings in the same request."""
    pipeline = ObservedPipeline(generate_llm=FailingLLM(fail_times=99))
    ok_pipeline = ObservedPipeline(tracer=pipeline.tracer)
    ok_pipeline.run(QUESTIONS[0])
    result = pipeline.run(QUESTIONS[1])
    view = MetricsView(pipeline.tracer.spans)
    assert not result.ok
    assert view.error_rate == pytest.approx(0.5)
    assert view.span_error_rate < view.error_rate
    assert view.errors_by_type()["LLMError"] == 3, "the call, the generate step and the root"


def test_tokens_per_second_ignores_cache_hits():
    tracer, cache = Tracer(), ResponseCache()
    client = InstrumentedLLM(EchoLLM(), tracer=tracer, cache=cache)
    client.complete(PROMPT)
    real_only = MetricsView(list(tracer.spans)).tokens_per_second
    for _ in range(5):
        client.complete(PROMPT)
    assert MetricsView(tracer.spans).tokens_per_second == pytest.approx(real_only)


def test_step_table_does_not_double_count_the_nested_model_call():
    pipeline = ObservedPipeline()
    for q in QUESTIONS[:3]:
        pipeline.run(q)
    steps = MetricsView(pipeline.tracer.spans).by_step()
    assert steps["generate"] and len(steps["generate"]) == 3, "3 requests, 3 generate steps"
    assert "request" not in steps, "the root span is not a step"


def test_cost_is_attributed_per_request_and_per_trace():
    pipeline = ObservedPipeline()
    a = pipeline.run(QUESTIONS[0], request_id="r-a")
    b = pipeline.run(QUESTIONS[1], request_id="r-b")
    view = MetricsView(pipeline.tracer.spans)
    per_request = view.cost_per_request()
    assert set(per_request) == {"r-a", "r-b"}
    assert per_request["r-a"] == pytest.approx(a.cost_usd, abs=1e-9)
    assert view.cost_per_trace()[b.trace_id] == pytest.approx(b.cost_usd, abs=1e-9)
    assert view.total_cost_usd == pytest.approx(a.cost_usd + b.cost_usd, abs=1e-9)


def test_span_export_round_trips_through_jsonl(tmp_path):
    pipeline = ObservedPipeline()
    result = pipeline.run(QUESTIONS[0])
    path = tmp_path / "spans.jsonl"
    written = export_spans_jsonl(pipeline.tracer, str(path))
    rows = load_spans_jsonl(str(path))
    assert written == len(rows) == len(pipeline.tracer.spans)
    assert all(json.dumps(r) for r in rows)
    assert {r["trace_id"] for r in rows} == {result.trace_id}
    assert any(r["attributes"].get("llm.cost_usd", 0) > 0 for r in rows)


def test_waterfall_shows_nesting_and_marks_the_failed_step():
    pipeline = ObservedPipeline(generate_llm=FailingLLM(fail_times=99))
    result = pipeline.run(QUESTIONS[0])
    text = render_waterfall(pipeline.tracer.spans, result.trace_id)
    assert "request [error]" in text
    assert "  retrieve" in text and "    llm.call" in text, "two levels of indentation"
    assert "generate [error]" in text
    assert render_waterfall(pipeline.tracer.spans, "nope").startswith("no spans")


def test_dashboard_renders_from_recorded_spans_only():
    pipeline = ObservedPipeline()
    pipeline.run(QUESTIONS[0])
    text = render_dashboard(MetricsView(pipeline.tracer.spans))
    assert "requests 1" in text
    assert "generate" in text and "retrieve" in text
    assert "cost" in text
