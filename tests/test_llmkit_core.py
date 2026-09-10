"""Tests for the shared core. Everything here runs offline."""
import json

import pytest

from llmkit import (
    BM25, CircuitBreaker, EchoLLM, FailingLLM, InMemoryVectorStore, RetryPolicy,
    RetryExhausted, ScriptedLLM, chunk_text, cosine, count_tokens, estimate_cost,
    get_embedder, get_llm, percentile, reciprocal_rank_fusion, retry, truncate_to_tokens,
)
from llmkit.corpus import chunks, gold_questions
from llmkit.tracing import Tracer


def test_default_provider_is_offline_and_deterministic():
    llm = get_llm()
    assert llm.name == "echo"
    a = llm.complete([{"role": "user", "content": "explain hybrid search"}])
    b = llm.complete([{"role": "user", "content": "explain hybrid search"}])
    assert a.text == b.text
    assert a.total_tokens > 0


def test_echo_grounds_answers_in_supplied_evidence():
    llm = EchoLLM()
    resp = llm.complete([
        {"role": "system", "content": "[S1] Tokens expire after 90 days.\n[S2] Regions cannot be changed."},
        {"role": "user", "content": "When do tokens expire?"},
    ])
    assert "[S1]" in resp.text


def test_echo_honours_a_json_schema():
    llm = EchoLLM()
    schema = {"type": "object", "properties": {"sentiment": {"type": "string", "enum": ["pos", "neg"]},
                                               "score": {"type": "number", "minimum": 0, "maximum": 1}}}
    obj = json.loads(llm.complete([{"role": "user", "content": "classify"}], json_schema=schema).text)
    assert obj["sentiment"] in ("pos", "neg")
    assert 0.0 <= obj["score"] <= 1.0


def test_echo_fault_rate_produces_broken_output():
    llm = EchoLLM(fault_rate=1.0)
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    text = llm.complete([{"role": "user", "content": "x"}], json_schema=schema).text
    with pytest.raises(Exception):
        json.loads(text)


def test_scripted_llm_returns_in_order():
    llm = ScriptedLLM(["one", "two"])
    assert llm.complete([{"role": "user", "content": "a"}]).text == "one"
    assert llm.complete([{"role": "user", "content": "b"}]).text == "two"


def test_token_estimate_and_truncation():
    text = "the quick brown fox jumps over the lazy dog " * 10
    assert count_tokens(text) > 50
    trimmed = truncate_to_tokens(text, 20)
    assert count_tokens(trimmed) <= 20
    assert truncate_to_tokens(text, 0) == ""


def test_cost_is_zero_for_free_tiers():
    assert estimate_cost("echo", 1000, 1000) == 0.0
    assert estimate_cost("large", 1000, 1000) > 0.0


def test_chunking_overlaps_and_covers():
    text = " ".join(f"Sentence number {i} carries a distinct fact." for i in range(60))
    parts = chunk_text(text, target_tokens=60, overlap_tokens=15)
    assert len(parts) > 1
    assert all(count_tokens(p) <= 90 for p in parts)


def test_vector_store_retrieves_the_right_document():
    store = InMemoryVectorStore()
    store.add(chunks())
    hits = store.search("how long before a token expires", k=3)
    assert any(h.chunk.doc_id == "auth-rotation" for h in hits)


def test_vector_store_round_trips(tmp_path):
    store = InMemoryVectorStore()
    store.add(chunks())
    path = tmp_path / "index.json"
    store.save(str(path))
    reloaded = InMemoryVectorStore.load(str(path))
    assert len(reloaded) == len(store)


def test_bm25_beats_random_on_the_gold_set():
    index = BM25()
    for c in chunks():
        index.add(c.doc_id, c.text)
    hits = sum(1 for g in gold_questions()
               if g["doc_id"] in [d for d, _ in index.search(g["question"], k=3)])
    assert hits >= len(gold_questions()) * 0.6


def test_rrf_rewards_agreement():
    fused = dict(reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]]))
    assert fused["b"] > fused["c"]


def test_cosine_edges():
    assert cosine([0, 0], [1, 1]) == 0.0
    assert cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        cosine([1], [1, 2])


def test_embedder_is_normalised():
    v = get_embedder().embed_one("meridian rate limits")
    assert sum(x * x for x in v) == pytest.approx(1.0, abs=1e-6)


def test_retry_recovers_then_succeeds():
    llm = FailingLLM(fail_times=2)
    resp = retry(lambda: llm.complete([{"role": "user", "content": "hi"}]),
                 policy=RetryPolicy(attempts=4, base_delay=0), sleep=lambda _: None)
    assert llm.attempts == 3
    assert resp.text


def test_retry_gives_up_and_reports():
    llm = FailingLLM(fail_times=99)
    with pytest.raises(RetryExhausted):
        retry(lambda: llm.complete([{"role": "user", "content": "hi"}]),
              policy=RetryPolicy(attempts=3, base_delay=0), sleep=lambda _: None)


def test_circuit_breaker_opens_and_recovers():
    now = [0.0]
    cb = CircuitBreaker(failure_threshold=2, cooldown_s=10, clock=lambda: now[0])
    cb.record_failure(); cb.record_failure()
    assert cb.state == "open" and not cb.allow()
    now[0] = 11.0
    assert cb.state == "half_open" and cb.allow()
    cb.record_success()
    assert cb.state == "closed"


def test_tracer_nests_spans_and_summarises():
    t = Tracer("test")
    with t.span("outer"):
        with t.span("inner", kind="llm") as sp:
            t.record_llm(sp, EchoLLM().complete([{"role": "user", "content": "x"}]))
    assert len(t.spans) == 2
    inner = [s for s in t.spans if s.name == "inner"][0]
    outer = [s for s in t.spans if s.name == "outer"][0]
    assert inner.parent_id == outer.span_id and inner.trace_id == outer.trace_id
    assert t.summary()["inner"]["tokens"] > 0


def test_tracer_records_errors():
    t = Tracer()
    with pytest.raises(ValueError):
        with t.span("boom"):
            raise ValueError("nope")
    assert t.spans[0].status == "error"


def test_percentile():
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([], 90) == 0.0
