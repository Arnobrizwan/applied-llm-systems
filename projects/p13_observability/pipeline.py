"""A realistic four step pipeline, fully instrumented.

retrieve -> rerank -> generate -> judge

Four steps rather than one model call because the interesting observability
questions only exist in a pipeline: which step owns the p95, which step is
generating the cost, whether an error in step three is visible as a failed
request in step one. A single call needs a stopwatch, not a tracing system.

The steps are real. Retrieval is BM25 over the Meridian corpus, reranking and
judging are model calls on the cheap tier, generation is a model call on the
expensive tier. Every step opens a span on the same trace, so the waterfall shows
nesting and every metric in `metrics.py` is computed from those spans.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from llmkit import BM25, EchoLLM, LLMError, LLMProvider, Tracer, system, user
from llmkit.corpus import by_id, chunks

from .client import InstrumentedLLM, PrivacyPolicy, ResponseCache, observed_request, step_span

RERANK_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["best_index", "confidence"],
    "properties": {
        "best_index": {"type": "integer", "minimum": 1, "maximum": 4},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
}

ANSWER_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["answer", "grounded"],
    "properties": {"answer": {"type": "string"}, "grounded": {"type": "boolean"}},
}

JUDGE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["faithfulness", "reason"],
    "properties": {
        "faithfulness": {"type": "integer", "minimum": 1, "maximum": 5},
        "reason": {"type": "string"},
    },
}


@dataclass
class PipelineResult:
    request_id: str
    trace_id: str
    ok: bool
    answer: str = ""
    error: str = ""
    duration_ms: float = 0.0
    cost_usd: float = 0.0
    total_tokens: int = 0
    llm_calls: int = 0
    cached_calls: int = 0
    step_ms: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"request_id": self.request_id, "trace_id": self.trace_id, "ok": self.ok,
                "duration_ms": round(self.duration_ms, 2), "cost_usd": round(self.cost_usd, 6),
                "total_tokens": self.total_tokens, "llm_calls": self.llm_calls,
                "cached_calls": self.cached_calls,
                "step_ms": {k: round(v, 2) for k, v in self.step_ms.items()},
                "error": self.error}


class ObservedPipeline:
    """retrieve, rerank, generate, judge, all on one trace.

    Providers are injected per step so a demo or a test can make one step slow or
    make one step fail without touching the other three. That is also how it
    works in production: the cheap tier and the expensive tier are different
    upstreams with different failure modes.
    """

    def __init__(self, tracer: Optional[Tracer] = None,
                 rerank_llm: Optional[LLMProvider] = None,
                 generate_llm: Optional[LLMProvider] = None,
                 judge_llm: Optional[LLMProvider] = None,
                 cache: Optional[ResponseCache] = None,
                 privacy: Optional[PrivacyPolicy] = None,
                 generate_tier: str = "large"):
        self.tracer = tracer or Tracer("rag-service")
        self.cache = cache
        privacy = privacy or PrivacyPolicy()
        self.rerank = InstrumentedLLM(rerank_llm or EchoLLM(model="echo-small"), tier="small",
                                      tracer=self.tracer, privacy=privacy, cache=cache)
        self.generate = InstrumentedLLM(generate_llm or EchoLLM(model="echo-large"),
                                        tier=generate_tier, tracer=self.tracer,
                                        privacy=privacy, cache=cache)
        self.judge = InstrumentedLLM(judge_llm or EchoLLM(model="echo-small"), tier="small",
                                     tracer=self.tracer, privacy=privacy, cache=cache)
        self._docs = by_id()
        self._index = BM25()
        for c in chunks():
            self._index.add(c.doc_id, c.text)

    def run(self, question: str, request_id: Optional[str] = None,
            tenant: str = "acme", context_repeat: int = 1) -> PipelineResult:
        """One request. Never raises: a failed request is a result, not an exception.

        `context_repeat` simulates a runaway context (a retry loop that appends
        the previous attempt), which is the cheapest way to reproduce a real cost
        incident without a real bill.
        """
        rid = request_id or uuid.uuid4().hex[:12]
        started = time.perf_counter()
        holder: Dict[str, Any] = {}
        ok, answer, error = True, "", ""

        try:
            with observed_request(self.tracer, "request", rid,
                                  tenant=tenant, question_chars=len(question)) as root:
                holder["span"] = root
                answer = self._steps(question, context_repeat)
        except LLMError as exc:
            # Upstream failures are expected and are recorded, not swallowed. The
            # root span is already marked errored by the tracer on the way out.
            ok, error = False, f"{type(exc).__name__}: {exc}"
        except Exception as exc:  # pragma: no cover - defensive
            ok, error = False, f"{type(exc).__name__}: {exc}"

        root_span = holder.get("span")
        trace_id = root_span.trace_id if root_span else ""
        spans = [s for s in self.tracer.spans if s.trace_id == trace_id]
        return PipelineResult(
            request_id=rid,
            trace_id=trace_id,
            ok=ok,
            answer=answer,
            error=error,
            duration_ms=(time.perf_counter() - started) * 1000.0,
            cost_usd=sum(float(s.attributes.get("llm.cost_usd", 0.0) or 0.0) for s in spans),
            total_tokens=sum(int(s.attributes.get("llm.total_tokens", 0) or 0) for s in spans),
            llm_calls=sum(1 for s in spans if "llm.tier" in s.attributes),
            # Counted separately because a cache hit is not a model latency
            # sample. Feeding hits into a latency baseline makes the baseline
            # chase the hit rate instead of tracking the upstream.
            cached_calls=sum(1 for s in spans if s.attributes.get("llm.cache") == "hit"),
            step_ms={str(s.attributes.get("step", s.name)): s.duration_ms
                     for s in spans if s.parent_id is not None},
        )

    def _steps(self, question: str, context_repeat: int) -> str:
        with step_span(self.tracer, "retrieve", k=4) as span:
            hits = self._index.search(question, k=4)
            span.attributes["retrieve.hits"] = len(hits)
            span.attributes["retrieve.top_score"] = round(hits[0][1], 3) if hits else 0.0
            candidates = [self._docs[doc_id].text for doc_id, _ in hits]

        with step_span(self.tracer, "rerank"):
            if candidates:
                self.rerank.complete(
                    [system("Pick the single most relevant passage."),
                     user(question + "\n" + "\n".join(
                         f"[{i}] {c[:200]}" for i, c in enumerate(candidates, 1)))],
                    step="rerank", json_schema=RERANK_SCHEMA,
                )

        evidence = "\n".join(f"[S{i}] {c}" for i, c in enumerate(candidates[:2], 1))
        evidence = "\n".join([evidence] * max(1, context_repeat))

        with step_span(self.tracer, "generate"):
            resp = self.generate.complete(
                [system("Answer from the evidence only.\n" + evidence), user(question)],
                step="generate", json_schema=ANSWER_SCHEMA,
            )
            answer = resp.text

        with step_span(self.tracer, "judge"):
            self.judge.complete(
                [system("Grade the answer for faithfulness to the evidence."),
                 user(f"Question: {question}\nAnswer: {answer[:400]}")],
                step="judge", json_schema=JUDGE_SCHEMA,
            )
        return answer


QUESTIONS: List[str] = [
    "How long is a Meridian token valid before it expires?",
    "What happens if I go over the request rate limit?",
    "How are webhook deliveries authenticated?",
    "Which role is allowed to export the audit log?",
    "How long are request logs kept?",
    "How much notice is given before scheduled maintenance?",
    "What is the maximum page size on list endpoints?",
    "How is a failed invoice payment retried?",
]
