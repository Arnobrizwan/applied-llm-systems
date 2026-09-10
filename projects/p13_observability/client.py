"""The instrumented LLM client: one span per call, with the attributes an
incident actually needs.

The rule this module is built around: if it is not on the span, it does not
exist at 3am. Provider, model, tier, prompt and completion tokens, latency,
cost, cache hit, error, request id and trace id all go on every call, because
every one of them has been the answer to "why did the bill triple" or "which
step got slow" at some point.

Privacy is the other half. A trace store is the easiest place in a system to
leak customer data: it is high volume, it is retained for weeks, it is usually
readable by everyone on call, and it is frequently a third party SaaS. So prompt
text is hashed by default here, never captured raw, and full capture is an
explicit opt-in that is recorded on the span itself so an audit can see which
spans hold text.
"""
from __future__ import annotations

import hashlib
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence

from llmkit import LLMProvider, LLMResponse, Tracer, count_message_tokens, estimate_cost
from llmkit.providers import _as_messages

_CURRENT = threading.local()


def current_request_id() -> Optional[str]:
    """The request id of the enclosing `observed_request`, if any."""
    return getattr(_CURRENT, "request_id", None)


@dataclass
class PrivacyPolicy:
    """How much prompt text reaches the trace store.

    Modes:
      hash     - sha256 prefix only. The default. Two identical prompts have the
                 same fingerprint, which is all you need to spot a hot cache key
                 or a retry storm, and none of the text.
      truncate - first `truncate_chars` characters. Useful in a staging
                 environment where the inputs are synthetic.
      full     - the whole prompt. Legitimate for internal tools with no customer
                 data. It is opt-in, and `privacy.mode` is written to every span
                 so a later audit can find exactly which spans carry text.

    `salt` exists because an unsalted hash of a short prompt is reversible by
    anyone who can guess the prompt, which for a fixed set of support questions
    is trivial. Salting per environment breaks that.
    """

    mode: str = "hash"
    truncate_chars: int = 64
    salt: str = ""

    VALID = ("hash", "truncate", "full")

    def __post_init__(self) -> None:
        if self.mode not in self.VALID:
            raise ValueError(f"privacy mode must be one of {self.VALID}")

    def capture(self, text: str) -> Dict[str, Any]:
        digest = hashlib.sha256((self.salt + text).encode("utf-8")).hexdigest()[:16]
        out: Dict[str, Any] = {
            "privacy.mode": self.mode,
            "prompt.sha256": digest,
            "prompt.chars": len(text),
        }
        if self.mode == "truncate":
            out["prompt.preview"] = text[: self.truncate_chars]
        elif self.mode == "full":
            out["prompt.text"] = text
        return out


class ResponseCache:
    """Content addressed cache keyed on the exact prompt.

    Deliberately exact match rather than semantic similarity. A semantic cache
    returns an answer to a question the user did not ask, and the failure is
    invisible in the metrics: cache hit rate goes up and quality goes down with
    no signal connecting the two. Exact match is boring and safe, and the hit
    rate it reports is a hit rate you can trust.
    """

    def __init__(self) -> None:
        self._store: Dict[str, LLMResponse] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(messages: Sequence, **kwargs: Any) -> str:
        payload = "|".join(f"{m.role}:{m.content}" for m in _as_messages(messages))
        extra = ",".join(f"{k}={v}" for k, v in sorted(kwargs.items()) if k != "request_id")
        return hashlib.sha256((payload + "||" + extra).encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[LLMResponse]:
        hit = self._store.get(key)
        if hit is None:
            self.misses += 1
            return None
        self.hits += 1
        # A copy, with cached=True. Returning the stored object would let a
        # caller mutate a shared response and corrupt every later hit.
        return LLMResponse(
            text=hit.text, model=hit.model, provider=hit.provider,
            prompt_tokens=hit.prompt_tokens, completion_tokens=hit.completion_tokens,
            latency_ms=0.0, cached=True, finish_reason=hit.finish_reason,
        )

    def put(self, key: str, response: LLMResponse) -> None:
        self._store[key] = response

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def __len__(self) -> int:
        return len(self._store)


@contextmanager
def observed_request(tracer: Tracer, name: str, request_id: Optional[str] = None,
                     **attributes: Any) -> Iterator[Any]:
    """Open the root span for one request and bind its id for nested spans.

    The request id is carried in a thread local rather than threaded through
    every function signature. That is the same trade every tracing library makes:
    it is invisible plumbing that survives being passed through code you do not
    own, at the cost of not crossing a thread or a task boundary by itself.
    """
    rid = request_id or uuid.uuid4().hex[:12]
    previous = getattr(_CURRENT, "request_id", None)
    _CURRENT.request_id = rid
    try:
        with tracer.span(name, **{"request.id": rid, **attributes}) as span:
            yield span
    finally:
        _CURRENT.request_id = previous


class InstrumentedLLM:
    """Wraps any `llmkit` provider and records every call as a span.

    Wrapping rather than subclassing so the same instrumentation works for every
    provider in `llmkit` and for anything added later. The wrapper is transparent:
    it returns the provider's own `LLMResponse` and re-raises the provider's own
    exception, because instrumentation that changes behaviour is instrumentation
    people turn off.
    """

    def __init__(self, provider: LLMProvider, tier: str = "small",
                 tracer: Optional[Tracer] = None,
                 privacy: Optional[PrivacyPolicy] = None,
                 cache: Optional[ResponseCache] = None,
                 span_name: str = "llm.call"):
        self.provider = provider
        self.tier = tier
        self.tracer = tracer or Tracer("llm")
        self.privacy = privacy or PrivacyPolicy()
        self.cache = cache
        self.span_name = span_name
        self.calls = 0

    @property
    def model(self) -> str:
        return self.provider.model

    def complete(self, messages: Sequence, step: Optional[str] = None,
                 **kwargs: Any) -> LLMResponse:
        self.calls += 1
        msgs = _as_messages(messages)
        prompt_text = "\n".join(m.content for m in msgs)
        # `is not None`, not a truthiness check: ResponseCache defines __len__,
        # so an empty cache is falsy and the first call of the process would have
        # been stored under an empty key and never hit again. Caught by a test.
        cache_key = ResponseCache.key(messages, **kwargs) if self.cache is not None else ""

        attrs: Dict[str, Any] = {
            "llm.tier": self.tier,
            "llm.provider": self.provider.name,
            "llm.model": self.provider.model,
            "step": step or self.span_name,
            "request.id": current_request_id(),
            "llm.estimated_prompt_tokens": count_message_tokens(msgs),
        }
        attrs.update(self.privacy.capture(prompt_text))

        with self.tracer.span(self.span_name, **attrs) as span:
            if self.cache is not None:
                hit = self.cache.get(cache_key)
                if hit is not None:
                    # A cache hit still costs a span. Hits that are invisible make
                    # the p95 look better than the user experience actually is.
                    self.tracer.record_llm(span, hit, cost_usd=0.0)
                    span.attributes["llm.cache"] = "hit"
                    span.attributes["llm.latency_ms"] = 0.0
                    return hit
                span.attributes["llm.cache"] = "miss"

            started = time.perf_counter()
            try:
                response = self.provider.complete(messages, **kwargs)
            except Exception as exc:
                # The span is marked errored by the tracer's context manager. What
                # it cannot know is the taxonomy, and "which error class spiked"
                # is the first question during an incident.
                span.attributes["error.type"] = type(exc).__name__
                span.attributes["error.retryable"] = bool(getattr(exc, "retryable", False))
                span.attributes["error.status"] = getattr(exc, "status", None)
                span.attributes["llm.latency_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
                raise
            elapsed_ms = (time.perf_counter() - started) * 1000.0

            cost = estimate_cost(self.tier, response.prompt_tokens, response.completion_tokens)
            self.tracer.record_llm(span, response, cost_usd=cost)
            span.attributes["llm.latency_ms"] = round(elapsed_ms, 3)
            span.attributes["llm.tokens_per_second"] = (
                round(response.completion_tokens / (elapsed_ms / 1000.0), 2) if elapsed_ms > 0 else 0.0
            )
            if self.cache is not None:
                self.cache.put(cache_key, response)
            return response


def step_span(tracer: Tracer, name: str, **attributes: Any):
    """A span for a non-LLM pipeline step (retrieval, reranking, post-processing).

    Non-model steps get spans too. Blaming the model for latency that was really
    a slow vector search is a classic, and it is only avoidable if the retrieval
    step is on the same trace.
    """
    return tracer.span(name, **{"request.id": current_request_id(), "step": name, **attributes})
