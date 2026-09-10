"""Metric aggregation over recorded spans.

Everything here is computed from spans the instrumented client actually wrote.
There is no separate metrics pipeline and no counter incremented by hand, which
is deliberate: the most common way a dashboard lies is that the counter and the
trace were updated in two different places and one of them was missed on a code
path added later.

Percentiles use `llmkit.percentile` (nearest rank). Nearest rank rather than
linear interpolation because an interpolated p99 reports a latency no request
ever had, and during an incident someone will go looking for that request.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from llmkit import Span, percentile


def _attr(span: Span, key: str, default: Any = None) -> Any:
    return span.attributes.get(key, default)


@dataclass
class LatencyStats:
    count: int = 0
    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    max: float = 0.0
    mean: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {"count": self.count, "p50": round(self.p50, 2), "p95": round(self.p95, 2),
                "p99": round(self.p99, 2), "max": round(self.max, 2), "mean": round(self.mean, 2)}


def latency_stats(values: Sequence[float]) -> LatencyStats:
    if not values:
        return LatencyStats()
    return LatencyStats(
        count=len(values),
        p50=percentile(list(values), 50),
        p95=percentile(list(values), 95),
        p99=percentile(list(values), 99),
        max=max(values),
        mean=sum(values) / len(values),
    )


@dataclass
class MetricsView:
    """A read-only view over a list of spans.

    Constructed from `tracer.spans`, so the same object works for a live process
    and for spans reloaded from a JSONL export.
    """

    spans: List[Span] = field(default_factory=list)

    # -- selection -------------------------------------------------------
    @property
    def llm_spans(self) -> List[Span]:
        return [s for s in self.spans if "llm.tier" in s.attributes]

    @property
    def request_spans(self) -> List[Span]:
        """Root spans, one per request. Identified by having no parent."""
        return [s for s in self.spans if s.parent_id is None]

    def by_step(self) -> Dict[str, List[Span]]:
        """Pipeline steps only: no root spans, no nested `llm.call` spans.

        The nested model call carries the same `step` attribute as the step span
        that wraps it, so grouping naively on that attribute counts every model
        call twice and reports 52 generate calls for 26 requests. The step span
        is the right unit for "how long did this stage take", because it includes
        the prompt assembly and response handling either side of the call.
        """
        out: Dict[str, List[Span]] = {}
        for s in self.spans:
            if s.parent_id is None or s.name == "llm.call":
                continue
            out.setdefault(str(_attr(s, "step", s.name)), []).append(s)
        return out

    def llm_by_tier(self) -> Dict[str, Dict[str, Any]]:
        """Model calls grouped by price tier: the view the bill is written in."""
        out: Dict[str, Dict[str, Any]] = {}
        for s in self.llm_spans:
            tier = str(_attr(s, "llm.tier", "unknown"))
            agg = out.setdefault(tier, {"calls": 0, "tokens": 0, "cost_usd": 0.0,
                                        "cached": 0, "latency": []})
            agg["calls"] += 1
            agg["tokens"] += int(_attr(s, "llm.total_tokens", 0) or 0)
            agg["cost_usd"] += float(_attr(s, "llm.cost_usd", 0.0) or 0.0)
            agg["cached"] += 1 if _attr(s, "llm.cache") == "hit" else 0
            agg["latency"].append(s.duration_ms)
        for agg in out.values():
            agg["p95_ms"] = round(percentile(agg.pop("latency"), 95), 2)
            agg["cost_usd"] = round(agg["cost_usd"], 6)
        return dict(sorted(out.items()))

    # -- latency ---------------------------------------------------------
    def latency(self, spans: Optional[Iterable[Span]] = None) -> LatencyStats:
        chosen = list(spans) if spans is not None else self.request_spans
        return latency_stats([s.duration_ms for s in chosen])

    def latency_by_step(self) -> Dict[str, LatencyStats]:
        return {name: latency_stats([s.duration_ms for s in group])
                for name, group in sorted(self.by_step().items())}

    # -- reliability -----------------------------------------------------
    @property
    def error_rate(self) -> float:
        """Errors over requests, not over spans.

        Span level error rate is diluted by every healthy sibling span, so a
        pipeline with more steps looks more reliable than the same pipeline with
        fewer. The user experiences one failed request either way.
        """
        roots = self.request_spans
        if not roots:
            return 0.0
        return sum(1 for s in roots if s.status == "error") / len(roots)

    @property
    def span_error_rate(self) -> float:
        return (sum(1 for s in self.spans if s.status == "error") / len(self.spans)
                if self.spans else 0.0)

    def errors_by_type(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for s in self.spans:
            if s.status == "error":
                key = str(_attr(s, "error.type") or (s.error or "unknown").split(":")[0])
                out[key] = out.get(key, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    # -- tokens and cost -------------------------------------------------
    @property
    def total_tokens(self) -> int:
        return sum(int(_attr(s, "llm.total_tokens", 0) or 0) for s in self.llm_spans)

    @property
    def total_cost_usd(self) -> float:
        return sum(float(_attr(s, "llm.cost_usd", 0.0) or 0.0) for s in self.llm_spans)

    @property
    def tokens_per_second(self) -> float:
        """Completion tokens divided by time spent inside model calls.

        Model time, not wall clock. Dividing by wall clock mixes in retrieval and
        idle time, and the number then drops whenever something unrelated gets
        slow, which makes it useless as a model health signal.
        """
        busy_ms = sum(s.duration_ms for s in self.llm_spans if not _attr(s, "llm.cached"))
        completion = sum(int(_attr(s, "llm.completion_tokens", 0) or 0)
                         for s in self.llm_spans if not _attr(s, "llm.cached"))
        return completion / (busy_ms / 1000.0) if busy_ms > 0 else 0.0

    def cost_per_request(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for s in self.llm_spans:
            rid = str(_attr(s, "request.id") or "unknown")
            out[rid] = out.get(rid, 0.0) + float(_attr(s, "llm.cost_usd", 0.0) or 0.0)
        return out

    def cost_per_trace(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for s in self.llm_spans:
            out[s.trace_id] = out.get(s.trace_id, 0.0) + float(_attr(s, "llm.cost_usd", 0.0) or 0.0)
        return out

    @property
    def mean_cost_per_request(self) -> float:
        costs = list(self.cost_per_request().values())
        return sum(costs) / len(costs) if costs else 0.0

    # -- cache -----------------------------------------------------------
    @property
    def cache_hit_rate(self) -> float:
        marked = [s for s in self.llm_spans if _attr(s, "llm.cache") in ("hit", "miss")]
        if not marked:
            return 0.0
        return sum(1 for s in marked if _attr(s, "llm.cache") == "hit") / len(marked)

    @property
    def cache_saved_usd(self) -> float:
        """What the hits would have cost at the tier they were served from.

        This is an estimate: it prices a hit using the token counts of the cached
        response, which is exactly what a miss would have cost. It is the only
        honest way to report cache savings without re-running the misses.
        """
        from llmkit import estimate_cost
        total = 0.0
        for s in self.llm_spans:
            if _attr(s, "llm.cache") == "hit":
                total += estimate_cost(str(_attr(s, "llm.tier", "small")),
                                       int(_attr(s, "llm.prompt_tokens", 0) or 0),
                                       int(_attr(s, "llm.completion_tokens", 0) or 0))
        return total

    # -- rollup ----------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        req = self.latency()
        return {
            "requests": len(self.request_spans),
            "spans": len(self.spans),
            "llm_calls": len(self.llm_spans),
            "latency_ms": req.to_dict(),
            "error_rate": round(self.error_rate, 4),
            "span_error_rate": round(self.span_error_rate, 4),
            "errors_by_type": self.errors_by_type(),
            "total_tokens": self.total_tokens,
            "tokens_per_second": round(self.tokens_per_second, 2),
            "total_cost_usd": round(self.total_cost_usd, 6),
            "mean_cost_per_request_usd": round(self.mean_cost_per_request, 6),
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "cache_saved_usd": round(self.cache_saved_usd, 6),
        }
