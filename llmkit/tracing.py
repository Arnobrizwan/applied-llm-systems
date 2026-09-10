"""Minimal tracing: spans, token/cost accounting, JSONL export.

Deliberately OpenTelemetry-shaped (trace id, span id, parent id, attributes,
status) so a real exporter can be dropped in later, but with no dependency and
no collector to run. Project 13 builds the full observability stack on top.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str
    parent_id: Optional[str] = None
    start_ms: float = 0.0
    end_ms: float = 0.0
    status: str = "ok"
    error: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return max(0.0, self.end_ms - self.start_ms)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["duration_ms"] = round(self.duration_ms, 3)
        return d


class Tracer:
    """Thread-safe span collector. `export_jsonl` writes one span per line."""

    def __init__(self, service: str = "llmkit"):
        self.service = service
        self.spans: List[Span] = []
        self._lock = threading.Lock()
        self._local = threading.local()

    # -- context --------------------------------------------------------
    def _stack(self) -> List[Span]:
        if not hasattr(self._local, "stack"):
            self._local.stack = []
        return self._local.stack

    @property
    def current(self) -> Optional[Span]:
        stack = self._stack()
        return stack[-1] if stack else None

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Span]:
        parent = self.current
        sp = Span(
            name=name,
            trace_id=parent.trace_id if parent else uuid.uuid4().hex,
            span_id=uuid.uuid4().hex[:16],
            parent_id=parent.span_id if parent else None,
            start_ms=time.perf_counter() * 1000.0,
            attributes=dict(attributes),
        )
        self._stack().append(sp)
        try:
            yield sp
        except Exception as exc:
            sp.status = "error"
            sp.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            sp.end_ms = time.perf_counter() * 1000.0
            self._stack().pop()
            with self._lock:
                self.spans.append(sp)

    def record_llm(self, span: Span, resp: Any, cost_usd: float = 0.0) -> None:
        """Attach standard LLM attributes to a span."""
        span.attributes.update(
            {
                "llm.provider": getattr(resp, "provider", None),
                "llm.model": getattr(resp, "model", None),
                "llm.prompt_tokens": getattr(resp, "prompt_tokens", 0),
                "llm.completion_tokens": getattr(resp, "completion_tokens", 0),
                "llm.total_tokens": getattr(resp, "total_tokens", 0),
                "llm.cached": getattr(resp, "cached", False),
                "llm.cost_usd": round(cost_usd, 6),
            }
        )

    # -- output ---------------------------------------------------------
    def export_jsonl(self, path: str) -> int:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with self._lock, open(path, "w", encoding="utf-8") as f:
            for sp in self.spans:
                f.write(json.dumps({"service": self.service, **sp.to_dict()}) + "\n")
        return len(self.spans)

    def summary(self) -> Dict[str, Any]:
        by_name: Dict[str, Dict[str, Any]] = {}
        for sp in self.spans:
            agg = by_name.setdefault(sp.name, {"count": 0, "errors": 0, "total_ms": 0.0, "tokens": 0, "cost_usd": 0.0})
            agg["count"] += 1
            agg["errors"] += 1 if sp.status == "error" else 0
            agg["total_ms"] += sp.duration_ms
            agg["tokens"] += int(sp.attributes.get("llm.total_tokens", 0) or 0)
            agg["cost_usd"] += float(sp.attributes.get("llm.cost_usd", 0.0) or 0.0)
        for agg in by_name.values():
            agg["avg_ms"] = round(agg["total_ms"] / agg["count"], 3) if agg["count"] else 0.0
            agg["total_ms"] = round(agg["total_ms"], 3)
            agg["cost_usd"] = round(agg["cost_usd"], 6)
        return by_name

    def clear(self) -> None:
        with self._lock:
            self.spans.clear()


def percentile(values: List[float], p: float) -> float:
    """Nearest-rank percentile. p in [0, 100]."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round((p / 100.0) * len(ordered) + 0.5)) - 1))
    return ordered[idx]


tracer = Tracer()
