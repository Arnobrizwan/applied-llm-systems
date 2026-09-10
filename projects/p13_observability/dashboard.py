"""Output surfaces: a JSONL span export, a text dashboard and a trace waterfall.

Three outputs because they answer three different questions.

The dashboard answers "is the system healthy right now". The waterfall answers
"where did this one slow request spend its time", which an aggregate can never
answer: a p95 tells you something is slow and nothing about which step. The
JSONL export answers "what happened three hours ago", and it is line delimited
so it can be grepped, piped into jq, or bulk loaded without a parser.

The export format matches what `llmkit.Tracer.export_jsonl` writes, which is
OpenTelemetry shaped (trace id, span id, parent id, attributes, status). Nothing
here needs a collector running, and swapping in a real exporter later is a change
in one function rather than a change to every instrumented call site.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence

from llmkit import Span, Tracer

from .anomaly import Alert
from .metrics import MetricsView


def export_spans_jsonl(tracer: Tracer, path: str) -> int:
    """Write every span as one JSON object per line. Returns the count."""
    return tracer.export_jsonl(path)


def load_spans_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read an export back. The dashboard can run over a file from another host."""
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _bar(fraction: float, width: int = 24) -> str:
    filled = max(0, min(width, int(round(fraction * width))))
    return "#" * filled + "." * (width - filled)


def render_dashboard(view: MetricsView, alerts: Optional[Sequence[Alert]] = None,
                     title: str = "LLM SERVICE DASHBOARD") -> str:
    """A text dashboard. Every number comes from a recorded span."""
    snap = view.snapshot()
    lines: List[str] = [title, "=" * len(title)]

    lines.append(f"requests {snap['requests']}   spans {snap['spans']}   "
                 f"llm calls {snap['llm_calls']}")
    lat = snap["latency_ms"]
    lines.append(f"request latency ms   p50 {lat['p50']:8.1f}   p95 {lat['p95']:8.1f}   "
                 f"p99 {lat['p99']:8.1f}   max {lat['max']:8.1f}")
    lines.append(f"error rate           {snap['error_rate']:.1%} of requests "
                 f"({snap['span_error_rate']:.1%} of spans)")
    if snap["errors_by_type"]:
        lines.append("  by type            " +
                     ", ".join(f"{k} x{v}" for k, v in snap["errors_by_type"].items()))
    lines.append(f"throughput           {snap['tokens_per_second']:.1f} completion tokens/s "
                 f"inside model calls")
    lines.append(f"tokens               {snap['total_tokens']} total")
    lines.append(f"cost                 ${snap['total_cost_usd']:.6f} total, "
                 f"${snap['mean_cost_per_request_usd']:.6f} mean per request")
    lines.append(f"cache                {snap['cache_hit_rate']:.1%} hit rate, "
                 f"${snap['cache_saved_usd']:.6f} avoided")

    lines.append("")
    lines.append(f"{'step':<16}{'calls':>7}{'p50 ms':>10}{'p95 ms':>10}{'max ms':>10}  share")
    lines.append("-" * 72)
    by_step = view.latency_by_step()
    worst = max((s.p95 for s in by_step.values()), default=0.0) or 1.0
    for name, stats in sorted(by_step.items(), key=lambda kv: -kv[1].p95):
        lines.append(f"{name[:15]:<16}{stats.count:>7}{stats.p50:>10.1f}{stats.p95:>10.1f}"
                     f"{stats.max:>10.1f}  {_bar(stats.p95 / worst)}")

    tiers = view.llm_by_tier()
    if tiers:
        lines.append("")
        lines.append(f"{'tier':<16}{'calls':>7}{'cached':>8}{'p95 ms':>10}{'tokens':>9}{'cost usd':>12}")
        lines.append("-" * 62)
        for tier, agg in tiers.items():
            lines.append(f"{tier:<16}{agg['calls']:>7}{agg['cached']:>8}{agg['p95_ms']:>10.1f}"
                         f"{agg['tokens']:>9}{agg['cost_usd']:>12.6f}")

    costs = sorted(view.cost_per_trace().items(), key=lambda kv: -kv[1])[:5]
    if costs:
        lines.append("")
        lines.append("most expensive traces")
        for trace_id, cost in costs:
            lines.append(f"  {trace_id[:16]}  ${cost:.6f}")

    if alerts:
        lines.append("")
        lines.append(f"alerts ({len(alerts)})")
        for a in alerts:
            extra = (f"  [+{a.suppressed_since_last} suppressed]"
                     if a.suppressed_since_last else "")
            label = ",".join(f"{k}={v}" for k, v in sorted(a.labels.items()))
            lines.append(f"  {a.severity.upper():<8} {a.rule:<22} {label:<16} "
                         f"{a.message}{extra}")
    return "\n".join(lines)


def render_waterfall(spans: Iterable[Span], trace_id: str, width: int = 40) -> str:
    """Print one trace as a nested waterfall with real offsets and durations.

    Offsets are relative to the root span's start, which is the only reading that
    survives being looked at on a different machine than it was recorded on.
    """
    chosen = [s for s in spans if s.trace_id == trace_id]
    if not chosen:
        return f"no spans for trace {trace_id}"
    root_start = min(s.start_ms for s in chosen)
    total = max(s.end_ms for s in chosen) - root_start or 1.0

    children: Dict[Optional[str], List[Span]] = {}
    for s in chosen:
        children.setdefault(s.parent_id, []).append(s)
    for group in children.values():
        group.sort(key=lambda s: s.start_ms)

    known_ids = {s.span_id for s in chosen}
    roots = [s for s in chosen if s.parent_id is None or s.parent_id not in known_ids]

    lines = [f"trace {trace_id}   total {total:.1f} ms   spans {len(chosen)}"]
    lines.append(f"{'span':<34}{'start':>9}{'dur ms':>10}  timeline")

    def walk(span: Span, depth: int) -> None:
        offset = span.start_ms - root_start
        lead = int(round(offset / total * width))
        length = max(1, int(round(span.duration_ms / total * width)))
        bar = " " * lead + ("!" if span.status == "error" else "=") * min(length, width - lead)
        name = ("  " * depth) + span.name + (" [error]" if span.status == "error" else "")
        lines.append(f"{name[:33]:<34}{offset:>9.1f}{span.duration_ms:>10.2f}  |{bar:<{width}}|")
        for child in children.get(span.span_id, []):
            walk(child, depth + 1)

    for root in roots:
        walk(root, 0)
    return "\n".join(lines)


def write_report(path: str, view: MetricsView, alerts: Sequence[Alert]) -> str:
    """Machine-readable rollup, for a scrape endpoint or a CI artefact."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload = {
        "metrics": view.snapshot(),
        "latency_by_step": {k: v.to_dict() for k, v in view.latency_by_step().items()},
        "llm_by_tier": view.llm_by_tier(),
        "cost_per_request_usd": {k: round(v, 6) for k, v in view.cost_per_request().items()},
        "alerts": [a.to_dict() for a in alerts],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return path
