"""Project 13: LLM observability stack.

Tracing, metrics, anomaly detection and alerting for LLM calls, built on
`llmkit.tracing`. Public API:

    from projects.p13_observability import (
        InstrumentedLLM, PrivacyPolicy, ResponseCache, observed_request,
        MetricsView, AlertManager, default_rules, render_dashboard, render_waterfall,
    )

    tracer = Tracer("my-service")
    llm = InstrumentedLLM(EchoLLM(), tier="large", tracer=tracer)
    with observed_request(tracer, "request", request_id="req-1"):
        llm.complete(messages, step="generate")
    print(render_dashboard(MetricsView(tracer.spans)))
"""
from .anomaly import (
    Alert, AlertManager, AlertRule, EwmaBaseline, RollingRate, default_rules, fingerprint,
)
from .client import (
    InstrumentedLLM, PrivacyPolicy, ResponseCache, current_request_id, observed_request, step_span,
)
from .dashboard import (
    export_spans_jsonl, load_spans_jsonl, render_dashboard, render_waterfall, write_report,
)
from .metrics import LatencyStats, MetricsView, latency_stats
from .pipeline import ObservedPipeline, PipelineResult

__all__ = [
    "InstrumentedLLM", "PrivacyPolicy", "ResponseCache", "observed_request", "step_span",
    "current_request_id",
    "MetricsView", "LatencyStats", "latency_stats",
    "AlertManager", "AlertRule", "Alert", "EwmaBaseline", "RollingRate",
    "default_rules", "fingerprint",
    "render_dashboard", "render_waterfall", "export_spans_jsonl", "load_spans_jsonl",
    "write_report",
    "ObservedPipeline", "PipelineResult",
]
