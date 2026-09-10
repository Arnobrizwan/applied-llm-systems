"""End to end demo: instrument a pipeline, break it twice, watch the alerts.

Run it with:  python3 projects/p13_observability/demo.py

Four phases against one shared tracer, cache and alert manager:
  A steady state      - the baseline the detector learns from
  B latency spike     - the generate step gets slow
  C error burst       - the generate upstream starts failing
  D context runaway   - a retry loop inflates the prompt and the bill

Every number printed is computed from spans recorded in this process.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from llmkit import EchoLLM, FailingLLM, Tracer  # noqa: E402

from projects.p13_observability.anomaly import (  # noqa: E402
    AlertManager, RollingRate, default_rules,
)
from projects.p13_observability.client import PrivacyPolicy, ResponseCache  # noqa: E402
from projects.p13_observability.dashboard import (  # noqa: E402
    export_spans_jsonl, render_dashboard, render_waterfall, write_report,
)
from projects.p13_observability.metrics import MetricsView  # noqa: E402
from projects.p13_observability.pipeline import ObservedPipeline, QUESTIONS  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(ROOT, "artifacts", "p13_observability")

STEADY, SPIKE, ERRORS, RUNAWAY = 12, 5, 6, 3


def rule(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tracer = Tracer("rag-service")
    cache = ResponseCache()
    privacy = PrivacyPolicy(mode="hash", salt="demo-env")
    alerts = AlertManager(default_rules())
    error_window = RollingRate(window=20)
    emitted = []
    results = []

    def observe(result, phase):
        """Feed one request into the detectors. Order matters: latency first so
        the error burst does not update the latency baseline with zeros."""
        out = []
        if result.ok and result.cached_calls == 0:
            out += alerts.observe("latency_ms", result.step_ms.get("generate", 0.0),
                                  {"step": "generate"})
        out += alerts.observe("error_rate", error_window.observe(not result.ok),
                              {"service": "rag"})
        out += alerts.observe("cost_usd", result.cost_usd, {"tenant": "acme"})
        for a in out:
            print(f"    ALERT [{a.severity}] {a.rule}: {a.message}")
        emitted.extend(out)
        results.append((phase, result))

    rule("PHASE A: STEADY STATE")
    healthy = ObservedPipeline(tracer=tracer, cache=cache, privacy=privacy)
    for i in range(STEADY):
        observe(healthy.run(QUESTIONS[i % len(QUESTIONS)], request_id=f"a-{i:02d}"), "A")
    print(f"{STEADY} requests, {len(QUESTIONS)} distinct questions, "
          f"cache {cache.hits} hits / {cache.misses} misses")

    rule("PHASE B: LATENCY SPIKE ON THE GENERATE STEP")
    slow = ObservedPipeline(tracer=tracer, cache=cache, privacy=privacy,
                            generate_llm=EchoLLM(model="echo-large", latency_ms=30))
    for i in range(SPIKE):
        observe(slow.run(f"{QUESTIONS[i % len(QUESTIONS)]} (deployment canary)",
                         request_id=f"b-{i:02d}"), "B")

    rule("PHASE C: ERROR BURST FROM THE GENERATE UPSTREAM")
    broken = ObservedPipeline(tracer=tracer, cache=cache, privacy=privacy,
                              generate_llm=FailingLLM(fail_times=999, status=503))
    for i in range(ERRORS):
        observe(broken.run(f"{QUESTIONS[i % len(QUESTIONS)]} (during incident)",
                           request_id=f"c-{i:02d}"), "C")
    print(f"    {ERRORS} consecutive failures produced "
          f"{len([a for a in emitted if a.rule == 'error_rate_high'])} error_rate_high alert(s)")

    rule("PHASE D: CONTEXT RUNAWAY")
    for i in range(RUNAWAY):
        observe(healthy.run(f"{QUESTIONS[i]} (retry loop)", request_id=f"d-{i:02d}",
                            context_repeat=40), "D")

    view = MetricsView(tracer.spans)

    rule("DASHBOARD")
    print(render_dashboard(view, emitted))

    rule("PER PHASE")
    print(f"{'phase':<8}{'reqs':>6}{'ok':>5}{'mean ms':>10}{'cost usd':>12}{'tokens':>9}")
    print("-" * 50)
    for phase in ("A", "B", "C", "D"):
        rows = [r for p, r in results if p == phase]
        okc = sum(1 for r in rows if r.ok)
        mean_ms = sum(r.duration_ms for r in rows) / len(rows)
        print(f"{phase:<8}{len(rows):>6}{okc:>5}{mean_ms:>10.1f}"
              f"{sum(r.cost_usd for r in rows):>12.6f}{sum(r.total_tokens for r in rows):>9}")

    rule("ALERTING")
    summary = alerts.summary()
    print(f"emitted {summary['alerts_emitted']} alerts across "
          f"{summary['distinct_fingerprints']} distinct fingerprints")
    print(f"suppressed {summary['alerts_suppressed']} repeat alerts inside the cooldown")
    print(f"by severity: {summary['by_severity']}")
    print(f"by rule: {summary['by_rule']}")
    total_events = summary["alerts_emitted"] + summary["alerts_suppressed"]
    if total_events:
        print(f"without cooldown and dedup this run would have paged "
              f"{total_events} times instead of {summary['alerts_emitted']}")
    for fp, count in alerts.pending_suppressed().items():
        print(f"  fingerprint {fp} still has {count} suppressed events inside its cooldown")

    rule("TRACE WATERFALL: A HEALTHY REQUEST")
    healthy_trace = next(r.trace_id for p, r in results if p == "A" and r.ok
                         and r.cached_calls == 0)
    print(render_waterfall(tracer.spans, healthy_trace))

    rule("TRACE WATERFALL: A FAILED REQUEST")
    failed_trace = next(r.trace_id for p, r in results if not r.ok)
    print(render_waterfall(tracer.spans, failed_trace))

    rule("PRIVACY")
    sample = next(s for s in tracer.spans if "prompt.sha256" in s.attributes)
    print(f"privacy mode: {sample.attributes['privacy.mode']} (salted per environment)")
    print(f"a captured span carries prompt.sha256={sample.attributes['prompt.sha256']} "
          f"and prompt.chars={sample.attributes['prompt.chars']}")
    leaked = [s for s in tracer.spans if "prompt.text" in s.attributes]
    print(f"spans carrying raw prompt text: {len(leaked)} of {len(tracer.spans)}")

    rule("EXPORT")
    span_path = os.path.join(OUT_DIR, "spans.jsonl")
    count = export_spans_jsonl(tracer, span_path)
    report_path = write_report(os.path.join(OUT_DIR, "report.json"), view, emitted)
    print(f"{count} spans -> {os.path.relpath(span_path, ROOT)}")
    print(f"rollup + alerts -> {os.path.relpath(report_path, ROOT)}")


if __name__ == "__main__":
    main()
