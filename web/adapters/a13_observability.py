"""Web adapter for project 13: the observability stack.

The visitor picks a fault, the page runs a real traced pipeline through a steady
warm-up and then the fault, and reports what the instrumentation saw: the trace
waterfall, the cost and token table, the latency percentiles and the alerts
(including the ones the cooldown swallowed).

Read only: spans stay in memory, nothing is exported to disk.
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

NUMBER = 13
SLUG = "observability"
TITLE = "LLM Observability Stack"
TAGLINE = ("Break a traced AI pipeline on purpose and see the traces, the bill, the "
           "latency and the alerts it produced.")

WHAT_IT_DOES = """Type spike, errors, cost or clean to choose what goes wrong. The page
runs a real four step pipeline (find documents, rerank them, write an answer, grade the
answer) a dozen times to establish normal behaviour, then runs it again with the fault you
picked.

Every step is timed and recorded as a trace. You get the waterfall for one request, showing
which step nested inside which and how many milliseconds each one took, plus the token and
money totals broken down by model tier, and the p50 and p95 latency for the whole request
and for each step.

The last part is the alerting. Latency is compared against a baseline the system learns as
it goes, rather than a fixed number someone guessed. Repeat alerts about the same problem
are held back by a cooldown and counted, so one incident pages you once instead of forty
times, and the page shows you both numbers."""

INPUT_LABEL = "What should go wrong"
PLACEHOLDER = "spike, errors, cost or clean"
EXAMPLES = ["spike", "errors", "cost", "clean"]
SOURCE = "projects/p13_observability"

STEADY = 12
FAULT = 5

_MODES = {
    "spike": "the answer-writing step suddenly takes 30 ms instead of a fraction of one",
    "errors": "the answer-writing upstream returns HTTP 503 on every call",
    "cost": "a retry loop repeats the evidence 40 times and inflates the prompt",
    "clean": "nothing is broken at all",
}

_ALIASES = {
    "spike": "spike", "slow": "spike", "latency": "spike", "lag": "spike", "spikes": "spike",
    "errors": "errors", "error": "errors", "fail": "errors", "failure": "errors",
    "failing": "errors", "outage": "errors", "down": "errors", "503": "errors",
    "cost": "cost", "runaway": "cost", "spend": "cost", "bill": "cost", "money": "cost",
    "expensive": "cost", "tokens": "cost",
    "clean": "clean", "healthy": "clean", "ok": "clean", "fine": "clean",
    "nothing": "clean", "normal": "clean", "steady": "clean", "good": "clean",
}


def _pick(user_input: str):
    words = "".join(c.lower() if c.isalnum() else " " for c in (user_input or "")).split()
    for word in words:
        if word in _ALIASES:
            return _ALIASES[word], ""
    if not words:
        return "spike", "no input given, so the latency spike was run"
    return "spike", (f"'{' '.join(words)[:40]}' is not one of spike, errors, cost or clean, "
                     "so the latency spike was run")


def _run(user_input: str) -> str:
    from llmkit import EchoLLM, FailingLLM, Tracer
    from projects.p13_observability.anomaly import AlertManager, RollingRate, default_rules
    from projects.p13_observability.client import PrivacyPolicy, ResponseCache
    from projects.p13_observability.dashboard import render_waterfall
    from projects.p13_observability.metrics import MetricsView
    from projects.p13_observability.pipeline import QUESTIONS, ObservedPipeline

    mode, note = _pick(user_input)

    tracer = Tracer("rag-service")
    cache = ResponseCache()
    privacy = PrivacyPolicy(mode="hash", salt="demo-env")
    alerts = AlertManager(default_rules())
    error_window = RollingRate(window=20)
    fired = []
    results = []

    healthy = ObservedPipeline(tracer=tracer, cache=cache, privacy=privacy)

    def observe(result, phase):
        # Latency first, so an error burst does not teach the latency baseline
        # that zero milliseconds is normal. A cache hit is not a latency sample.
        out = []
        if result.ok and result.cached_calls == 0:
            out += alerts.observe("latency_ms", result.step_ms.get("generate", 0.0),
                                  {"step": "generate"})
        out += alerts.observe("error_rate", error_window.observe(not result.ok),
                              {"service": "rag"})
        out += alerts.observe("cost_usd", result.cost_usd, {"tenant": "acme"})
        fired.extend(out)
        results.append((phase, result))

    # Warm up. Twelve requests over eight questions, so the last four are cache
    # hits and the learned baseline has eight real latency samples behind it.
    for i in range(STEADY):
        observe(healthy.run(QUESTIONS[i % len(QUESTIONS)], request_id=f"warm-{i:02d}"), "warm")

    steady_alerts = len(fired)

    if mode == "spike":
        pipeline = ObservedPipeline(tracer=tracer, cache=cache, privacy=privacy,
                                    generate_llm=EchoLLM(model="echo-large", latency_ms=30))
        for i in range(FAULT):
            observe(pipeline.run(f"{QUESTIONS[i % len(QUESTIONS)]} (canary)",
                                 request_id=f"fault-{i:02d}"), "fault")
    elif mode == "errors":
        pipeline = ObservedPipeline(tracer=tracer, cache=cache, privacy=privacy,
                                    generate_llm=FailingLLM(fail_times=10 ** 6, status=503))
        for i in range(FAULT):
            observe(pipeline.run(f"{QUESTIONS[i % len(QUESTIONS)]} (during incident)",
                                 request_id=f"fault-{i:02d}"), "fault")
    elif mode == "cost":
        for i in range(4):
            observe(healthy.run(f"{QUESTIONS[i]} (retry loop)", request_id=f"fault-{i:02d}",
                                context_repeat=40), "fault")
    else:
        for i in range(FAULT):
            observe(healthy.run(f"{QUESTIONS[i % len(QUESTIONS)]} (more traffic)",
                                request_id=f"fault-{i:02d}"), "fault")

    view = MetricsView(tracer.spans)
    snap = view.snapshot()
    fault_rows = [r for p, r in results if p == "fault"]

    out = []
    if note:
        out.append(f"note: {note}")
        out.append("")

    out.append("WHAT WAS RUN")
    out.append(f"  warm up      {STEADY} requests on the healthy pipeline, so the detector "
               "learns what normal looks like")
    out.append(f"  then         {len(fault_rows)} requests where {_MODES[mode]}")
    out.append(f"  recorded     {snap['requests']} requests, {snap['spans']} spans, "
               f"{snap['llm_calls']} model calls, all measured in this process")

    out.append("")
    out.append("ALERTS")
    if fired:
        for a in fired:
            labels = ",".join(f"{k}={v}" for k, v in sorted(a.labels.items()))
            extra = (f"  (+{a.suppressed_since_last} earlier repeats were suppressed)"
                     if a.suppressed_since_last else "")
            out.append(f"  {a.severity.upper():<9}{a.rule:<23}{labels:<18}{a.message}{extra}")
    else:
        out.append("  nothing fired, which is the correct answer for a healthy run")
    summary = alerts.summary()
    total_events = summary["alerts_emitted"] + summary["alerts_suppressed"]
    out.append(f"  {summary['alerts_emitted']} alert(s) sent across "
               f"{summary['distinct_fingerprints']} distinct problem(s); "
               f"{summary['alerts_suppressed']} repeat(s) held back by the cooldown")
    if summary["alerts_suppressed"]:
        out.append(f"  without the cooldown this run would have paged {total_events} times "
                   f"instead of {summary['alerts_emitted']}")
    for fp, count in alerts.pending_suppressed().items():
        out.append(f"  problem {fp} still has {count} suppressed repeat(s) inside its cooldown")
    if steady_alerts == 0 and mode != "clean" and fired:
        out.append("  the warm up produced no alerts at all, so everything above came "
                   "from the fault")
    if any(a.rule.startswith("llm_latency") for a in fired):
        out.append("  the sigma number is enormous because a sub-millisecond baseline has "
                   "almost no spread; the detection is reliable, the size of it is not")

    lat = snap["latency_ms"]
    out.append("")
    out.append("LATENCY")
    out.append(f"  whole request   p50 {lat['p50']:.1f} ms   p95 {lat['p95']:.1f} ms   "
               f"p99 {lat['p99']:.1f} ms   worst {lat['max']:.1f} ms")
    out.append(f"{'  step':<18}{'calls':>7}{'p50 ms':>10}{'p95 ms':>10}{'worst ms':>11}")
    out.append("  " + "-" * 54)
    by_step = view.latency_by_step()
    for name, stats in sorted(by_step.items(), key=lambda kv: -kv[1].p95):
        out.append(f"  {name[:15]:<16}{stats.count:>7}{stats.p50:>10.1f}{stats.p95:>10.1f}"
                   f"{stats.max:>11.1f}")

    out.append("")
    out.append("COST AND TOKENS, BY MODEL TIER")
    out.append(f"{'  tier':<18}{'calls':>7}{'cached':>8}{'tokens':>9}{'cost usd':>12}")
    out.append("  " + "-" * 52)
    for tier, agg in view.llm_by_tier().items():
        out.append(f"  {tier:<16}{agg['calls']:>7}{agg['cached']:>8}{agg['tokens']:>9}"
                   f"{agg['cost_usd']:>12.6f}")
    out.append(f"  total {snap['total_tokens']} tokens, ${snap['total_cost_usd']:.6f}, "
               f"mean ${snap['mean_cost_per_request_usd']:.6f} per request")
    out.append(f"  cache {snap['cache_hit_rate']:.0%} hit rate, "
               f"${snap['cache_saved_usd']:.6f} of calls avoided")
    out.append(f"  errors {snap['error_rate']:.0%} of requests "
               f"({snap['span_error_rate']:.0%} of spans, because one bad step "
               "sits next to healthy siblings)")
    if snap["errors_by_type"]:
        out.append("  failures by type: " +
                   ", ".join(f"{k} x{v}" for k, v in snap["errors_by_type"].items()))

    expensive = sorted(view.cost_per_request().items(), key=lambda kv: -kv[1])[:3]
    if expensive:
        out.append("  most expensive requests: " +
                   ", ".join(f"{rid} ${cost:.6f}" for rid, cost in expensive))

    out.append("")
    out.append("ONE REQUEST, TRACED END TO END")
    failed = next((r for r in fault_rows if not r.ok), None)
    chosen = failed or next((r for r in reversed(fault_rows) if r.cached_calls == 0),
                            fault_rows[-1] if fault_rows else None)
    if chosen is not None:
        out.append(f"  request {chosen.request_id}, "
                   f"{'failed' if not chosen.ok else 'served'}"
                   f"{', ' + chosen.error if chosen.error else ''}")
        out.append(render_waterfall(tracer.spans, chosen.trace_id, width=26))
    else:
        out.append("  no request was recorded")

    sample = next((s for s in tracer.spans if "prompt.sha256" in s.attributes), None)
    leaked = sum(1 for s in tracer.spans if "prompt.text" in s.attributes)
    out.append("")
    out.append("PRIVACY")
    if sample is not None:
        out.append(f"  prompts are stored as a salted hash "
                   f"({sample.attributes['prompt.sha256']}, "
                   f"{sample.attributes['prompt.chars']} characters) and not as text")
    out.append(f"  spans holding raw prompt text: {leaked} of {len(tracer.spans)}")
    return "\n".join(out)


def run(user_input: str) -> str:
    try:
        return _run(user_input)
    except Exception as exc:  # an adapter must never take the page down
        return (f"This demo could not complete: {type(exc).__name__}: {exc}\n"
                "Try typing spike, errors, cost or clean.")
