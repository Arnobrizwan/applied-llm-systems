# 13. LLM Observability Stack

**Run it live: [https://applied-llm-systems.vercel.app/s/observability](https://applied-llm-systems.vercel.app/s/observability)** - the hosted page runs this code and shows the real output.

Tracing for prompts, tokens, latency and cost per request, with metric aggregation, anomaly detection and alerting that does not page you four hundred times for one incident.

## The problem

An LLM feature goes to production with a request log and nothing else. Three things then happen in order.

The bill triples and nobody can say which endpoint, which tenant or which prompt did it, because cost was never attributed to a request. Someone says "the model is slow" and the real cause is a retrieval step, but there is no trace linking the two so the model gets blamed and swapped. Then the upstream degrades at 2am, a static threshold alert fires on every request for six minutes, 400 pages land in one channel, and the one alert that mattered is buried in the other 399.

The fourth thing is quieter and worse: someone adds prompt capture to debug an incident, and six months of customer support conversations are now sitting in a third party trace store that half the company can read.

## What this builds

1. **Instrumented client** (`client.py`) - wraps any `llmkit` provider, records every call as a span with provider, model, tier, prompt and completion tokens, latency, cost, cache state, error taxonomy, request id and trace id.
2. **Privacy-aware capture** (`client.py`) - prompt text is salted-hashed by default, with truncate and full modes as documented opt-ins, and the mode written onto every span so an audit can find the spans that hold text.
3. **Response cache** (`client.py`) - exact-match, content addressed, counted in the metrics so the hit rate and the money it saved are both visible.
4. **Metric aggregation** (`metrics.py`) - p50/p95/p99 latency overall and per step, error rate per request and per span, tokens per second, cost per request and per trace, cache hit rate. All computed from recorded spans, never from a hand-maintained counter.
5. **Anomaly detection** (`anomaly.py`) - EWMA baselines with z-scores for latency, a rolling rate for errors, alert rules with severity, per-fingerprint cooldown and suppression counting.
6. **Output** (`dashboard.py`) - JSONL span export, a text dashboard, and a trace waterfall showing nesting and durations.
7. **Instrumented pipeline** (`pipeline.py`) - retrieve, rerank, generate, judge, on one trace.

## Architecture

```
  request
     |
     v
  observed_request(tracer, "request", request_id)   <- root span, binds request id
     |
     +-- span: retrieve   (BM25 over the corpus, not a model call, still traced)
     +-- span: rerank ----+-- span: llm.call  tier=small
     +-- span: generate --+-- span: llm.call  tier=large
     +-- span: judge -----+-- span: llm.call  tier=small
                              |
                              | every llm.call span carries:
                              | provider, model, tier, prompt/completion tokens,
                              | latency, cost, cache hit/miss, error type,
                              | request.id, prompt.sha256, privacy.mode
                              v
                        tracer.spans
                              |
        +---------------------+----------------------+
        v                     v                      v
   MetricsView          AlertManager            dashboard.py
   p50/p95/p99          EWMA baseline           spans.jsonl
   error rate           z-score / rate rules    text dashboard
   tokens/s             fingerprint dedup       trace waterfall
   cost per req/trace   per-fingerprint cooldown
   cache hit rate
```

## Design decisions

**Every metric is derived from spans, not from separate counters.** Rejected: incrementing a counter next to each call. Two write paths get out of sync the first time someone adds a code path and updates only one of them, and the dashboard then lies in a way nobody can reproduce. The cost is that the metrics can only be as good as the span coverage, which is a cost worth paying because span coverage is checkable.

**Prompt text is hashed by default and full capture is opt-in and recorded.** Rejected: capturing everything and redacting later. Redaction is a filter someone has to maintain against inputs they have not seen, and the data is already in the store by the time it fails. The hash is salted per environment because an unsalted hash of a short prompt from a known set is reversible by anyone who can guess the prompt.

**Latency uses a learned EWMA baseline, errors use a flat rate.** A static latency threshold is wrong the day after you set it. But errors are rare, so the standard deviation of the error series is near zero and a z-score makes every single failure look like a twenty sigma event. Different signals need different detectors, and using one mechanism everywhere is how you get an alerting system nobody believes.

**Cooldown is keyed on a fingerprint of rule name plus labels, not on the value.** Including the value in the fingerprint gives every occurrence a new identity, which is the bug that produces 400 alerts for one incident. The suppressed count is carried onto the next alert for that fingerprint, so silence is never mistaken for recovery.

**Errors are counted per request, not per span.** Span level error rate is diluted by every healthy sibling in the same trace, so adding a step to a pipeline makes it look more reliable while the user experience is unchanged. Both numbers are reported, with the request-level one as the headline.

**The instrumented client wraps rather than subclasses, and re-raises unchanged.** Instrumentation that alters behaviour, swallows exceptions or changes the return type is instrumentation that gets switched off during the first incident it causes.

## Running it

```bash
python3 projects/p13_observability/demo.py
python3 -m pytest projects/p13_observability -q     # 29 tests
```

The demo runs four phases against one shared tracer, cache and alert manager: steady state, a latency spike on the generate step, an error burst from `llmkit.FailingLLM`, and a context runaway that inflates the prompt. It prints the alerts as they fire, then the dashboard, a per-phase table, the alerting summary, two trace waterfalls (one healthy, one failed) and the privacy summary. Spans and a JSON rollup are written to `artifacts/p13_observability/`.

## Results

From one run of `demo.py`: 26 requests, 196 spans, 72 model calls across the four phases.

| metric | value |
|---|---|
| requests | 26 (12 steady, 5 spiked, 6 failing, 3 runaway) |
| spans recorded | 196 |
| model calls | 72 (26 large tier, 46 small tier) |
| error rate | 23.1% of requests, 9.2% of spans |
| errors by type | LLMError x18 (the call, its step span and the root span, for each of 6 failed requests) |
| total tokens | 31,534 |
| total cost | $0.079875, mean $0.003072 per request |
| cache hit rate | 16.7% of model calls, $0.005919 avoided |

Cost by tier, which is the point of tracking tier on the span: the large tier took 26 of 72 calls and $0.078516 of $0.079875, so **36% of the calls are 98% of the bill**.

Per phase:

| phase | requests | ok | cost usd | tokens |
|---|---|---|---|---|
| A steady | 12 | 12 | 0.012049 | 6,840 |
| B latency spike | 5 | 5 | 0.008292 | 3,103 |
| C error burst | 6 | 0 | 0.000269 | 1,511 |
| D context runaway | 3 | 3 | 0.059265 | 20,080 |

Phase D is the number worth staring at: 3 requests out of 26 produced 74% of the total cost and 64% of the tokens, because a simulated retry loop repeated the evidence block 40 times. That is what a cost incident looks like from inside the trace, and per-request cost attribution is what makes it a one line finding instead of an afternoon.

Alerting, same run: 3 alerts emitted across 3 distinct fingerprints (`llm_latency_spike` warning, `error_rate_high` critical, `cost_per_request_high` warning), 4 repeat alerts suppressed inside their cooldowns. **7 alert events became 3 pages.** The 6 consecutive failures in phase C produced exactly 1 error alert. The suppressed count moves between 4 and 8 across repeated runs, because it depends on how many latency samples cross 3 sigma on a given machine.

Latency figures are wall-clock on the machine running the demo and vary substantially between runs, so they are quoted rather than tabulated. In the run above, request p50 was 0.3 ms and p95 was 38.3 ms, and the generate step owned the p95 (38.1 ms) as expected while retrieve, rerank and judge stayed under a millisecond at p50. The detector caught the injected 30 ms spike against a learned baseline of 7.7 ms. The sigma figure it reports swings between roughly 70 and 1300 across runs because the steady-state variance of a sub-millisecond series is tiny and noisy; the detection itself is stable, the magnitude is not, and a real deployment would clamp it before putting it in an alert message.

Privacy: 0 of 196 spans carried raw prompt text under the default policy, while every span carried a 16 character salted digest and a character count.

## Limits

`llmkit.EchoLLM` returns in well under a millisecond, so the throughput number (thousands of completion tokens per second) is a property of the offline reference model and not a benchmark of anything. The measurement is real; the value is not transferable. The latency percentiles are similarly dominated by process scheduling rather than by model time, which is why the injected spike had to be 30 ms to be visible at all.

Cost uses `llmkit.tokens.PRICE_PER_1K_USD`, a small editable tier table, with token counts from a word-based estimator rather than the provider's tokenizer. Expect the absolute cost to be within a rough factor of the real bill, and the relative attribution between tiers, requests and traces to be sound.

Spans are held in memory and exported at the end. There is no sampling, no batching, no back pressure and no retention policy, all of which a real deployment needs at volume. The export format is OpenTelemetry shaped so the swap to a real exporter is a change in `dashboard.py` rather than at every call site.

The request id is carried in a thread local, so it does not cross a thread or an asyncio task boundary on its own. That is the same limitation every context-propagation library has, and the fix is the same: pass the context explicitly across the boundary.

Alert thresholds (3 sigma, 6 sigma, 20 percent error rate, 2 cents per request) are legible round numbers, not tuned values. Tuning them needs production traffic, and shipping invented tuned numbers would be worse than shipping obvious ones. There is no paging policy, escalation or routing here on purpose: that belongs in an alert manager, not in the application.
