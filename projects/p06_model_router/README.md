# 06. Model Routing Gateway

Complexity-based routing across three model tiers, with per-tenant and per-endpoint policy, a budget guard that can refuse, a fallback chain that survives a tier outage, and quality-aware escalation.

## The problem

Every request goes to the biggest model, because that is the safe default and nobody has time to work out which ones do not need it. The bill is four to twenty times what it needs to be, and the reason is that a sentiment label and a migration plan are being served by the same model.

The obvious fix makes it worse in three ways. Routing on a single opaque score means nobody can explain why a customer's question went to the cheap model, so the first complaint ends the experiment. Routing with no fallback means the day the chosen tier rate-limits, requests fail rather than quietly serving from the next tier. And routing with no budget ceiling means a runaway loop on the expensive tier is discovered at the end of the month.

## What this builds

1. **Complexity scorer** (`complexity.py`) - explicit features: input length, task type across five classes, presence of math or code, tool requirement, ambiguity, required output length. Returns the features and each feature's contribution, not just a number.
2. **Policy table** (`policy.py`) - score bands to tiers, with per-tenant floors and ceilings, per-endpoint forcing, and a hard pin on the request. Fixed precedence, recorded on every decision.
3. **Cost tracking and budget guard** (`budget.py`) - a ledger attributing every charge to a request, tenant and tier through `llmkit.estimate_cost`, and a guard that downgrades at 80 percent of a daily budget and refuses at 100 percent.
4. **Fallback chain** (`gateway.py`) - `llmkit.retry` for the transient case, `llmkit.CircuitBreaker` for the sustained one, with the reason for every attempt recorded on the result.
5. **Quality-aware escalation** (`gateway.py`) - answer on the cheap tier, judge it with project 04's `RubricJudge`, escalate only when the judge rejects.
6. **Workload** (`workload.py`) - 44 hand written requests across five task types, four endpoints and five tenants.

## Architecture

```
  Request(prompt, tenant, endpoint, pin)
        |
        v
  score_complexity()  -> ComplexityFeatures
        |                  task_type, input_tokens, output_tokens,
        |                  math, code, tools, ambiguity, per-feature contributions
        v
  RoutingPolicy.decide()
        |   precedence: hard pin > endpoint force > tenant floor > tenant ceiling > band
        v
  BudgetGuard.check(tenant, tier)
        |
        +-- refuse   -> GatewayResult(refused=True, ok=False, cost 0, no upstream call)
        +-- downgrade-> tier := small
        v
  fallback_chain(tier)          e.g. medium -> [medium, large, small]
        |
        +-- breaker open?  -> record "circuit_open", skip without a call
        +-- retry(attempts=2) -> failed?  record error, open breaker, next candidate
        v
  answer + ledger.charge(request, tenant, tier, tokens)
        |
        v
  QualityJudge (project 04 RubricJudge)
        |
        +-- accepted -> done
        +-- rejected -> escalate one tier up through the same chain, charge again
```

## Design decisions

**The scorer returns features, not a score.** Rejected: a single float, or an embedding-similarity model trained on past routing. Both are more accurate in principle and neither can answer "why did this go to the cheap tier", which is the first question anyone asks. Every decision here prints as one line naming the top three contributing features.

**Fixed precedence rather than rule priorities.** A priority number per rule is more flexible, and after the third rule nobody can predict the outcome. Pin beats endpoint beats tenant beats band, always, and the decision records the chain that fired.

**Fallback prefers the nearest capability, tie-breaking upward.** When medium is down the chain goes to large, not small. Rejected: falling to the cheapest available tier, which is cheaper and converts every upstream incident into a silent quality incident nobody is watching for.

**Retry is capped at 2 attempts and the breaker opens after 2 failures.** Retry alone turns a tier outage into a slower tier outage, because every request pays every retry before failing over. The measured effect below is that only 2 requests out of 44 ever paid for a failed call.

**Refusal is a distinct outcome, not an exception and not an infinite downgrade.** A guard that silently downgrades forever turns a budget problem into a quality problem nobody attributes to the budget. A refused result has `refused=True`, `ok=False`, `tier_used=None`, zero cost, and makes no upstream call at all.

**Escalation is judged after the cheap answer, not predicted before it.** Predicting which requests the cheap tier will get wrong is the same hard problem as the routing itself. Judging the answer costs a cheap call plus a judge call for the requests that then escalate, which only pays when the cheap tier is right most of the time, and the demo reports the escalation rate so that trade is visible instead of assumed.

**The escalation judge is imported from project 04, not reimplemented.** A second, subtly different rubric judge in this project would drift from the one the eval harness uses, and the two systems would then disagree about what a good answer is.

## Running it

```bash
python3 projects/p06_model_router/demo.py
python3 -m pytest projects/p06_model_router -q     # 25 tests
```

The demo prints six sections: the workload profile, seven annotated routing decisions covering every precedence path, the tier and cost table against both baselines, escalation, a full large-tier outage, and a tenant spending its daily budget down to refusal. A JSON report is written to `artifacts/p06_model_router/workload_report.json`.

## Results

44 requests. Task mix as classified by the scorer: summarisation 16, classification 9, reasoning 8, extraction 6, code 5. Complexity scores ranged from 0.094 to 0.802 with a median of 0.356.

Tier distribution and cost, with the baselines measured by running the same workload through a fixed policy (hard pins stripped, since a pin is an escape hatch that exists under any policy):

| strategy | small | medium | large | cost | vs always-large |
|---|---|---|---|---|---|
| routed | 15 | 14 | 15 | $0.008106 | **55.7% saved** |
| always-small | 44 | 0 | 0 | $0.000781 | 95.7% |
| always-large | 0 | 0 | 44 | $0.018306 | baseline |

Routing saved $0.010200 of $0.018306 and cost 10.4x the always-small bill. Cost by tier on the routed run: small 15 calls $0.000266, medium 14 calls $0.000878, large 15 calls $0.006963. So **34% of the calls are 86% of the spend**, which is the whole argument for routing in one number.

Escalation: 29 of 44 answers were served below the top tier and went to the judge, 20 of them escalated, an escalation rate of **69.0%**. That pushed the bill from $0.008106 to $0.012934, still 29.3% below always-large. A 69 percent escalation rate would be a failed configuration in production, and here it is a property of the offline reference judge (see Limits) rather than of the router. The number is reported because it is the number that decides whether escalation is worth running at all: above roughly 30 percent, paying for a cheap call plus a judge call plus the expensive call costs more than routing straight to the expensive tier.

Fallback, with the large tier returning HTTP 503 on every call: **44 of 44 requests served, 0 failures, 0 refusals**. 15 fallback events, of which 2 were real failed calls (each 2 retries, so 4 upstream calls total) and 13 were requests that skipped the dead tier entirely because the breaker was open. The large tier's breaker ended in the `open` state and traffic redistributed to 15 small and 29 medium.

Budget guard, replaying acme's 18 requests three times as one day against a $0.003000 daily budget: **26 allowed, 17 downgraded, 11 refused**. Spend stopped at $0.003007. The $0.000007 overshoot is the documented cost of checking the budget before a call whose completion length is not yet known.

## Limits

All three tiers are the same `llmkit.EchoLLM` behind different names, so token counts are identical across tiers and the cost comparison isolates price alone. That is the right control for measuring the routing and accounting logic and it is not a claim about answer quality: with real models, the cheap tier would produce different and usually shorter completions, which moves the savings in both directions.

The escalation rate of 69 percent is a measurement of the reference judge, not of the router. `EchoLLM` draws its rubric integers from a seeded uniform distribution inside the schema bounds, so roughly two thirds of answers score below a 0.6 threshold regardless of quality. With a real judge the escalation rate would be a function of the band edges, and tuning those edges against the observed escalation rate is exactly the feedback loop this design is built for.

The complexity weights and band edges (0.30 and 0.60) are hand set. They encode defensible claims ("reasoning is harder than classification") that have not been fitted to anything. The right calibration is a labelled sample of production traffic where the cheap tier's answers were graded, then fit the weights to predict where the cheap tier failed. That is not possible in a repo with no production traffic, and inventing tuned numbers would be worse than shipping legible ones.

The task classifier is regex based and abstains more often than it looks: **7 of the 44 requests matched no task pattern at all** and were routed on length and ambiguity alone. That is a measured weakness, not a hidden one, and it is why `has_explicit_task_signal` is part of the public API. The cheapest real fix is not a better regex, it is letting the caller declare the task type, which the API already supports.

The cost ledger is per process. A real deployment needs a shared store, because a per-tenant daily cap enforced in memory lets a tenant spend the cap once per replica. The accounting logic is identical either way.

Circuit breaker state is also per process and per tier, not per tier and per replica coordinated. That is usually acceptable, since each replica independently learns the upstream is down within two requests.
