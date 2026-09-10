# 12. Prompt Versioning and A/B System

A registry for prompts and configs with content-addressed versions, environment
labels, deterministic traffic splitting, per-arm outcome tracking, a statistical
promotion gate and one-call rollback.

## The problem

Prompts are code that ships without any of the machinery code gets.

A prompt is edited in a dashboard on Friday. On Monday, quality is down. Nobody
can say which text was live on Friday, because the version label was assigned by
hand and someone edited in place under the same label. Nobody can say what the
change was worth either, because the A/B that justified it re-randomised users on
every request, so half the "treatment" group also saw the control. And nobody can
roll it back quickly, because staging and prod hold separate copies of the text
that have quietly drifted apart.

Three specific failures, three specific mechanisms:

- An id that is not derived from the content is an id that can lie.
- A split that is not a pure function of the unit id measures nothing.
- A promotion decision without a sample floor is a decision made by whoever
  refreshes the dashboard first.

## What this builds

- `templates.py` - `PromptVersion`, frozen, whose `version_id` is a SHA-256 hash
  of the name, template, declared variables and config. Rendering raises on a
  missing or an unexpected variable.
- `registry.py` - versions, environment labels as pointers, an audit trail of
  every label move, one-call rollback, and JSON persistence that refuses to load
  a file whose content no longer matches its ids.
- `splitter.py` - deterministic weighted assignment from
  `sha256(salt:experiment:unit)`, with no assignment table.
- `outcomes.py` - per-request outcomes and per-arm rollups: success rate, p50 and
  p95 latency, tokens, cost, sample count.
- `stats.py` - the two-proportion z-test, two normal CDF implementations, the
  Wilson interval and a sample-size calculator, all in pure Python.
- `gate.py` - the promotion gate: sample floor, significance, minimum lift, with
  optional cost and latency ceilings.
- `workload.py` - the real workload: two variants answering `llmkit.corpus` gold
  questions through `EchoLLM`, judged on whether the output contains the gold
  phrase.

## Architecture

```
  register(name, template, variables, config)
        |
        v
  version_id = sha256(name + template + variables + config)[:12]
        |
        +--> versions{}          immutable, content addressed
        |
  set_label(prod, version_id, actor, reason)
        |
        +--> labels{prompt -> {dev, staging, prod} -> version_id}
        +--> audit[]  (who, when, from, to)  --> rollback() reads this
        |
  request(unit_id)
        |
        v
  arm = cumulative_weights(sha256(salt:experiment:unit) / BUCKETS)
        |
        v
  render -> EchoLLM -> judge -> Outcome(success, latency, tokens, cost)
        |
        v
  PromotionGate: samples >= floor  AND  p < alpha  AND  lift >= bar
        |                                    (optional cost/latency ceilings)
        +-- promote: set_label(prod, challenger)
        +-- refuse:  prod is untouched, the reason is recorded
```

## Design decisions

**The version id is the content hash, and config is inside the hash.**
A hand-assigned "v3" that someone tweaked in place is worse than no version at
all: it makes the log look trustworthy while it lies. Config is included because
the same prompt at temperature 0.2 and at 0.9 are different systems, and an
experiment that treats them as one version cannot explain its own results. A
side-effect worth having: registration becomes idempotent, so a deploy that
registers prompts on every boot does not fill the registry with duplicates.

**Rendering fails on an unexpected variable, not just a missing one.**
The unexpected case is the one people argue about, and it is the one that bites.
Passing a variable the template does not use is almost always a rename applied on
one side only, and the symptom is a prompt that silently stops including the
customer's name. Failing at render time turns a quality regression into a stack
trace.

**Labels are pointers and rollback reads the audit trail.**
Copying prompt text between environments lets staging and prod drift into
near-identical strings that differ by a trailing space nobody can see. Rollback
walks the audit trail rather than a "previous version" field, because a
previous-pointer is a second source of truth that goes stale the moment a label
moves twice.

**Assignment is a hash, not a stored table.**
A table would also be sticky, at the cost of a read on the request path and a
consistency problem behind it. Hashing gives stickiness with no state, survives a
process restart, and lets offline analysis recompute assignment from a log, which
is what makes an A/B result auditable. The experiment id is inside the hash so
concurrent experiments do not correlate, and a salt allows deliberate
re-randomisation on relaunch without renaming the experiment.

**The gate has a sample floor as well as a p-value.**
This is the decision that does the most work in practice. In the demo, the
40-request checkpoint already shows p = 0.031, below the 0.05 threshold, with an
apparent lift of 33.8 points. The full sample settles at 13.8 points. The p-value
alone would have shipped a number that was more than double the real effect. The
floor is a commitment made before the numbers arrive.

**Two-sided test, and a minimum absolute lift on top of significance.**
A one-sided test is tempting because you only want to detect a winner, and it
doubles the false-positive rate in the direction the experimenter already
believes. Significance on a large sample can also certify a real but pointless
difference, so a lift below the deploy bar is refused even when p is tiny.

**Cost and latency are tracked always and enforced optionally.**
"Better and slower" is a judgement call that belongs to a human, so the default
gate reports the ratios without blocking. The demo shows the same winning result
held by a cost ceiling, because a quality gate that ignores cost approves changes
a finance review then reverses.

**Pure-Python statistics.**
The test is about twenty lines. Importing a numerical stack to obtain a normal
CDF hides the one part of a promotion decision a reviewer should be able to read
and disagree with. Both a `math.erf` implementation and the Abramowitz and Stegun
26.2.17 polynomial are provided, and a test asserts they agree.

## Running it

```bash
python3 projects/p12_prompt_registry/demo.py
python3 -m pytest projects/p12_prompt_registry -q
```

The demo registers both variants, shows that an edit produces a new id, shows
rendering failing on missing and unexpected variables, sets the three
environment labels, verifies stickiness, runs 600 requests through the splitter
against `EchoLLM`, evaluates the gate at 40 requests and again at 600, evaluates
it once more under a cost ceiling, promotes, rolls back, prints the audit trail,
round-trips the registry through JSON and compares the two normal CDFs.

## Results

All figures are printed by `demo.py`. The workload is 600 units answering gold
questions from `llmkit.corpus` through `llmkit.EchoLLM`; a request counts as a
success when the model's output contains that question's gold phrase. Costs are
priced at `llmkit`'s `small` tier so the arithmetic is real; the provider the demo
actually runs on is free.

**The experiment.** Control shows the single best-matching document. Challenger
shows three tagged evidence blocks and asks the model to quote and cite the one
it used.

| Arm | Samples | Successes | Success rate | Mean tokens | Cost USD |
|---|---|---|---|---|---|
| control | 303 | 97 | 32.0% | 136.3 | 0.0095 |
| challenger | 297 | 136 | 45.8% | 326.1 | 0.0209 |

The demo also prints p50 and p95 latency per arm. Those are omitted here on
purpose: they are wall-clock timings of a local rule engine, they land in the
hundredths of a millisecond, and they change between runs, so quoting them as a
result would be quoting noise. The plumbing that records and rolls them up is
real and the cost ceiling below uses the token and cost numbers, which are
deterministic.

**Stickiness.** 600 of 600 units land in the same arm when asked twice, and 600
of 600 still match after rebuilding the splitter from scratch, with no assignment
table anywhere. A different experiment id reshuffles the same population: 299 of
600 assignments coincide, 49.8 percent against the 50 percent expected by chance.

**Split evenness.** 50.5 percent control and 49.5 percent challenger against an
intended 50/50, a largest deviation of 0.50 points over 600 units. Not exactly
even, which is the point: the correct alert on a hash split is a deviation
threshold, not equality.

**The gate at 40 requests.** control 6/22, challenger 11/18, apparent lift +33.8
points, z = 2.154, p = 0.03126. Significant, and refused: the gate requires 200
samples per arm. The sample-size calculator puts the requirement for this effect
size at 1312 per arm.

**The gate at 600 requests.** control 97/303 = 32.0 percent, 95 percent Wilson
interval [27.0, 37.5]; challenger 136/297 = 45.8 percent, interval [40.2, 51.5].
Absolute lift +13.8 points, z = 3.462, p = 0.00054. Promoted.

**The same numbers under a cost ceiling.** The challenger sends 2.39 times the
tokens and costs 2.20 times as much. With `max_cost_ratio=1.5` the gate holds the
same statistically significant win.

**Rollback.** One call moved prod back to the control version, and the audit
trail ends with five entries covering the three initial label sets, the
gate-driven promotion with its p-value in the reason field, and the rollback with
its actor.

**Persistence.** The registry round-trips through JSON with 3 versions and 5
audit entries and the prod pointer intact. A hand-edited file whose template no
longer hashes to its stored id is rejected on load, which is tested.

**Normal CDF agreement.** Largest gap between `math.erf` and Abramowitz and
Stegun 26.2.17 across the sampled points was 7.00e-08 at z = 1.96, inside the
7.5e-08 bound the approximation claims.

**Tests: 35**, covering content addressing including config, frozen versions,
both rendering failure modes, both template declaration errors, label moves,
rollback and its refusal case, audit contents, tamper detection on load,
stickiness across instances, weighted splits, experiment independence, salt
re-randomisation, a hand-worked z-test, degenerate inputs, Wilson bounds, and
five distinct gate refusal paths against one promotion path.

## Limits

- **The success judge is a substring check.** A gold phrase in the output is a
  usable proxy on this corpus and it is not a quality metric. A real deployment
  scores outcomes with human labels, an LLM judge or a downstream business event,
  which is what project 06 in this repo is for. The registry does not care where
  the boolean comes from.
- **`EchoLLM` is a deterministic rule engine.** The measured 13.8-point gap is a
  real difference in how the reference model responds to one evidence block
  versus three, and it is not evidence about how a frontier model responds. What
  the demo proves is that the pipeline measures a real difference and gates on
  it, not that tagged evidence is worth 13.8 points.
- **A z-test on one metric is not an experiment framework.** There is no
  sequential testing correction, so evaluating the gate repeatedly as data
  arrives inflates the false-positive rate; the sample floor is a blunt guard
  against exactly that, not a proper solution. There is no multiple-comparison
  correction across arms, and no guardrail metrics beyond cost and latency.
- **The split is stateless, which cuts both ways.** Changing the arm weights
  reassigns some already-exposed units, because assignment is a function of the
  weights. Mid-experiment reweighting therefore contaminates the sample, and this
  implementation does not detect that.
- **JSON file storage.** Single writer, no locking, no concurrent access story.
  The interface is `load`, `save` and an audit list, so a real datastore is a
  contained change, but that change brings the concurrency questions this
  deliberately avoids.
