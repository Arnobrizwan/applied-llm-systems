# 04. LLM Evaluation Harness

**Run it live: [https://applied-llm-systems.vercel.app/s/eval-harness](https://applied-llm-systems.vercel.app/s/eval-harness)** - the hosted page runs this code and shows the real output.

A golden dataset, deterministic scorers, an LLM judge with its bias controls, and a CI gate that blocks a merge on a quality regression it can actually defend.

## The problem

Two failures, and they are opposites.

The first is having no gate. A prompt edit ships, retrieval quietly gets worse for one class of question, and nobody notices until a customer does. There is no artefact anyone can point at to say the change was safe.

The second is having a gate nobody trusts. Someone wires up "fail if the score drops 2 points" against a 40 case eval set. One flipped case is 2.5 points, so the gate fires on unrelated pull requests, people learn to re-run CI until it goes green, and within a month the gate is disabled or ignored. A gate that fires on noise protects nothing and costs everyone time.

This project is built for the second problem as much as the first. A regression blocks a merge only when the drop is larger than the threshold **and** the evidence supports calling it a regression.

## What this builds

1. **Dataset** (`dataset.py`) - a typed `EvalCase` with input, expected, tags, difficulty and the scorers it wants, persisted as JSONL so adding a case is a one line diff. The 15 gold questions come from `llmkit.corpus`; 8 adversarial cases are hand written and shipped in `data/adversarial.jsonl`.
2. **Deterministic scorers** (`scorers.py`) - exact match, contains, regex, JSON schema validity (with a standard library subset validator) and token F1. These run first because they are free and unambiguous.
3. **LLM judges** (`judges.py`) - a rubric judge scoring faithfulness, relevance and completeness, and a pairwise judge that compares two systems in both orders. Both go through `llmkit.EchoLLM`'s `json_schema` path.
4. **Statistics** (`stats.py`) - a percentile bootstrap and a correlation coefficient, in pure Python.
5. **Harness** (`harness.py`) - runs a system over the dataset, produces a JSON report with a confidence interval on every metric plus per slice pass rates.
6. **CI gate** (`gate.py`) - compares a report against a committed baseline, applies per metric thresholds and hard floors, writes a machine readable report and exits non-zero.
7. **Systems under test** (`systems.py`) - a real hybrid retrieval QA system over the Meridian corpus, plus two worse variants.

## Architecture

```
   data/adversarial.jsonl        llmkit.corpus.gold_questions()
              |                              |
              +--------------+---------------+
                             v
                      build_dataset()  ->  23 EvalCase
                             |
                             v
   system(case) -> answer -> DETERMINISTIC SCORERS  (free, run first)
                             |          exact | contains | regex | json_schema | token_f1
                             |
                             +--> settled? -----> yes: skip the judge
                             |
                             v no
                       RUBRIC JUDGE (EchoLLM, json_schema)
                       faithfulness / relevance / completeness
                             |
                             v
                   AGGREGATE + BOOTSTRAP CI (2000 resamples)
                             |
                             v
            EvalReport JSON  ->  CIGate  ->  exit 0 / exit 1
                                    ^
                                    |
                       data/baseline.json (committed)

   PairwiseJudge (separate path): system A vs system B, asked twice with the
   order swapped; disagreement between the two orders is recorded as a tie and
   counted as a position bias event.
```

## Design decisions

**A regression needs a threshold breach and statistical support.** The gate blocks only when the drop exceeds the metric threshold and the baseline mean falls outside the candidate run's bootstrap interval. Rejected: a plain threshold on the point estimate, which is what most teams write first and which fires constantly at the sample sizes eval sets actually have. Measured here: the noisy variant's pass rate drop of 0.087 is 4x the 0.02 threshold and is correctly not blocked, while the degraded variant's 0.522 drop is blocked.

**Two escape hatches, because the significance rule can be abused.** `min_values` is a hard floor with no significance test, for metrics that are contractual rather than statistical (schema validity has to be 1.000, and "the drop was not significant on n=1" is not an answer). And a metric present in the baseline but missing from the candidate is an automatic failure, because deleting the failing metric is the cheapest way to make any gate green.

**Deterministic checks run before the judge.** Rejected: judging every case, which is what most harnesses do. On the baseline run, 16 of 23 cases were settled by string operations, so switching the judge policy from `all` to `on_failure` cuts the judge bill from 23 calls to 7 for the same verdict. That ratio is what makes a judged eval affordable per commit rather than per release.

**Pairwise judging always asks twice with the order swapped.** Rejected: randomising the order per case, which halves the cost but turns a measurable bias into unmeasurable variance. Swapping makes the flip rate a reported number, and any case where the two orders disagree is recorded as a tie rather than a win.

**Verbosity bias is measured, not asserted.** The harness correlates judge score against answer token length across the run and flags the run when the absolute correlation exceeds 0.3. On a set of short factual questions, a strong positive correlation means the judge is measuring length.

**Sentence level evidence, chosen by measurement.** The first version retrieved whole documents. `llmkit.EchoLLM` answers extractively from the first sentence of each evidence block, so the answer bearing sentence in the middle of a document never reached the answer and the pass rate was capped at 0.348. Sentence level chunks lifted it to 0.696 with no other change. This is a property of the offline reference model rather than of retrieval, and the README says so rather than presenting 0.696 as a retrieval result.

## Running it

```bash
python3 projects/p04_eval_harness/demo.py
python3 -m pytest projects/p04_eval_harness -q          # 26 tests

# the gate as CI would invoke it
python3 projects/p04_eval_harness/gate.py \
  --baseline projects/p04_eval_harness/data/baseline.json \
  --candidate artifacts/p04_eval_harness/report_degraded.json \
  --out artifacts/p04_eval_harness/gate_report.json
echo $?    # 1
```

The demo prints seven sections: dataset composition, three systems scored side by side, a retrieval depth sweep, the rubric judge with its bias diagnostics, the pairwise judge with its position bias rate, a passing gate and a failing gate. It writes per system reports and the gate report to `artifacts/p04_eval_harness/`.

## Results

All numbers below are printed by `demo.py` on the 23 case dataset (15 gold, 8 adversarial) using `llmkit.EchoLLM` as both the answering model and the judge.

Three systems, deterministic scorers only:

| system | pass rate | 95% CI | contains | token F1 |
|---|---|---|---|---|
| baseline (k=2, abstention on) | 0.696 | [0.522, 0.870] | 0.700 | 0.494 |
| noisy (k=3) | 0.609 | [0.391, 0.783] | 0.650 | 0.407 |
| degraded (no abstention, 6 word answers) | 0.174 | [0.043, 0.348] | 0.100 | 0.180 |

Baseline slices: adversarial 0.875 (n=8), gold 0.600 (n=15), easy 0.700, medium 0.625, hard 0.800.

Retrieval depth sweep, pass rate by number of evidence sentences: k=1 0.435, k=2 0.696, k=3 0.609, k=4 0.565. More context is worse here, which is why the baseline is k=2.

Judge, run over all 23 cases: faithfulness 0.478, relevance 0.467, completeness 0.391, overall 0.446 with 95% CI [0.355, 0.533]. 23 judge calls, 0 parse errors. 16 of 23 cases were settled by the cheap deterministic checks, so the `on_failure` policy would have made 7 judge calls instead of 23. Verbosity bias r = -0.094, not flagged.

Pairwise, baseline against degraded, 46 judge calls over 23 cases: baseline wins 1, degraded wins 2, ties 20. The verdict flipped when the order was swapped on **18 of 23 cases (78.3%)**. Read that as a statement about the offline reference judge, which picks its verdict from a hash of the prompt and therefore has no stable preference at all. The point stands either way: a single order pairwise judge would have reported a winner on 18 cases where the winner was an artefact of the ordering, and the swap control caught every one of them.

Gate, both directions, against the committed baseline with a 0.02 threshold:

| candidate | pass rate delta | verdict | exit code |
|---|---|---|---|
| noisy | -0.087 | PASS with 3 noise warnings | 0 |
| degraded | -0.522 | FAIL, 3 blocking breaches | 1 |

The failing run blocks on contains (-0.600), pass_rate (-0.522) and token_f1 (-0.314). It does not block on regex, which dropped 0.500, because n=2 and the interval spans [0.000, 1.000]. That is the gate refusing to make a claim two cases cannot support, and it is a real limitation rather than a feature: the fix is more cases, not a looser rule.

## Limits

The judge numbers measure `llmkit.EchoLLM`, which is a deterministic rule engine, not a language model. Its rubric scores come from a seeded random draw inside the schema bounds and its pairwise verdicts come from a hash of the prompt. The judge plumbing, the bias controls and the aggregation are real and would be unchanged with a frontier model behind them; the judge score values themselves would not.

The gate's significance test is one sample: it asks whether the baseline mean sits inside the candidate's interval. The correct test for two runs over the same cases is a paired bootstrap on the per case differences, which is more sensitive. It is not implemented because the baseline file deliberately stores only summary metrics, and it errs toward under-reporting regressions rather than inventing them.

23 cases is small. The intervals are wide because they honestly reflect that. A real gate wants a few hundred cases per protected slice, at which point the same code reports far tighter intervals and the significance rule stops rescuing changes it should be blocking.

The abstention rule in `systems.py` is a vocabulary heuristic, not a calibrated confidence estimate. On this corpus the unanswerable case scores 0.67 unknown terms against a maximum of 0.40 across all 15 gold questions, so the 0.60 threshold separates them, but that margin is a property of one corpus and would need re-measuring on any other.

With a real model and a budget, the changes are: swap `EchoLLM` for a judge from a different model family than the system under test to reduce self-preference bias, calibrate the rubric against human labels on a few hundred cases before trusting it, and run the judge with `judge_policy="on_failure"` so per commit cost tracks the failure count.
