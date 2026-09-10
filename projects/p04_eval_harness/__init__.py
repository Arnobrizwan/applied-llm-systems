"""Project 04: LLM evaluation harness with a CI quality gate.

Public API, deliberately small so the other projects in this repo can depend on
it without depending on its internals. Project 06 uses `RubricJudge` as the
escalation judge in its routing gateway.

    from projects.p04_eval_harness import EvalCase, EvalHarness, RubricJudge

    harness = EvalHarness(judge=RubricJudge(), judge_policy="on_failure")
    report = harness.run(my_system, build_dataset(), name="my-system")
    result = CIGate().check(load_report("baseline.json"), report.to_dict())
    sys.exit(result.exit_code)

`my_system` is any callable taking an `EvalCase` and returning a string.
"""
from .dataset import (
    EvalCase, adversarial_cases, build_dataset, gold_cases, load_jsonl, save_jsonl, slice_by,
)
from .gate import Breach, CIGate, GateResult, write_baseline
from .harness import CaseResult, EvalHarness, EvalReport, load_report
from .judges import (
    PairwiseJudge, PairwiseSummary, PairwiseVerdict, RubricJudge, RubricVerdict,
    aggregate_rubric, verbosity_bias,
)
from .scorers import (
    DETERMINISTIC_SCORERS, ScoreResult, contains, exact_match, extract_json,
    json_schema, regex, score_case, token_f1, validate_schema,
)
from .stats import bootstrap_ci, mean, pearson_r, summarise

__all__ = [
    "EvalCase", "build_dataset", "gold_cases", "adversarial_cases", "load_jsonl",
    "save_jsonl", "slice_by",
    "EvalHarness", "EvalReport", "CaseResult", "load_report",
    "DETERMINISTIC_SCORERS", "ScoreResult", "score_case", "exact_match", "contains",
    "regex", "token_f1", "json_schema", "validate_schema", "extract_json",
    "RubricJudge", "RubricVerdict", "PairwiseJudge", "PairwiseVerdict",
    "PairwiseSummary", "aggregate_rubric", "verbosity_bias",
    "bootstrap_ci", "summarise", "mean", "pearson_r",
    "CIGate", "GateResult", "Breach", "write_baseline",
]
