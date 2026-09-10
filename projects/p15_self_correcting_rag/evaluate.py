"""Four-outcome evaluation, against single-shot RAG on the same questions.

The four outcomes
-----------------
**correct**   answered right on the first retrieval, no escalation needed.
**corrected** answered right, but only after the loop escalated. This is the
              column that justifies the whole project: every question in it is
              one that single-shot RAG would have got wrong.
**abstained** declined to answer. Right on an unanswerable question, a miss on an
              answerable one, so it is reported per question set and never
              aggregated across them.
**wrong**     answered, and the answer was not right. The only outcome that is
              bad in every context.

Splitting "correct" from "corrected" matters because a single accuracy number
hides the mechanism. Two systems can both score 0.8 where one is a good retriever
and the other is a mediocre retriever with a working repair loop, and those two
systems behave very differently when the corpus changes.

Correctness is judged on citations, not on answer text
------------------------------------------------------
An answer counts as correct when the gold document appears among the documents it
cited. Grading the prose would be grading `EchoLLM`, which is a rule engine, and
the resulting number would say nothing about a real deployment. Citation-level
grading measures the part of the system that is real code: retrieval, escalation,
critique and validation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from projects.p01_rag_pipeline.pipeline import RagPipeline

from .agent import AgentResult, SelfCorrectingRAG

CORRECT, CORRECTED, ABSTAINED, WRONG = "correct", "corrected", "abstained", "wrong"
OUTCOMES = (CORRECT, CORRECTED, ABSTAINED, WRONG)


@dataclass
class Outcome:
    question: str
    expected: str
    outcome: str
    steps: int
    winning_step: int
    confidence: float
    cited: List[str] = field(default_factory=list)
    escalations: List[str] = field(default_factory=list)
    kind: str = ""


@dataclass
class Confusion:
    label: str
    counts: Dict[str, int] = field(default_factory=lambda: {o: 0 for o in OUTCOMES})
    n: int = 0

    def add(self, outcome: str) -> None:
        self.counts[outcome] += 1
        self.n += 1

    def rate(self, outcome: str) -> float:
        return self.counts[outcome] / self.n if self.n else 0.0


def classify_agent(result: AgentResult, gold: Dict[str, str]) -> str:
    expects_abstain = gold.get("expect") == "abstain"
    if result.abstained:
        return ABSTAINED
    if expects_abstain:
        return WRONG  # answered a question with no answer
    if gold["doc_id"] in result.cited_doc_ids:
        return CORRECTED if result.corrected else CORRECT
    return WRONG


def classify_single_shot(answer, gold: Dict[str, str]) -> str:
    """Single-shot RAG has no repair loop, so it can never land in `corrected`."""
    expects_abstain = gold.get("expect") == "abstain"
    if answer.refused:
        return ABSTAINED
    if expects_abstain:
        return WRONG
    return CORRECT if gold["doc_id"] in answer.cited_doc_ids else WRONG


def evaluate_agent(
    agent: SelfCorrectingRAG,
    questions: Sequence[Dict[str, str]],
    label: str = "self-correcting",
) -> tuple:
    confusion = Confusion(label)
    rows: List[Outcome] = []
    for gold in questions:
        result = agent.answer(gold["question"])
        outcome = classify_agent(result, gold)
        confusion.add(outcome)
        rows.append(
            Outcome(
                question=gold["question"], expected=gold.get("expect", "answer"),
                outcome=outcome, steps=result.step_count,
                winning_step=result.winning_step, confidence=result.confidence,
                cited=result.cited_doc_ids, escalations=result.escalations,
                kind=gold.get("kind", ""),
            )
        )
    return confusion, rows


def evaluate_single_shot(
    pipeline: RagPipeline,
    questions: Sequence[Dict[str, str]],
    label: str = "single-shot (p01)",
) -> tuple:
    confusion = Confusion(label)
    rows: List[Outcome] = []
    for gold in questions:
        answer = pipeline.ask(gold["question"])
        outcome = classify_single_shot(answer, gold)
        confusion.add(outcome)
        rows.append(
            Outcome(
                question=gold["question"], expected=gold.get("expect", "answer"),
                outcome=outcome, steps=1, winning_step=1, confidence=answer.grounding,
                cited=answer.cited_doc_ids, kind=gold.get("kind", ""),
            )
        )
    return confusion, rows


def format_confusion(confusions: Sequence[Confusion]) -> str:
    header = f"{'system':<22}{'n':>4}" + "".join(f"{o:>12}" for o in OUTCOMES)
    lines = [header, "-" * len(header)]
    for confusion in confusions:
        cells = "".join(
            f"{confusion.counts[o]:>7} {confusion.rate(o):>4.2f}" for o in OUTCOMES
        )
        lines.append(f"{confusion.label:<22}{confusion.n:>4}{cells}")
    return "\n".join(lines)


def escalation_histogram(rows: Sequence[Outcome]) -> Dict[int, int]:
    """How many questions were answered from each rung's evidence.

    Keyed on the winning step rather than the number of steps run, because the
    interesting quantity is which rung produced the evidence that was used, not
    how far the loop walked before giving up on improving.
    """
    histogram: Dict[int, int] = {}
    for row in rows:
        histogram[row.winning_step] = histogram.get(row.winning_step, 0) + 1
    return dict(sorted(histogram.items()))


def rows_with_outcome(rows: Sequence[Outcome], outcome: str) -> List[Outcome]:
    return [r for r in rows if r.outcome == outcome]


@dataclass
class CalibrationRow:
    threshold: float
    answered_correctly: int
    answerable_abstained: int
    answerable_wrong: int
    adversarial_abstained: int
    adversarial_answered: int

    @property
    def score(self) -> int:
        """Correct decisions: right answers plus justified refusals."""
        return self.answered_correctly + self.adversarial_abstained


def calibrate_abstention(
    agent: SelfCorrectingRAG,
    answerable: Sequence[Dict[str, str]],
    unanswerable: Sequence[Dict[str, str]],
    thresholds: Sequence[float] = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65),
) -> List[CalibrationRow]:
    """Sweep the abstention threshold over both labelled sets.

    The abstention floor is the single most consequential number in this system
    and there is no principled way to pick it from first principles: it depends on
    the corpus, the retriever and the relative cost of a wrong answer against a
    refusal. So it is swept against labelled data and the chosen value is shown
    with the sweep that produced it, rather than appearing in a constructor as a
    round number with no justification.

    Each question is run once with abstention effectively disabled and the
    resulting confidences are re-thresholded. Re-running the agent per threshold
    would produce the same confidences at ten times the cost, because the
    threshold only affects the final decision, not retrieval or critique.
    """
    probe = SelfCorrectingRAG(
        pipeline=agent.pipeline, critic=agent.critic, rewriter=agent.rewriter,
        search_tool=agent.search_tool, llm=agent.llm, max_steps=agent.max_steps,
        accept_above=agent.accept_above, abstain_below=0.0, k=agent.k,
        tracer=agent.tracer,
    )
    answerable_runs = [(probe.answer(g["question"]), g) for g in answerable]
    unanswerable_runs = [probe.answer(g["question"]) for g in unanswerable]

    rows: List[CalibrationRow] = []
    for threshold in thresholds:
        correct = abstained = wrong = 0
        for result, gold in answerable_runs:
            if result.confidence < threshold:
                abstained += 1
            elif gold["doc_id"] in result.cited_doc_ids:
                correct += 1
            else:
                wrong += 1
        refused = sum(1 for r in unanswerable_runs if r.confidence < threshold)
        rows.append(
            CalibrationRow(
                threshold=threshold, answered_correctly=correct,
                answerable_abstained=abstained, answerable_wrong=wrong,
                adversarial_abstained=refused,
                adversarial_answered=len(unanswerable_runs) - refused,
            )
        )
    return rows


@dataclass
class AcceptanceRow:
    accept_above: float
    correct: int
    corrected: int
    wrong: int
    abstained: int
    steps_run: int
    judge_calls: int

    @property
    def answered_right(self) -> int:
        return self.correct + self.corrected


def sweep_acceptance(
    build_agent,
    answerable: Sequence[Dict[str, str]],
    thresholds: Sequence[float] = (0.50, 0.60, 0.70, 0.80),
) -> List[AcceptanceRow]:
    """Sweep the early-stopping threshold, which is purely an accuracy/cost dial.

    `accept_above` does not change what the agent is capable of finding; it
    changes how early it stops looking. Raising it buys accuracy with model calls
    and latency. Sweeping it is the only way to know the exchange rate, and the
    exchange rate is what a team actually needs to decide the value.

    `build_agent` is a callable taking the threshold and returning a fresh agent,
    so each row gets its own judge-call counter.
    """
    rows: List[AcceptanceRow] = []
    for threshold in thresholds:
        agent = build_agent(threshold)
        confusion, outcomes = evaluate_agent(agent, answerable)
        rows.append(
            AcceptanceRow(
                accept_above=threshold,
                correct=confusion.counts[CORRECT],
                corrected=confusion.counts[CORRECTED],
                wrong=confusion.counts[WRONG],
                abstained=confusion.counts[ABSTAINED],
                steps_run=sum(o.steps for o in outcomes),
                judge_calls=agent.critic.judge_calls,
            )
        )
    return rows


def format_acceptance(rows: Sequence[AcceptanceRow]) -> str:
    header = (f"{'accept':>8}{'correct':>9}{'corrected':>11}{'wrong':>7}"
              f"{'abstained':>11}{'right':>7}{'steps':>7}{'judge calls':>13}")
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row.accept_above:>8.2f}{row.correct:>9}{row.corrected:>11}"
            f"{row.wrong:>7}{row.abstained:>11}{row.answered_right:>7}"
            f"{row.steps_run:>7}{row.judge_calls:>13}"
        )
    return "\n".join(lines)


def format_calibration(rows: Sequence[CalibrationRow]) -> str:
    header = (f"{'floor':>7}{'correct':>9}{'abstained':>11}{'wrong':>7}"
              f"{'adv refused':>13}{'adv answered':>14}{'score':>7}")
    lines = [header, "-" * len(header)]
    best = max(r.score for r in rows) if rows else 0
    for row in rows:
        marker = "  <- best" if row.score == best else ""
        lines.append(
            f"{row.threshold:>7.2f}{row.answered_correctly:>9}"
            f"{row.answerable_abstained:>11}{row.answerable_wrong:>7}"
            f"{row.adversarial_abstained:>13}{row.adversarial_answered:>14}"
            f"{row.score:>7}{marker}"
        )
    return "\n".join(lines)
