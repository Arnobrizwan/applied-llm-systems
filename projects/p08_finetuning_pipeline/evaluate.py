"""Metrics and the prompt-only baseline.

Accuracy alone is not enough on this dataset. The label distribution is
uneven after filtering (measured, and printed by the demo), so a classifier
that ignores the smallest class can still post a respectable accuracy. Macro-F1
averages F1 per class with equal weight, so ignoring a small class is
expensive. Both are reported and macro-F1 is the one to argue with.

The confusion matrix is here because the two aggregate numbers cannot tell you
*which* classes a model confuses, and on an intent classifier that is the only
thing a product owner cares about: routing a cancellation to billing is a
different business problem from routing a bug report to feature requests.

The prompt-only baseline is a genuine no-training system, not a placeholder.
Each intent is supplied to EchoLLM as a numbered evidence block and the model
is asked which one applies. EchoLLM's grounded path selects the block with the
strongest lexical overlap with the query and cites it, so the prediction is a
real decision made by the provider rather than a random draw. It is a lexical
decision, which is exactly what a small no-training baseline should be, and it
is what the two trained systems have to beat.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from llmkit import EchoLLM, LLMProvider

from .data import INTENT_DESCRIPTIONS, INTENTS, Example

_CITATION_RE = re.compile(r"\[S(\d+)\]")


@dataclass
class Evaluation:
    name: str
    accuracy: float = 0.0
    macro_f1: float = 0.0
    per_class: Dict[str, Dict[str, float]] = field(default_factory=dict)
    confusion: List[List[int]] = field(default_factory=list)
    unparsed: int = 0
    n: int = 0

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "n": self.n,
            "accuracy": round(self.accuracy, 4),
            "macro_f1": round(self.macro_f1, 4),
            "unparsed": self.unparsed,
        }


def evaluate_predictions(name: str, y_true: Sequence[int], y_pred: Sequence[int],
                         unparsed: int = 0) -> Evaluation:
    """Accuracy, per-class precision/recall/F1, macro-F1 and a confusion matrix."""
    n_classes = len(INTENTS)
    confusion = [[0] * n_classes for _ in range(n_classes)]
    for t, p in zip(y_true, y_pred):
        if 0 <= p < n_classes:
            confusion[t][p] += 1
    correct = sum(confusion[c][c] for c in range(n_classes))
    total = len(y_true)

    per_class: Dict[str, Dict[str, float]] = {}
    f1s: List[float] = []
    for c, label in enumerate(INTENTS):
        tp = confusion[c][c]
        fp = sum(confusion[r][c] for r in range(n_classes)) - tp
        fn = sum(confusion[c]) - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        # A class with no test examples is excluded rather than counted as a
        # zero, which would silently drag macro-F1 down for a class that was
        # never given a chance. The stratified split makes this unlikely, but
        # a metric that lies when the split changes is not worth shipping.
        if sum(confusion[c]) > 0:
            f1s.append(f1)
        per_class[label] = {
            "support": sum(confusion[c]),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    return Evaluation(
        name=name,
        accuracy=correct / total if total else 0.0,
        macro_f1=sum(f1s) / len(f1s) if f1s else 0.0,
        per_class=per_class,
        confusion=confusion,
        unparsed=unparsed,
        n=total,
    )


def build_classification_prompt(text: str) -> List[Dict[str, str]]:
    """Every candidate intent as a numbered evidence block, plus the message.

    The question goes first and the candidate blocks second, which looks
    backwards. It is deliberate. EchoLLM finds evidence blocks by scanning the
    concatenation of every message with a regex whose last block runs to the
    end of the text, so any prose placed after the final block is absorbed
    into it. With the question last, block S6 would silently contain the
    customer message itself, score a perfect lexical overlap against it every
    single time, and the baseline would predict `request_feature` for all 48
    test rows. That was the first result this function produced. Reading
    llmkit/providers.py rather than trusting the output is what found it.
    """
    blocks = "\n".join(
        f"[S{i + 1}] {label.replace('_', ' ')} covers {INTENT_DESCRIPTIONS[label]}"
        for i, label in enumerate(INTENTS)
    )
    return [
        {"role": "user", "content": f"Which single category best matches this customer message: {text}"},
        {"role": "system", "content": blocks},
    ]


def prompt_only_baseline(
    examples: Sequence[Example],
    llm: Optional[LLMProvider] = None,
) -> Tuple[List[int], int]:
    """No training at all. Returns (predictions, unparsed_count).

    An unparseable response is recorded as prediction -1, which the metric
    counts as wrong. Silently retrying or defaulting to the majority class
    would flatter the baseline and make the comparison dishonest.
    """
    model = llm or EchoLLM()
    preds: List[int] = []
    unparsed = 0
    for ex in examples:
        text = model.complete(build_classification_prompt(ex.text)).text
        match = _CITATION_RE.search(text)
        if not match:
            unparsed += 1
            preds.append(-1)
            continue
        index = int(match.group(1)) - 1
        if 0 <= index < len(INTENTS):
            preds.append(index)
        else:
            unparsed += 1
            preds.append(-1)
    return preds, unparsed


def format_confusion(evaluation: Evaluation, width: int = 7) -> str:
    """Confusion matrix as text. Rows are true labels, columns are predictions."""
    label_col = max(len(label) for label in INTENTS) + 8
    short = [label[:width] for label in INTENTS]
    header = " " * label_col + "".join(f"{s:>{width + 2}}" for s in short)
    lines = [header, " " * label_col + "-" * (len(short) * (width + 2))]
    for i, label in enumerate(INTENTS):
        row = "".join(f"{evaluation.confusion[i][j]:>{width + 2}}" for j in range(len(INTENTS)))
        lines.append(f"  true {label:<{label_col - 7}}{row}")
    return "\n".join(lines)


def format_per_class(evaluation: Evaluation) -> str:
    lines = [f"  {'label':<22}{'support':>8}{'precision':>11}{'recall':>8}{'f1':>8}"]
    lines.append("  " + "-" * 55)
    for label in INTENTS:
        row = evaluation.per_class.get(label, {})
        lines.append(f"  {label:<22}{int(row.get('support', 0)):>8}"
                     f"{row.get('precision', 0.0):>11.3f}{row.get('recall', 0.0):>8.3f}"
                     f"{row.get('f1', 0.0):>8.3f}")
    return "\n".join(lines)
