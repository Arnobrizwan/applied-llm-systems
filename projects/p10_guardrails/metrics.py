"""Confusion matrix and the three numbers that come off it.

Accuracy is reported but should be ignored: on a fixture set that is 55 percent
benign, a detector that allows everything scores 55 percent accuracy and catches
no attacks. Precision and recall are the numbers that describe the trade being
made, and for a guardrail they pull in opposite directions. Recall is what stops
an incident; precision is what stops the feature being switched off because it
keeps rejecting real customers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

__all__ = ["Confusion", "confusion_matrix", "format_matrix"]


@dataclass
class Confusion:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.total if self.total else 0.0


def confusion_matrix(pairs: Iterable[Tuple[bool, bool]]) -> Confusion:
    """`pairs` is (predicted_positive, actually_positive)."""
    matrix = Confusion()
    for predicted, actual in pairs:
        if predicted and actual:
            matrix.tp += 1
        elif predicted and not actual:
            matrix.fp += 1
        elif not predicted and actual:
            matrix.fn += 1
        else:
            matrix.tn += 1
    return matrix


def format_matrix(matrix: Confusion, title: str) -> List[str]:
    return [
        f"{title}",
        f"{'':<18}{'actually attack':>18}{'actually benign':>18}",
        f"{'flagged':<18}{matrix.tp:>18}{matrix.fp:>18}",
        f"{'passed':<18}{matrix.fn:>18}{matrix.tn:>18}",
        "",
        f"  precision {matrix.precision:.3f}   recall {matrix.recall:.3f}   "
        f"F1 {matrix.f1:.3f}   accuracy {matrix.accuracy:.3f}   n={matrix.total}",
    ]
