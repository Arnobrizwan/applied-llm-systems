"""Rank and alpha ablation.

Two knobs, and they are not independent: the effective update magnitude is
scaled by alpha / r, so holding alpha fixed while raising r shrinks the
per-step perturbation, and holding the ratio fixed while raising r buys
capacity at constant magnitude. The grid is laid out to separate those two
effects: it sweeps rank at a fixed scaling of 1.0, then sweeps alpha at a
fixed rank so the scaling changes on its own.

Every configuration starts from a fresh copy of the same frozen base and the
same seed, so the only thing varying is the adapter. Reusing a mutated head
between runs is the standard way an ablation table ends up measuring run
order instead of the parameter it names.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .evaluate import evaluate_predictions
from .model import LinearHead, LoRAHead
from .train import train_lora

Vector = List[float]


@dataclass
class AblationRow:
    rank: int
    alpha: float
    scaling: float
    trainable: int
    percent_of_full: float
    val_acc: float
    test_acc: float
    test_macro_f1: float
    seconds: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "scaling": round(self.scaling, 3),
            "trainable": self.trainable,
            "percent_of_full": round(self.percent_of_full, 1),
            "val_acc": round(self.val_acc, 4),
            "test_acc": round(self.test_acc, 4),
            "test_macro_f1": round(self.test_macro_f1, 4),
            "seconds": round(self.seconds, 3),
        }


DEFAULT_GRID: Tuple[Tuple[int, float], ...] = (
    (1, 1.0), (2, 2.0), (4, 4.0), (8, 8.0),      # rank sweep, scaling held at 1.0
    (4, 1.0), (4, 8.0), (4, 16.0),               # alpha sweep, rank held at 4
)


def run_ablation(
    base: LinearHead,
    xs: Sequence[Vector],
    ys: Sequence[int],
    val_xs: Sequence[Vector],
    val_ys: Sequence[int],
    test_xs: Sequence[Vector],
    test_ys: Sequence[int],
    grid: Sequence[Tuple[int, float]] = DEFAULT_GRID,
    epochs: int = 14,
    lr: float = 0.2,
    seed: int = 3,
) -> List[AblationRow]:
    full_params = base.trainable_parameters
    base_test = [base.forward(x) for x in test_xs]
    rows: List[AblationRow] = []
    for rank, alpha in grid:
        started = time.perf_counter()
        head = LoRAHead(base, rank=rank, alpha=alpha, seed=seed)
        result = train_lora(head, xs, ys, val_xs, val_ys, epochs=epochs, lr=lr, seed=seed)
        preds = [head.predict(x, base_test[i]) for i, x in enumerate(test_xs)]
        evaluation = evaluate_predictions(f"lora-r{rank}-a{alpha:g}", test_ys, preds)
        rows.append(AblationRow(
            rank=rank,
            alpha=alpha,
            scaling=head.scaling,
            trainable=head.trainable_parameters,
            percent_of_full=100.0 * head.trainable_parameters / full_params,
            val_acc=result.final_val_acc,
            test_acc=evaluation.accuracy,
            test_macro_f1=evaluation.macro_f1,
            seconds=time.perf_counter() - started,
        ))
    return rows


def format_ablation(rows: Sequence[AblationRow]) -> str:
    header = (f"  {'rank':>5}{'alpha':>7}{'a/r':>7}{'params':>9}{'%full':>8}"
              f"{'val acc':>10}{'test acc':>10}{'macro f1':>10}{'sec':>7}")
    lines = [header, "  " + "-" * (len(header) - 2)]
    for r in rows:
        lines.append(f"  {r.rank:>5}{r.alpha:>7.0f}{r.scaling:>7.2f}{r.trainable:>9}"
                     f"{r.percent_of_full:>7.1f}%{r.val_acc:>10.3f}{r.test_acc:>10.3f}"
                     f"{r.test_macro_f1:>10.3f}{r.seconds:>7.2f}")
    return "\n".join(lines)
