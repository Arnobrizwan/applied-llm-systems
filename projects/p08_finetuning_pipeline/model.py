"""Linear classifier, LoRA adapter, and hand-written backprop in plain Python.

torch and peft are unavailable in this environment, so the maths is written
out. That is a constraint, but it is also the reason this file is worth
reading: the LoRA update rule is four lines of calculus and it is almost
always described rather than shown.

Forward pass, full fine-tune:

    z = W x + b                      W is (C, D), x is (D,), b is (C,)

Forward pass, LoRA:

    z = W x + b + (alpha / r) * B (A x)

    W and b are FROZEN. A is (r, D), B is (C, r). Only A and B receive
    gradients. alpha / r is the standard scaling, and it exists so that
    changing r does not change the effective magnitude of the update: without
    it, doubling the rank doubles the size of the perturbation and the
    learning rate has to be retuned for every rank.

Initialisation follows peft: A is small random, B is exactly zero. B at zero
means B(Ax) = 0 at step 0, so the adapted model starts as an exact copy of the
base. Initialising both randomly would perturb a pretrained model before
training even began, which is the one thing an adapter must not do.

Gradients, with p = softmax(z) and y the true class, for cross-entropy:

    dz      = p - onehot(y)
    dW      = dz x^T                 full fine-tune only
    db      = dz                     full fine-tune only
    h       = A x                    (r,)
    dB      = s * dz h^T             s = alpha / r
    dh      = s * B^T dz
    dA      = dh x^T

Merging is exact arithmetic, not an approximation:

    W_merged = W + s * (B A)

so the merged model is one plain matrix with zero inference overhead. The
test suite asserts merged logits equal adapted logits to 1e-9, because "the
adapter merges cleanly" is a claim that is easy to make and easy to get wrong
by dropping the scaling factor.

One performance note that is also a real property of LoRA: because the base is
frozen, W x + b is constant for a given example across the whole of training.
The trainer computes it once per example and reuses it every epoch. On a GPU
this is not how it is done (memory beats recompute), but the fact that no
gradient ever flows into the base is exactly why LoRA training is cheap.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

Vector = List[float]
Matrix = List[List[float]]


def matvec(m: Matrix, v: Vector) -> Vector:
    return [sum(row[i] * v[i] for i in range(len(v))) for row in m]


def softmax(z: Vector) -> Vector:
    """Shifted by the max for numerical stability: exp(800) overflows."""
    top = max(z)
    exps = [math.exp(v - top) for v in z]
    total = sum(exps)
    return [e / total for e in exps]


def cross_entropy(probs: Vector, target: int) -> float:
    # Clamped so a confidently wrong prediction gives a large finite loss
    # rather than inf, which would poison every average downstream.
    return -math.log(max(probs[target], 1e-12))


class LinearHead:
    """A dense classifier head. This is the full-fine-tune model."""

    def __init__(self, in_dim: int, out_dim: int, seed: int = 0, scale: Optional[float] = None):
        rng = random.Random(seed)
        # Xavier-ish scale: too large and softmax saturates before the first
        # gradient step, too small and every class starts indistinguishable.
        s = scale if scale is not None else math.sqrt(1.0 / in_dim)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W: Matrix = [[rng.uniform(-s, s) for _ in range(in_dim)] for _ in range(out_dim)]
        self.b: Vector = [0.0] * out_dim

    def forward(self, x: Vector) -> Vector:
        return [sum(self.W[c][i] * x[i] for i in range(self.in_dim)) + self.b[c]
                for c in range(self.out_dim)]

    def predict(self, x: Vector) -> int:
        z = self.forward(x)
        return max(range(self.out_dim), key=lambda c: z[c])

    def apply_gradients(self, x: Vector, dz: Vector, lr: float, weight_decay: float = 0.0) -> None:
        for c in range(self.out_dim):
            g = dz[c]
            if g == 0.0 and weight_decay == 0.0:
                continue
            row = self.W[c]
            for i in range(self.in_dim):
                row[i] -= lr * (g * x[i] + weight_decay * row[i])
            self.b[c] -= lr * g

    @property
    def trainable_parameters(self) -> int:
        return self.in_dim * self.out_dim + self.out_dim

    def copy(self) -> "LinearHead":
        clone = LinearHead(self.in_dim, self.out_dim)
        clone.W = [row[:] for row in self.W]
        clone.b = self.b[:]
        return clone


class LoRAHead:
    """A frozen base head plus a trainable rank-r adapter."""

    def __init__(self, base: LinearHead, rank: int = 4, alpha: float = 8.0, seed: int = 0):
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base = base                      # frozen, never written to
        self.in_dim = base.in_dim
        self.out_dim = base.out_dim
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / rank
        rng = random.Random(seed)
        s = math.sqrt(1.0 / self.in_dim)
        self.A: Matrix = [[rng.uniform(-s, s) for _ in range(self.in_dim)] for _ in range(rank)]
        self.B: Matrix = [[0.0] * rank for _ in range(self.out_dim)]   # zero, per peft

    # -- forward ---------------------------------------------------------
    def base_logits(self, x: Vector) -> Vector:
        return self.base.forward(x)

    def forward(self, x: Vector, base_logits: Optional[Vector] = None) -> Tuple[Vector, Vector]:
        """Returns (logits, h) where h = A x is needed again during backprop."""
        z0 = base_logits if base_logits is not None else self.base.forward(x)
        h = [sum(self.A[k][i] * x[i] for i in range(self.in_dim)) for k in range(self.rank)]
        z = list(z0)
        for c in range(self.out_dim):
            row = self.B[c]
            z[c] += self.scaling * sum(row[k] * h[k] for k in range(self.rank))
        return z, h

    def predict(self, x: Vector, base_logits: Optional[Vector] = None) -> int:
        z, _ = self.forward(x, base_logits)
        return max(range(self.out_dim), key=lambda c: z[c])

    # -- backward --------------------------------------------------------
    def apply_gradients(self, x: Vector, dz: Vector, h: Vector, lr: float,
                        weight_decay: float = 0.0) -> None:
        """Update A and B only. The base is never touched, by construction."""
        s = self.scaling
        # dh must be computed from B *before* B is updated, otherwise the
        # gradient flowing into A is taken with respect to the wrong weights.
        dh = [s * sum(self.B[c][k] * dz[c] for c in range(self.out_dim)) for k in range(self.rank)]
        for c in range(self.out_dim):
            g = s * dz[c]
            row = self.B[c]
            for k in range(self.rank):
                row[k] -= lr * (g * h[k] + weight_decay * row[k])
        for k in range(self.rank):
            g = dh[k]
            if g == 0.0 and weight_decay == 0.0:
                continue
            row = self.A[k]
            for i in range(self.in_dim):
                row[i] -= lr * (g * x[i] + weight_decay * row[i])

    # -- reporting and merge ---------------------------------------------
    @property
    def trainable_parameters(self) -> int:
        return self.rank * self.in_dim + self.out_dim * self.rank

    @property
    def frozen_parameters(self) -> int:
        return self.base.trainable_parameters

    def delta_weights(self) -> Matrix:
        """s * (B @ A), the low-rank update as a full (C, D) matrix."""
        s = self.scaling
        return [[s * sum(self.B[c][k] * self.A[k][i] for k in range(self.rank))
                 for i in range(self.in_dim)] for c in range(self.out_dim)]

    def merge(self) -> LinearHead:
        """Fold the adapter into the base. Predictions must be identical."""
        merged = self.base.copy()
        delta = self.delta_weights()
        for c in range(self.out_dim):
            row, drow = merged.W[c], delta[c]
            for i in range(self.in_dim):
                row[i] += drow[i]
        return merged


@dataclass
class EpochStat:
    epoch: int
    train_loss: float
    train_acc: float
    val_acc: float
    seconds: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "epoch": self.epoch,
            "train_loss": round(self.train_loss, 4),
            "train_acc": round(self.train_acc, 4),
            "val_acc": round(self.val_acc, 4),
            "seconds": round(self.seconds, 3),
        }


@dataclass
class TrainResult:
    curve: List[EpochStat] = field(default_factory=list)
    trainable_parameters: int = 0
    frozen_parameters: int = 0
    seconds: float = 0.0

    @property
    def final_val_acc(self) -> float:
        return self.curve[-1].val_acc if self.curve else 0.0
