"""Training loops for the full fine-tune and the LoRA adapter.

Plain SGD with a shuffled pass per epoch and a fixed seed. Adam would converge
in fewer epochs but it needs two extra state tensors per parameter, and the
comparison this project is making is about which parameters are updated, not
about the optimiser. Keeping the optimiser identical across all three systems
is what makes the accuracy difference attributable to the adaptation method.

Both loops share `_step_loss`, so the only difference between them is which
parameters receive the gradient. That is the entire distinction between full
fine-tuning and LoRA, and keeping it visible in the code is the point.
"""
from __future__ import annotations

import random
import time
from typing import List, Optional, Sequence, Tuple

from llmkit import HashingEmbedder

from .data import INTENTS, Example
from .model import (
    EpochStat,
    LinearHead,
    LoRAHead,
    TrainResult,
    cross_entropy,
    softmax,
)

Vector = List[float]


def encode(examples: Sequence[Example], embedder: HashingEmbedder) -> Tuple[List[Vector], List[int]]:
    """Features once, up front. Re-embedding every epoch would dominate runtime."""
    xs = embedder.embed([e.text for e in examples])
    ys = [INTENTS.index(e.label) for e in examples]
    return xs, ys


def _step_loss(logits: Vector, target: int) -> Tuple[float, Vector, int]:
    """Cross-entropy loss, its gradient wrt the logits, and the argmax."""
    probs = softmax(logits)
    loss = cross_entropy(probs, target)
    dz = probs[:]
    dz[target] -= 1.0
    pred = max(range(len(logits)), key=lambda c: logits[c])
    return loss, dz, pred


def accuracy_full(head: LinearHead, xs: Sequence[Vector], ys: Sequence[int]) -> float:
    if not xs:
        return 0.0
    return sum(1 for x, y in zip(xs, ys) if head.predict(x) == y) / len(xs)


def accuracy_lora(head: LoRAHead, xs: Sequence[Vector], ys: Sequence[int],
                  base: Optional[Sequence[Vector]] = None) -> float:
    if not xs:
        return 0.0
    hits = 0
    for i, (x, y) in enumerate(zip(xs, ys)):
        z0 = base[i] if base is not None else None
        if head.predict(x, z0) == y:
            hits += 1
    return hits / len(xs)


def train_full(
    head: LinearHead,
    xs: Sequence[Vector],
    ys: Sequence[int],
    val_xs: Sequence[Vector],
    val_ys: Sequence[int],
    epochs: int = 12,
    lr: float = 0.5,
    weight_decay: float = 0.0,
    seed: int = 3,
) -> TrainResult:
    """Every weight in the head is updated. This is the full fine-tune baseline."""
    rng = random.Random(seed)
    order = list(range(len(xs)))
    result = TrainResult(trainable_parameters=head.trainable_parameters, frozen_parameters=0)
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        rng.shuffle(order)
        total_loss = 0.0
        hits = 0
        for idx in order:
            x, y = xs[idx], ys[idx]
            loss, dz, pred = _step_loss(head.forward(x), y)
            total_loss += loss
            hits += 1 if pred == y else 0
            head.apply_gradients(x, dz, lr, weight_decay)
        result.curve.append(EpochStat(
            epoch=epoch,
            train_loss=total_loss / max(1, len(order)),
            train_acc=hits / max(1, len(order)),
            val_acc=accuracy_full(head, val_xs, val_ys),
            seconds=time.perf_counter() - epoch_started,
        ))
    result.seconds = time.perf_counter() - started
    return result


def train_lora(
    head: LoRAHead,
    xs: Sequence[Vector],
    ys: Sequence[int],
    val_xs: Sequence[Vector],
    val_ys: Sequence[int],
    epochs: int = 12,
    lr: float = 0.5,
    weight_decay: float = 0.0,
    seed: int = 3,
) -> TrainResult:
    """Only A and B are updated. The base head is read, never written.

    Base logits are computed once for the training and validation sets and
    reused every epoch. That is sound precisely because the base is frozen; if
    any gradient reached it the cache would go stale on the first step. It is
    also the cheap-training property LoRA is famous for, made literal.
    """
    rng = random.Random(seed)
    order = list(range(len(xs)))
    base_train = [head.base_logits(x) for x in xs]
    base_val = [head.base_logits(x) for x in val_xs]
    result = TrainResult(
        trainable_parameters=head.trainable_parameters,
        frozen_parameters=head.frozen_parameters,
    )
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        rng.shuffle(order)
        total_loss = 0.0
        hits = 0
        for idx in order:
            x, y = xs[idx], ys[idx]
            logits, h = head.forward(x, base_train[idx])
            loss, dz, pred = _step_loss(logits, y)
            total_loss += loss
            hits += 1 if pred == y else 0
            head.apply_gradients(x, dz, h, lr, weight_decay)
        result.curve.append(EpochStat(
            epoch=epoch,
            train_loss=total_loss / max(1, len(order)),
            train_acc=hits / max(1, len(order)),
            val_acc=accuracy_lora(head, val_xs, val_ys, base_val),
            seconds=time.perf_counter() - epoch_started,
        ))
    result.seconds = time.perf_counter() - started
    return result


def pretrain_base(
    xs: Sequence[Vector],
    ys: Sequence[int],
    in_dim: int,
    out_dim: int,
    epochs: int = 10,
    lr: float = 0.5,
    seed: int = 1,
) -> LinearHead:
    """Train the stand-in "pretrained" head on the out-of-domain bank, then freeze it.

    Both adaptation methods start from this identical head, which is what makes
    the comparison between them fair. Starting LoRA from a random base instead
    would be measuring something else entirely: the capacity of a rank-r matrix
    to learn a task from scratch, rather than its capacity to adapt a model that
    already works.
    """
    head = LinearHead(in_dim, out_dim, seed=seed)
    rng = random.Random(seed + 1)
    order = list(range(len(xs)))
    for _ in range(epochs):
        rng.shuffle(order)
        for idx in order:
            x, y = xs[idx], ys[idx]
            _, dz, _ = _step_loss(head.forward(x), y)
            head.apply_gradients(x, dz, lr)
    return head
