"""The LoRA arithmetic: init, gradients, scaling and the merge."""
import math
import random

import pytest

from projects.p08_finetuning_pipeline.model import (
    LinearHead,
    LoRAHead,
    cross_entropy,
    softmax,
)
from projects.p08_finetuning_pipeline.train import _step_loss

IN_DIM, OUT_DIM = 12, 4


def _x(seed=0):
    rng = random.Random(seed)
    return [rng.uniform(-1, 1) for _ in range(IN_DIM)]


def test_adapter_starts_as_an_exact_copy_of_the_base():
    """B is initialised to zero, so training begins from the base's behaviour."""
    base = LinearHead(IN_DIM, OUT_DIM, seed=1)
    lora = LoRAHead(base, rank=3, alpha=6.0, seed=2)
    assert all(v == 0.0 for row in lora.B for v in row)
    x = _x()
    logits, _ = lora.forward(x)
    assert logits == pytest.approx(base.forward(x), abs=1e-12)


def test_scaling_is_alpha_over_r_and_normalises_the_update_magnitude():
    base = LinearHead(IN_DIM, OUT_DIM, seed=1)
    assert LoRAHead(base, rank=4, alpha=8.0).scaling == 2.0
    assert LoRAHead(base, rank=8, alpha=16.0).scaling == 2.0    # same effective magnitude
    assert LoRAHead(base, rank=2, alpha=8.0).scaling == 4.0
    with pytest.raises(ValueError):
        LoRAHead(base, rank=0)


def test_analytic_gradients_match_finite_differences():
    """The only test that can catch a wrong derivative. Everything else can pass anyway."""
    base = LinearHead(IN_DIM, OUT_DIM, seed=4)
    lora = LoRAHead(base, rank=3, alpha=6.0, seed=5)
    # B starts at zero, which makes dA identically zero; perturb it first so the
    # check exercises the real path rather than a degenerate one.
    rng = random.Random(0)
    for row in lora.B:
        for k in range(len(row)):
            row[k] = rng.uniform(-0.3, 0.3)

    x, target = _x(3), 2
    logits, h = lora.forward(x)
    _, dz, _ = _step_loss(logits, target)

    s = lora.scaling
    grad_B = [[s * dz[c] * h[k] for k in range(lora.rank)] for c in range(OUT_DIM)]
    dh = [s * sum(lora.B[c][k] * dz[c] for c in range(OUT_DIM)) for k in range(lora.rank)]
    grad_A = [[dh[k] * x[i] for i in range(IN_DIM)] for k in range(lora.rank)]

    def loss_now() -> float:
        z, _ = lora.forward(x)
        return cross_entropy(softmax(z), target)

    eps = 1e-6
    for c in (0, OUT_DIM - 1):
        for k in range(lora.rank):
            original = lora.B[c][k]
            lora.B[c][k] = original + eps
            up = loss_now()
            lora.B[c][k] = original - eps
            down = loss_now()
            lora.B[c][k] = original
            assert (up - down) / (2 * eps) == pytest.approx(grad_B[c][k], abs=1e-5)

    for k in (0, lora.rank - 1):
        for i in (0, IN_DIM - 1):
            original = lora.A[k][i]
            lora.A[k][i] = original + eps
            up = loss_now()
            lora.A[k][i] = original - eps
            down = loss_now()
            lora.A[k][i] = original
            assert (up - down) / (2 * eps) == pytest.approx(grad_A[k][i], abs=1e-5)


def test_full_finetune_gradients_match_finite_differences():
    head = LinearHead(IN_DIM, OUT_DIM, seed=6)
    x, target = _x(7), 1
    _, dz, _ = _step_loss(head.forward(x), target)
    eps = 1e-6
    for c in (0, OUT_DIM - 1):
        for i in (0, IN_DIM - 1):
            original = head.W[c][i]
            head.W[c][i] = original + eps
            up = cross_entropy(softmax(head.forward(x)), target)
            head.W[c][i] = original - eps
            down = cross_entropy(softmax(head.forward(x)), target)
            head.W[c][i] = original
            assert (up - down) / (2 * eps) == pytest.approx(dz[c] * x[i], abs=1e-5)


def test_training_never_writes_to_the_frozen_base():
    base = LinearHead(IN_DIM, OUT_DIM, seed=8)
    snapshot = [row[:] for row in base.W]
    bias = base.b[:]
    lora = LoRAHead(base, rank=2, alpha=4.0, seed=9)
    for step in range(40):
        x = _x(step)
        logits, h = lora.forward(x)
        _, dz, _ = _step_loss(logits, step % OUT_DIM)
        lora.apply_gradients(x, dz, h, lr=0.3)
    assert base.W == snapshot
    assert base.b == bias
    assert any(v != 0.0 for row in lora.B for v in row)     # the adapter did move


def test_merged_weights_reproduce_adapted_predictions_exactly():
    """W + (alpha/r) * B @ A must be the same function, not approximately."""
    base = LinearHead(IN_DIM, OUT_DIM, seed=10)
    lora = LoRAHead(base, rank=3, alpha=6.0, seed=11)
    for step in range(60):
        x = _x(step + 100)
        logits, h = lora.forward(x)
        _, dz, _ = _step_loss(logits, step % OUT_DIM)
        lora.apply_gradients(x, dz, h, lr=0.05)

    merged = lora.merge()
    assert merged.trainable_parameters == base.trainable_parameters
    assert any(v != 0.0 for row in lora.B for v in row)
    for step in range(30):
        x = _x(step + 500)
        adapted, _ = lora.forward(x)
        # Relative tolerance: the two paths sum the same terms in a different
        # order, so they agree to floating point precision, not bit for bit.
        assert merged.forward(x) == pytest.approx(adapted, rel=1e-12, abs=1e-12)
        assert merged.predict(x) == lora.predict(x)


def test_trainable_parameter_counts_are_r_times_in_plus_out():
    base = LinearHead(256, 6, seed=1)
    assert base.trainable_parameters == 256 * 6 + 6
    for rank in (1, 2, 4, 8):
        lora = LoRAHead(base, rank=rank, alpha=float(rank))
        assert lora.trainable_parameters == rank * (256 + 6)
        assert lora.frozen_parameters == base.trainable_parameters
    assert LoRAHead(base, rank=1).trainable_parameters < base.trainable_parameters


def test_softmax_is_numerically_stable_and_cross_entropy_is_finite():
    probs = softmax([1000.0, 999.0, 998.0])         # would overflow without the shift
    assert sum(probs) == pytest.approx(1.0)
    assert all(math.isfinite(p) for p in probs)
    assert probs[0] > probs[1] > probs[2]
    assert math.isfinite(cross_entropy([1.0, 0.0], 1))    # clamped, not inf
