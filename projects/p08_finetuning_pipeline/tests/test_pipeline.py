"""End-to-end behaviour: training helps, metrics are honest, the ablation is fair."""
import pytest

from llmkit import HashingEmbedder

from projects.p08_finetuning_pipeline.ablation import run_ablation
from projects.p08_finetuning_pipeline.data import INTENTS, build_dataset
from projects.p08_finetuning_pipeline.evaluate import (
    evaluate_predictions,
    prompt_only_baseline,
)
from projects.p08_finetuning_pipeline.model import LoRAHead
from projects.p08_finetuning_pipeline.train import (
    accuracy_full,
    accuracy_lora,
    encode,
    pretrain_base,
    train_full,
    train_lora,
)

DIM = 96


@pytest.fixture(scope="module")
def fitted():
    """One small pipeline run shared by the tests in this module."""
    data = build_dataset(per_intent_target=40, per_intent_pretrain=24)
    embedder = HashingEmbedder(dim=DIM)
    pre = encode(data["pretrain"], embedder)
    tr = encode(data["train"], embedder)
    va = encode(data["val"], embedder)
    te = encode(data["test"], embedder)
    base = pretrain_base(pre[0], pre[1], DIM, len(INTENTS), epochs=5, lr=0.5)
    return data, base, tr, va, te


def test_both_adaptation_methods_beat_the_frozen_base(fitted):
    data, base, tr, va, te = fitted
    base_acc = accuracy_full(base, te[0], te[1])

    full = base.copy()
    train_full(full, tr[0], tr[1], va[0], va[1], epochs=10, lr=0.2)

    lora = LoRAHead(base, rank=4, alpha=4.0, seed=3)
    train_lora(lora, tr[0], tr[1], va[0], va[1], epochs=10, lr=0.2)
    base_test = [lora.base_logits(x) for x in te[0]]

    assert accuracy_full(full, te[0], te[1]) > base_acc
    assert accuracy_lora(lora, te[0], te[1], base_test) > base_acc


def test_training_loss_decreases_monotonically_enough_to_be_real(fitted):
    data, base, tr, va, te = fitted
    result = train_full(base.copy(), tr[0], tr[1], va[0], va[1], epochs=8, lr=0.2)
    losses = [s.train_loss for s in result.curve]
    assert len(losses) == 8
    assert losses[-1] < losses[0] / 2
    assert result.curve[-1].train_acc >= result.curve[0].train_acc
    assert all(s.seconds >= 0 for s in result.curve)


def test_lora_trains_far_fewer_parameters_than_the_full_finetune(fitted):
    data, base, tr, va, te = fitted
    full_result = train_full(base.copy(), tr[0], tr[1], va[0], va[1], epochs=2, lr=0.2)
    lora = LoRAHead(base, rank=1, alpha=1.0, seed=3)
    lora_result = train_lora(lora, tr[0], tr[1], va[0], va[1], epochs=2, lr=0.2)
    assert lora_result.trainable_parameters < full_result.trainable_parameters
    assert lora_result.frozen_parameters == full_result.trainable_parameters


def test_prompt_only_baseline_is_a_real_system_not_a_coin_flip(fitted):
    data, base, tr, va, te = fitted
    preds, unparsed = prompt_only_baseline(data["test"])
    evaluation = evaluate_predictions("prompt-only", te[1], preds, unparsed)
    assert unparsed == 0
    assert len(set(preds)) > 1                        # it does not collapse to one class
    assert evaluation.accuracy > 1.0 / len(INTENTS)   # better than guessing
    assert evaluation.accuracy < 1.0                  # and not solved without training


def test_macro_f1_punishes_ignoring_a_small_class():
    """Accuracy alone would hide this, which is why both numbers are reported."""
    # 18 of class 0, 2 of class 1; the model never predicts class 1.
    y_true = [0] * 18 + [1] * 2
    y_pred = [0] * 20
    evaluation = evaluate_predictions("majority", y_true, y_pred)
    assert evaluation.accuracy == pytest.approx(0.9)
    assert evaluation.macro_f1 < 0.5
    assert evaluation.per_class[INTENTS[1]]["recall"] == 0.0


def test_confusion_matrix_rows_sum_to_the_support_of_each_class():
    y_true = [0, 0, 1, 1, 2]
    y_pred = [0, 1, 1, 2, 2]
    evaluation = evaluate_predictions("x", y_true, y_pred)
    assert [sum(row) for row in evaluation.confusion[:3]] == [2, 2, 1]
    assert evaluation.confusion[0][1] == 1
    assert evaluation.accuracy == pytest.approx(3 / 5)


def test_unparseable_baseline_responses_count_as_wrong_not_as_missing():
    evaluation = evaluate_predictions("x", [0, 1, 2], [0, -1, 2], unparsed=1)
    assert evaluation.n == 3
    assert evaluation.accuracy == pytest.approx(2 / 3)
    assert evaluation.unparsed == 1


def test_every_ablation_configuration_starts_from_the_same_frozen_base(fitted):
    """Otherwise the table measures run order instead of rank and alpha."""
    data, base, tr, va, te = fitted
    snapshot = [row[:] for row in base.W]
    rows = run_ablation(base, tr[0], tr[1], va[0], va[1], te[0], te[1],
                        grid=((1, 1.0), (4, 4.0), (4, 16.0)), epochs=4, lr=0.2)
    assert base.W == snapshot                          # base survived untouched
    assert [r.rank for r in rows] == [1, 4, 4]
    assert rows[0].trainable < rows[1].trainable
    assert rows[1].trainable == rows[2].trainable      # same rank, different alpha
    assert rows[1].scaling == 1.0 and rows[2].scaling == 4.0
    assert all(0.0 <= r.test_macro_f1 <= 1.0 for r in rows)
    assert all(r.percent_of_full > 0 for r in rows)
