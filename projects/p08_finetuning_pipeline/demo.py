"""End-to-end adaptation pipeline: generate, filter, split, train, compare, ablate.

Three systems on one held-out test set:
  1. prompt-only     EchoLLM, no training at all
  2. full fine-tune  every weight in the head updated
  3. LoRA            base frozen, only a rank-r adapter trained

All three start from the same features and the same frozen base, so the
difference between them is the adaptation method and nothing else.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import time

from llmkit import HashingEmbedder

from projects.p08_finetuning_pipeline.ablation import format_ablation, run_ablation
from projects.p08_finetuning_pipeline.data import INTENTS, build_dataset
from projects.p08_finetuning_pipeline.evaluate import (
    evaluate_predictions,
    format_confusion,
    format_per_class,
    prompt_only_baseline,
)
from projects.p08_finetuning_pipeline.model import LoRAHead
from projects.p08_finetuning_pipeline.train import (
    accuracy_full,
    encode,
    pretrain_base,
    train_full,
    train_lora,
)

EMBED_DIM = 256
EPOCHS = 14
LR = 0.2
RANK = 4
ALPHA = 4.0


def rule(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main():
    started = time.perf_counter()

    rule("1. SYNTHETIC DATA  templates plus schema-constrained EchoLLM style selection")
    data = build_dataset(per_intent_target=95, per_intent_pretrain=45)
    target_report = data["report"]
    pretrain_report = data["pretrain_report"]
    print(f"  intents            {len(INTENTS)}: {', '.join(INTENTS)}")
    print(f"  pretrain bank      {pretrain_report.generated} generated -> "
          f"{pretrain_report.kept} kept (formal phrasings, stands in for pretraining)")
    print(f"  target bank        {target_report.generated} generated -> "
          f"{target_report.kept} kept (colloquial phrasings, the domain to adapt to)")
    print("\n  quality filter on the target bank")
    print(f"    too short        {target_report.dropped_short}")
    print(f"    too long         {target_report.dropped_long}")
    print(f"    near duplicate   {target_report.dropped_duplicate} "
          f"(hashed-feature cosine >= 0.97, within label)")
    print(f"    kept             {target_report.kept}")
    print(f"\n  label balance      {target_report.per_label}")
    print(f"    ratio            {target_report.balance_ratio:.2f} to 1, "
          f"within tolerance: {target_report.balanced}")
    if not target_report.balanced:
        print("    the check reports rather than deletes: the fix for imbalance is")
        print("    generating more of the small class, not discarding the large one")
    train, val, test = data["train"], data["val"], data["test"]
    print(f"\n  stratified split   train {len(train)} / val {len(val)} / test {len(test)}, seed fixed")
    print("  sample rows")
    for ex in train[:4]:
        print(f"    {ex.label:<22} {ex.text}")

    rule("2. FEATURES AND FROZEN BASE")
    embedder = HashingEmbedder(dim=EMBED_DIM)
    pre_xs, pre_ys = encode(data["pretrain"], embedder)
    train_xs, train_ys = encode(train, embedder)
    val_xs, val_ys = encode(val, embedder)
    test_xs, test_ys = encode(test, embedder)
    t0 = time.perf_counter()
    base = pretrain_base(pre_xs, pre_ys, EMBED_DIM, len(INTENTS), epochs=6, lr=0.5)
    pretrain_s = time.perf_counter() - t0
    base_test_acc = accuracy_full(base, test_xs, test_ys)
    print(f"  embedder           HashingEmbedder(dim={EMBED_DIM}), lexical, no download")
    print(f"  base head          {base.trainable_parameters} weights, trained {pretrain_s:.2f} s "
          f"on the formal bank, then frozen")
    print(f"  base test accuracy {base_test_acc:.3f} on the colloquial test set "
          f"(the domain gap the adapters have to close)")

    rule("3. PROMPT-ONLY BASELINE  no training")
    t0 = time.perf_counter()
    prompt_preds, unparsed = prompt_only_baseline(test)
    prompt_s = time.perf_counter() - t0
    prompt_eval = evaluate_predictions("prompt-only", test_ys, prompt_preds, unparsed)
    print(f"  every intent supplied as an evidence block, EchoLLM picks one, {prompt_s:.2f} s")
    print(f"  accuracy {prompt_eval.accuracy:.3f}   macro-F1 {prompt_eval.macro_f1:.3f}   "
          f"unparseable responses {prompt_eval.unparsed}")
    print(f"  random guessing over {len(INTENTS)} classes would be {1 / len(INTENTS):.3f}")

    rule("4. FULL FINE-TUNE  every weight updated")
    full = base.copy()
    full_result = train_full(full, train_xs, train_ys, val_xs, val_ys, epochs=EPOCHS, lr=LR)
    full_preds = [full.predict(x) for x in test_xs]
    full_eval = evaluate_predictions("full-finetune", test_ys, full_preds)
    print(f"  trainable parameters {full_result.trainable_parameters}   "
          f"wall clock {full_result.seconds:.2f} s")
    print(f"\n  {'epoch':>6}{'train loss':>13}{'train acc':>12}{'val acc':>10}{'sec':>8}")
    print("  " + "-" * 47)
    for stat in full_result.curve:
        print(f"  {stat.epoch:>6}{stat.train_loss:>13.4f}{stat.train_acc:>12.3f}"
              f"{stat.val_acc:>10.3f}{stat.seconds:>8.2f}")

    rule(f"5. LoRA  base frozen, rank {RANK}, alpha {ALPHA:g}")
    lora = LoRAHead(base, rank=RANK, alpha=ALPHA, seed=3)
    lora_result = train_lora(lora, train_xs, train_ys, val_xs, val_ys, epochs=EPOCHS, lr=LR)
    base_test_logits = [lora.base_logits(x) for x in test_xs]
    lora_preds = [lora.predict(x, base_test_logits[i]) for i, x in enumerate(test_xs)]
    lora_eval = evaluate_predictions("lora", test_ys, lora_preds)
    print(f"  frozen parameters    {lora_result.frozen_parameters}")
    print(f"  trainable parameters {lora_result.trainable_parameters} "
          f"({100.0 * lora_result.trainable_parameters / full_result.trainable_parameters:.1f} "
          f"percent of full fine-tuning)")
    print(f"  wall clock           {lora_result.seconds:.2f} s")
    print(f"\n  {'epoch':>6}{'train loss':>13}{'train acc':>12}{'val acc':>10}{'sec':>8}")
    print("  " + "-" * 47)
    for stat in lora_result.curve:
        print(f"  {stat.epoch:>6}{stat.train_loss:>13.4f}{stat.train_acc:>12.3f}"
              f"{stat.val_acc:>10.3f}{stat.seconds:>8.2f}")

    print("\n  base weights untouched during training: "
          f"{all(a == b for ra, rb in zip(lora.base.W, base.W) for a, b in zip(ra, rb))}")

    rule("6. MERGE  folding s * (B @ A) back into the base")
    merged = lora.merge()
    max_logit_delta = 0.0
    for x in test_xs:
        adapted, _ = lora.forward(x)
        for a, b in zip(adapted, merged.forward(x)):
            max_logit_delta = max(max_logit_delta, abs(a - b))
    merged_preds = [merged.predict(x) for x in test_xs]
    print(f"  largest logit difference across the test set  {max_logit_delta:.3e}")
    print(f"  predictions identical                         {merged_preds == lora_preds}")
    print(f"  merged model parameters                       {merged.trainable_parameters} "
          f"(one matrix, no adapter, no inference overhead)")

    rule("7. RESULTS  three systems, one held-out test set")
    print(f"  {'system':<26}{'params trained':>16}{'accuracy':>11}{'macro F1':>11}")
    print("  " + "-" * 62)
    rows = [
        ("prompt-only (no training)", 0, prompt_eval),
        ("frozen base (no adapter)", 0, evaluate_predictions(
            "base", test_ys, [base.predict(x) for x in test_xs])),
        ("full fine-tune", full_result.trainable_parameters, full_eval),
        (f"LoRA r={RANK} alpha={ALPHA:g}", lora_result.trainable_parameters, lora_eval),
    ]
    for name, params, ev in rows:
        print(f"  {name:<26}{params:>16}{ev.accuracy:>11.3f}{ev.macro_f1:>11.3f}")

    print(f"\n  per class, LoRA r={RANK}")
    print(format_per_class(lora_eval))
    print(f"\n  confusion matrix, LoRA r={RANK} (rows true, columns predicted)")
    print(format_confusion(lora_eval))
    print(f"\n  confusion matrix, prompt-only baseline")
    print(format_confusion(prompt_eval))

    rule("8. ABLATION  rank and alpha")
    t0 = time.perf_counter()
    ablation_rows = run_ablation(base, train_xs, train_ys, val_xs, val_ys, test_xs, test_ys,
                                 epochs=EPOCHS, lr=LR)
    print(format_ablation(ablation_rows))
    print(f"\n  {len(ablation_rows)} configurations in {time.perf_counter() - t0:.2f} s")
    best = max(ablation_rows, key=lambda r: (r.test_macro_f1, -r.trainable))
    cheapest_match = min((r for r in ablation_rows if r.test_macro_f1 >= best.test_macro_f1),
                         key=lambda r: r.trainable)
    print(f"  best macro-F1        rank {best.rank} alpha {best.alpha:g} -> "
          f"{best.test_macro_f1:.3f}")
    print(f"  cheapest to match it rank {cheapest_match.rank} alpha {cheapest_match.alpha:g}, "
          f"{cheapest_match.trainable} parameters "
          f"({cheapest_match.percent_of_full:.1f} percent of full fine-tuning)")

    print(f"\ntotal demo runtime {time.perf_counter() - started:.1f} s")


if __name__ == "__main__":
    main()
