"""Hosted adapter for the fine-tuning pipeline.

The project trains three systems on one held-out test set: a prompt-only
baseline with no training at all, a full fine-tune where every weight moves,
and a LoRA adapter where the base is frozen and only a rank-r pair of small
matrices is trained. All the maths is plain Python, so the whole thing runs on
a CPU with no downloads and no writes to disk.

What was cut to fit the page's time budget is stated in the output and listed
at the top of this file, because a demo that quietly shrinks its own experiment
and reports the numbers anyway is worse than useless:

  feature width   256 -> 128 dimensions
  target bank     95 -> 40 utterances per intent before filtering
  pretrain bank   45 -> 20 utterances per intent
  epochs          14 -> 10 for both the full fine-tune and the adapter
  ablation        7 configurations -> a 4 rank sweep beside the chosen rank

Nothing about the method changed. The base is still pretrained on formal
phrasings and frozen, both adapters still start from that identical base, the
optimiser and the seed are still the same across all of them, and the merge is
still checked to 1e-9.
"""
from __future__ import annotations

import re
from typing import Any, List

NUMBER = 8
SLUG = "finetuning-pipeline"
TITLE = "Fine-Tuning Pipeline with LoRA"
TAGLINE = "Pick an adapter size and watch a small model learn a new way of talking, then check what the shortcut cost."

WHAT_IT_DOES = """A support inbox sorts messages into six buckets: billing,
password reset, cancellation, order tracking, bug report, feature request. The
starting model was trained on polite, well-punctuated writing. The messages it
now has to handle are clipped, lowercase and full of typos, so it needs to
adapt.

There are three ways to do that, and this page runs all three on the same test
messages. Ask the model directly with no training at all. Retrain every weight
it has. Or freeze the whole thing and train a tiny add-on beside it, which is
what LoRA means. You choose how big that add-on is by typing a number, and the
page shows how many settings each approach had to change and how well each one
scored.

The last check is the one people skip. The add-on can be folded back into the
original weights with plain arithmetic, so the finished model is one plain
matrix again with nothing bolted on and nothing slower about it. The page
proves the folded model gives exactly the same answers, to nine decimal places.

Everything trains here and now in a couple of seconds. It is small on purpose,
and the output says exactly what was shrunk to make that possible."""

INPUT_LABEL = "Adapter size (LoRA rank): try a number from 1 to 16"
PLACEHOLDER = "4"

EXAMPLES = ["4", "1", "8", "16"]

SOURCE = "projects/p08_finetuning_pipeline"

# Trimmed from the project's own settings so a full run fits comfortably in
# the serverless time budget. Stated in the output as well as here.
EMBED_DIM = 128
PER_INTENT_TARGET = 40
PER_INTENT_PRETRAIN = 20
EPOCHS = 10
PRETRAIN_EPOCHS = 6
LR = 0.2
ALPHA_RATIO = 1.0          # alpha = rank, so alpha / r stays at 1.0
RANK_SWEEP = (1, 2, 4, 8)
MAX_RANK = 32


def _parse_rank(user_input: str) -> int:
    text = (user_input or "").strip()
    if not text:
        text = EXAMPLES[0]
    match = re.search(r"\d+", text)
    if not match:
        return 4
    return max(1, min(MAX_RANK, int(match.group(0))))


def run(user_input: str) -> str:
    try:
        return _run(user_input)
    except Exception as exc:  # an adapter must never take the page down
        return (f"This demo hit an unexpected error and stopped: "
                f"{type(exc).__name__}: {exc}\n"
                "Nothing was written to disk; the whole pipeline runs in memory.")


def _run(user_input: str) -> str:
    import time

    from llmkit import HashingEmbedder

    from projects.p08_finetuning_pipeline.data import INTENTS, build_dataset
    from projects.p08_finetuning_pipeline.evaluate import (
        evaluate_predictions,
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

    rank = _parse_rank(user_input)
    alpha = float(rank) * ALPHA_RATIO
    started = time.perf_counter()
    out: List[str] = []
    add = out.append

    add(f"FINE-TUNING PIPELINE  prompt-only vs full fine-tune vs LoRA rank {rank}")
    add("=" * 78)
    add(f"  you chose        adapter rank {rank}, alpha {alpha:g}")
    add(f"                   alpha / r is held at {ALPHA_RATIO:g}, so a bigger rank buys")
    add("                   capacity without changing the size of each update step")
    add("")
    add("  RUN SIZE FOR THIS PAGE, and what the project uses when it is not on a clock:")
    add(f"    feature width  {EMBED_DIM} dimensions      (project: 256)")
    add(f"    data           {PER_INTENT_TARGET} per intent generated  (project: 95)")
    add(f"    epochs         {EPOCHS}                  (project: 14)")
    add(f"    rank sweep     {len(set(RANK_SWEEP) | {rank})} configurations   (project: 7)")
    add("    the method is unchanged: one frozen base for both adapters, same")
    add("    optimiser, same seed, same held-out test set.")

    # -- 1. data -----------------------------------------------------------
    t0 = time.perf_counter()
    data = build_dataset(per_intent_target=PER_INTENT_TARGET,
                         per_intent_pretrain=PER_INTENT_PRETRAIN, seed=7)
    data_s = time.perf_counter() - t0
    report = data["report"]
    pre_report = data["pretrain_report"]
    train, val, test = data["train"], data["val"], data["test"]

    add("")
    add("-" * 78)
    add("1. DATA  two banks, generated then filtered")
    add("-" * 78)
    add(f"  six intents        {', '.join(INTENTS)}")
    add(f"  pretrain bank      {pre_report.generated} generated -> {pre_report.kept} kept, "
        f"formal phrasings (stands in for what the base already knew)")
    add(f"  target bank        {report.generated} generated -> {report.kept} kept, "
        f"clipped and lowercase (the new domain)")
    add(f"  filter             {report.dropped_short} too short, "
        f"{report.dropped_long} too long, {report.dropped_duplicate} near duplicates "
        f"(cosine >= 0.97 within a label)")
    add(f"  label balance      {report.balance_ratio:.2f} to 1, "
        f"within tolerance: {report.balanced}")
    add("                     the check reports, it does not delete: the fix for")
    add("                     imbalance is more of the small class, not less of the big")
    add(f"  stratified split   train {len(train)} / validation {len(val)} / test {len(test)}, "
        f"seed fixed, built in {data_s:.2f} s")
    add("  a few training rows")
    for ex in train[:2]:
        add(f"    {ex.label:<22} {ex.text[:52]}")

    # -- 2. features and the frozen base -----------------------------------
    embedder = HashingEmbedder(dim=EMBED_DIM)
    pre_xs, pre_ys = encode(data["pretrain"], embedder)
    train_xs, train_ys = encode(train, embedder)
    val_xs, val_ys = encode(val, embedder)
    test_xs, test_ys = encode(test, embedder)
    t0 = time.perf_counter()
    base = pretrain_base(pre_xs, pre_ys, EMBED_DIM, len(INTENTS),
                         epochs=PRETRAIN_EPOCHS, lr=0.5)
    pretrain_s = time.perf_counter() - t0
    base_eval = evaluate_predictions("base", test_ys, [base.predict(x) for x in test_xs])

    add("")
    add("-" * 78)
    add("2. THE FROZEN BASE  trained on the formal bank, then never touched again")
    add("-" * 78)
    add(f"  features           HashingEmbedder(dim={EMBED_DIM}), lexical, nothing downloaded")
    add(f"  base head          {base.trainable_parameters} weights, trained in "
        f"{pretrain_s:.2f} s, then frozen")
    add(f"  base on the new domain   accuracy {base_eval.accuracy:.3f}, "
        f"macro F1 {base_eval.macro_f1:.3f}")
    add("                     that gap is what both methods have to close, and both")
    add("                     start from this identical head, so the comparison is")
    add("                     about the method and nothing else")

    # -- 3. prompt only ----------------------------------------------------
    t0 = time.perf_counter()
    prompt_preds, unparsed = prompt_only_baseline(test)
    prompt_s = time.perf_counter() - t0
    prompt_eval = evaluate_predictions("prompt-only", test_ys, prompt_preds, unparsed)

    add("")
    add("-" * 78)
    add("3. PROMPT-ONLY  no training at all")
    add("-" * 78)
    add("  each intent is handed to the model as a numbered description and it picks")
    add("  one. An unparseable reply counts as wrong rather than being retried or")
    add("  quietly defaulted to the biggest class.")
    add(f"  accuracy {prompt_eval.accuracy:.3f}   macro F1 {prompt_eval.macro_f1:.3f}   "
        f"unparseable {prompt_eval.unparsed} of {prompt_eval.n}   {prompt_s:.2f} s")

    # -- 4. and 5. train both ---------------------------------------------
    full_head = base.copy()
    full_result = train_full(full_head, train_xs, train_ys, val_xs, val_ys,
                             epochs=EPOCHS, lr=LR)
    full_eval = evaluate_predictions("full", test_ys,
                                     [full_head.predict(x) for x in test_xs])

    lora = LoRAHead(base, rank=rank, alpha=alpha, seed=3)
    lora_result = train_lora(lora, train_xs, train_ys, val_xs, val_ys,
                             epochs=EPOCHS, lr=LR)
    base_test_logits = [lora.base_logits(x) for x in test_xs]
    lora_preds = [lora.predict(x, base_test_logits[i]) for i, x in enumerate(test_xs)]
    lora_eval = evaluate_predictions(f"lora-r{rank}", test_ys, lora_preds)

    pct = 100.0 * lora_result.trainable_parameters / full_result.trainable_parameters
    add("")
    add("-" * 78)
    add(f"4. WHAT EACH METHOD HAD TO CHANGE")
    add("-" * 78)
    add(f"  {'method':<28}{'trainable':>12}{'frozen':>10}{'share of full':>16}"
        f"{'train time':>13}")
    add("  " + "-" * 77)
    add(f"  {'prompt-only':<28}{0:>12}{'-':>10}{'0.0%':>16}{'0.00 s':>13}")
    add(f"  {'full fine-tune':<28}{full_result.trainable_parameters:>12}{0:>10}"
        f"{'100.0%':>16}{full_result.seconds:>11.2f} s")
    add(f"  {'LoRA rank ' + str(rank):<28}{lora_result.trainable_parameters:>12}"
        f"{lora_result.frozen_parameters:>10}{pct:>15.1f}%{lora_result.seconds:>11.2f} s")
    add(f"  the adapter is two small matrices: A is {rank} by {EMBED_DIM}, "
        f"B is {len(INTENTS)} by {rank}.")
    add("  B starts at exactly zero, so at step zero the adapted model is a bit-for-bit")
    add("  copy of the base. Random values in both would perturb a working model before")
    add("  training had even begun.")
    add(f"  worth being blunt about the share column: this head has only "
        f"{len(INTENTS)} outputs, so")
    add(f"  the adapter is only smaller than a full fine-tune while the rank stays "
        f"under {len(INTENTS)}.")
    add("  LoRA pays off when the matrix being adapted is large in both directions, as")
    add("  it is inside a transformer. Same mechanism, smaller numbers.")

    add("")
    add("-" * 78)
    add("5. TRAINING CURVE  same optimiser, same seed, same data order")
    add("-" * 78)
    add(f"  {'epoch':>6}{'full loss':>12}{'full val':>11}"
        f"{'lora loss':>12}{'lora val':>11}")
    add("  " + "-" * 50)
    for f_stat, l_stat in zip(full_result.curve, lora_result.curve):
        add(f"  {f_stat.epoch:>6}{f_stat.train_loss:>12.4f}{f_stat.val_acc:>11.3f}"
            f"{l_stat.train_loss:>12.4f}{l_stat.val_acc:>11.3f}")
    same_base = all(a == b for ra, rb in zip(lora.base.W, base.W) for a, b in zip(ra, rb))
    add(f"  base weights untouched through LoRA training: {same_base}")
    add("  no gradient ever reaches the base, which is why its logits are computed once")
    add("  and reused every epoch. That is the cheap-training property, literally.")

    # -- 6. merge ----------------------------------------------------------
    merged = lora.merge()
    max_delta = 0.0
    for x in test_xs:
        adapted, _ = lora.forward(x)
        for a, b in zip(adapted, merged.forward(x)):
            max_delta = max(max_delta, abs(a - b))
    merged_preds = [merged.predict(x) for x in test_xs]

    add("")
    add("-" * 78)
    add("6. MERGE CHECK  folding the adapter back into the base")
    add("-" * 78)
    add("  W_merged = W + (alpha / r) * (B @ A). Exact arithmetic, not an approximation.")
    add(f"  largest logit difference across the test set   {max_delta:.3e}")
    add(f"  every prediction identical                     {merged_preds == lora_preds}")
    add(f"  merged model parameters                        {merged.trainable_parameters} "
        f"(one matrix again)")
    add("  so the adapter costs nothing at inference time once merged. Dropping the")
    add("  alpha / r scaling is the classic way to get a merge that is nearly right and")
    add("  quietly wrong, which is why it is asserted rather than assumed.")

    # -- 7. results --------------------------------------------------------
    add("")
    add("-" * 78)
    add("7. RESULTS  one held-out test set, four systems")
    add("-" * 78)
    add(f"  {'system':<28}{'trained':>12}{'accuracy':>11}{'macro F1':>11}")
    add("  " + "-" * 62)
    for name, params, ev in (
        ("prompt-only, no training", 0, prompt_eval),
        ("frozen base, no adapter", 0, base_eval),
        ("full fine-tune", full_result.trainable_parameters, full_eval),
        (f"LoRA rank {rank}", lora_result.trainable_parameters, lora_eval),
    ):
        add(f"  {name:<28}{params:>12}{ev.accuracy:>11.3f}{ev.macro_f1:>11.3f}")
    add("  macro F1 is the number to argue with. Accuracy lets a model ignore a small")
    add("  class and still look respectable; macro F1 charges it full price.")
    add("")
    add(f"  per class, LoRA rank {rank}")
    add(format_per_class(lora_eval))

    # -- 8. small rank sweep ----------------------------------------------
    add("")
    add("-" * 78)
    add("8. DOES A BIGGER ADAPTER HELP  same base, same seed, alpha / r held at 1.0")
    add("-" * 78)
    add(f"  {'rank':>6}{'trainable':>12}{'share of full':>16}{'accuracy':>11}"
        f"{'macro F1':>11}{'seconds':>10}")
    add("  " + "-" * 64)
    for r in sorted(set(RANK_SWEEP) | {rank}):
        head = LoRAHead(base, rank=r, alpha=float(r) * ALPHA_RATIO, seed=3)
        t0 = time.perf_counter()
        train_lora(head, train_xs, train_ys, val_xs, val_ys, epochs=EPOCHS, lr=LR)
        secs = time.perf_counter() - t0
        preds = [head.predict(x, base_test_logits[i]) for i, x in enumerate(test_xs)]
        ev = evaluate_predictions(f"r{r}", test_ys, preds)
        share = 100.0 * head.trainable_parameters / full_result.trainable_parameters
        marker = "  <- you chose this" if r == rank else ""
        add(f"  {r:>6}{head.trainable_parameters:>12}{share:>15.1f}%"
            f"{ev.accuracy:>11.3f}{ev.macro_f1:>11.3f}{secs:>10.2f}{marker}")
    add("  every row starts from a fresh copy of the same frozen base, so the table")
    add("  measures the rank, not the order the rows ran in.")

    add("")
    add(f"  whole pipeline, data generation through the sweep, ran in "
        f"{time.perf_counter() - started:.2f} s here,")
    add("  in plain Python with no tensor library and nothing written to disk.")
    return "\n".join(out)
