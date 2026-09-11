# 08. Fine-Tuning Pipeline

**Run it live: [https://applied-llm-systems.vercel.app/s/finetuning-pipeline](https://applied-llm-systems.vercel.app/s/finetuning-pipeline)** - the hosted page runs this code and shows the real output.

A LoRA adaptation pipeline with the maths written out by hand, benchmarked
against a full fine-tune and a prompt-only baseline on a held-out test set.

## What this is, and what it is not

**This is not a GPU fine-tune of a transformer.** `torch`, `peft` and
`transformers` cannot be installed in this environment, so the model being
adapted is a linear classifier over `llmkit.HashingEmbedder` features, and
every gradient is computed by hand with plain Python lists.

**The training is real.** There is no simulation layer and no pretend loss
curve. Forward pass, softmax cross-entropy, backpropagation and SGD are
implemented and the numbers below come from running them.

**The LoRA mechanism is the actual mechanism.** A frozen base weight matrix, a
trainable low-rank pair `A` and `B`, `B` initialised to zero, `alpha / r`
scaling, and an exact merge of `B @ A` back into the base at the end. That is
the whole of LoRA. What changes on a GPU is which matrices it is applied to
(attention and MLP projections inside every transformer block instead of one
classifier head) and how many there are, not what the update rule is. The
`peft` configuration that corresponds to this code is given below.

## The problem

Prompting stops being enough somewhere between "the model mostly gets it" and
"the model has to get it, on our data, every time". Fine-tuning is the answer,
and the two things that go wrong are:

1. **The data, not the training.** Synthetic or scraped training sets are full
   of near-duplicates. A duplicate that survives filtering lands on both sides
   of the split, validation accuracy goes up, production accuracy does not, and
   the gap is invisible until launch.
2. **Full fine-tuning is expensive to train, store and serve.** Every task
   needs its own full copy of the weights. LoRA trains a small adapter over a
   frozen base, and the interesting question is not whether it is cheaper but
   how much accuracy that costs, which is a measurement, not an opinion.

## What this builds

- `data.py` two template banks (formal for pretraining, colloquial for the
  target domain), `EchoLLM` style selection under a JSON schema, a three-stage
  quality filter and a seeded stratified split.
- `model.py` `LinearHead`, `LoRAHead`, softmax, cross-entropy, hand-written
  gradients for both, and an exact merge.
- `train.py` SGD loops for full fine-tuning and LoRA, sharing everything except
  which parameters receive the gradient.
- `evaluate.py` accuracy, per-class precision/recall/F1, macro-F1, a confusion
  matrix, and a prompt-only baseline that makes a genuine decision.
- `ablation.py` a rank and alpha grid, each configuration starting from the
  same frozen base.

## Architecture

```
  templates (formal)          templates (colloquial)
        |                             |
        v                             v
   EchoLLM style choice under a JSON schema (deterministic)
        |                             |
        v                             v
   quality filter:  length bounds -> near-duplicate cosine -> balance report
        |                             |
        v                             v
   pretrain bank                 stratified split (seeded)
        |                        train / val / test
        v                             |
   train LinearHead                   |
        |                             |
        v                             |
   [ FROZEN BASE W, b ] <-------------+
        |            |                |
        |            |                +--> prompt-only baseline (EchoLLM, no training)
        |            |
        |            +--> copy -> full fine-tune: all 1542 weights updated
        |
        +--> LoRAHead:  z = Wx + b + (alpha/r) * B(Ax)
                        only A (r x D) and B (C x r) updated
                        merge: W' = W + (alpha/r) * B@A
```

## The maths

With `x` the feature vector, `W` the frozen base, `s = alpha / r`, `h = A x`,
`p = softmax(z)` and `y` the true class:

```
forward    z  = W x + b + s * B h
loss       L  = -log p[y]
gradients  dz = p - onehot(y)
           dB = s * dz h^T
           dh = s * B^T dz
           dA = dh x^T
merge      W' = W + s * (B A)
```

`B` starts at zero so `s * B h = 0` at step 0 and the adapted model is an exact
copy of the base before training. `dh` is computed from `B` before `B` is
updated; doing it in the other order takes the gradient with respect to the
wrong weights, and the resulting model still trains, just worse. There is a
finite-difference test that catches exactly that class of error.

## Design decisions

**Two template banks, and the base is pretrained on the other one.** The frozen
base is trained on formal, well-punctuated phrasings and both adapters are
evaluated on clipped, lowercase ones. If the base had seen the target
distribution the adapter would have nothing to learn and the experiment would
flatter itself. Starting LoRA from a random base instead would measure
something different again: the ability of a rank-r matrix to learn a task from
scratch, not to adapt a model that already works.

**The dedup threshold was set by measurement.** At cosine 0.93 the filter also
removes utterances that differ only in a slot value, which is variation the
classifier needs; on this generator that discarded 168 of 264 target examples
and left the labels imbalanced 5.75 to 1. At 0.97 it removes near-identical
restatements and keeps slot variation. The numbers below use 0.97.

**The balance check reports, it does not delete.** Rebalancing by deleting from
the large class throws away real data to fix a number. The report says the
ratio is 2.43 to 1 and outside the 1.5 tolerance, and the fix is generating more
of the small class.

**Base logits are cached across epochs during LoRA training.** This is only
sound because no gradient ever reaches the base, and it is the cheap-training
property of LoRA made literal. On a GPU the tradeoff goes the other way
(recompute beats memory), but the reason it is possible at all is the same.

**Macro-F1 is reported next to accuracy.** The filtered label distribution is
uneven, so a classifier that ignores the smallest class still posts decent
accuracy. There is a test asserting a majority-class predictor scores 0.9
accuracy and under 0.5 macro-F1 on a deliberately skewed set.

**The prompt-only baseline records unparseable output as wrong.** Retrying or
defaulting to the majority class would flatter the baseline that the trained
systems have to beat.

## Running it

```bash
python3 projects/p08_finetuning_pipeline/demo.py
python3 -m pytest projects/p08_finetuning_pipeline -q
```

## Results

Measured by running `demo.py` on Python 3.13, macOS, CPU only, single process.
Configuration: `HashingEmbedder(dim=256)`, 6 intents, 14 epochs, SGD at learning
rate 0.2, LoRA rank 4 and alpha 4, seeds fixed. Wall-clock figures move with
machine load; accuracies and parameter counts are deterministic.

**Data.** Pretrain bank 270 generated, 135 kept. Target bank 570 generated,
249 kept after removing 321 near-duplicates (0 too short, 0 too long). Label
counts after filtering: billing 56, request_feature 49, report_bug 48,
track_order 40, cancel_subscription 33, password_reset 23, a 2.43 to 1 ratio
that the balance check flags. Stratified split: 150 train, 51 validation,
48 test.

**Frozen base.** 1542 weights, trained in 0.08 s on the formal bank, scoring
0.812 accuracy on the colloquial test set. That gap is what the adapters exist
to close.

**Three systems on the same 48-example held-out test set:**

| system | parameters trained | accuracy | macro F1 |
| --- | --- | --- | --- |
| prompt-only, no training | 0 | 0.688 | 0.704 |
| frozen base, no adapter | 0 | 0.812 | 0.831 |
| full fine-tune | 1542 | 1.000 | 1.000 |
| LoRA r=4, alpha=4 | 1048 | 1.000 | 1.000 |

Random guessing over 6 classes would be 0.167. The prompt-only baseline
produced 0 unparseable responses. Across three consecutive runs, full
fine-tuning took 0.23 to 0.24 s of wall clock and LoRA 0.17 s, the seven-row
ablation 1.19 to 1.22 s, and the whole demo 1.8 to 1.9 s.

**Prompt-only confusion matrix** (rows true, columns predicted):

```
                     billing passwor cancel_ track_o report_ request
  billing_question        11       0       0       0       0       0
  password_reset           0       4       0       0       0       0
  cancel_subscription      2       0       4       0       0       0
  track_order              0       0       0       8       0       0
  report_bug               3       0       0       2       4       0
  request_feature          6       0       0       0       2       2
```

The baseline is lexical, so it handles the classes with distinctive vocabulary
(billing, password, tracking) and collapses `request_feature` into
`billing_question`. Both trained systems produce a clean diagonal on this test
set.

**Merge.** After training, folding `s * (B @ A)` into the base changed the
largest logit on the test set by 1.776e-15 and changed zero predictions. The
merged model is a single 1542-weight matrix with no adapter and no inference
overhead. The base weights were verified untouched at the end of training.

**Ablation** (each row from a fresh adapter over the same frozen base):

```
   rank  alpha    a/r   params   %full   val acc  test acc  macro f1    sec
      1      1   1.00      262   17.0%     0.941     0.979     0.966   0.06
      2      2   1.00      524   34.0%     1.000     1.000     1.000   0.10
      4      4   1.00     1048   68.0%     1.000     1.000     1.000   0.18
      8      8   1.00     2096  135.9%     1.000     1.000     1.000   0.33
      4      1   0.25     1048   68.0%     0.941     0.979     0.983   0.18
      4      8   2.00     1048   68.0%     1.000     1.000     1.000   0.18
      4     16   4.00     1048   68.0%     1.000     1.000     1.000   0.18
```

Rank 1 is the only configuration that loses accuracy, and rank 2 at 34 percent
of the full parameter count already matches the full fine-tune. Holding rank at
4 and dropping alpha to 1 (scaling 0.25) costs as much accuracy as dropping to
rank 1, which is the point of the `alpha / r` scaling: the effective magnitude
of the update matters as much as the capacity. Rank 8 trains *more* parameters
than the full fine-tune, because with only 6 output classes the adapter cost
`r * (D + C)` overtakes the dense cost `D * C` at `r > C`. That is a real
property of LoRA at small output dimension and it is the reason the savings
here are modest.

**Where the savings actually come from.** LoRA replaces `D * C` weights with
`r * (D + C)`, so the saving scales with `min(D, C)`. Here `C = 6`, so rank 4
is 68 percent of full. On a transformer attention projection with
`D = C = 4096`, rank 8 would be `8 * 8192 = 65,536` against `4096 * 4096 =
16,777,216`, or 0.39 percent. That last pair of figures is arithmetic from the
same formula, not something measured here, and is labelled as such.

**Tests.** 23 tests, including analytic gradients for both `A` and `B` checked
against central finite differences, the base proven byte-identical after 40
LoRA update steps, an exact merge check, and an assertion that every ablation
row starts from the same unmodified base.

## Limits

- Both trained systems reach 1.000 on this test set. That is a property of a
  synthetic 6-class problem with lexically distinctive intents and 48 test
  examples, not evidence that LoRA equals full fine-tuning in general. The
  interesting number here is the rank-1 row, which is the only configuration
  the task is hard enough to separate.
- 48 test examples means accuracy moves in steps of about 0.021. Any difference
  smaller than that is noise, and no claim above rests on one.
- The features are lexical. `HashingEmbedder` is character and word hashing, so
  a paraphrase with no shared sub-words is invisible to it. A real sentence
  encoder would change the absolute numbers and would not change the comparison
  between the three systems.
- `EchoLLM` is not a language model. Its role in data generation is a real
  schema-constrained call that produces deterministic surface variation, and
  its role as the prompt-only baseline is a real lexical decision, but a
  frontier model would make the baseline much stronger and would narrow the gap
  the adapters are closing.
- SGD, not Adam, and no learning-rate schedule, no early stopping, no gradient
  clipping. The optimiser is held identical across all three systems so the
  difference is attributable to the adaptation method; it is not tuned.
- Only the output projection is adapted, because there is only one matrix. A
  transformer has an adapter per targeted projection per layer, and choosing
  which modules to target is a real decision this scale cannot pose.

## The equivalent GPU run

Same mechanism, applied to a transformer, with `peft`:

```python
# pip install torch transformers peft datasets accelerate
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_id = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=6)

config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=4,                       # the `rank` in this project
    lora_alpha=4,              # the `alpha`; scaling is lora_alpha / r
    lora_dropout=0.05,
    bias="none",               # the frozen base bias, as here
    target_modules=["q_proj", "v_proj"],   # this project has one matrix to adapt
    modules_to_save=["score"],
)

model = get_peft_model(model, config)
model.print_trainable_parameters()
# the peft equivalent of this project's "trainable 1048 of 1542" line
```

For QLoRA, the base is loaded in 4-bit and everything else is unchanged:

```python
from transformers import BitsAndBytesConfig
import torch

quant = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForSequenceClassification.from_pretrained(
    model_id, num_labels=6, quantization_config=quant, device_map="auto")
```

Training and merging, the two commands that map onto `train_lora` and `merge`
in this repo:

```bash
python -m trl.scripts.sft --model_name_or_path meta-llama/Llama-3.2-1B \
  --dataset_name ./intents --use_peft --lora_r 4 --lora_alpha 4 \
  --lora_target_modules q_proj v_proj --learning_rate 2e-4 \
  --num_train_epochs 14 --per_device_train_batch_size 8 --bf16 \
  --output_dir ./out-lora
```

```python
merged = model.merge_and_unload()   # W <- W + (lora_alpha / r) * B @ A
merged.save_pretrained("./out-merged")
```

`merge_and_unload` is `LoRAHead.merge` in `model.py`, and the identity it
relies on is the same one the test suite asserts to floating point precision.
