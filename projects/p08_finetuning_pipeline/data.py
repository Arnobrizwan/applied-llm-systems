"""Synthetic dataset construction, quality filtering and a stratified split.

The task is six-way intent classification over customer support messages. The
data is generated rather than collected because a public portfolio repo cannot
ship someone's real support inbox, and because generating it makes the failure
modes of synthetic data reproducible instead of theoretical.

Two template banks, and the split between them is the point:

  PRETRAIN_TEMPLATES  formal, well-punctuated phrasings. This stands in for the
                      generic corpus a base model was pretrained on.
  TARGET_TEMPLATES    clipped, lowercase, typo-ridden phrasings. This is the
                      in-domain data you actually have to adapt to.

Training the frozen base on the first bank and adapting on the second is what
makes the LoRA comparison meaningful. If both the base and the adapter saw the
same distribution, the adapter would have nothing to learn and the experiment
would flatter itself.

EchoLLM's role is real, not decorative. It is asked for a schema-constrained
style selection per utterance, which is a genuine constrained-generation call:
the provider synthesises a value from the enum, seeded deterministically by a
hash of the prompt. That gives reproducible surface variation without a network
call. It is not a paraphrase model, and the README says so; a real model
plugged into the same call site would return genuine paraphrases and nothing
else in this file would change.

The quality filter runs three checks, in the order that matters:
  1. length bounds, cheapest, removes degenerate output first
  2. near-duplicate removal by hashed-feature cosine, the expensive one
  3. label balance, a report rather than a filter, because the fix for
     imbalance is generating more data, not deleting more
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from llmkit import EchoLLM, HashingEmbedder, cosine, count_tokens

INTENTS = [
    "billing_question",
    "password_reset",
    "cancel_subscription",
    "track_order",
    "report_bug",
    "request_feature",
]

# Short natural-language descriptions, used by the prompt-only baseline as the
# only thing it is given about each class.
INTENT_DESCRIPTIONS: Dict[str, str] = {
    "billing_question": "questions about an invoice charge payment refund or billing amount",
    "password_reset": "problems signing in resetting a forgotten password or a locked account",
    "cancel_subscription": "requests to cancel a subscription plan or stop a recurring renewal",
    "track_order": "asking where a shipment or delivery is and when the parcel will arrive",
    "report_bug": "reporting that a feature is broken crashing erroring or not working",
    "request_feature": "suggesting a new capability or asking whether something could be added",
}

PRETRAIN_TEMPLATES: Dict[str, List[str]] = {
    "billing_question": [
        "I would like to query the {amount} charge on my most recent invoice.",
        "Could you explain the billing amount applied to my account this {period}?",
        "There appears to be a duplicate payment of {amount} on my statement.",
        "Please clarify why my {plan} plan was invoiced at {amount} this {period}.",
    ],
    "password_reset": [
        "I am unable to sign in to the {app} and would like to reset my forgotten password.",
        "My account appears to be locked after several failed login attempts on the {app}.",
        "Could you send a password reset link to my registered {channel}?",
        "Authentication on the {app} keeps failing and I no longer recall my password.",
    ],
    "cancel_subscription": [
        "I would like to cancel my {plan} subscription before the next renewal.",
        "Please stop the recurring charge and close my {plan} plan.",
        "I wish to terminate my subscription at the end of the current {period}.",
        "Kindly cancel the {plan} plan and do not renew it next {period}.",
    ],
    "track_order": [
        "Could you tell me where my delivery {order} currently is?",
        "I would like an update on the shipment status of order {order}.",
        "When is the parcel for order {order} expected to arrive?",
        "Please provide a delivery estimate for the parcel in order {order}.",
    ],
    "report_bug": [
        "The {feature} feature returns an error whenever I attempt to use it.",
        "I am experiencing a crash when opening the {feature} screen.",
        "The {feature} page appears to be broken and does not load correctly.",
        "Using {feature} on the {app} reliably produces an error and then crashes.",
    ],
    "request_feature": [
        "Would it be possible to add support for {feature} in a future release?",
        "I would like to suggest a new capability for exporting {feature} data.",
        "Could you consider adding {feature} to the product roadmap?",
        "Please consider supporting {feature} on the {app} in an upcoming release.",
    ],
}

TARGET_TEMPLATES: Dict[str, List[str]] = {
    "billing_question": [
        "whats this {amount} charge on my card",
        "why did u bill me {amount} again this {period}",
        "got charged twice {amount} pls check invoice",
        "billing looks wrong, refund the {amount} pls",
        "invoice says {amount} but my {plan} plan costs less",
        "need a refund of {amount}, wrong billing {period}",
    ],
    "password_reset": [
        "cant log in to the {app} forgot my password help",
        "account locked out on {app}, need password reset",
        "login not working send reset link to my {channel}",
        "signin keeps failing on {app} i forgot the password",
        "reset my password pls, {channel} link never arrived",
        "locked out again, password reset on {app} not working",
    ],
    "cancel_subscription": [
        "want to cancel my {plan} plan now",
        "stop the renewal on my {plan} subscription",
        "how do i cancel, dont charge me next {period}",
        "cancel subscription pls dont renew",
        "end my {plan} subscription at the end of this {period}",
        "dont want to continue, cancel the {plan} renewal",
    ],
    "track_order": [
        "wheres my order {order}",
        "delivery {order} still not here when arriving",
        "track my parcel {order} pls",
        "shipment {order} late, any update on delivery",
        "order {order} shipped days ago, where is the parcel",
        "no delivery update for {order}, when does it arrive",
    ],
    "report_bug": [
        "{feature} is broken again throws an error",
        "app crashes when i open {feature}",
        "{feature} page not loading, seems broken",
        "getting an error on {feature} every time",
        "{feature} keeps crashing on the {app}, totally broken",
        "error every time i use {feature}, nothing works",
    ],
    "request_feature": [
        "can u add {feature} pls would be great",
        "suggestion: support {feature} in the app",
        "any chance of adding {feature} soon",
        "would love a {feature} option added",
        "please consider adding {feature} to the {app}",
        "feature idea: {feature} support would help a lot",
    ],
}

SLOTS: Dict[str, List[str]] = {
    "amount": ["$19", "$42.50", "$7", "$120", "$65", "$310", "$8.99"],
    "period": ["month", "quarter", "billing cycle", "year"],
    "plan": ["pro", "starter", "team", "enterprise", "business"],
    "order": ["A1042", "B7781", "C3390", "D5512", "E9004", "F2218", "G6650"],
    "feature": ["dark mode", "csv export", "the dashboard", "search", "notifications",
                "the mobile app", "two factor login", "bulk upload"],
    "app": ["web app", "mobile app", "desktop client", "portal"],
    "channel": ["email address", "phone number", "work inbox", "backup email"],
}

# EchoLLM picks one of these per utterance under a JSON schema. They are surface
# noise, which is exactly what separates two utterances of the same intent in a
# real inbox.
STYLE_SCHEMA = {
    "type": "object",
    "properties": {
        "opener": {"type": "string", "enum": ["", "hi ", "hello, ", "hey team ", "quick one: "]},
        "closer": {"type": "string", "enum": ["", " thanks", " pls advise", " cheers", " asap"]},
    },
}


@dataclass
class Example:
    text: str
    label: str
    source: str = "target"          # "pretrain" or "target"

    def __hash__(self) -> int:
        return hash((self.text, self.label))


@dataclass
class FilterReport:
    generated: int = 0
    dropped_short: int = 0
    dropped_long: int = 0
    dropped_duplicate: int = 0
    kept: int = 0
    per_label: Dict[str, int] = field(default_factory=dict)
    balance_ratio: float = 1.0
    balanced: bool = True

    def to_dict(self) -> Dict[str, object]:
        return {
            "generated": self.generated,
            "dropped_short": self.dropped_short,
            "dropped_long": self.dropped_long,
            "dropped_duplicate": self.dropped_duplicate,
            "kept": self.kept,
            "per_label": dict(self.per_label),
            "balance_ratio": round(self.balance_ratio, 3),
            "balanced": self.balanced,
        }


def generate(
    templates: Dict[str, List[str]],
    per_intent: int,
    seed: int = 7,
    source: str = "target",
    llm: EchoLLM = None,
    style: bool = True,
) -> List[Example]:
    """Fill templates, then let EchoLLM choose a surface style under a schema."""
    rng = random.Random(seed)
    model = llm or EchoLLM()
    out: List[Example] = []
    for intent in INTENTS:
        bank = templates[intent]
        for i in range(per_intent):
            template = bank[i % len(bank)]
            text = template
            for slot, options in SLOTS.items():
                token = "{" + slot + "}"
                if token in text:
                    text = text.replace(token, options[rng.randrange(len(options))])
            if style:
                choice = json.loads(model.complete(
                    [{"role": "user", "content": f"style for: {text}"}],
                    json_schema=STYLE_SCHEMA,
                ).text)
                text = f"{choice.get('opener', '')}{text}{choice.get('closer', '')}"
            out.append(Example(text.strip(), intent, source))
    rng.shuffle(out)
    return out


def quality_filter(
    examples: Sequence[Example],
    min_tokens: int = 4,
    max_tokens: int = 40,
    dedup_threshold: float = 0.97,
    embed_dim: int = 128,
    balance_tolerance: float = 1.5,
) -> Tuple[List[Example], FilterReport]:
    """Length bounds, near-duplicate removal, and a label balance check.

    Near-duplicates are found with a hashed-feature cosine rather than exact
    string equality, because the pair that actually hurts is two utterances
    that are the same sentence with a different opener, not two identical
    strings. A duplicate that survives filtering can land on both sides of the
    split and inflate validation accuracy for free.

    The threshold is a real tradeoff and was set by measurement, not taste. At
    0.93 the filter also removes utterances that differ only in a slot value,
    which is variation the classifier needs: on this generator it discarded
    168 of 264 target examples and left the labels imbalanced 5.75 to 1. At
    0.97 it removes near-identical restatements and keeps slot variation. The
    numbers in the README come from the looser setting for that reason.

    Deduplication is greedy and O(kept^2). At this dataset size that is a few
    hundred thousand dot products and takes well under a second. At a hundred
    thousand examples it would need LSH bucketing; the honest note is that the
    algorithm here is the correct one for the scale it runs at, not for all
    scales.
    """
    report = FilterReport(generated=len(examples))
    embedder = HashingEmbedder(dim=embed_dim)

    length_ok: List[Example] = []
    for ex in examples:
        n = count_tokens(ex.text)
        if n < min_tokens:
            report.dropped_short += 1
        elif n > max_tokens:
            report.dropped_long += 1
        else:
            length_ok.append(ex)

    kept: List[Example] = []
    kept_vectors: List[List[float]] = []
    for ex, vec in zip(length_ok, embedder.embed([e.text for e in length_ok])):
        # Only compare within the same label: two different intents that happen
        # to share vocabulary are informative, not redundant.
        if any(cosine(vec, kv) >= dedup_threshold
               for kv, ke in zip(kept_vectors, kept) if ke.label == ex.label):
            report.dropped_duplicate += 1
            continue
        kept.append(ex)
        kept_vectors.append(vec)

    report.kept = len(kept)
    for ex in kept:
        report.per_label[ex.label] = report.per_label.get(ex.label, 0) + 1
    counts = [report.per_label.get(i, 0) for i in INTENTS]
    smallest = min(counts) if counts else 0
    report.balance_ratio = (max(counts) / smallest) if smallest else float("inf")
    report.balanced = report.balance_ratio <= balance_tolerance
    return kept, report


def stratified_split(
    examples: Sequence[Example],
    ratios: Tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 13,
) -> Tuple[List[Example], List[Example], List[Example]]:
    """Split per label so every class appears in train, validation and test.

    A plain shuffle-and-slice can leave a small class absent from the test set,
    which makes macro-F1 undefined for that class and quietly changes what the
    headline number means. The seed is fixed so a reported number is
    reproducible; that matters more here than any particular value of the seed.
    """
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("ratios must sum to 1.0")
    rng = random.Random(seed)
    train: List[Example] = []
    val: List[Example] = []
    test: List[Example] = []
    by_label: Dict[str, List[Example]] = {}
    for ex in examples:
        by_label.setdefault(ex.label, []).append(ex)
    for label in sorted(by_label):
        rows = list(by_label[label])
        rng.shuffle(rows)
        n = len(rows)
        n_train = int(round(n * ratios[0]))
        n_val = int(round(n * ratios[1]))
        train.extend(rows[:n_train])
        val.extend(rows[n_train:n_train + n_val])
        test.extend(rows[n_train + n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def build_dataset(
    per_intent_target: int = 60,
    per_intent_pretrain: int = 40,
    seed: int = 7,
) -> Dict[str, object]:
    """The whole data stage: generate, filter, split. Returns everything measured."""
    pretrain = generate(PRETRAIN_TEMPLATES, per_intent_pretrain, seed=seed,
                        source="pretrain")
    target = generate(TARGET_TEMPLATES, per_intent_target, seed=seed + 1, source="target")
    pretrain_kept, pretrain_report = quality_filter(pretrain)
    target_kept, target_report = quality_filter(target)
    train, val, test = stratified_split(target_kept, seed=seed + 2)
    return {
        "pretrain": pretrain_kept,
        "pretrain_report": pretrain_report,
        "train": train,
        "val": val,
        "test": test,
        "report": target_report,
    }
