"""Data generation, quality filtering and the stratified split."""
import pytest

from llmkit import count_tokens

from projects.p08_finetuning_pipeline.data import (
    INTENTS,
    TARGET_TEMPLATES,
    Example,
    build_dataset,
    generate,
    quality_filter,
    stratified_split,
)


def test_generation_is_deterministic_and_covers_every_intent():
    a = generate(TARGET_TEMPLATES, 6, seed=5)
    b = generate(TARGET_TEMPLATES, 6, seed=5)
    assert [(e.text, e.label) for e in a] == [(e.text, e.label) for e in b]
    assert {e.label for e in a} == set(INTENTS)
    assert all("{" not in e.text for e in a)          # every slot was filled


def test_quality_filter_enforces_length_bounds():
    rows = [
        Example("hi", "report_bug"),                                   # too short
        Example("the dashboard is broken and throws an error", "report_bug"),
        Example("word " * 80, "report_bug"),                           # too long
    ]
    kept, report = quality_filter(rows, min_tokens=4, max_tokens=40)
    assert report.dropped_short == 1
    assert report.dropped_long == 1
    assert [e.text for e in kept] == ["the dashboard is broken and throws an error"]
    assert all(4 <= count_tokens(e.text) <= 40 for e in kept)


def test_near_duplicates_are_removed_but_only_within_a_label():
    """Two intents sharing vocabulary is signal; the same sentence twice is not."""
    rows = [
        Example("cant log in forgot my password help", "password_reset"),
        Example("cant log in forgot my password help", "password_reset"),   # exact
        Example("cant log in forgot my password  help", "password_reset"),  # near
        Example("cant log in forgot my password help", "report_bug"),       # other label
    ]
    kept, report = quality_filter(rows, dedup_threshold=0.97)
    assert report.dropped_duplicate == 2
    assert len(kept) == 2
    assert {e.label for e in kept} == {"password_reset", "report_bug"}


def test_balance_check_reports_rather_than_deletes():
    rows = [Example(f"billing question number {i} about an invoice charge", "billing_question")
            for i in range(10)]
    rows += [Example("cant log in forgot my password help", "password_reset")]
    kept, report = quality_filter(rows, balance_tolerance=1.5)
    assert report.balanced is False
    assert report.balance_ratio > 1.5
    assert len(kept) == len(rows)                     # nothing was deleted for imbalance


def test_stratified_split_puts_every_class_in_every_partition():
    rows = [Example(f"{label} message variant {i}", label)
            for label in INTENTS for i in range(10)]
    train, val, test = stratified_split(rows, ratios=(0.6, 0.2, 0.2), seed=1)
    for part in (train, val, test):
        assert {e.label for e in part} == set(INTENTS)
    assert len(train) + len(val) + len(test) == len(rows)


def test_split_is_reproducible_and_leak_free():
    rows = [Example(f"{label} message variant {i}", label)
            for label in INTENTS for i in range(10)]
    first = stratified_split(rows, seed=42)
    second = stratified_split(rows, seed=42)
    assert [e.text for e in first[0]] == [e.text for e in second[0]]
    train_texts = {e.text for e in first[0]}
    assert train_texts.isdisjoint({e.text for e in first[1]})
    assert train_texts.isdisjoint({e.text for e in first[2]})
    with pytest.raises(ValueError):
        stratified_split(rows, ratios=(0.5, 0.3, 0.3))


def test_build_dataset_keeps_the_two_banks_separate():
    """The frozen base must not be pretrained on the split it is evaluated against."""
    data = build_dataset(per_intent_target=12, per_intent_pretrain=8)
    pretrain_texts = {e.text for e in data["pretrain"]}
    for part in ("train", "val", "test"):
        assert pretrain_texts.isdisjoint({e.text for e in data[part]})
    assert all(e.source == "pretrain" for e in data["pretrain"])
    assert all(e.source == "target" for e in data["train"])
