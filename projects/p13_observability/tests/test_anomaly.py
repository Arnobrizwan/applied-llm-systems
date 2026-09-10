"""Baselines, alert rules, cooldown and fingerprint deduplication."""
import pytest

from projects.p13_observability.anomaly import (
    AlertManager, AlertRule, EwmaBaseline, RollingRate, default_rules, fingerprint,
)


def steady(manager, metric, value, times, labels=None):
    for _ in range(times):
        manager.observe(metric, value, labels or {})


def test_ewma_scores_a_value_before_folding_it_into_the_baseline():
    """Scoring after updating hides the spike inside its own baseline."""
    baseline = EwmaBaseline(alpha=0.2, min_samples=5)
    for _ in range(20):
        baseline.observe(100.0 + (1 if _ % 2 else -1))
    z = baseline.observe(400.0)
    assert z > 3.0, "a 4x spike must be far outside the baseline"
    assert baseline.mean < 200.0, "one spike must not move the baseline to the spike"


def test_a_cold_baseline_scores_zero_rather_than_guessing():
    baseline = EwmaBaseline(min_samples=8)
    for _ in range(7):
        assert baseline.observe(50.0) == 0.0
    assert not baseline.ready
    baseline.observe(50.0)
    assert baseline.ready


def test_flat_series_does_not_produce_infinite_z_scores():
    baseline = EwmaBaseline(min_samples=3)
    for _ in range(10):
        baseline.observe(10.0)
    z = baseline.zscore(20.0)
    assert z != float("inf") and z > 0


def test_alpha_must_be_a_valid_smoothing_factor():
    with pytest.raises(ValueError):
        EwmaBaseline(alpha=0.0)
    with pytest.raises(ValueError):
        EwmaBaseline(alpha=1.5)


def test_rolling_rate_tracks_only_the_window():
    window = RollingRate(window=4)
    for _ in range(4):
        window.observe(True)
    assert window.rate == 1.0
    for _ in range(4):
        window.observe(False)
    assert window.rate == 0.0, "the old failures must fall out of the window"


def test_one_incident_emits_one_alert_not_four_hundred():
    """The whole point of a cooldown, proved with a controlled clock."""
    now = [1000.0]
    rule = AlertRule(name="latency", metric="latency_ms", mode="threshold",
                     threshold=500.0, cooldown_s=300.0)
    manager = AlertManager([rule], clock=lambda: now[0])

    emitted = []
    for _ in range(400):
        emitted += manager.observe("latency_ms", 900.0, {"step": "generate"})

    assert len(emitted) == 1
    assert manager.suppressed_total == 399
    assert manager.pending_suppressed()[emitted[0].fingerprint] == 399


def test_the_next_alert_after_the_cooldown_carries_the_suppressed_count():
    now = [0.0]
    rule = AlertRule(name="latency", metric="latency_ms", mode="threshold",
                     threshold=500.0, cooldown_s=60.0)
    manager = AlertManager([rule], clock=lambda: now[0])
    for _ in range(10):
        manager.observe("latency_ms", 900.0, {"step": "generate"})
    now[0] = 61.0
    again = manager.observe("latency_ms", 900.0, {"step": "generate"})
    assert len(again) == 1
    assert again[0].suppressed_since_last == 9, "silence must not look like recovery"
    assert manager.pending_suppressed() == {}


def test_fingerprints_separate_series_so_cooldowns_are_independent():
    now = [0.0]
    rule = AlertRule(name="latency", metric="latency_ms", mode="threshold",
                     threshold=500.0, cooldown_s=300.0)
    manager = AlertManager([rule], clock=lambda: now[0])
    first = manager.observe("latency_ms", 900.0, {"step": "generate"})
    second = manager.observe("latency_ms", 900.0, {"step": "judge"})
    third = manager.observe("latency_ms", 900.0, {"step": "generate"})
    assert len(first) == len(second) == 1
    assert first[0].fingerprint != second[0].fingerprint
    assert third == [], "the generate series is still inside its cooldown"


def test_fingerprint_is_stable_across_label_ordering():
    assert fingerprint("r", {"a": "1", "b": "2"}) == fingerprint("r", {"b": "2", "a": "1"})
    assert fingerprint("r", {"a": "1"}) != fingerprint("r", {"a": "2"})


def test_a_zscore_rule_stays_quiet_until_it_has_a_baseline():
    rule = AlertRule(name="spike", metric="latency_ms", mode="zscore",
                     threshold=3.0, min_samples=10, cooldown_s=0.0)
    manager = AlertManager([rule], clock=lambda: 0.0)
    assert manager.observe("latency_ms", 5000.0, {}) == [], "no baseline, no claim"
    steady(manager, "latency_ms", 100.0, 12)
    assert manager.observe("latency_ms", 5000.0, {}) != []


def test_below_direction_alerts_on_a_collapse():
    rule = AlertRule(name="throughput_drop", metric="tokens_per_s", severity="warning",
                     mode="threshold", direction="below", threshold=10.0, cooldown_s=0.0)
    manager = AlertManager([rule], clock=lambda: 0.0)
    assert manager.observe("tokens_per_s", 50.0, {}) == []
    assert manager.observe("tokens_per_s", 2.0, {}) != []


def test_rules_reject_nonsense_configuration():
    with pytest.raises(ValueError):
        AlertRule(name="r", metric="m", severity="apocalyptic")
    with pytest.raises(ValueError):
        AlertRule(name="r", metric="m", mode="vibes")
    with pytest.raises(ValueError):
        AlertRule(name="r", metric="m", direction="sideways")


def test_default_rules_cover_latency_errors_and_cost():
    metrics = {r.metric for r in default_rules()}
    assert {"latency_ms", "error_rate", "cost_usd"} <= metrics
    assert any(r.severity == "critical" for r in default_rules())


def test_summary_counts_emitted_and_suppressed_separately():
    now = [0.0]
    manager = AlertManager([AlertRule(name="r", metric="m", mode="threshold",
                                      threshold=1.0, cooldown_s=100.0)],
                           clock=lambda: now[0])
    for _ in range(5):
        manager.observe("m", 9.0, {"tenant": "a"})
    manager.observe("m", 9.0, {"tenant": "b"})
    summary = manager.summary()
    assert summary["alerts_emitted"] == 2
    assert summary["alerts_suppressed"] == 4
    assert summary["distinct_fingerprints"] == 2
