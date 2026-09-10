"""The bounded channel and its two full-buffer policies."""
import threading
import time

import pytest

from projects.p11_streaming.backpressure import POLICY_BLOCK, POLICY_DROP, BoundedChannel


def test_drop_policy_discards_and_counts_gap_runs_not_individual_drops():
    ch = BoundedChannel(maxsize=2, policy=POLICY_DROP)
    assert ch.put("a") and ch.put("b")
    assert ch.put("c") is False                 # full
    assert ch.put("d") is False                 # same gap, not a second one
    assert ch.stats.dropped == 2
    assert ch.stats.gaps == 1
    ch.get()
    assert ch.put("e") is True                  # room again
    assert ch.put("f") is False                 # a new, separate gap
    assert ch.stats.gaps == 2
    assert ch.stats.max_depth == 2


def test_block_policy_never_loses_an_item_and_waits_for_room():
    ch = BoundedChannel(maxsize=2, policy=POLICY_BLOCK)
    ch.put("a")
    ch.put("b")
    produced = []

    def producer():
        produced.append(ch.put("c"))            # must wait for the consumer

    t = threading.Thread(target=producer)
    t.start()
    time.sleep(0.05)
    assert produced == []                       # still parked in put()
    assert ch.get() == "a"
    t.join(timeout=2)
    assert produced == [True]
    assert ch.stats.dropped == 0
    assert ch.stats.block_events == 1
    assert ch.stats.blocked_s > 0


def test_cancel_unblocks_a_parked_producer_so_the_thread_does_not_leak():
    """A client disconnect must free the producer, not strand it on a full buffer."""
    ch = BoundedChannel(maxsize=1, policy=POLICY_BLOCK)
    ch.put("a")
    outcome = []
    t = threading.Thread(target=lambda: outcome.append(ch.put("b")))
    t.start()
    time.sleep(0.05)
    ch.cancel()
    t.join(timeout=2)
    assert t.is_alive() is False
    assert outcome == [False]
    assert ch.cancelled and ch.finished


def test_close_lets_the_consumer_drain_but_cancel_does_not():
    closed = BoundedChannel(maxsize=4)
    closed.put("a")
    closed.close()
    assert closed.finished is False             # one item still buffered
    assert closed.get() == "a"
    assert closed.finished is True

    cancelled = BoundedChannel(maxsize=4)
    cancelled.put("a")
    cancelled.cancel()
    assert cancelled.get() is None              # buffer abandoned
    assert cancelled.finished is True


def test_get_returns_none_on_an_idle_channel_which_is_what_triggers_a_heartbeat():
    ch = BoundedChannel(maxsize=2)
    started = time.perf_counter()
    assert ch.get(timeout=0.05) is None
    assert time.perf_counter() - started >= 0.04
    assert ch.finished is False                 # idle is not finished


def test_bad_configuration_fails_at_construction():
    with pytest.raises(ValueError):
        BoundedChannel(maxsize=0)
    with pytest.raises(ValueError):
        BoundedChannel(maxsize=4, policy="explode")


def test_depth_is_bounded_under_a_fast_producer_and_a_slow_consumer():
    """The property an unbounded queue cannot offer: memory is capped."""
    ch = BoundedChannel(maxsize=3, policy=POLICY_DROP)
    for i in range(500):
        ch.put(i)
    assert ch.depth() == 3
    assert ch.stats.max_depth == 3
    assert ch.stats.accepted + ch.stats.dropped == 500
