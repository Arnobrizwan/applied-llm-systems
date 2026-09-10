"""Server behaviour over a real loopback socket: resume, cancellation, heartbeats."""
import time
import urllib.parse

import pytest

from projects.p11_streaming.client import SSEClient, slow_socket_consume
from projects.p11_streaming.corpus import EVIDENCE, QUESTION
from projects.p11_streaming.server import StreamRegistry, serve


@pytest.fixture()
def server():
    httpd, base_url, registry = serve(StreamRegistry(system_prompt=EVIDENCE))
    try:
        yield httpd, base_url, registry
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_a_complete_stream_frames_open_tokens_and_done(server):
    _, base_url, registry = server
    result = SSEClient(base_url).consume(params={"prompt": QUESTION, "stream_id": "s"})
    kinds = [e.event for e in result.events]
    assert kinds[0] == "open"
    assert kinds[-1] == "done"
    assert result.completed
    assert len(result.tokens) == len(registry.get("s").tokens) > 50
    assert result.metrics.ttft_ms > 0
    assert result.token_ids == list(range(len(result.tokens)))


def test_resume_continues_from_last_event_id_instead_of_restarting(server):
    """The test the reconnect path exists for: no duplicates, no gaps, same text."""
    _, base_url, registry = server
    client = SSEClient(base_url)
    interrupted = client.consume(params={"prompt": QUESTION, "stream_id": "r", "delay_ms": 1},
                                 kill_after=15)
    clean = client.consume(params={"prompt": QUESTION, "stream_id": "r2"})

    assert interrupted.metrics.reconnects == 1
    assert registry.get("r").connections == 2
    assert interrupted.duplicate_ids == []
    assert interrupted.token_ids == list(range(len(interrupted.token_ids)))
    assert interrupted.text == clean.text
    # The second connection announced where it picked up.
    assert '"resumed": true' in interrupted.open_frames[1]["data"]
    assert '"start_index": 15' in interrupted.open_frames[1]["data"]


def test_resume_does_not_regenerate_the_answer(server):
    """Re-running the model on resume would bill twice and could diverge."""
    _, base_url, registry = server
    llm = registry.llm
    SSEClient(base_url).consume(params={"prompt": QUESTION, "stream_id": "once", "delay_ms": 1},
                                kill_after=10)
    assert registry.get("once").connections == 2
    assert llm.call_count == 1              # one generation across two connections


def test_client_disconnect_stops_the_producer(server):
    """Otherwise the model keeps generating into a socket nobody is reading."""
    _, base_url, registry = server
    client = SSEClient(base_url, max_reconnects=0)
    result = client.consume(params={"prompt": QUESTION, "stream_id": "c", "delay_ms": 10},
                            kill_after=4)
    time.sleep(0.4)
    record = registry.get("c")
    assert len(result.tokens) == 4
    assert record.cancelled is True
    assert record.produced < len(record.tokens) / 2
    assert not result.completed


def test_heartbeats_keep_an_idle_connection_alive_through_a_stall(server):
    _, base_url, _ = server
    result = SSEClient(base_url).consume(
        params={"prompt": QUESTION, "stream_id": "h", "stall_at": 5, "stall_ms": 300,
                "heartbeat_ms": 50})
    assert result.metrics.heartbeats >= 3
    assert result.completed
    assert result.metrics.summary()["gap_max_ms"] > 250     # the stall really happened


def test_block_policy_delivers_everything_and_drop_policy_reports_its_loss(server):
    """Both policies exercised through genuine TCP flow control, not a simulation."""
    httpd, _, registry = server
    port = httpd.server_address[1]
    outcomes = {}
    for policy in ("block", "drop"):
        query = urllib.parse.urlencode({"prompt": QUESTION, "stream_id": f"bp-{policy}",
                                        "policy": policy, "maxsize": 4, "delay_ms": 1,
                                        "sndbuf": 2048})
        result = slow_socket_consume("127.0.0.1", port, query, recv_buffer_bytes=2048,
                                     per_token_delay_s=0.006)
        outcomes[policy] = (result, registry.get(f"bp-{policy}"))

    block_result, block_record = outcomes["block"]
    drop_result, drop_record = outcomes["drop"]
    total = len(block_record.tokens)

    assert len(block_result.tokens) == total
    assert block_record.channel_stats["dropped"] == 0
    assert block_record.channel_stats["blocked_s"] > 0
    assert block_record.channel_stats["max_depth"] <= 4     # memory stayed bounded

    assert drop_record.channel_stats["dropped"] > 0
    assert len(drop_result.tokens) < total
    assert drop_record.channel_stats["max_depth"] <= 4
    assert drop_record.channel_stats["blocked_s"] == 0.0


def test_unknown_path_is_404_and_a_bad_last_event_id_starts_from_zero(server):
    import urllib.error
    import urllib.request

    _, base_url, _ = server
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(base_url + "/nope", timeout=5)
    assert exc.value.code == 404

    result = SSEClient(base_url).consume(
        params={"prompt": QUESTION, "stream_id": "bad", "last_event_id": "not-a-number"})
    assert result.token_ids[0] == 0
    assert result.completed
