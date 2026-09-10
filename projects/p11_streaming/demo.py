"""End-to-end streaming demo over a real loopback socket.

Five scenarios, all against one running server:
  1. a normal stream, with TTFT and inter-token percentiles
  2. heartbeats holding an idle connection open through a mid-stream stall
  3. a forced mid-stream disconnect, reconnect and resume from Last-Event-ID
  4. the two backpressure policies under an identical slow consumer
  5. a client that walks away, proving the producer stops
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import time
import urllib.parse

from llmkit import percentile

from projects.p11_streaming.client import SSEClient, slow_socket_consume
from projects.p11_streaming.corpus import EVIDENCE, QUESTION
from projects.p11_streaming.metrics import aggregate
from projects.p11_streaming.server import StreamRegistry, serve

# One slow-consumer profile, reused for both policies so the comparison is fair.
SLOW = {"maxsize": 4, "producer_delay_ms": 1, "consumer_delay_s": 0.006,
        "sndbuf": 2048, "rcvbuf": 2048}


def rule(title):
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def main():
    registry = StreamRegistry(system_prompt=EVIDENCE)
    httpd, base_url, registry = serve(registry)
    host, port = "127.0.0.1", httpd.server_address[1]
    client = SSEClient(base_url)
    print(f"SSE server listening on {base_url}")

    try:
        rule("1. BASELINE STREAM  time to first token and cadence")
        runs = []
        for i in range(5):
            r = client.consume(params={"prompt": QUESTION, "stream_id": f"base-{i}", "delay_ms": 1})
            runs.append(r.metrics)
        first = runs[0].summary()
        total_tokens = first["tokens"]
        print(f"  answer      {total_tokens} tokens, streamed one SSE frame each")
        print(f"  first run   ttft {first['ttft_ms']} ms, total {first['total_ms']} ms, "
              f"{first['tokens_per_s']} tokens/s")
        print(f"  gaps        p50 {first['gap_p50_ms']} ms, p95 {first['gap_p95_ms']} ms, "
              f"max {first['gap_max_ms']} ms")
        agg = aggregate(runs)
        print(f"  {agg['streams']} streams  ttft p50 {agg['ttft_p50_ms']} ms / p95 {agg['ttft_p95_ms']} ms, "
              f"gap p50 {agg['gap_p50_ms']} ms / p95 {agg['gap_p95_ms']} ms")
        ttfts = [m.ttft_ms for m in runs]
        totals = [m.total_ms for m in runs]
        share = percentile(ttfts, 50) / (percentile(totals, 50) or 1) * 100
        print(f"  median ttft is {share:.1f} percent of median total stream time: a reader starts "
              f"seeing output almost immediately, which is the whole point of streaming")

        rule("2. HEARTBEAT  a 400 ms mid-stream stall on a 60 ms heartbeat")
        r = client.consume(params={"prompt": QUESTION, "stream_id": "stall", "delay_ms": 0,
                                   "stall_at": 10, "stall_ms": 400, "heartbeat_ms": 60})
        s = r.metrics.summary()
        print(f"  tokens {s['tokens']}, heartbeat comments received {s['heartbeats']}, "
              f"largest inter-token gap {s['gap_max_ms']} ms")
        print("  the comments carry no data; they exist so no proxy reaps an idle connection")

        rule("3. RECONNECT AND RESUME  connection killed after 20 tokens")
        r = client.consume(params={"prompt": QUESTION, "stream_id": "resume", "delay_ms": 1},
                           kill_after=20)
        ids = r.token_ids
        print(f"  tokens received      {len(r.tokens)}")
        print(f"  reconnects           {r.metrics.reconnects}")
        print(f"  server connections   {registry.get('resume').connections}")
        print(f"  event ids            {ids[:3]} ... {ids[-3:]}")
        print(f"  contiguous 0..{len(ids) - 1}     {ids == list(range(len(ids)))}")
        print(f"  duplicated ids       {r.duplicate_ids}")
        print(f"  stream completed     {r.completed}")
        for frame in r.open_frames:
            print(f"  open frame           {frame['data']}")
        baseline = client.consume(params={"prompt": QUESTION, "stream_id": "resume-check", "delay_ms": 0})
        print(f"  text identical to an uninterrupted stream: {r.text == baseline.text}")

        rule("4. BACKPRESSURE  same slow consumer, two policies")
        print(f"  buffer {SLOW['maxsize']} items, producer {SLOW['producer_delay_ms']} ms/token, "
              f"consumer {SLOW['consumer_delay_s'] * 1000:.0f} ms/token")
        print(f"  socket buffers shrunk to {SLOW['sndbuf']} bytes so real TCP flow control "
              f"is reached on loopback\n")
        results = {}
        for policy in ("block", "drop"):
            sid = f"bp-{policy}"
            query = urllib.parse.urlencode({
                "prompt": QUESTION, "stream_id": sid, "policy": policy,
                "maxsize": SLOW["maxsize"], "delay_ms": SLOW["producer_delay_ms"],
                "sndbuf": SLOW["sndbuf"],
            })
            res = slow_socket_consume(host, port, query,
                                      recv_buffer_bytes=SLOW["rcvbuf"],
                                      per_token_delay_s=SLOW["consumer_delay_s"])
            rec = registry.get(sid)
            results[policy] = (res, rec)
            stats = rec.channel_stats
            print(f"  policy={policy}")
            print(f"    tokens delivered   {len(res.tokens)} of {len(rec.tokens)}")
            print(f"    buffer high water  {stats['max_depth']} of {SLOW['maxsize']}")
            print(f"    dropped            {stats['dropped']} in {stats['gaps']} gap run(s)")
            print(f"    producer blocked   {stats['blocked_s']} s over {stats['block_events']} waits")
            print(f"    gap events seen    {len(res.gaps)}")
            print(f"    wall clock         {res.metrics.total_ms:.0f} ms")
        blk, drp = results["block"][0], results["drop"][0]
        print(f"\n  block kept every token and took {blk.metrics.total_ms:.0f} ms")
        print(f"  drop finished in {drp.metrics.total_ms:.0f} ms and lost "
              f"{results['drop'][1].channel_stats['dropped']} tokens")
        print("  neither is free. an unbounded queue would look like block with no wait")
        print("  and no loss, while growing without limit for the life of the connection")

        rule("5. CANCELLATION  client disconnects after 5 tokens")
        cancel_client = SSEClient(base_url, max_reconnects=0)
        r = cancel_client.consume(params={"prompt": QUESTION, "stream_id": "cancel", "delay_ms": 10},
                                  kill_after=5)
        rec = registry.get("cancel")
        time.sleep(0.4)                       # give the producer a chance to overrun
        print(f"  full answer length   {len(rec.tokens)} tokens")
        print(f"  client received      {len(r.tokens)}")
        print(f"  server produced      {rec.produced}")
        print(f"  producer cancelled   {rec.cancelled}")
        print(f"  wasted generation    {rec.produced - len(r.tokens)} tokens "
              f"({(rec.produced - len(r.tokens)) / max(1, len(rec.tokens)) * 100:.1f} percent "
              f"of the answer)")
        print("  without the cancel check the producer would have generated all "
              f"{len(rec.tokens)} into a dead socket")
    finally:
        httpd.shutdown()
        httpd.server_close()
    print("\nserver stopped")


if __name__ == "__main__":
    main()
