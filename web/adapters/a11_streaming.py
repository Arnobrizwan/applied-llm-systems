"""Hosted adapter for the streaming and backpressure system.

A serverless function cannot hold an open connection, so nothing here binds a
socket. What it does instead is run the three pieces that actually contain the
behaviour: `sse.format_event` for the wire frames, `backpressure.BoundedChannel`
for the buffer between producer and consumer, and `server._produce`, the real
producer thread body, including its cancellation check. The consumer is a loop
in this file that mirrors the writer loop in `server.py` line for line, and the
frames it produces are fed straight back through `sse.parse_stream`, so the
bytes on the page are parsed by the same parser a browser's EventSource would
be imitating.

Resume uses `StreamRegistry`, which caches the token list per stream id. That
is the whole point of the design: a reconnect is served out of the cache from
the offset after the last event id the client acknowledged, so nothing is
generated or billed twice and no token is repeated.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional, Tuple

NUMBER = 11
SLUG = "streaming"
TITLE = "Streaming, Backpressure and Resume"
TAGLINE = "See what a token stream really sends, what happens when the reader is too slow, and how a dropped connection picks up where it left off."

WHAT_IT_DOES = """When an assistant types its answer out word by word, there is
a small protocol underneath doing the work. This page shows you that protocol:
the exact text frames that go down the wire, how long the first word took to
arrive, and how evenly the rest followed.

Then it shows the awkward part. The model produces words faster than a slow
reader can take them, so something has to hold the difference. The buffer here
is deliberately small, and you choose what it does when it fills up: make the
model wait, which keeps every word but slows everything down, or throw words
away, which stays fast but leaves holes in the answer. You see both, with the
cost of each measured.

Last, it kills the connection halfway through and reconnects. The reader tells
the server the id of the last word it actually received, and the server carries
on from the next one, so the answer arrives complete with nothing repeated.

One honest note: this page shows the protocol and the buffer behaviour, not a
live open connection. The server this runs on is a serverless function that
cannot hold a stream open, so the frames, the timings and the buffer decisions
are produced in memory and printed. Everything you see is real code doing the
real thing, just not across a socket."""

INPUT_LABEL = "Type block, drop, compare, or disconnect"
PLACEHOLDER = "block"

EXAMPLES = ["block", "drop", "disconnect", "compare"]

SOURCE = "projects/p11_streaming"

# Trimmed for the time budget. The project's own demo streams the full 277
# token answer five times over a real socket with a 6 ms per token consumer,
# which is roughly ten seconds of wall clock. Here the answer is cut to the
# first TOKEN_CAP tokens and the slow consumer is 3 ms per token, which is
# still slower than the producer by a wide margin and still fills a 4 slot
# buffer, so every policy decision is genuinely exercised.
TOKEN_CAP = 70
BUFFER = 4
CONSUMER_DELAY_S = 0.003
PRODUCER_DELAY_S = 0.0005
KILL_AFTER = 18


def _show_frame(frame: bytes, add: Any) -> None:
    """Print one frame, keeping the single blank line that terminates it."""
    text = frame.decode("utf-8")
    for line in text.rstrip("\n").split("\n"):
        add("    | " + line)
    add("    |")


def _pick(user_input: str) -> str:
    text = (user_input or "").strip().lower()
    if not text:
        text = EXAMPLES[0]
    if "disconnect" in text or "resume" in text or "reconnect" in text or "kill" in text:
        return "disconnect"
    if "compare" in text or "both" in text:
        return "compare"
    if "drop" in text or "discard" in text or "lose" in text:
        return "drop"
    return "block"


class _Result:
    def __init__(self) -> None:
        self.frames: List[bytes] = []
        self.tokens: List[str] = []
        self.token_ids: List[int] = []
        self.gap_frames: List[int] = []
        self.heartbeats = 0
        self.last_event_id: Optional[str] = None
        self.completed = False
        self.stats: Dict[str, Any] = {}
        self.metrics: Any = None
        self.wall_ms = 0.0

    @property
    def text(self) -> str:
        return "".join(self.tokens)

    @property
    def duplicate_ids(self) -> List[int]:
        seen, dupes = set(), []
        for i in self.token_ids:
            if i in seen:
                dupes.append(i)
            seen.add(i)
        return dupes


def _drive(record: Any, start_index: int, policy: str, maxsize: int,
           consumer_delay_s: float, producer_delay_s: float,
           kill_after: Optional[int] = None, stall_at: int = -1,
           stall_s: float = 0.0, heartbeat_s: float = 0.0) -> _Result:
    """One connection, in process.

    The producer is the project's own `_produce`, running on a thread exactly
    as it does under the HTTP server. The loop below is the writer half of
    `server.Handler._stream`, with `write()` replaced by appending to a list
    instead of pushing bytes at a socket.
    """
    from projects.p11_streaming.backpressure import BoundedChannel
    from projects.p11_streaming.metrics import StreamMetrics
    from projects.p11_streaming.server import DEFAULT_RETRY_MS, _produce
    from projects.p11_streaming.sse import format_comment, format_event

    res = _Result()
    metrics = StreamMetrics("in-process").start()
    res.metrics = metrics
    channel = BoundedChannel(maxsize=maxsize, policy=policy)
    producer = threading.Thread(
        target=_produce,
        args=(record, channel, start_index, producer_delay_s, stall_at, stall_s),
        name="produce", daemon=True)

    started = time.perf_counter()
    producer.start()

    res.frames.append(format_event(
        data=f'{{"stream_id": "{record.stream_id}", '
             f'"resumed": {str(start_index > 0).lower()}, '
             f'"start_index": {start_index}, "policy": "{policy}", "buffer": {maxsize}}}',
        event="open", retry_ms=DEFAULT_RETRY_MS))

    alive = True
    last_write = time.perf_counter()
    while alive and not channel.finished:
        item = channel.get(timeout=0.02)
        if item is None:
            if heartbeat_s and time.perf_counter() - last_write >= heartbeat_s:
                res.frames.append(format_comment("heartbeat"))
                res.heartbeats += 1
                last_write = time.perf_counter()
            continue
        if item["skipped"]:
            res.frames.append(format_event(
                data=f'{{"dropped": {item["skipped"]}}}', event="gap",
                event_id=str(item["index"] - 1)))
            res.gap_frames.append(item["skipped"])
        res.frames.append(format_event(data=item["text"], event="token",
                                       event_id=str(item["index"])))
        metrics.record_token()
        res.tokens.append(item["text"])
        res.token_ids.append(item["index"])
        res.last_event_id = str(item["index"])
        last_write = time.perf_counter()
        if consumer_delay_s:
            time.sleep(consumer_delay_s)
        if kill_after is not None and len(res.tokens) >= kill_after:
            # The client walked away. cancel() is what a broken pipe triggers
            # in the real handler, and it is what unblocks a producer parked
            # inside put() instead of leaking the thread.
            alive = False

    if not alive:
        channel.cancel()
    producer.join(timeout=2.0)
    with record.lock:
        record.channel_stats = channel.stats.to_dict()
    res.stats = channel.stats.to_dict()

    if alive:
        res.completed = True
        res.frames.append(format_event(
            data=f'{{"tokens": {len(record.tokens)}, "delivered_from": {start_index}, '
                 f'"dropped": {channel.stats.dropped}}}',
            event="done",
            event_id=str(len(record.tokens) - 1) if record.tokens else "0"))
    metrics.stop()
    res.wall_ms = (time.perf_counter() - started) * 1000.0
    return res


def _parsed_back(res: _Result) -> Tuple[int, int]:
    """Feed the produced bytes through the project's own parser."""
    from projects.p11_streaming.sse import parse_stream

    lines: List[bytes] = []
    for frame in res.frames:
        for line in frame.split(b"\n"):
            lines.append(line + b"\n")
    events = list(parse_stream(iter(lines), include_comments=True))
    tokens = sum(1 for e in events if e.event == "token")
    return len(events), tokens


def run(user_input: str) -> str:
    try:
        return _run(user_input)
    except Exception as exc:  # an adapter must never take the page down
        return (f"This demo hit an unexpected error and stopped: "
                f"{type(exc).__name__}: {exc}\n"
                "No connection was opened; everything here runs in memory.")


def _run(user_input: str) -> str:
    from projects.p11_streaming.corpus import EVIDENCE, QUESTION
    from projects.p11_streaming.server import StreamRegistry

    choice = _pick(user_input)
    out: List[str] = []
    add = out.append

    registry = StreamRegistry(system_prompt=EVIDENCE)

    def record_for(stream_id: str) -> Any:
        rec = registry.record(stream_id, QUESTION)
        del rec.tokens[TOKEN_CAP:]      # trimmed to fit the page's time budget
        return rec

    full_tokens = len(record_for("probe").tokens)

    add("STREAMING, BACKPRESSURE AND RESUME  protocol and buffers, no socket")
    add("=" * 78)
    add(f"  you chose           {choice}")
    add(f"  question            {QUESTION!r}")
    add(f"  answer length       {full_tokens} tokens, one SSE frame each")
    add(f"  buffer              {BUFFER} slots between the producer and the reader")
    add(f"  slow reader         {CONSUMER_DELAY_S * 1000:.0f} ms per token, against a "
        f"producer that needs {PRODUCER_DELAY_S * 1000:.1f} ms")
    add("")
    add("  TRIMMED FOR THIS PAGE: the project's own demo streams the full 277 token")
    add(f"  answer five times over a loopback socket with a 6 ms reader, about ten")
    add(f"  seconds of wall clock. Here the answer is cut to its first {TOKEN_CAP} tokens and")
    add("  the reader is 3 ms per token. The buffer still fills, both policies still")
    add("  make their decision, and the resume still replays from the last event id.")

    # -- 1. the wire ------------------------------------------------------
    baseline = _drive(record_for("baseline"), 0, "block", 64, 0.0, PRODUCER_DELAY_S)
    add("")
    add("-" * 78)
    add("1. WHAT ACTUALLY GOES DOWN THE WIRE")
    add("-" * 78)
    add("  a fast reader, so nothing is queued. The first frames, exactly as bytes,")
    add("  each one closed by a blank line:")
    add("")
    for frame in baseline.frames[:4]:
        _show_frame(frame, add)
    add("    | ...")
    add("")
    add("  and the closing frame")
    _show_frame(baseline.frames[-1], add)
    add("")
    events, parsed_tokens = _parsed_back(baseline)
    add(f"  fed back through the project's own parser: {events} frames read, "
        f"{parsed_tokens} of them tokens")
    add("  the id on each frame is the token index. That is the only reason a resume")
    add("  can work: the reader can name exactly how far it got.")

    # -- 2. timing --------------------------------------------------------
    s = baseline.metrics.summary()
    add("")
    add("-" * 78)
    add("2. TIMING  a fast reader, nothing in the way")
    add("-" * 78)
    add(f"  time to first token   {s['ttft_ms']:.2f} ms")
    add(f"  whole answer          {s['total_ms']:.2f} ms")
    add(f"  share spent waiting for the first token   "
        f"{100.0 * s['ttft_ms'] / max(s['total_ms'], 0.001):.1f} percent")
    add(f"  between tokens        p50 {s['gap_p50_ms']:.2f} ms, "
        f"p95 {s['gap_p95_ms']:.2f} ms, worst {s['gap_max_ms']:.2f} ms")
    add(f"  throughput            {s['tokens_per_s']:.0f} tokens per second")
    add("  time to first token is the number a reader can feel. An answer that takes")
    add("  four seconds but starts printing in 200 ms reads as fast; one that arrives")
    add("  all at once after two seconds of a spinner reads as slow.")
    add("  p95 is tracked separately from the average because one 900 ms stall in the")
    add("  middle of an otherwise smooth stream is what makes a stream look broken.")

    # -- 3. backpressure --------------------------------------------------
    add("")
    add("-" * 78)
    add(f"3. THE BUFFER UNDER A SLOW READER  {BUFFER} slots, "
        f"{CONSUMER_DELAY_S * 1000:.0f} ms per token")
    add("-" * 78)
    policies = ("block", "drop") if choice in ("compare", "disconnect") else (
        choice, "drop" if choice == "block" else "block")
    runs: Dict[str, _Result] = {}
    for policy in policies:
        runs[policy] = _drive(record_for(f"bp-{policy}"), 0, policy, BUFFER,
                              CONSUMER_DELAY_S, PRODUCER_DELAY_S)
    add(f"  {'policy':<8}{'delivered':>11}{'dropped':>9}{'gap runs':>10}"
        f"{'high water':>12}{'producer waited':>17}{'wall ms':>10}")
    add("  " + "-" * 77)
    for policy in ("block", "drop"):
        r = runs.get(policy)
        if r is None:
            continue
        st = r.stats
        add(f"  {policy:<8}{len(r.tokens):>7} of {full_tokens:<2}{st['dropped']:>9}"
            f"{st['gaps']:>10}{st['max_depth']:>7} of {BUFFER:<2}"
            f"{st['blocked_s'] * 1000:>12.0f} ms{r.wall_ms:>10.0f}")
    add("")
    primary = runs.get(choice if choice in runs else policies[0])
    if primary is not None and primary.gap_frames:
        add("  under drop the loss is announced, never silent. A gap frame looks like:")
        for frame in primary.frames:
            if b"event: gap" in frame:
                _show_frame(frame, add)
                break
        add("  a dropped token nobody is told about is a silently truncated answer.")
    if "block" in runs and "drop" in runs:
        blk, drp = runs["block"], runs["drop"]
        add(f"  block kept all {len(blk.tokens)} tokens and took {blk.wall_ms:.0f} ms, "
            f"waiting {blk.stats['blocked_s'] * 1000:.0f} ms across "
            f"{blk.stats['block_events']} pauses.")
        add(f"  drop finished in {drp.wall_ms:.0f} ms and lost "
            f"{drp.stats['dropped']} tokens in {drp.stats['gaps']} runs.")
    add("  neither is free. An unbounded queue looks like block with no wait and no")
    add("  loss, right up to the point where one slow reader per stream turns the")
    add("  difference in speed into resident memory and the process is killed.")
    add("  block is right for an answer someone is reading. drop is only right when")
    add("  the payload is refreshable state, a progress percentage or a preview.")

    # -- 4. disconnect and resume -----------------------------------------
    add("")
    add("-" * 78)
    add(f"4. DISCONNECT AND RESUME  the connection dies after {KILL_AFTER} tokens")
    add("-" * 78)
    rec = record_for("resume")
    leg1 = _drive(rec, 0, "block", 16, 0.0, PRODUCER_DELAY_S, kill_after=KILL_AFTER)
    add(f"  first connection      {len(leg1.tokens)} tokens, last event id "
        f"{leg1.last_event_id}, completed={leg1.completed}")
    add(f"  producer stopped      the server generated {rec.produced} tokens then saw the")
    add(f"                        cancel and stopped, instead of generating all "
        f"{len(rec.tokens)} into a dead connection")
    resume_from = int(leg1.last_event_id or "-1") + 1
    leg2 = _drive(rec, resume_from, "block", 16, 0.0, PRODUCER_DELAY_S)
    add(f"  reconnect sends       Last-Event-ID: {leg1.last_event_id}")
    add(f"  second connection     resumed at index {resume_from}, "
        f"{len(leg2.tokens)} more tokens, completed={leg2.completed}")
    combined_ids = leg1.token_ids + leg2.token_ids
    combined_text = leg1.text + leg2.text
    whole = _drive(record_for("uninterrupted"), 0, "block", 64, 0.0, PRODUCER_DELAY_S)
    add("")
    add(f"  event ids received    {combined_ids[:4]} ... {combined_ids[-4:]}")
    add(f"  contiguous 0..{len(combined_ids) - 1}       "
        f"{combined_ids == list(range(len(combined_ids)))}")
    add(f"  duplicated ids        {leg1.duplicate_ids + leg2.duplicate_ids or 'none'}")
    add(f"  text matches a stream that was never interrupted   "
        f"{combined_text == whole.text}")
    add("  the resume is served out of the cached token list, so the model is not run")
    add("  a second time. Re-running it would bill the caller twice, and at any")
    add("  temperature above zero the continuation would not match what the reader")
    add("  already has on screen.")

    add("")
    add("-" * 78)
    add("  Nothing above opened a connection. The SSE frames were built by the")
    add("  project's own formatter and read back by its own parser, the buffer is the")
    add("  project's BoundedChannel, and the producer is the project's own thread body")
    add("  including the cancellation check. A serverless function cannot hold a")
    add("  stream open, so the protocol is shown rather than performed.")
    return "\n".join(out)
