# 11. Streaming Response Infrastructure

**Run it live: [https://applied-llm-systems.vercel.app/s/streaming](https://applied-llm-systems.vercel.app/s/streaming)** - the hosted page runs this code and shows the real output.

Server-sent events with real backpressure, resume-from-offset reconnects,
producer cancellation and measured time to first token.

## The problem

Streaming looks like three lines of code until it meets a network. The four
things that break, all of them in production rather than in the demo:

1. **A slow reader becomes a memory leak.** Put an unbounded queue between the
   decoder and the socket and a client reading at half the generation speed
   accumulates the difference in RAM for the whole life of the connection.
   Multiply by concurrent streams and the process is killed by the OOM killer,
   not by the slow client.
2. **A dropped connection restarts the generation.** A mobile client loses
   signal at token 400 of 600, reissues the original request, and the user sees
   the answer start again from the beginning while the operator pays twice.
3. **A client that walks away leaves a producer running.** The decoder keeps
   generating into a socket nobody is reading, holding a worker and burning
   tokens for output that will never be seen.
4. **An idle connection gets reaped.** The model spends 30 seconds on a tool
   call, a proxy sees no bytes, and the stream is closed under both parties.

## What this builds

- `sse.py` spec-correct framing and an incremental parser: `event`, `id`,
  `retry`, multi-line `data`, and comment heartbeats.
- `backpressure.py` a bounded channel between producer and socket with two
  implemented policies, block and drop, plus the stats to tell them apart.
- `metrics.py` TTFT and inter-token gaps on a monotonic clock, with p50 and p95
  from `llmkit.tracing.percentile`.
- `server.py` an `http.server` SSE endpoint: producer thread, heartbeats,
  resume from `Last-Event-ID`, cancellation on a broken pipe.
- `client.py` a client that tracks the last event id and resumes, plus a raw
  socket consumer with a small receive window used to force genuine TCP flow
  control.

## Architecture

```
  producer thread                 bounded channel              handler thread
  ---------------                 ---------------              --------------
  llm.stream() deltas  --put()-->  [ | | | | ]  --get()-->  frame as SSE
  checks cancelled                  maxsize N               write + flush
  before every token                                              |
        ^                          full? policy:                  v
        |                            block: wait              socket
        |                            drop:  discard,             |
        |                                   count a gap          |
        +------------- cancel() <--------------------------- write raised
                                                             BrokenPipeError

  client                                       server
  ------                                       ------
  remembers last event id  --reconnect-->  Last-Event-ID: 19
                                                 |
                                                 v
                                           cached token list for this stream id,
                                           resume at index 20, no regeneration
```

## Design decisions

**A bounded channel with an explicit policy, not an unbounded queue.** An
unbounded queue does not remove backpressure, it converts it into resident
memory. Both honest responses to a full buffer are implemented and the choice
is a request parameter. Block keeps every token and runs generation at the
consumer's speed, which is correct when the payload is an append-only token
sequence. Drop keeps generation at full speed and bounds memory, which is only
acceptable for refreshable state such as a progress figure or a partial render.
A third option, failing the stream when the buffer fills, was rejected as
uninteresting: it is drop with a lower tolerance and it hides the tradeoff.

**Dropped tokens emit a `gap` event.** Silently discarding deltas ships a
truncated answer that nobody finds out about. When a put succeeds after one or
more drops, the count of skipped deltas is sent as its own event before the
next token, so the loss is in the protocol rather than in a server-side metric.

**Resume serves a cached token list, it does not re-run the model.**
Regenerating and skipping is wrong twice: the operator pays for the tokens a
second time, and at any temperature above zero the resumed text is the tail of
a different answer rather than a continuation of what the client already has.
The registry holds the token sequence per stream id; in production that is
Redis with a TTL and the interface is the same.

**Event ids are the token index.** An id has to be monotonic within a stream and
meaningful to the server, because the server is the party that has to act on
it. An index satisfies both in the smallest possible payload. A UUID per event
satisfies neither.

**The producer checks a cancellation flag before every token.** That single
check is the entire cancellation mechanism. The handler sets it when a socket
write raises, and `cancel()` also wakes a producer parked inside `put()` under
the block policy, which is what stops the thread leaking instead of merely
stopping the generation.

**The demo shrinks socket buffers rather than faking a slow writer.** With
default buffers the kernel absorbs a whole short stream on loopback, the
server's write never blocks, the channel never fills and the policy never has
to decide anything. Setting `SO_SNDBUF` on the server connection and `SO_RCVBUF`
on the client before connect produces real TCP flow control in a few kilobytes.
A `sleep` in the write loop would have produced a nicer-looking demo that proved
nothing.

## Running it

```bash
python3 projects/p11_streaming/demo.py
python3 -m pytest projects/p11_streaming -q
```

The demo starts the SSE server on an ephemeral loopback port and runs five
scenarios in process over real sockets: a baseline stream with TTFT and cadence
percentiles, a mid-stream stall covered by heartbeats, a forced disconnect with
resume, both backpressure policies under an identical slow consumer, and a
client disconnect that stops the producer.

## Results

Measured by running `demo.py` on Python 3.13, macOS, over `127.0.0.1`, with
`EchoLLM` answering a grounded prompt. The answer is 277 tokens, one SSE frame
each. Figures below are from one run, with the spread across four consecutive
runs given where it matters: token counts, event ids and the resume behaviour
are stable, while latencies and drop counts depend on thread scheduling.

**Baseline.** First stream: TTFT 9.49 ms, total 360.26 ms, 786.9 tokens per
second, inter-token gap p50 1.27 ms, p95 1.38 ms. Across 5 streams in that run:
TTFT p50 3.27 ms, p95 9.49 ms. Over four runs, TTFT p50 stayed between 2.88 and
3.27 ms and gap p50 was 1.27 ms every time. The p95 TTFT is always the first
connection, which carries socket setup; on a sample of five that single cold
connection is the 95th percentile. Median TTFT is 0.9 percent of median total
stream time, which is the entire argument for streaming.

**Heartbeats.** A 400 ms mid-stream stall with a 60 ms heartbeat interval
produced 3 heartbeat comments and a largest inter-token gap of 404.7 ms. The
stream still completed with all 277 tokens.

**Reconnect and resume.** The client was killed after 20 tokens. It reconnected
once, the server logged 2 connections for that stream id, and the client
finished with 277 tokens whose event ids were contiguous 0 to 276 with zero
duplicates. The second `open` frame reported `"resumed": true, "start_index":
20`. The reassembled text was byte-identical to an uninterrupted stream. All of
that was identical on every run. A test additionally asserts
`EchoLLM.call_count == 1` across both connections, proving the resume did not
regenerate.

**Backpressure, identical slow consumer** (buffer 4 items, producer 1 ms per
token, consumer 6 ms per token, both socket buffers 2048 bytes):

| policy | delivered | buffer high water | dropped | producer blocked | wall clock |
| --- | --- | --- | --- | --- | --- |
| block | 277 of 277 | 4 of 4 | 0 | 0.1254 s over 3 waits | 2063 ms |
| drop  | 177 of 277 | 4 of 4 | 100 in 2 gap runs | 0 s | 1304 ms |

Across four runs, block delivered 277 of 277 every time with 0 drops and
blocked between 0.1254 and 0.1566 s; drop delivered between 177 and 183 tokens,
discarding 94 to 100, and finished in 1300 to 1350 ms against block's 2038 to
2063 ms. The exact drop count moves because it depends on when the consumer's
6 ms sleep interleaves with the producer's 1 ms pace, which is the honest
character of this policy rather than a defect in the measurement.

Under drop the client received 2 `gap` events carrying the number of tokens
skipped, so the loss reaches the client in the protocol rather than only in a
server-side counter. Both policies held the buffer at 4 items. An unbounded
queue would have looked like the block row with no wait and no loss, while
holding the entire undelivered remainder in memory for the life of the stream.

**Cancellation.** The client disconnected after 5 tokens of a 277 token answer
with the producer pacing at 10 ms per token. The server produced 8 tokens and
then stopped, with `cancelled = True`, on every run. Three tokens of overrun is
the write already in flight plus one loop iteration. Without the check the
producer would have generated all 277 into a dead socket.

**Tests.** 21 tests, including spec edge cases in the parser, a producer parked
in `put()` being freed by `cancel()` so the thread does not leak, resume with no
duplicate ids over a real socket, and both backpressure policies driven through
genuine TCP flow control rather than a simulation.

## Limits

- The resume registry is an in-process dict with no TTL. Behind more than one
  instance a reconnect can land on a node that has never seen the stream id, so
  this needs a shared store before it is horizontally scalable.
- `EchoLLM` produces its whole answer instantly, so generation cadence is
  imposed by a `delay_ms` parameter. TTFT, inter-token gaps, buffer depth,
  blocking time and drop counts are all genuinely measured; the pace being
  measured is synthetic. With a real provider the same code paths run and only
  the numbers move.
- Heartbeat interval, buffer size and policy are per-request query parameters
  because that makes the demo legible. In production they belong in server
  configuration, not in a URL a caller controls.
- There is no authentication, no per-tenant limit and no cost accounting here.
  That is project 07, and the two would be composed in a real service.
- Cancellation is detected when a write fails, which means it is detected on the
  next token rather than instantly. For a slow generation that is fine; for a
  long stall it would need a separate socket readability poll.
