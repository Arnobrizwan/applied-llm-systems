"""The SSE server: producer thread, bounded channel, socket writer, resume.

Shape of one request:

    producer thread            bounded channel            handler thread
    pull deltas from  ---->  put(item) under a  ---->  get(), frame as SSE,
    llm.stream()             block/drop policy         write to the socket

The two threads exist so that generation speed and consumption speed are
decoupled, which is the entire subject of this project. If the handler wrote
tokens directly out of the generation loop, a slow reader would throttle the
model and hold the worker for the duration.

Resume is built on a cached, deterministic token list per stream id. When a
client reconnects with Last-Event-ID: 11, the server serves index 12 onward
from that cached list rather than re-running the model. The alternative,
re-generating and skipping, is wrong twice: it pays for the tokens again, and
with a non-zero temperature the resumed text would not be a continuation of
what the client already has, it would be the tail of a different answer.

Event ids are the token index. Ids must be monotonic within a stream and
meaningful to the *server*, and an index is the smallest thing that satisfies
both. A UUID per event would satisfy neither.
"""
from __future__ import annotations

import socket
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple

from llmkit import EchoLLM, LLMProvider

from .backpressure import POLICY_BLOCK, POLICIES, BoundedChannel
from .sse import format_comment, format_event

DEFAULT_RETRY_MS = 750


@dataclass
class StreamRecord:
    """Server-side bookkeeping for one stream id, kept so resume and tests work."""

    stream_id: str
    tokens: List[str]
    produced: int = 0                  # deltas the producer actually emitted
    served: int = 0                    # frames written to a socket
    connections: int = 0
    cancelled: bool = False
    channel_stats: Dict[str, Any] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)


class StreamRegistry:
    """Caches the token list per stream id. This is what makes resume cheap.

    In production this belongs in Redis with a TTL, keyed by stream id, holding
    the tokens emitted so far. The in-memory dict here has the same interface
    and the same semantics for a single process.
    """

    def __init__(self, llm: Optional[LLMProvider] = None, system_prompt: Optional[str] = None):
        self.llm = llm or EchoLLM()
        # An optional grounding block. EchoLLM answers extractively from `[S1]`
        # style evidence, so supplying a long document is how the demo gets a
        # stream with enough tokens for cadence, resume and buffer pressure to
        # be visible. Without it EchoLLM echoes the prompt and every stream is
        # a few dozen tokens long.
        self.system_prompt = system_prompt
        self._records: Dict[str, StreamRecord] = {}
        self._lock = threading.Lock()

    def record(self, stream_id: str, prompt: str) -> StreamRecord:
        with self._lock:
            rec = self._records.get(stream_id)
            if rec is None:
                messages = []
                if self.system_prompt:
                    messages.append({"role": "system", "content": self.system_prompt})
                messages.append({"role": "user", "content": prompt})
                tokens = [t for t in self.llm.stream(messages) if t]
                rec = StreamRecord(stream_id=stream_id, tokens=tokens)
                self._records[stream_id] = rec
            return rec

    def get(self, stream_id: str) -> Optional[StreamRecord]:
        with self._lock:
            return self._records.get(stream_id)

    def clear(self) -> None:
        with self._lock:
            self._records.clear()


def _produce(
    record: StreamRecord,
    channel: BoundedChannel,
    start_index: int,
    token_delay_s: float,
    stall_at: int,
    stall_s: float,
) -> None:
    """Producer thread body.

    Checks `channel.cancelled` before every token so a client disconnect stops
    generation instead of leaving this thread writing into a dead stream. That
    check is the whole cancellation mechanism, and the test that proves it
    works asserts on `record.produced`.
    """
    skipped_since_last_delivery = 0
    for index in range(start_index, len(record.tokens)):
        if channel.cancelled:
            with record.lock:
                record.cancelled = True
            break
        if token_delay_s:
            time.sleep(token_delay_s)
        if stall_at >= 0 and index == start_index + stall_at and stall_s:
            # A real generator stalls: a tool call, a scheduler preemption, a
            # long prefill on the next chunk. The heartbeat exists for this.
            time.sleep(stall_s)
        with record.lock:
            record.produced += 1
        item = {"index": index, "text": record.tokens[index], "skipped": skipped_since_last_delivery}
        if channel.put(item):
            skipped_since_last_delivery = 0
        else:
            if channel.cancelled:
                with record.lock:
                    record.cancelled = True
                break
            skipped_since_last_delivery += 1
    channel.close()


def make_handler(registry: StreamRegistry):
    class Handler(BaseHTTPRequestHandler):
        # HTTP/1.1 with an explicit Connection: close. An SSE body has no
        # Content-Length, so keep-alive would require chunked encoding for no
        # benefit: the connection ends when the stream ends either way.
        protocol_version = "HTTP/1.1"
        server_version = "sse-stream/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            """Silence the default access log; timing output would be noise here."""

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path != "/stream":
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.send_header("Connection", "close")
                self.end_headers()
                return
            self._stream(urllib.parse.parse_qs(parsed.query))

        def _stream(self, q: Dict[str, List[str]]) -> None:
            def one(name: str, default: str) -> str:
                return (q.get(name) or [default])[0]

            prompt = one("prompt", "hello")
            stream_id = one("stream_id", "s1")
            policy = one("policy", POLICY_BLOCK)
            if policy not in POLICIES:
                policy = POLICY_BLOCK
            maxsize = max(1, int(one("maxsize", "8")))
            token_delay_s = float(one("delay_ms", "0")) / 1000.0
            heartbeat_s = float(one("heartbeat_ms", "15000")) / 1000.0
            stall_at = int(one("stall_at", "-1"))
            stall_s = float(one("stall_ms", "0")) / 1000.0
            sndbuf = int(one("sndbuf", "0"))

            if sndbuf:
                # Shrinking the kernel send buffer is how the demo reaches real
                # TCP flow control on loopback in a few kilobytes instead of a
                # few hundred. Without it the kernel happily absorbs an entire
                # short stream and the socket writer never blocks, so the
                # bounded channel never fills and the backpressure policy never
                # gets to make a decision. This is the genuine mechanism, not a
                # sleep pretending to be one.
                try:
                    self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, sndbuf)
                except OSError:  # pragma: no cover - platform dependent
                    pass

            record = registry.record(stream_id, prompt)

            # Resume: Last-Event-ID wins, the query parameter is a fallback for
            # clients that cannot set the header.
            last_id = self.headers.get("Last-Event-ID") or (q.get("last_event_id") or [None])[0]
            start_index = 0
            resumed = False
            if last_id is not None:
                try:
                    start_index = int(last_id) + 1
                    resumed = True
                except ValueError:
                    start_index = 0
            start_index = max(0, min(start_index, len(record.tokens)))

            with record.lock:
                record.connections += 1

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache, no-transform")
            self.send_header("Connection", "close")
            # nginx buffers proxied responses by default, which holds every token
            # until the response completes and turns a stream into a slow request.
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()

            channel = BoundedChannel(maxsize=maxsize, policy=policy)
            producer = threading.Thread(
                target=_produce,
                args=(record, channel, start_index, token_delay_s, stall_at, stall_s),
                name=f"produce-{stream_id}",
                daemon=True,
            )
            producer.start()

            def write(payload: bytes) -> bool:
                """Returns False once the client is gone. That is the disconnect signal."""
                try:
                    self.wfile.write(payload)
                    self.wfile.flush()
                    return True
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return False

            alive = write(format_event(
                data=f'{{"stream_id": "{stream_id}", "resumed": {str(resumed).lower()}, '
                     f'"start_index": {start_index}, "policy": "{policy}", "buffer": {maxsize}}}',
                event="open",
                retry_ms=DEFAULT_RETRY_MS,
            ))

            last_write = time.perf_counter()
            while alive and not channel.finished:
                item = channel.get(timeout=min(0.05, heartbeat_s) if heartbeat_s else 0.05)
                if item is None:
                    if heartbeat_s and time.perf_counter() - last_write >= heartbeat_s:
                        alive = write(format_comment("heartbeat"))
                        last_write = time.perf_counter()
                    continue
                if item["skipped"]:
                    # Loss is made visible. A dropped token that nobody is told
                    # about is a silently truncated answer.
                    alive = write(format_event(
                        data=f'{{"dropped": {item["skipped"]}}}', event="gap",
                        event_id=str(item["index"] - 1)))
                    if not alive:
                        break
                alive = write(format_event(data=item["text"], event="token", event_id=str(item["index"])))
                last_write = time.perf_counter()
                with record.lock:
                    record.served += 1

            if not alive:
                channel.cancel()          # unblocks a producer parked in put()
            producer.join(timeout=2.0)
            with record.lock:
                record.channel_stats = channel.stats.to_dict()

            if alive:
                write(format_event(
                    data=f'{{"tokens": {len(record.tokens)}, "delivered_from": {start_index}, '
                         f'"dropped": {channel.stats.dropped}}}',
                    event="done",
                    event_id=str(len(record.tokens) - 1) if record.tokens else "0",
                ))

    return Handler


def serve(
    registry: Optional[StreamRegistry] = None,
    host: str = "127.0.0.1",
    port: int = 0,
) -> Tuple[ThreadingHTTPServer, str, StreamRegistry]:
    """Start on an ephemeral loopback port. Returns (server, base_url, registry)."""
    reg = registry or StreamRegistry()
    httpd = ThreadingHTTPServer((host, port), make_handler(reg))
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, name="sse-server", daemon=True).start()
    return httpd, f"http://{host}:{httpd.server_address[1]}", reg
