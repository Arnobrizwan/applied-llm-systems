"""An SSE client that resumes rather than restarts.

The behaviour that matters: on a dropped connection the client reconnects and
sends `Last-Event-ID` with the id of the last event it actually processed. A
client that reconnects without it gets the stream from the beginning, so the
user sees the answer duplicated, and the server pays for the generation twice.

`kill_after` deliberately closes the socket mid-stream. Simulating a network
failure by patching a method would prove nothing about the server; closing a
real socket exercises the actual path, including the server noticing the
broken pipe and stopping its producer.

The consumer delay is how a slow reader is simulated. It is a sleep in the
read loop, which is exactly what a client doing real work per token (rendering,
persisting, re-tokenising) looks like from the server's side of the socket.
"""
from __future__ import annotations

import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .metrics import StreamMetrics
from .sse import SSEEvent, parse_stream


@dataclass
class ConsumeResult:
    tokens: List[str] = field(default_factory=list)
    token_ids: List[int] = field(default_factory=list)
    events: List[SSEEvent] = field(default_factory=list)
    metrics: StreamMetrics = field(default_factory=lambda: StreamMetrics("client"))
    last_event_id: Optional[str] = None
    completed: bool = False
    gaps: List[int] = field(default_factory=list)
    open_frames: List[Dict[str, str]] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "".join(self.tokens)

    @property
    def duplicate_ids(self) -> List[int]:
        """Non-empty means resume replayed something the client already had."""
        seen, dupes = set(), []
        for i in self.token_ids:
            if i in seen:
                dupes.append(i)
            seen.add(i)
        return dupes


class SSEClient:
    def __init__(self, base_url: str, timeout: float = 15.0, max_reconnects: int = 3):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_reconnects = max_reconnects

    def url_for(self, **params) -> str:
        return f"{self.base_url}/stream?" + urllib.parse.urlencode(
            {k: v for k, v in params.items() if v is not None})

    def consume(
        self,
        *,
        params: Optional[Dict[str, object]] = None,
        kill_after: Optional[int] = None,
        consumer_delay_s: float = 0.0,
        count_heartbeats: bool = True,
        on_token: Optional[Callable[[int, str], None]] = None,
    ) -> ConsumeResult:
        """Read a stream to completion, reconnecting and resuming if it breaks."""
        url = self.url_for(**(params or {}))
        result = ConsumeResult()
        result.metrics.start()
        killed_once = False
        attempts = 0

        while attempts <= self.max_reconnects:
            attempts += 1
            request = urllib.request.Request(url)
            if result.last_event_id is not None:
                request.add_header("Last-Event-ID", result.last_event_id)
                result.metrics.record_reconnect()
            try:
                response = urllib.request.urlopen(request, timeout=self.timeout)
            except urllib.error.URLError:
                break

            broke = False
            try:
                lines = iter(response.readline, b"")
                for event in parse_stream(lines, include_comments=count_heartbeats):
                    if event.event == "comment":
                        result.metrics.record_heartbeat()
                        continue
                    result.events.append(event)
                    if event.event == "open":
                        result.open_frames.append({"data": event.data})
                        continue
                    if event.event == "gap":
                        result.gaps.append(len(result.tokens))
                        if event.id is not None:
                            result.last_event_id = event.id
                        continue
                    if event.event == "done":
                        result.completed = True
                        break
                    if event.event == "token":
                        result.metrics.record_token()
                        result.tokens.append(event.data)
                        if event.id is not None:
                            result.last_event_id = event.id
                            result.token_ids.append(int(event.id))
                        if on_token:
                            on_token(len(result.tokens) - 1, event.data)
                        if consumer_delay_s:
                            time.sleep(consumer_delay_s)
                        if kill_after is not None and not killed_once and len(result.tokens) >= kill_after:
                            killed_once = True
                            broke = True
                            break
            finally:
                # Closing here is what the server sees as a client disconnect.
                response.close()

            if result.completed:
                break
            if not broke and attempts > self.max_reconnects:
                break
            if not broke and not result.tokens:
                break

        result.metrics.stop()
        return result


def slow_socket_consume(
    host: str,
    port: int,
    query: str,
    *,
    recv_buffer_bytes: int = 2048,
    per_token_delay_s: float = 0.02,
    timeout: float = 20.0,
) -> ConsumeResult:
    """Consume a stream over a raw socket with a deliberately small receive buffer.

    This is the only honest way to demonstrate backpressure end to end on
    loopback. urllib gives no control over SO_RCVBUF, and with the default
    buffer the kernel absorbs a whole short stream before the reader touches
    it, so the server's socket write never blocks and the bounded channel never
    fills. Setting a small receive window before connect, then reading slowly,
    produces genuine TCP flow control: the receive window closes, the server's
    send buffer fills, `wfile.write` blocks, the channel backs up, and the
    configured policy actually has to decide something.
    """
    import socket as _socket

    sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    # SO_RCVBUF must be set before connect so the advertised window reflects it.
    sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_RCVBUF, recv_buffer_bytes)
    sock.settimeout(timeout)
    sock.connect((host, port))
    request = (
        f"GET /stream?{query} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        "Accept: text/event-stream\r\n"
        "Connection: close\r\n\r\n"
    )
    sock.sendall(request.encode("ascii"))

    result = ConsumeResult()
    result.metrics.start()
    stream = sock.makefile("rb")
    try:
        while True:                     # skip the status line and headers
            line = stream.readline()
            if line in (b"\r\n", b"\n", b""):
                break
        for event in parse_stream(iter(stream.readline, b""), include_comments=True):
            if event.event == "comment":
                result.metrics.record_heartbeat()
                continue
            result.events.append(event)
            if event.event == "gap":
                result.gaps.append(len(result.tokens))
            elif event.event == "token":
                result.metrics.record_token()
                result.tokens.append(event.data)
                if event.id is not None:
                    result.last_event_id = event.id
                    result.token_ids.append(int(event.id))
                if per_token_delay_s:
                    time.sleep(per_token_delay_s)
            elif event.event == "done":
                result.completed = True
                break
    finally:
        stream.close()
        sock.close()
        result.metrics.stop()
    return result
