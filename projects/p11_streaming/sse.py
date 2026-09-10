"""Server-sent events framing and parsing, to the spec rather than by feel.

SSE looks trivial until it is wrong in a way that only shows up behind a proxy.
The rules that actually matter and that this module enforces:

* A field line is `name: value`, and a frame ends with a blank line. Forgetting
  the terminating blank line is the classic bug: the browser buffers the event
  forever and the stream appears to hang after the first token.
* `data` may not contain a newline. Multi-line payloads must be split into one
  `data:` line per line, and the client rejoins them with "\\n". Emitting a raw
  newline inside data silently truncates the event.
* A leading colon is a comment. Comments are the standard heartbeat: they keep
  the TCP connection and any intermediary proxy from reaping an idle stream,
  and they are ignored by every conforming client so they cannot be mistaken
  for a token.
* `id` is what makes resume possible. The client remembers the last id it saw
  and replays it in the Last-Event-ID header on reconnect.
* `retry` tells the client how long to wait before reconnecting, in
  milliseconds. Without it, every client picks its own default and a server
  restart turns into a synchronised reconnect storm.

The parser here is a real incremental parser over a byte stream, not a
`split("\\n\\n")` over a fully buffered response, because buffering the whole
response to parse it defeats the entire purpose of streaming.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional


@dataclass
class SSEEvent:
    """One parsed event. `event` defaults to "message" per the spec."""

    data: str = ""
    event: str = "message"
    id: Optional[str] = None
    retry: Optional[int] = None
    raw_fields: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.event == "done"


def format_event(
    data: str = "",
    event: Optional[str] = None,
    event_id: Optional[str] = None,
    retry_ms: Optional[int] = None,
) -> bytes:
    """Encode one frame. Field order follows the spec's examples: retry, id, event, data."""
    lines: List[str] = []
    if retry_ms is not None:
        lines.append(f"retry: {int(retry_ms)}")
    if event_id is not None:
        lines.append(f"id: {event_id}")
    if event is not None:
        lines.append(f"event: {event}")
    # An empty data payload still needs a data line, otherwise the frame has no
    # fields at all and a conforming client discards it.
    for line in (data.split("\n") if data else [""]):
        lines.append(f"data: {line}")
    return ("\n".join(lines) + "\n\n").encode("utf-8")


def format_comment(text: str = "") -> bytes:
    """A heartbeat. Ignored by clients, but it keeps the socket and proxies alive."""
    return (f": {text}\n\n").encode("utf-8")


def parse_stream(lines: Iterable[bytes], include_comments: bool = False) -> Iterator[SSEEvent]:
    """Incremental parser over raw byte lines. Yields one SSEEvent per frame.

    Accepts an iterable of lines (each terminated or not) so it can be driven
    directly from `http.client.HTTPResponse.readline`, which returns as soon as
    a line is available rather than waiting for the body to complete.

    `include_comments` surfaces heartbeats as SSEEvent(event="comment"). A real
    client must ignore them; this project counts them, because "did the
    heartbeat actually fire" is one of the things worth proving.
    """
    fields: Dict[str, List[str]] = {}
    for raw in lines:
        line = raw.decode("utf-8", "replace").rstrip("\n").rstrip("\r")
        if line.startswith(":"):
            if include_comments:
                yield SSEEvent(data=line[1:].strip(), event="comment")
            continue                      # a conforming client ignores these
        if line == "":
            if fields:
                yield _build(fields)
                fields = {}
            continue
        if ":" in line:
            name, value = line.split(":", 1)
            if value.startswith(" "):     # exactly one leading space is stripped
                value = value[1:]
        else:
            name, value = line, ""
        fields.setdefault(name, []).append(value)
    if fields:
        yield _build(fields)


def _build(fields: Dict[str, List[str]]) -> SSEEvent:
    retry = None
    if fields.get("retry"):
        try:
            retry = int(fields["retry"][-1])
        except ValueError:
            retry = None
    return SSEEvent(
        data="\n".join(fields.get("data", [])),
        event=(fields.get("event") or ["message"])[-1],
        id=(fields.get("id") or [None])[-1],
        retry=retry,
        raw_fields=dict(fields),
    )
