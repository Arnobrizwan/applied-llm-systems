"""SSE wire format. Getting this wrong produces a stream that silently hangs."""
from projects.p11_streaming.sse import SSEEvent, format_comment, format_event, parse_stream


def test_frame_ends_with_a_blank_line_and_orders_fields_correctly():
    frame = format_event(data="hello", event="token", event_id="7", retry_ms=750)
    assert frame == b"retry: 750\nid: 7\nevent: token\ndata: hello\n\n"
    assert frame.endswith(b"\n\n")           # missing this is the classic hang


def test_multiline_data_becomes_one_data_line_per_line():
    frame = format_event(data="line one\nline two", event="token")
    assert frame.count(b"data: ") == 2
    parsed = list(parse_stream(frame.splitlines(keepends=True)))
    assert parsed[0].data == "line one\nline two"


def test_empty_payload_still_emits_a_data_field():
    """A frame with no fields at all is discarded by a conforming client."""
    frame = format_event(data="", event="ping")
    assert b"data: " in frame
    assert list(parse_stream(frame.splitlines(keepends=True)))[0].event == "ping"


def test_parser_skips_comments_unless_asked_for_them():
    raw = format_comment("heartbeat") + format_event(data="x", event="token", event_id="0")
    lines = raw.splitlines(keepends=True)
    assert [e.event for e in parse_stream(lines)] == ["token"]
    with_comments = list(parse_stream(lines, include_comments=True))
    assert [e.event for e in with_comments] == ["comment", "token"]
    assert with_comments[0].data == "heartbeat"


def test_parser_is_incremental_over_a_line_iterator():
    """Frames are yielded as they complete, not after the body is buffered."""
    def dribble():
        yield b"event: token\n"
        yield b"id: 1\n"
        yield b"data: first\n"
        yield b"\n"
        yield b"event: token\n"
        yield b"data: second\n"

    events = list(parse_stream(dribble()))
    assert [e.data for e in events] == ["first", "second"]
    assert events[0].id == "1"
    assert events[1].id is None


def test_field_parsing_handles_spec_edge_cases():
    raw = b"data:no-space\n\nname-only\n\nretry: notanumber\ndata: x\n\n"
    events = list(parse_stream(raw.splitlines(keepends=True)))
    assert events[0].data == "no-space"                  # exactly one leading space is optional
    assert events[1].raw_fields["name-only"] == [""]     # a bare field name is legal
    assert events[2].retry is None                       # a bad retry is ignored, not fatal
    assert events[0].event == "message"                  # default event name


def test_terminal_event_is_recognisable():
    done = list(parse_stream(format_event(data="{}", event="done").splitlines(keepends=True)))[0]
    assert done.is_terminal
    assert not SSEEvent(event="token").is_terminal
