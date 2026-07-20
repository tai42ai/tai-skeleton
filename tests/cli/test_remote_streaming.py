"""Streaming remote commands render SSE frames incrementally, never buffered.

The agent-run and interactions doors are SSE streams; their commands must print
each frame as it arrives. Incrementality is proven with a stream that emits frames
and then FAILS mid-stream: a buffering client would surface nothing, so seeing the
earlier frames in the output proves they reached stdout before the stream ended.
"""

from __future__ import annotations

import json

import httpx
import pytest

from tests.cli.remote_harness import run_cli, sse_response


def test_agents_run_streams_each_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = [{"type": "agent.step", "n": 1}, {"type": "agent.step", "n": 2}, {"type": "stream.end"}]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/agents/researcher/runs"
        assert request.headers["accept"] == "text/event-stream"
        assert json.loads(request.content) == {"query": "weather"}
        return sse_response(frames)

    result = run_cli(monkeypatch, handler, ["agents", "run", "researcher", "--input", '{"query":"weather"}'])
    assert result.exit_code == 0, result.output
    lines = [line for line in result.output.splitlines() if line.strip()]
    assert len(lines) == len(frames)
    assert json.loads(lines[0]) == frames[0]
    assert json.loads(lines[-1]) == frames[-1]


def test_agents_authored_run_streams(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/agents/authored/my_agent/runs"
        return sse_response([{"type": "stream.end"}])

    result = run_cli(monkeypatch, handler, ["agents", "authored-run", "my_agent", "--input", "{}"])
    assert result.exit_code == 0, result.output
    assert '"stream.end"' in result.output


def test_stream_is_incremental_not_buffered(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_body():
        yield b'data: {"type": "agent.step", "n": 1}\n\n'
        yield b'data: {"type": "agent.step", "n": 2}\n\n'
        raise httpx.ReadError("connection dropped mid-stream")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=failing_body())

    result = run_cli(monkeypatch, handler, ["agents", "run", "researcher", "--input", "{}"])
    # The run failed mid-stream, but the frames emitted before the failure must
    # already be in the output — proof the frames were rendered as they arrived,
    # not held until the whole run completed.
    assert '"n": 1' in result.output
    assert '"n": 2' in result.output


def test_streamed_frame_control_chars_are_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    # A frame is the most attacker-influenced data the CLI prints; a terminal escape
    # smuggled into a ``data`` line must be stripped before it reaches stdout.
    def handler(request: httpx.Request) -> httpx.Response:
        body = "data: hello\x1b[31minjected\x07world\n\n"
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())

    result = run_cli(monkeypatch, handler, ["agents", "run", "researcher", "--input", "{}"])
    assert result.exit_code == 0, result.output
    assert "\x1b" not in result.output  # the ESC that arms an ANSI sequence is gone
    assert "\x07" not in result.output  # BEL is gone
    assert "hello[31minjectedworld" in result.output  # inert printable remainder survives


def test_interactions_list_consumes_backlog_then_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    # Two backlog add frames, then the empty ``backlog_done`` marker frame; the
    # list command must print the backlog and stop at the marker.
    frames = [
        {"interaction_id": "i1", "question": "one?"},
        {"interaction_id": "i2", "question": "two?"},
        {},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/interactions/stream"
        return sse_response(frames)

    result = run_cli(monkeypatch, handler, ["interactions", "list"])
    assert result.exit_code == 0, result.output
    lines = [line for line in result.output.splitlines() if line.strip()]
    assert len(lines) == 2
    assert json.loads(lines[0])["interaction_id"] == "i1"
    assert json.loads(lines[1])["interaction_id"] == "i2"


def _typed_sse_body(frames: list[tuple[str, dict]]) -> bytes:
    """An SSE body whose frames carry their type on an out-of-band ``event:`` line,
    as the interactions stream frames them."""
    return "".join(f"event: {event}\ndata: {json.dumps(data)}\n\n" for event, data in frames).encode()


def test_interactions_stream_distinguishes_event_types(monkeypatch: pytest.MonkeyPatch) -> None:
    # An answered and a removed interaction carry byte-identical data; only the
    # out-of-band event type tells them apart, so it must reach the printed output.
    payload = {"interaction_id": "i1", "group_id": "g1"}
    body = _typed_sse_body([("interaction.answered", payload), ("interaction.removed", payload)])

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/interactions/stream"
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    result = run_cli(monkeypatch, handler, ["interactions", "stream"])
    assert result.exit_code == 0, result.output
    lines = [line for line in result.output.splitlines() if line.strip()]
    assert len(lines) == 2
    answered, removed = json.loads(lines[0]), json.loads(lines[1])
    assert answered["event"] == "interaction.answered"
    assert removed["event"] == "interaction.removed"
    # The discriminator is what distinguishes them — the data alone is identical.
    assert answered["data"] == removed["data"] == payload


def test_interactions_list_stops_at_backlog_done_event(monkeypatch: pytest.MonkeyPatch) -> None:
    # The backlog_done marker also arrives as an ``event:`` line plus ``{}`` data;
    # the list must print the backlog and stop at the marker, ignoring any live tail.
    body = _typed_sse_body(
        [
            ("interaction.add", {"interaction_id": "i1", "question": "one?"}),
            ("interaction.backlog_done", {}),
            ("interaction.add", {"interaction_id": "i2", "question": "live tail"}),
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    result = run_cli(monkeypatch, handler, ["interactions", "list"])
    assert result.exit_code == 0, result.output
    lines = [line for line in result.output.splitlines() if line.strip()]
    assert len(lines) == 1
    frame = json.loads(lines[0])
    assert frame["event"] == "interaction.add"
    assert frame["data"]["interaction_id"] == "i1"


def test_framed_event_control_chars_are_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    # A terminal escape smuggled into a typed frame must still be stripped, and the
    # event type must survive so the frame stays attributable.
    body = 'event: interaction.answered\ndata: {"interaction_id": "i\x1b[31m1\x07"}\n\n'

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())

    result = run_cli(monkeypatch, handler, ["interactions", "stream"])
    assert result.exit_code == 0, result.output
    assert "\x1b" not in result.output  # the ESC that arms an ANSI sequence is gone
    assert "\x07" not in result.output  # BEL is gone
    assert "interaction.answered" in result.output  # the event type still identifies the frame


def test_interactions_answer_posts_json_value(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/interactions/i1/answer"
        assert json.loads(request.content) == {"answer": "yes"}
        return httpx.Response(200, json={"data": {"interaction_id": "i1", "status": "answered"}})

    result = run_cli(monkeypatch, handler, ["interactions", "answer", "i1", "--answer", '"yes"'])
    assert result.exit_code == 0, result.output
    assert "answered" in result.output


def test_interactions_list_tolerates_non_json_frame_before_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    # A frame whose ``data`` is not JSON cannot be the empty-object backlog marker,
    # so the ``until_empty`` listing prints it verbatim and keeps consuming until the
    # real ``{}`` marker arrives, then stops.
    body = 'data: plain text line\n\ndata: {}\n\ndata: {"live": "tail"}\n\n'

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/interactions/stream"
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())

    result = run_cli(monkeypatch, handler, ["interactions", "list"])
    assert result.exit_code == 0, result.output
    lines = [line for line in result.output.splitlines() if line.strip()]
    assert lines == ["plain text line"]
