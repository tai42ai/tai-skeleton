"""The shared HTTP client: api-key auth, the ``{data}``/``{error}`` envelope,
typed errors per status, and incremental SSE streaming — all against a fake
httpx transport, never a real network.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import httpx
import pytest

from tai42_skeleton.cli.client import (
    ApiClient,
    ApiError,
    AuthError,
    BadRequestError,
    ConflictError,
    NotFoundError,
)


def _client(handler) -> ApiClient:
    return ApiClient("http://tai.test", "secret-key", transport=httpx.MockTransport(handler))


def test_get_unwraps_data_and_sends_api_key() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["x-api-key"] = request.headers["x-api-key"]
        seen["path"] = request.url.path
        return httpx.Response(200, json={"data": ["a", "b"]})

    with _client(handler) as client:
        result = client.get("/api/tools")

    assert result == ["a", "b"]
    assert seen["x-api-key"] == "secret-key"
    assert seen["path"] == "/api/tools"


def test_post_sends_json_body_and_unwraps() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content) == {"name": "x"}
        return httpx.Response(200, json={"data": {"ok": True}})

    with _client(handler) as client:
        assert client.post("/api/tools", json={"name": "x"}) == {"ok": True}


def test_no_content_returns_none() -> None:
    with _client(lambda request: httpx.Response(204)) as client:
        assert client.delete("/api/keys/u") is None


@pytest.mark.parametrize(
    ("status", "error_cls"),
    [
        (401, AuthError),
        (404, NotFoundError),
        (409, ConflictError),
        (400, BadRequestError),
    ],
)
def test_error_status_raises_typed_error(status: int, error_cls: type[ApiError]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": "boom detail"})

    with _client(handler) as client, pytest.raises(error_cls) as excinfo:
        client.get("/api/tools")

    assert excinfo.value.status_code == status
    if status == 401:
        assert "not authenticated" in excinfo.value.message
    else:
        assert "boom detail" in excinfo.value.message


def test_unmapped_status_raises_base_api_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "kaboom"})

    with _client(handler) as client, pytest.raises(ApiError) as excinfo:
        client.get("/api/tools")

    assert type(excinfo.value) is ApiError
    assert excinfo.value.status_code == 500
    assert "kaboom" in excinfo.value.message


def test_malformed_success_envelope_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": 1})

    with _client(handler) as client, pytest.raises(ApiError, match="malformed success envelope"):
        client.get("/api/tools")


def test_non_json_success_body_raises_malformed_envelope() -> None:
    # A 2xx whose body is not JSON (a proxy's HTML page) must surface as the typed
    # ApiError, not a raw json.JSONDecodeError traceback.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>proxy interstitial</html>")

    with _client(handler) as client, pytest.raises(ApiError, match="malformed success envelope"):
        client.get("/api/tools")


def test_request_raw_returns_body_without_unwrapping() -> None:
    # A download route answers outside the ``{"data": ...}`` envelope; request_raw
    # returns the raw response carrying the auth header, unwrapping nothing.
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["x-api-key"] = request.headers["x-api-key"]
        return httpx.Response(200, text="id,name\n1,alpha\n")

    with _client(handler) as client:
        response = client.request_raw("GET", "/api/obs/export")

    assert response.text == "id,name\n1,alpha\n"
    assert seen["x-api-key"] == "secret-key"


def test_request_raw_raises_typed_error_on_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "no such export"})

    with _client(handler) as client, pytest.raises(NotFoundError, match="no such export"):
        client.request_raw("GET", "/api/obs/export")


def test_error_without_envelope_falls_back_to_body_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="plain not found")

    with _client(handler) as client, pytest.raises(NotFoundError, match="plain not found"):
        client.get("/api/tools")


class _RecordingStream(httpx.SyncByteStream):
    """A byte stream that logs each chunk as it is pulled, so a test can prove
    the client consumes frames incrementally rather than buffering the run."""

    def __init__(self, chunks: list[bytes], log: list[tuple[str, str]]) -> None:
        self._chunks = chunks
        self._log = log

    def __iter__(self) -> Iterator[bytes]:
        for chunk in self._chunks:
            self._log.append(("emit", chunk.decode()))
            yield chunk

    def close(self) -> None:  # pragma: no cover - no resources to release
        pass


def test_stream_yields_sse_frames_incrementally() -> None:
    log: list[tuple[str, str]] = []
    frames = [b'data: {"type": "a"}\n\n', b": keepalive\n\n", b'data: {"type": "b"}\n\n']

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["accept"] == "text/event-stream"
        return httpx.Response(200, stream=_RecordingStream(frames, log))

    with _client(handler) as client:
        for event, data in client.stream("POST", "/api/agents/x/runs", json={}):
            # A runs-style frame carries its type inside the JSON, so no ``event:``
            # line arrives and the surfaced event is None.
            assert event is None
            log.append(("frame", data))

    # Only the two data frames surface; the keepalive comment is dropped.
    yielded = [value for kind, value in log if kind == "frame"]
    assert yielded == ['{"type": "a"}', '{"type": "b"}']
    # The first frame is yielded before the last chunk is emitted — incremental,
    # not buffered.
    assert log.index(("frame", '{"type": "a"}')) < log.index(("emit", frames[-1].decode()))


def test_stream_surfaces_out_of_band_event_type() -> None:
    # The interactions stream frames its type on an out-of-band ``event:`` line while
    # the payload rides on ``data:``; the parser must surface that event type.
    body = b'event: interaction.answered\ndata: {"interaction_id": "i1"}\n\n'

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    with _client(handler) as client:
        frames = list(client.stream("GET", "/api/interactions/stream"))

    assert frames == [("interaction.answered", '{"interaction_id": "i1"}')]


def test_iter_sse_data_mixes_typed_and_untyped_frames() -> None:
    from tai42_skeleton.cli.client import iter_sse_data

    lines = [
        "event: interaction.removed",
        'data: {"interaction_id": "i2"}',
        "",
        'data: {"type": "agent.step"}',
        "",
    ]

    assert list(iter_sse_data(iter(lines))) == [
        ("interaction.removed", '{"interaction_id": "i2"}'),
        (None, '{"type": "agent.step"}'),
    ]


def test_stream_raises_typed_error_before_any_frame() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "nope"})

    with _client(handler) as client, pytest.raises(AuthError):
        list(client.stream("POST", "/api/agents/x/runs", json={}))
