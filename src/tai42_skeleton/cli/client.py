"""Shared HTTP client for the ``tai`` CLI's remote commands.

Every remote command talks to the skeleton's ``/api/*`` surface — the SAME
routes the Studio calls — through this one client, so the envelope contract and
authentication live in exactly one place (mirroring the Studio api-client).

The wire contract:

* the api key travels in the ``x-api-key`` header;
* a success body is ``{"data": ...}`` and is unwrapped to the inner value;
* a failure body is ``{"error": "<message>"}`` and, together with the HTTP
  status, is raised as a typed :class:`ApiError` (401/404/409/400 each get their
  own subclass so callers can distinguish them);
* streaming runs are consumed frame by frame off an SSE response — never
  buffered whole.
"""

import json
from collections.abc import Iterator, Mapping
from typing import Any

import httpx


class ApiError(Exception):
    """A non-2xx response from the skeleton API.

    ``message`` is the server's ``{"error": ...}`` text where present, else the
    raw response body. ``status_code`` is the HTTP status.
    """

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class AuthError(ApiError):
    """A 401 — the request carried no API key the server would accept."""


class NotFoundError(ApiError):
    """A 404 — the addressed resource does not exist."""


class ConflictError(ApiError):
    """A 409 — the request conflicts with the current server state."""


class BadRequestError(ApiError):
    """A 400 — the server rejected the request payload."""


_STATUS_ERRORS: dict[int, type[ApiError]] = {
    400: BadRequestError,
    401: AuthError,
    404: NotFoundError,
    409: ConflictError,
}


def _server_message(response: httpx.Response) -> str | None:
    """The ``{"error": ...}`` message from a response body, or ``None`` when the
    body is not the JSON error envelope."""
    try:
        body = response.json()
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(body, Mapping):
        error = body.get("error")
        if isinstance(error, str):
            return error
    return None


def _error_from_response(response: httpx.Response) -> ApiError:
    status = response.status_code
    message = _server_message(response) or response.text.strip()
    if status == 401:
        detail = message or "no valid API key was accepted by the server"
        return AuthError(f"not authenticated: {detail}", status_code=401)
    if not message:
        message = f"HTTP {status}"
    error_cls = _STATUS_ERRORS.get(status, ApiError)
    return error_cls(message, status_code=status)


def _unwrap(response: httpx.Response) -> Any:
    """Return the ``data`` payload of a success response, raising on any failure
    status or a malformed success envelope."""
    if response.status_code >= 400:
        raise _error_from_response(response)
    if response.status_code == 204 or not response.content:
        return None
    try:
        body = response.json()
    except (json.JSONDecodeError, ValueError) as exc:
        # A 2xx whose body is not JSON (a proxy's HTML page, say) is as malformed as
        # a missing ``data`` key — surface it as the same typed error, not a raw decode.
        raise ApiError(
            f"malformed success envelope (body is not JSON): {response.text!r}",
            status_code=response.status_code,
        ) from exc
    if not isinstance(body, Mapping) or "data" not in body:
        raise ApiError(
            f"malformed success envelope (expected a 'data' key): {body!r}",
            status_code=response.status_code,
        )
    return body["data"]


def iter_sse_data(lines: Iterator[str]) -> Iterator[tuple[str | None, str]]:
    """Yield ``(event, data)`` for each SSE frame from a line iterator.

    The skeleton emits one JSON object per frame as a single ``data:`` line
    terminated by a blank line. A frame's type may ride INSIDE that JSON (the
    agents/runs stream sends no ``event:`` line) OR arrive OUT OF BAND on an
    ``event:`` line (the interactions stream), so the parser surfaces the
    ``event:`` value when present and yields ``None`` for it otherwise.
    Comment/keepalive lines (a leading ``:``) and any other SSE fields are
    ignored; multi-line ``data:`` values are joined with newlines.
    """
    event: str | None = None
    data_parts: list[str] = []
    for line in lines:
        if line == "":
            if data_parts:
                yield event, "\n".join(data_parts)
            event = None
            data_parts = []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value
        if field == "data":
            data_parts.append(value)
        elif field == "event":
            event = value
    if data_parts:
        yield event, "\n".join(data_parts)


class ApiClient:
    """A thin httpx wrapper that owns the api-key header and the envelope.

    Construct with a ``base_url`` and an ``api_key``; inject ``transport`` in tests
    to serve responses from a fake without any network. ``api_key=None`` builds an
    ANONYMOUS client that sends no credential header — the one public door the CLI
    calls (``tai auth claim`` exchanging a claim token, which the caller has no key for
    yet); every other command passes a real key.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 30.0,
    ) -> None:
        # No key → no ``x-api-key`` header at all, so a public route is never handed a
        # stale/wrong credential (which its always-public middleware would ignore, but a
        # protected route would 401 on).
        headers = {"x-api-key": api_key} if api_key is not None else {}
        self._client = httpx.Client(
            base_url=base_url,
            timeout=timeout,
            transport=transport,
            headers=headers,
        )

    def __enter__(self) -> "ApiClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def request(
        self,
        method: str,
        path: str,
        *,
        json: Any | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> Any:
        response = self._client.request(method, path, json=json, params=params)
        return _unwrap(response)

    def request_raw(
        self,
        method: str,
        path: str,
        *,
        json: Any | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> httpx.Response:
        """Perform the auth'd request and return the RAW response, unwrapping no
        envelope, but still raising the typed error on a failure status.

        The download routes (a bare backup document, a CSV/JSON export) answer
        outside the ``{"data": ...}`` envelope, so their callers read the body
        directly while keeping this module's auth, base URL, and typed errors.
        """
        response = self._client.request(method, path, json=json, params=params)
        if response.status_code >= 400:
            raise _error_from_response(response)
        return response

    def get(self, path: str, *, params: Mapping[str, Any] | None = None) -> Any:
        return self.request("GET", path, params=params)

    def post(
        self,
        path: str,
        *,
        json: Any | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> Any:
        return self.request("POST", path, json=json, params=params)

    def patch(
        self,
        path: str,
        *,
        json: Any | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> Any:
        return self.request("PATCH", path, json=json, params=params)

    def put(
        self,
        path: str,
        *,
        json: Any | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> Any:
        return self.request("PUT", path, json=json, params=params)

    def delete(self, path: str, *, params: Mapping[str, Any] | None = None) -> Any:
        return self.request("DELETE", path, params=params)

    def stream(
        self,
        method: str,
        path: str,
        *,
        json: Any | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> Iterator[tuple[str | None, str]]:
        """Yield each SSE frame as ``(event, data)`` from a streaming run, incrementally.

        The response body is consumed frame by frame — a run's output reaches
        the caller as it arrives, never after the whole run completes. ``event``
        is the frame's out-of-band ``event:`` type when the server sends one (the
        interactions stream) and ``None`` otherwise (the runs stream, whose type
        rides inside the JSON ``data``). A failure status raises the typed error
        before any frame is yielded.
        """
        with self._client.stream(
            method,
            path,
            json=json,
            params=params,
            headers={"accept": "text/event-stream"},
        ) as response:
            if response.status_code >= 400:
                response.read()
                raise _error_from_response(response)
            yield from iter_sse_data(response.iter_lines())
