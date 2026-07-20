"""Shared tool-call request parsing for the tool doors.

Lives in its own ROUTE-FREE module (no ``@tai42_app.http.custom_route`` decorators)
so the synchronous ``/api/run-tool`` door and the background ``/api/tool-runs``
submit door can share the parser WITHOUT importing each other's route module —
importing a route module registers its routes as a side effect, so a shared
helper must not sit in one.
"""

from __future__ import annotations

from starlette.requests import Request


class ToolCallRequestError(Exception):
    """A malformed tool-call request body — carries the loud ``(message,
    status_code)`` the door returns unchanged. Shared by the synchronous
    ``/api/run-tool`` door and the background ``/api/tool-runs`` submit door so
    both reject a bad body identically."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


async def read_tool_call(request: Request) -> tuple[str, dict[str, object]]:
    """Parse and validate a tool-call body ``{tool_name, arguments}``.

    Both tool-execution doors share this one field shape — the explicit pair that
    matches the run-record fields. Returns ``(tool_name, arguments)``;
    ``arguments`` defaults to ``{}`` when absent. Raises
    :class:`ToolCallRequestError` on invalid JSON, a non-object body, a
    missing/empty ``tool_name``, or a non-object ``arguments`` — the caller maps
    it to the same loud 4xx both doors share."""
    try:
        body = await request.json()
    except ValueError as exc:
        raise ToolCallRequestError("invalid JSON body", 400) from exc
    if not isinstance(body, dict):
        raise ToolCallRequestError("body must be a JSON object", 400)
    name = body.get("tool_name", "")
    if not isinstance(name, str) or not name:
        raise ToolCallRequestError("body must contain a non-empty 'tool_name'", 400)
    arguments = body.get("arguments", {})
    # ``arguments`` feeds the tool's validated kwargs; a non-object (array/scalar)
    # is a malformed request — reject it as a loud 400 here rather than letting it
    # fail deeper as a 500.
    if not isinstance(arguments, dict):
        raise ToolCallRequestError("'arguments' must be a JSON object", 400)
    return name, arguments
