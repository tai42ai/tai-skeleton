"""App-level request body-size cap — the backstop on every route.

Public doors read their bodies bounded already (a 64 KiB-class cap); the AUTHED
routes read the whole body unbounded (``request.json()`` and friends). This
middleware caps EVERY route in one place: the running total of the ACTUAL body
bytes read is bounded (never a client-declared ``Content-Length``, which is
advisory), and an over-cap request is answered with 413, loudly — never a
silently truncated stream.

The escape signal is a module-private ``_BodyTooLarge`` that subclasses
``Exception`` DIRECTLY, never ``ValueError``: several routes wrap
``request.json()`` in ``except ValueError`` (``routers/backup.py``,
``routers/_tool_call.py``), so a ``ValueError``-based escape would be swallowed
into a 400 and the 413 would never fire. Raised from the wrapped ``receive`` and
caught around the inner app: before response-start it is converted to a 413,
after start it is re-raised (the stream cannot be un-sent).

This runs INSIDE the base app's own Starlette stack (via the base-app middleware
list, so it sits inside that app's ``ServerErrorMiddleware``): the over-cap escape
must reach this handler and become a 413 before any error middleware turns the
raised ``_BodyTooLarge`` into a 500. Always on; tune via
``TAI_BODY_LIMIT_MAX_BODY_BYTES``.
"""

from __future__ import annotations

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from tai_skeleton.settings.body_limit import body_limit_settings


class _BodyTooLarge(Exception):
    """Internal escape raised once the accumulated body exceeds the cap.

    Subclasses ``Exception`` DIRECTLY (never ``ValueError``) so a route's
    ``except ValueError`` around ``request.json()`` cannot swallow it into a 400.
    """


class BodyLimitMiddleware:
    """Caps every request body at ``TAI_BODY_LIMIT_MAX_BODY_BYTES`` actual bytes;
    over-cap answers 413. Non-http scopes pass straight through."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        cap = body_limit_settings().max_body_bytes

        # Up-front reject: a declared Content-Length already over the cap is
        # refused before a single body byte is read.
        headers = Headers(scope=scope)
        declared = headers.get("content-length")
        if declared is not None:
            try:
                declared_len = int(declared)
            except ValueError:
                declared_len = None
            if declared_len is not None and declared_len > cap:
                await self._reject(cap, scope, receive, send)
                return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > cap:
                    raise _BodyTooLarge
            return message

        response_started = False

        async def tracking_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except _BodyTooLarge:
            if response_started:
                # The response head is already on the wire; the stream cannot be
                # un-sent, so surface the over-cap loudly rather than truncate.
                raise
            await self._reject(cap, scope, receive, send)

    @staticmethod
    async def _reject(cap: int, scope: Scope, receive: Receive, send: Send) -> None:
        response = JSONResponse(
            {"error": f"request body exceeds TAI_BODY_LIMIT_MAX_BODY_BYTES ({cap} bytes)"},
            status_code=413,
            headers={"Cache-Control": "no-store"},
        )
        await response(scope, receive, send)
