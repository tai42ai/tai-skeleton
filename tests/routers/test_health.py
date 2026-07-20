"""The /health custom route returns a plain ``OK``."""

from __future__ import annotations

from typing import cast

from starlette.requests import Request
from starlette.responses import PlainTextResponse

from tai_skeleton.routers import health


async def test_health_check_returns_ok() -> None:
    # The handler ignores the request; starlette's concrete Request can't be
    # matched by a bare stand-in, so cast a placeholder object to it.
    response = await health.health_check(cast(Request, object()))
    assert isinstance(response, PlainTextResponse)
    assert response.body == b"OK"
    assert response.status_code == 200
