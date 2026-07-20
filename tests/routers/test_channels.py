"""Channels router: the authed catalog door lists the registered channel names
inside the ``{"data": ...}`` envelope."""

from __future__ import annotations

import json

import pytest
from starlette.requests import Request
from tai_contract.app import tai_app

from tai_skeleton.routers import channels as router
from tests._helpers import DeliverOnlyChannel


class _Chan(DeliverOnlyChannel):
    async def deliver(self, delivery) -> None:
        return None


@pytest.fixture
def registry():
    """The process app's channel registry, with ``tai_app`` bound to that same
    app so the route resolves channels from it; cleared after the test."""
    from tai_skeleton.app.instance import build_app

    app = build_app()
    tai_app.bind(app)
    reg = app._channel_registry
    reg.reset()
    try:
        yield reg
    finally:
        reg.reset()


def _get_request() -> Request:
    scope = {"type": "http", "method": "GET", "path": "/api/channels", "query_string": b"", "headers": []}
    return Request(scope)


async def test_list_channels_empty(registry) -> None:
    resp = await router.list_channels(_get_request())
    assert resp.status_code == 200
    assert json.loads(bytes(resp.body)) == {"data": {"channels": []}}


async def test_list_channels_returns_sorted_names(registry) -> None:
    tai_app.channels.register("zeta", _Chan())
    tai_app.channels.register("alpha", _Chan())
    resp = await router.list_channels(_get_request())
    assert json.loads(bytes(resp.body)) == {"data": {"channels": ["alpha", "zeta"]}}
