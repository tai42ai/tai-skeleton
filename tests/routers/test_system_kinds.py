"""System-kinds router: the authed door returns every pluggable-kind row inside
the ``{"data": [...]}`` envelope, each validating against ``KindStatus``."""

from __future__ import annotations

import json

import pytest
from starlette.requests import Request
from tai42_contract.app import tai42_app

from tai42_skeleton.app.instance import build_app
from tai42_skeleton.app.kind_status import KindStatus
from tai42_skeleton.manifest import Manifest
from tai42_skeleton.routers import system_kinds as router


@pytest.fixture
def bound_app(monkeypatch: pytest.MonkeyPatch):
    """The process app singleton bound to ``tai42_app`` with an empty live manifest,
    so the collector reads a started app the way the route does at runtime."""
    app = build_app()
    tai42_app.bind(app)
    monkeypatch.setattr(app, "_manifest", Manifest.model_validate({}), raising=False)
    return app


def _get_request() -> Request:
    scope = {"type": "http", "method": "GET", "path": "/api/system/kinds", "query_string": b"", "headers": []}
    return Request(scope)


async def test_returns_data_envelope_of_kind_rows(bound_app) -> None:
    resp = await router.list_system_kinds(_get_request())
    assert resp.status_code == 200
    body = json.loads(bytes(resp.body))
    assert set(body) == {"data"}
    rows = body["data"]
    assert isinstance(rows, list)
    # Every row round-trips through the model, and the door reports all nine kinds.
    validated = [KindStatus.model_validate(row) for row in rows]
    assert {row.kind for row in validated} == {
        "identity",
        "accounts",
        "monitoring",
        "storage",
        "backend",
        "channels",
        "webhook_verifiers",
        "config",
        "studio_plugins",
    }
