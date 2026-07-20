"""Extension router: the flat ``{name, kind}`` listing for the picker."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import cast

import pytest
from starlette.requests import Request
from tai_contract.app import tai_app

from tai_skeleton.routers import extensions as router


def _req() -> Request:
    return cast(Request, SimpleNamespace(path_params={}))


def _json(resp) -> dict:
    return json.loads(bytes(resp.body))


class _FakeExtensions:
    def __init__(self, items):
        self._items = items

    def available_extensions(self):
        return self._items


@pytest.fixture
def install(monkeypatch):
    def _install(items):
        monkeypatch.setattr(tai_app, "_impl", SimpleNamespace(extensions=_FakeExtensions(items)))

    return _install


async def test_list_extensions(install):
    items = [{"name": "cache", "kind": "wrapper"}, {"name": "run_sync", "kind": "backend"}]
    install(items)
    resp = await router.list_extensions(_req())
    assert resp.status_code == 200
    assert _json(resp) == {"data": items}


async def test_list_extensions_empty(install):
    install([])
    resp = await router.list_extensions(_req())
    assert _json(resp) == {"data": []}
