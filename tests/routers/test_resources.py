"""Resources router: load a stored resource by id, with the text/media/404/400
surface and the adapter's media serialization over HTTP."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, cast

import pytest
from fastmcp.utilities.types import Image
from starlette.requests import Request

import tai42_skeleton.operations.resources as resources_ops
import tai42_skeleton.routers.resources as router
from tai42_skeleton.template.media import MediaBlock


def _req(body: dict | None = None) -> Request:
    async def _json() -> Any:
        return body or {}

    return cast(Request, SimpleNamespace(json=_json, path_params={}, query_params={}))


def _data(resp):
    return json.loads(bytes(resp.body))


class _ResourceManager:
    def __init__(self, *, loaded: str | MediaBlock = "", rendered: str = "", raise_exc: Exception | None = None):
        self._loaded = loaded
        self._rendered = rendered
        self._raise = raise_exc

    async def load_file(self, source: str) -> str | MediaBlock:
        if self._raise is not None:
            raise self._raise
        return self._loaded

    async def render_by_id_or_content(self, content=None, template_id=None, kwargs=None) -> str:
        return self._rendered


@pytest.fixture
def bind(monkeypatch):
    def _bind(manager: _ResourceManager) -> _ResourceManager:
        fake_app = SimpleNamespace(storage=SimpleNamespace(resource_manager=manager))
        monkeypatch.setattr(resources_ops, "tai42_app", fake_app)
        return manager

    return _bind


async def test_get_text_unrendered(bind):
    bind(_ResourceManager(loaded="raw {{ body }}"))
    resp = await router.get_resource_by_id(_req({"resource_id": "doc.txt"}))
    assert resp.status_code == 200
    assert _data(resp)["data"] == "raw {{ body }}"


async def test_get_renders_with_kwargs(bind):
    bind(_ResourceManager(loaded="Hello {{ name }}", rendered="Hello Ada"))
    resp = await router.get_resource_by_id(_req({"resource_id": "greet.j2", "template_kwargs": {"name": "Ada"}}))
    assert _data(resp)["data"] == "Hello Ada"


async def test_get_media_serializes_over_http(bind):
    # The adapter converts a fastmcp media return to its JSON-native MCP wire block.
    bind(_ResourceManager(loaded=Image(data=b"\x89PNG\r\n", format="png")))
    resp = await router.get_resource_by_id(_req({"resource_id": "logo.png"}))
    assert resp.status_code == 200
    block = _data(resp)["data"]
    assert block["type"] == "image"
    assert block["mimeType"] == "image/png"
    assert "data" in block  # base64 payload


async def test_get_missing_is_404(bind):
    bind(_ResourceManager(raise_exc=FileNotFoundError("gone")))
    resp = await router.get_resource_by_id(_req({"resource_id": "gone.txt"}))
    assert resp.status_code == 404
    assert "not found" in _data(resp)["error"]


async def test_render_media_is_400(bind):
    bind(_ResourceManager(loaded=Image(data=b"\x89PNG", format="png")))
    resp = await router.get_resource_by_id(_req({"resource_id": "logo.png", "template_kwargs": {"x": 1}}))
    assert resp.status_code == 400
    assert "Cannot render media" in _data(resp)["error"]
