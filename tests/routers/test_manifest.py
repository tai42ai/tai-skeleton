"""Manifest router: get-manifest, mcp-config write, mcp-status, mcp reload."""

from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace
from typing import cast

import pytest
from starlette.requests import Request
from tai42_contract.app import tai42_app

import tai42_skeleton.routers.manifest as router
from tai42_skeleton.app import instance
from tests._fakes.bus import FakeBus


def _req(body=None, **path_params) -> Request:
    async def _json():
        return body

    return cast(Request, SimpleNamespace(json=_json, path_params=path_params, query_params={}))


def _data(resp):
    return json.loads(bytes(resp.body))


class _FakeConfigManager:
    """The transactional config-manager seams the ConfigService pipeline drives.

    ``mutate_manifest`` runs the mutator on a copy and persists it only if the mutator
    returns without raising (a rejected mutation leaves the store untouched);
    ``replace_manifest`` swaps the whole document. ``written`` records the last
    persisted document (``None`` until a persist lands)."""

    def __init__(self, manifest):
        self._manifest = manifest
        self.written = None

    def read_manifest(self):
        return deepcopy(self._manifest)

    def write_manifest(self, m):
        self.written = m

    def read_manifest_preserved(self):
        return deepcopy(self._manifest)

    def read_env(self):
        return {}

    def mutate_manifest(self, mutator):
        document = deepcopy(self._manifest)
        mutator(document)  # a raise here propagates before any persist
        self._manifest = document
        self.written = document
        return document

    def replace_manifest(self, document):
        self._manifest = deepcopy(document)
        self.written = deepcopy(document)
        return deepcopy(document)


@pytest.fixture
def fake(monkeypatch):
    live = {"mcp": [{"title": "gh", "config": {"url": "https://x"}}], "user_tools": ["b", "a"]}
    cm = _FakeConfigManager({"mcp": [], "tools": []})
    admin = SimpleNamespace(
        live_manifest=live,
        live_mcp_status=lambda: {"bound": {"gh": ["t1"]}, "failed": []},
        reload_config=lambda: {"status": "ok", "env_keys": 3},
        reload_mcp=lambda title: {"title": title, "status": "ok", "tools": ["t1"]},
    )
    # No worker bus: reloads stay local-only (fan-out has its own test). Patch the
    # contract handle's impl so the router body and the fanout helper both resolve
    # ``tai42_app`` to this fake.
    fake_app = SimpleNamespace(
        admin=admin,
        config=SimpleNamespace(config_manager=cm),
        backends=SimpleNamespace(backend=None),
    )
    monkeypatch.setattr(tai42_app, "_impl", fake_app)
    bus = FakeBus(origin="serve-x")
    monkeypatch.setattr(instance.app, "_bus", bus)
    return SimpleNamespace(app=fake_app, cm=cm, live=live, bus=bus)


async def test_get_manifest(fake):
    resp = await router.get_manifest(_req())
    body = _data(resp)["data"]
    assert body["mcp"][0]["title"] == "gh"
    assert body["user_tools"] == ["a", "b"]  # sorted


async def test_mcp_config_schema_shape():
    # No app impl needed: the handler only calls a pydantic classmethod.
    resp = await router.get_mcp_config_schema(_req())
    data = _data(resp)["data"]
    assert isinstance(data, dict)
    # A JSON-Schema object for a model with a nested ``config`` sub-model.
    assert "properties" in data or "$ref" in data or "$defs" in data


async def test_mcp_config_schema_round_trip():
    from tai42_skeleton.manifest import Manifest

    # An entry shaped per the served schema's required fields (``title`` + a
    # ``config`` MCPConfig with exactly one transport) must pass full manifest
    # validation.
    entry = {"title": "x", "config": {"type": "streamable_http", "url": "https://example.com/mcp"}}
    manifest = Manifest.model_validate({"mcp": [entry], "tools": [{"title": "t", "module": "m"}]})
    assert manifest.mcp[0].title == "x"
    assert manifest.mcp[0].config.url == "https://example.com/mcp"


async def test_mcp_status(fake):
    resp = await router.get_mcp_status(_req())
    assert _data(resp)["data"]["bound"] == {"gh": ["t1"]}


async def test_mcp_reload_known(fake):
    resp = await router.reload_mcp(_req(title="gh"))
    data = _data(resp)["data"]
    # The response is the per-origin fleet report; this worker's re-probe result
    # rides its self-entry payload.
    assert data["op"] == "reload_mcp"
    assert data["results"][0]["payload"]["status"] == "ok"


async def test_mcp_reload_unknown_404(fake):
    resp = await router.reload_mcp(_req(title="nope"))
    assert resp.status_code == 404


async def test_mcp_config_missing_key_400(fake):
    resp = await router.set_mcp_config(_req({}))
    assert resp.status_code == 400


async def test_mcp_config_invalid_400(fake):
    # A non-list mcp fails Manifest validation loudly.
    resp = await router.set_mcp_config(_req({"mcp": "not-a-list"}))
    assert resp.status_code == 400
    assert fake.cm.written is None


async def test_mcp_config_valid_persists_and_reloads(fake):
    resp = await router.set_mcp_config(_req({"mcp": []}))
    assert resp.status_code == 200
    assert _data(resp)["data"] == {
        "status": "ok",
        "env_keys": 3,
        "fanout": {"mode": "local-only", "note": "no worker bus configured; only this worker reloaded"},
    }
    assert fake.cm.written is not None
    assert fake.cm.written["mcp"] == []


# -- the new mcp-status / manifest routes (C4 domain work) -------------------


def _query_req(query: str = "", **path_params) -> Request:
    from starlette.datastructures import QueryParams

    return cast(
        Request,
        SimpleNamespace(json=None, path_params=path_params, query_params=QueryParams(query)),
    )


@pytest.fixture
def fake_full(monkeypatch):
    """A live app whose admin answers the whole mcp-status surface, no worker bus."""
    live = {"mcp": [{"title": "gh"}], "user_tools": []}
    cm = _FakeConfigManager({"mcp": [], "tools": []})

    admin = SimpleNamespace(
        live_manifest=live,
        list_failed_mcps=lambda: [{"title": "gh", "status": "unavailable"}],
        reload_failed_mcps=lambda: [{"title": "gh", "status": "ok"}],
        deregister_mcp=lambda title: {"title": title, "status": "ok", "removed": [f"{title}_t"]},
        reload_mcp=lambda title: {"title": title, "status": "ok"},
        reload_config=lambda: {"status": "ok", "env_keys": 0},
    )
    fake_app = SimpleNamespace(
        admin=admin,
        config=SimpleNamespace(config_manager=cm),
        backends=SimpleNamespace(backend=None),
    )
    monkeypatch.setattr(tai42_app, "_impl", fake_app)
    bus = FakeBus(origin="serve-x")
    monkeypatch.setattr(instance.app, "_bus", bus)
    return SimpleNamespace(app=fake_app, cm=cm, bus=bus)


async def test_update_manifest_valid_replaces(fake_full):
    resp = await router.update_manifest(_req({"manifest_text": "mcp: []\n"}))
    assert resp.status_code == 200
    data = _data(resp)["data"]
    # Persist-through: the whole posted document is persisted and reloaded, and the
    # response embeds the fleet report (local-only here — no worker bus configured).
    assert data["fanout"] == {
        "mode": "local-only",
        "note": "no worker bus configured; only this worker reloaded",
    }
    assert fake_full.cm.written == {"mcp": []}


async def test_update_manifest_invalid_body_400(fake_full):
    # A non-mapping manifest fails ManifestReplace validation at the HTTP edge → 400.
    resp = await router.update_manifest(_req({"manifest": "not-a-mapping"}))
    assert resp.status_code == 400
    assert "invalid manifest" in _data(resp)["error"]


async def test_update_manifest_non_object_body_400(fake_full):
    resp = await router.update_manifest(_req(["not", "an", "object"]))
    assert resp.status_code == 400
    assert _data(resp)["error"] == "request body must be a JSON object"


async def test_list_failed_mcps_route(fake_full):
    resp = await router.list_failed_mcps(_query_req(""))
    assert resp.status_code == 200
    assert _data(resp)["data"]["results"][0]["payload"] == [{"title": "gh", "status": "unavailable"}]


async def test_reload_failed_mcps_route(fake_full):
    resp = await router.reload_failed_mcps(_req({"targets": None}))
    assert resp.status_code == 200
    assert _data(resp)["data"]["results"][0]["payload"] == [{"title": "gh", "status": "ok"}]


async def test_reload_failed_mcps_route_no_body(fake_full):
    # A POST with no body → targets None → applied on this worker, its result on the
    # self entry of the fleet report.
    resp = await router.reload_failed_mcps(_req(None))
    assert resp.status_code == 200
    assert _data(resp)["data"]["results"][0]["payload"] == [{"title": "gh", "status": "ok"}]


async def test_deregister_mcp_route(fake_full):
    resp = await router.deregister_mcp(_req(None, title="gh"))
    assert resp.status_code == 200
    assert _data(resp)["data"]["results"][0]["payload"] == {"title": "gh", "status": "ok", "removed": ["gh_t"]}


async def test_mcp_config_malformed_json_400(fake):
    # A body whose JSON does not parse is a loud 400 via the HTTP-edge extractor.
    async def _raise():
        raise ValueError("Expecting value: line 1 column 1 (char 0)")

    req = cast(Request, SimpleNamespace(json=_raise, path_params={}, query_params={}))
    resp = await router.set_mcp_config(req)
    assert resp.status_code == 400
    assert "Expecting value" in _data(resp)["error"]


async def test_reload_mcp_route_targeting_self(fake_full):
    # Targeting this worker by its origin → it re-probes locally and the report
    # carries its self entry.
    resp = await router.reload_mcp(_req({"targets": ["serve-x"]}, title="gh"))
    assert resp.status_code == 200
    data = _data(resp)["data"]
    assert data["results"][0]["origin"] == "serve-x"
    assert data["results"][0]["payload"] == {"title": "gh", "status": "ok"}
