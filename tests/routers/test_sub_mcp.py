"""Sub-MCP router: JSON-safe route listing, register, and unregister."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, cast

import pytest
from starlette.requests import Request
from tai42_contract.app import tai42_app
from tai42_contract.sub_mcp import RouteConfig

from tai42_skeleton.routers import sub_mcp as router


def _req(**path_params) -> Request:
    return cast(Request, SimpleNamespace(path_params=path_params))


def _body_req(body: bytes) -> Request:
    scope = {"type": "http", "method": "POST", "path": "/api/sub-mcp", "headers": [], "query_string": b""}
    delivered = {"done": False}

    async def receive():
        if delivered["done"]:
            return {"type": "http.disconnect"}
        delivered["done"] = True
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(scope, receive)


def _json(resp) -> dict:
    return json.loads(bytes(resp.body))


class _FakeRouter:
    def __init__(self, routes: dict[str, RouteConfig]):
        self.routes = routes
        self.registered: list[tuple] = []
        self.unregistered: list[str] = []
        # Set by the ``install`` fixture to the per-test in-memory store, so a test
        # can assert what the store-first service persisted.
        self.store: Any = None
        # Set by the ``install`` fixture to the per-test FakeRedis backing the policy
        # version counter, so a test can assert a mount mutation bumped the version.
        self.redis: Any = None

    async def register_sub_mcp_app(self, slug, tools, transport="http"):
        self.registered.append((slug, tools, transport))
        self.routes[slug] = RouteConfig(tools=tools, transport=transport)

    async def unregister_sub_mcp_app(self, slug):
        self.unregistered.append(slug)
        self.routes.pop(slug, None)


class _FakeTools:
    def __init__(self, registered):
        self._registered = set(registered)

    async def get_tools(self):
        return {name: SimpleNamespace(name=name) for name in self._registered}


@pytest.fixture
def install(monkeypatch):
    """Install a fake router + a FRESH in-memory registration store seeded to match.

    The routes go through the store-first service now, so a fresh per-test store is
    monkeypatched over the module-level (reset-exempt) singleton to keep tests
    isolated; it is seeded from ``routes`` so the store-backed GET/DELETE see the
    same state the fake router is primed with.
    """
    from tai42_skeleton.access_control import management as management_module
    from tai42_skeleton.sub_mcp import store as sub_mcp_store
    from tests.access_control.conftest import FakeRedis, make_client_ctx

    # A successful register/unregister bumps the policy version (so cached capability
    # projections re-read the new mount set), which goes through ``management``'s Redis —
    # point it at a per-test FakeRedis so the bump is observable and never hits a backend.
    redis = FakeRedis(strings={}, hashes={})
    monkeypatch.setattr(management_module, "client_ctx", make_client_ctx(redis))

    def _install(routes=None, registered=("get_forecast",)):
        routes = routes or {}
        fake = _FakeRouter(dict(routes))
        fresh_store = sub_mcp_store.InMemorySubMcpStore()
        for slug, config in routes.items():
            fresh_store._routes[slug] = config
        monkeypatch.setattr(sub_mcp_store, "_IN_MEMORY_STORE", fresh_store)
        monkeypatch.setattr(
            tai42_app,
            "_impl",
            SimpleNamespace(
                sub_app=SimpleNamespace(mcp_sub_app_router=fake),
                tools=_FakeTools(registered),
            ),
        )
        fake.store = fresh_store
        fake.redis = redis
        return fake

    return _install


# -- GET /api/sub-mcp --------------------------------------------------------


async def test_list_serializes_routes(install):
    install({"weather": RouteConfig(tools=["get_forecast"], transport="http")})
    resp = await router.list_sub_mcp(_req())
    assert resp.status_code == 200
    assert _json(resp) == {"data": {"weather": {"tools": ["get_forecast"], "transport": "http"}}}


async def test_list_empty(install):
    install({})
    resp = await router.list_sub_mcp(_req())
    assert _json(resp) == {"data": {}}


# -- POST /api/sub-mcp -------------------------------------------------------


async def test_register_happy(install):
    fake = install({})
    resp = await router.register_sub_mcp(_body_req(b'{"slug": "weather", "tools": ["get_forecast"]}'))
    assert resp.status_code == 200
    assert _json(resp) == {"data": {"slug": "weather", "tools": ["get_forecast"], "transport": "http"}}
    assert fake.registered == [("weather", ["get_forecast"], "http")]
    # The registration is durable: it persisted to the store, not just the router.
    stored = await fake.store.get_route("weather")
    assert stored is not None
    assert stored.tools == ["get_forecast"]


async def test_register_with_transport(install):
    fake = install({}, registered=("get_forecast",))
    resp = await router.register_sub_mcp(
        _body_req(b'{"slug": "weather", "tools": ["get_forecast"], "transport": "sse"}')
    )
    assert resp.status_code == 200
    assert _json(resp)["data"]["transport"] == "sse"
    assert fake.registered == [("weather", ["get_forecast"], "sse")]


async def test_register_invalid_transport_400(install):
    install({})
    resp = await router.register_sub_mcp(
        _body_req(b'{"slug": "weather", "tools": ["get_forecast"], "transport": "carrier-pigeon"}')
    )
    assert resp.status_code == 400
    assert "transport" in _json(resp)["error"]


async def test_register_bad_slug_400(install):
    # A slug with a '/' would register a route the dispatcher can never reach or
    # delete (a permanent phantom) — rejected up front.
    install({})
    resp = await router.register_sub_mcp(_body_req(b'{"slug": "a/b", "tools": ["get_forecast"]}'))
    assert resp.status_code == 400
    assert "slug" in _json(resp)["error"]


async def test_register_slug_with_trailing_newline_400(install):
    # ``\Z`` (not ``$``) anchors the slug so a trailing newline cannot slip through
    # and mint a phantom, unreachable route.
    install({})
    resp = await router.register_sub_mcp(_body_req(b'{"slug": "weather\\n", "tools": ["get_forecast"]}'))
    assert resp.status_code == 400
    assert "slug" in _json(resp)["error"]


async def test_register_unknown_tool_404(install):
    install({}, registered=("get_forecast",))
    resp = await router.register_sub_mcp(_body_req(b'{"slug": "weather", "tools": ["get_forecast", "ghost"]}'))
    assert resp.status_code == 404
    assert "ghost" in _json(resp)["error"]


async def test_register_missing_slug_400(install):
    install({})
    resp = await router.register_sub_mcp(_body_req(b'{"tools": ["x"]}'))
    assert resp.status_code == 400
    assert "slug" in _json(resp)["error"]


async def test_register_missing_tools_400(install):
    install({})
    resp = await router.register_sub_mcp(_body_req(b'{"slug": "weather"}'))
    assert resp.status_code == 400
    assert "tools" in _json(resp)["error"]


async def test_register_non_string_tools_400(install):
    install({})
    resp = await router.register_sub_mcp(_body_req(b'{"slug": "weather", "tools": [1, 2]}'))
    assert resp.status_code == 400


async def test_register_bad_json_400(install):
    install({})
    resp = await router.register_sub_mcp(_body_req(b"nope"))
    assert resp.status_code == 400
    assert "invalid JSON" in _json(resp)["error"]


async def test_register_non_object_body_400(install):
    # Valid JSON but not an object (e.g. a list) is rejected before any registration.
    install({})
    resp = await router.register_sub_mcp(_body_req(b"[1, 2]"))
    assert resp.status_code == 400
    assert "JSON object" in _json(resp)["error"]


# -- DELETE /api/sub-mcp/{slug} ----------------------------------------------


async def test_unregister_happy(install):
    fake = install({"weather": RouteConfig(tools=["get_forecast"])})
    resp = await router.unregister_sub_mcp(_req(slug="weather"))
    assert resp.status_code == 200
    assert _json(resp) == {"data": {"slug": "weather", "removed": True}}
    assert fake.unregistered == ["weather"]
    # Removed from the durable store, not just the local router.
    assert await fake.store.get_route("weather") is None


async def test_unregister_store_only_slug_succeeds(install):
    # A slug registered on a SIBLING worker is present in the shared store but not
    # bound in this worker's router. It must still be deletable from here.
    fake = install({})
    await fake.store.save_route("remote", RouteConfig(tools=["get_forecast"]))
    resp = await router.unregister_sub_mcp(_req(slug="remote"))
    assert resp.status_code == 200
    assert _json(resp) == {"data": {"slug": "remote", "removed": True}}
    # It was store-only, so nothing was torn down locally, but the store row is gone.
    assert fake.unregistered == []
    assert await fake.store.get_route("remote") is None


async def test_list_reads_store_not_local_cache(install):
    # A slug registered on a SIBLING worker lives in the shared store but is not
    # bound in this worker's router. GET must read the durable store so the list is
    # coherent across workers — the sibling's route shows up here even though the
    # local router never saw it.
    fake = install({})
    await fake.store.save_route("remote", RouteConfig(tools=["get_forecast"], transport="sse"))
    resp = await router.list_sub_mcp(_req())
    assert resp.status_code == 200
    assert _json(resp) == {"data": {"remote": {"tools": ["get_forecast"], "transport": "sse"}}}
    # It is store-only — the local router cache never bound it.
    assert "remote" not in fake.routes


async def test_unregister_unknown_404(install):
    fake = install({})
    resp = await router.unregister_sub_mcp(_req(slug="ghost"))
    assert resp.status_code == 404
    assert "not found" in _json(resp)["error"]
    assert fake.unregistered == []


# -- policy-version bump on mount mutation (projection invalidation) ----------


async def test_register_and_unregister_bump_policy_version(install):
    # A mount is a reachable surface, so registering or unregistering one must invalidate
    # cached capability projections exactly as a route-table edit does — by bumping the
    # policy version.
    from tai42_skeleton.access_control.settings import access_control_settings

    version_key = access_control_settings().policy_version_key
    fake = install({})

    v0 = int(fake.redis._strings.get(version_key, 0))
    resp = await router.register_sub_mcp(_body_req(b'{"slug": "weather", "tools": ["get_forecast"]}'))
    assert resp.status_code == 200
    v1 = int(fake.redis._strings[version_key])
    assert v1 > v0

    resp = await router.unregister_sub_mcp(_req(slug="weather"))
    assert resp.status_code == 200
    v2 = int(fake.redis._strings[version_key])
    assert v2 > v1


async def test_failed_register_and_unregister_do_not_bump_version(install):
    # A registration/unregistration that wrote nothing (an unknown tool 404, an unknown
    # slug 404) must NOT bump the version — no surface changed.
    from tai42_skeleton.access_control.settings import access_control_settings

    version_key = access_control_settings().policy_version_key
    fake = install({}, registered=("get_forecast",))

    resp = await router.register_sub_mcp(_body_req(b'{"slug": "weather", "tools": ["ghost"]}'))
    assert resp.status_code == 404
    assert version_key not in fake.redis._strings

    resp = await router.unregister_sub_mcp(_req(slug="ghost"))
    assert resp.status_code == 404
    assert version_key not in fake.redis._strings
