"""Shared fakes for the authz suite — reuses the access_control fakes and wires
the redis/pg client seams the verifier + policy enforcer read through."""

from __future__ import annotations

from contextlib import contextmanager

import pytest
from starlette.requests import Request
from starlette.responses import Response

from tai42_skeleton.access_control import policy as policy_module
from tai42_skeleton.access_control import store as store_module
from tai42_skeleton.access_control import verifier as verifier_module
from tai42_skeleton.access_control.role_gate import reset_route_index
from tai42_skeleton.app.route_registry import RouteAction, route_registry
from tests.access_control.conftest import FakeAccessControlPg, FakeRedis, make_client_ctx, make_pg_ctx

# The routes the suite's probe operations are registered at — all grantable ``write``.
PROBE_ROUTES = ("/api/things/wipe", "/api/things/write", "/api/things/read")
PROBE_METHOD = "POST"

# An admin-only route with a caller-filled path segment, plus a grantable route that
# SHADOWS one instantiation of it.
FENCED_TEMPLATE_ROUTE = "/api/things/{target}/fenced"
SHADOW_ROUTE = "/api/things/shadow/fenced"


class _FakeResourceManager:
    async def render_by_id_or_content(self, *, content, template_id, kwargs):
        return content


class _FakeStorage:
    def __init__(self) -> None:
        self.resource_manager = _FakeResourceManager()


class _FakeApp:
    def __init__(self) -> None:
        self.storage = _FakeStorage()


async def _probe_handler(request: Request) -> Response:
    """The probe route's handler; never serves a request. It exists so the recorded row is
    the one a real adapter registration produces."""
    return Response()


@contextmanager
def _recorded_routes(paths: tuple[str, ...], action: RouteAction):
    """Record ``paths`` as ``POST`` routes of class ``action`` for the duration.

    Only those rows are dropped afterwards — restoring a whole registry snapshot would
    delete the rows the routers record on their once-per-process import. The resolver index
    is rebuilt on both edges.
    """
    for path in paths:
        route_registry.record(
            path=path,
            methods=[PROBE_METHOD],
            name=f"probe{path.replace('/', '_')}",
            handler=_probe_handler,
            summary="The authz suite's probe route",
            tags=["things"],
            authed=True,
            request_model=None,
            response_model=None,
            action=action,
        )
    reset_route_index()
    try:
        yield
    finally:
        for path in paths:
            del route_registry._routes[path, (PROBE_METHOD,)]
        reset_route_index()


@pytest.fixture(autouse=True)
def probe_routes():
    """Record the suite's probe routes as the operations adapter does.

    The tool edge pins every dispatch to the operation's OWN registered route, so a probe
    whose route template no registered route answers would be denied.
    """
    with _recorded_routes(PROBE_ROUTES, "write"):
        yield


@pytest.fixture
def fenced_template_route():
    """The admin-only templated route the path-argument tests dispatch at, plus the
    grantable route shadowing one instantiation of it."""
    with _recorded_routes((FENCED_TEMPLATE_ROUTE,), "fenced"), _recorded_routes((SHADOW_ROUTE,), "write"):
        yield


@pytest.fixture
def bound_app():
    """Bind a minimal fake app onto ``tai42_app`` (the condition renderer the authz check
    reaches through), restoring whatever was bound before."""
    from tai42_contract.app import tai42_app

    app = _FakeApp()
    with tai42_app.bound(app):
        yield app


@pytest.fixture
def ac_env(monkeypatch):
    """A fake access-control backend: an empty PG (routes + policies) and a shared
    Redis, wired over the store/verifier/policy client seams. Returns the PG so a
    test seeds routes/policies on it."""
    pg = FakeAccessControlPg()
    redis = FakeRedis()
    monkeypatch.setattr(store_module, "client_ctx", make_pg_ctx(pg))
    monkeypatch.setattr(verifier_module, "client_ctx", make_client_ctx(redis))
    monkeypatch.setattr(policy_module, "client_ctx", make_client_ctx(redis))
    return pg
