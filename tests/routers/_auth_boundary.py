"""Shared wiring for the router auth-boundary tests.

Each boundary test stands a minimal Starlette app carrying only the router's
routes plus the real :class:`AuthAdapter` middleware stack, with access control
ENABLED. A fake access-control store resolves each route to either a protected
scope (AUTHED — an unauthenticated request is denied 401/403 before the handler
runs) or the public resource id (PUBLIC — the handler is reached with no
credential). The point is to pin the intended stance so a future accidental
auth-flip is caught.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import asynccontextmanager

from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from tai42_skeleton.access_control import store as store_module
from tai42_skeleton.access_control import verifier as verifier_module
from tai42_skeleton.access_control.adapter import AuthAdapter
from tai42_skeleton.access_control.settings import AccessControlSettings
from tests.access_control.conftest import FakeAccessControlPg, make_pg_ctx

AUTHED = "authed"
PUBLIC = "public"


class _VersionRedis:
    """A stand-in for the verifier's plain-Redis policy-version read: ``get`` on
    the version key answers ``None`` (version 0). Routes/patterns now come from the
    PG store, so this fake only serves the version counter."""

    async def get(self, key: str) -> str | None:
        return None


def wire_store_from_route_strings(monkeypatch, route_strings: Mapping[str, str]) -> None:
    """Seed the PG store the verifier now reads routes from.

    ``route_strings`` is a ``{url: scope_id}`` map; each key is the store row's
    ``url``. For boundary tests, call this so the route resolution hits the fake
    store, not a real Postgres."""
    pg = FakeAccessControlPg()
    for key, scope_id in route_strings.items():
        pg.add_route(key, scope_id)
    monkeypatch.setattr(store_module, "client_ctx", make_pg_ctx(pg))


def boundary_client(
    monkeypatch,
    routes: list[Route],
    stances: Mapping[str, str],
) -> TestClient:
    """Build a TestClient over ``routes`` guarded by the real auth stack.

    ``stances`` maps each route's path pattern (a regex, matched by the verifier)
    to :data:`AUTHED` or :data:`PUBLIC`. The pattern doubles as its own template
    key, so the PG store resolves an AUTHED pattern to a protected scope and a
    PUBLIC pattern to the settings' public resource id.
    """
    # ``path_patterns`` maps a route regex -> its template KEY; use the pattern
    # text itself as the key so the map stays one-to-one and readable.
    ac_settings = AccessControlSettings(path_patterns={pattern: pattern for pattern in stances})

    # The verifier resolves each matched pattern's template (the pattern text) to its
    # stored resource id through the PG store; seed one route row per pattern.
    pg = FakeAccessControlPg()
    for pattern, stance in stances.items():
        scope_id = ac_settings.public_resource_id if stance == PUBLIC else f"{pattern}-protected"
        pg.add_route(pattern, scope_id)

    @asynccontextmanager
    async def version_ctx(client_cls, settings=None, *, fresh=False, **kwargs):
        yield _VersionRedis()

    monkeypatch.setattr(verifier_module, "client_ctx", version_ctx)
    monkeypatch.setattr(store_module, "client_ctx", make_pg_ctx(pg))

    app = Starlette(routes=routes, middleware=AuthAdapter(ac_settings).get_middleware())
    return TestClient(app)
