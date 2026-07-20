"""The CLI↔route parity gate.

Every registered ``/api/*`` route the server exposes must be reachable from the
terminal. This asserts ``registered_routes - allowlist`` is a subset of
``covered_routes``, where the covered set is built from the ``@covers(...)``
attribution every remote command declares and the registered set comes from WP4's
shared enumeration primitive
:func:`load_api_routes` — the SAME primitive the OpenAPI coverage gate uses.

A newly added ``/api/*`` route with no CLI command FAILS this test loudly, so the
parity can never silently drift. A route deliberately withheld from the CLI is named
in the explicit allowlist below (its caller is a browser / external non-operator or
it serves assets — not an operator function).
"""

from __future__ import annotations

# Importing the app imports every remote command module, populating COVERED_ROUTES.
import tai42_skeleton.cli.app  # noqa: F401
from tai42_skeleton.app.route_registry import load_api_routes
from tai42_skeleton.cli.commands._common import COVERED_ROUTES

# Routes intentionally not CLI-exposed: their caller is a browser or an external
# non-operator party, or they serve assets — none are operator functions. Each entry
# is a ``(METHOD, path)`` pair, so an allowlisted path excuses ONLY its named
# method(s), never every method that path might also carry. (Internal health probes
# such as ``/health`` and ``/metrics`` are not ``/api/*`` routes, so
# ``load_api_routes`` never enumerates them and they need no entry here.)
ALLOWLIST: set[tuple[str, str]] = {
    ("POST", "/api/connectors/oauth/complete"),  # browser OAuth callback
    ("GET", "/api/interactions/callback/{ticket}"),  # unauthenticated external answer door
    ("POST", "/api/interactions/callback/{ticket}"),  # unauthenticated external answer door
    ("GET", "/api/plugins/{name}/studio/{path:path}"),  # studio SPA asset serving
    ("GET", "/api/login/methods"),  # public pre-auth login screen (browser)
    ("POST", "/api/auth/logout"),  # browser/session logout, not an operator function
    ("GET", "/api/auth/capabilities"),  # studio mint-capability gating (browser)
    ("GET", "/api/auth/roles"),  # studio users-admin role picker (browser)
}


def _registered_route_pairs() -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for meta in load_api_routes():
        for method in meta.methods:
            pair = (method, meta.path)
            if pair in ALLOWLIST:
                continue
            pairs.add(pair)
    return pairs


def test_every_registered_route_has_a_cli_command() -> None:
    registered = _registered_route_pairs()
    missing = sorted(pair for pair in registered if pair not in COVERED_ROUTES)
    assert not missing, f"registered /api routes with no CLI command: {missing}"


def test_no_attribution_targets_a_missing_route() -> None:
    # Every declared coverage entry must match a real registered route, so a
    # stale ``@covers`` (a route renamed/removed) is caught rather than masking a
    # genuine gap.
    registered_all = {(method, meta.path) for meta in load_api_routes() for method in meta.methods}
    stale = sorted(pair for pair in COVERED_ROUTES if pair not in registered_all)
    assert not stale, f"@covers attributions with no matching registered route: {stale}"


def test_allowlist_entries_are_registered() -> None:
    # An allowlist entry that no longer names a real ``(method, path)`` route is dead
    # — drop it, so the allowlist cannot silently excuse a future route by coincidence
    # of name.
    registered = {(method, meta.path) for meta in load_api_routes() for method in meta.methods}
    dead = sorted(pair for pair in ALLOWLIST if pair not in registered)
    assert not dead, f"allowlist entries that are not registered routes: {dead}"
