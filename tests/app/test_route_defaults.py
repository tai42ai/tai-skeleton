"""The default-router set is the FULL route-registering package, de-circularized.

The membership guard derives the route-registering module set by ITERATING the
real ``tai42_skeleton.routers`` package (importing each submodule against a
recording app and observing whether it registers any HTTP route), NEVER from
``DEFAULT_API_ROUTERS`` itself. So a newly-added route-registering router that is
missing from ``DEFAULT_API_ROUTERS`` FAILS this test — it can never be silently
omitted (which would recreate the dark-pages regression), and there is no
hand-maintained skip list for the route-less helpers (they are excluded by the
detection because they register nothing).
"""

from __future__ import annotations

import importlib
import pkgutil
import sys

import tai42_skeleton.routers as _routers_pkg
from tai42_skeleton.app.http import HttpSurface
from tai42_skeleton.app.route_defaults import DEFAULT_API_ROUTERS, STUDIO_SPA_ROUTER
from tai42_skeleton.app.route_registry import _SpecLifecycle


class _RecordingFastMCP:
    """Records every ``custom_route`` path. Every registration surface (a native
    ``@tai42_app.http.custom_route``, ``http_surface()``, and
    ``register_operation_route``) funnels through ``HttpSurface.custom_route`` and
    then ``self._app._fast_mcp.custom_route``, so this captures them all."""

    def __init__(self) -> None:
        self.paths: list[str] = []

    def custom_route(self, path, methods, name, include_in_schema):
        self.paths.append(path)
        return lambda fn: fn


class _RecordingApp:
    """A minimal offline app whose ``_fast_mcp`` records route registrations."""

    def __init__(self) -> None:
        self._fast_mcp = _RecordingFastMCP()
        self.http = HttpSurface(self)  # type: ignore[arg-type]
        self.lifecycle = _SpecLifecycle()


def _route_registering_modules() -> set[str]:
    """Every module under ``routers/`` that registers at least one HTTP route,
    discovered by re-importing each against a recording app."""
    from tai42_contract.app import tai42_app

    app = _RecordingApp()
    module_names = [info.name for info in pkgutil.iter_modules(_routers_pkg.__path__, _routers_pkg.__name__ + ".")]
    try:
        # The recording app is bound only for the probe, so the prior binding is back
        # as soon as it ends.
        with tai42_app.bound(app):
            registering: set[str] = set()
            for name in module_names:
                before = len(app._fast_mcp.paths)
                sys.modules.pop(name, None)
                importlib.import_module(name)
                if len(app._fast_mcp.paths) > before:
                    registering.add(name)
            return registering
    finally:
        # Drop the recording-bound router modules so the next consumer re-imports them
        # fresh, as the loader does on every boot.
        for name in module_names:
            sys.modules.pop(name, None)


def test_default_api_routers_excludes_the_spa_catch_all() -> None:
    assert STUDIO_SPA_ROUTER not in DEFAULT_API_ROUTERS


def test_default_api_routers_has_no_duplicates() -> None:
    assert len(DEFAULT_API_ROUTERS) == len(set(DEFAULT_API_ROUTERS))


def test_default_set_equals_the_discovered_route_registering_package() -> None:
    # The DE-CIRCULARIZING assertion: the default set + the catch-all EQUALS every
    # route-registering module discovered from the package. A new router missing
    # from DEFAULT_API_ROUTERS fails here; a route-less helper is excluded because
    # it registered nothing, not by a skip list.
    discovered = _route_registering_modules()
    assert set(DEFAULT_API_ROUTERS) | {STUDIO_SPA_ROUTER} == discovered
