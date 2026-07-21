"""The routers the skeleton mounts by default.

``DEFAULT_API_ROUTERS`` is the ordered set of every route-registering API router
module the skeleton mounts without an operator naming it, and
``STUDIO_SPA_ROUTER`` is the Studio SPA catch-all that must import LAST — it
matches ``/{spa_path:path}`` and would shadow any router registered after it.

``Manifest.default_routers`` selects how these compose:

- ``"all"`` — mount ``DEFAULT_API_ROUTERS``, then the manifest's own extras,
  then ``STUDIO_SPA_ROUTER`` last (the full-Studio deployment).
- ``"api"`` — mount ``DEFAULT_API_ROUTERS`` plus extras but NOT the SPA
  catch-all (a headless JSON ``/api`` deployment, no browser UI).
- ``"none"`` — mount nothing by default; ``routers_modules`` is authoritative
  (a fully-manual or MCP-only surface).

Membership is EVERY module under ``tai42_skeleton.routers`` that registers an
HTTP route, EXCEPT the SPA catch-all (force-appended last, never in this tuple)
and the route-less helper modules (``_tool_call``, ``metrics_settings``,
``observability_support``, ``tool_runs_settings``, ``prometheus``).
``tests/app/test_route_defaults.py`` re-derives that set by iterating the real
package, so a newly-added route-registering router missing here fails the test
rather than being silently un-mounted.
"""

from __future__ import annotations

# The Studio SPA catch-all. It registers ``GET /{spa_path:path}`` (matches any
# path) so it must import after every API router, else it shadows them. The
# loader force-appends it LAST under ``"all"``; the ordering-aware manifest patch
# inserts plugin routers BEFORE it. This is the ONE place the module path is
# spelled — every other consumer imports this constant.
STUDIO_SPA_ROUTER = "tai42_skeleton.routers.plugins"

# The 27 route-registering API router modules, mounted by default under
# ``"all"``/``"api"``. Ordered alphabetically; among these each owns a distinct
# ``/api/*`` (or ``/metrics``, ``/health``) prefix, so their relative order is
# not load-bearing — only the SPA catch-all's last position is.
DEFAULT_API_ROUTERS: tuple[str, ...] = (
    "tai42_skeleton.routers.agents",
    "tai42_skeleton.routers.api_keys",
    "tai42_skeleton.routers.backend",
    "tai42_skeleton.routers.backup",
    "tai42_skeleton.routers.channels",
    "tai42_skeleton.routers.config",
    "tai42_skeleton.routers.connectors",
    "tai42_skeleton.routers.extensions",
    "tai42_skeleton.routers.health",
    "tai42_skeleton.routers.hooks",
    "tai42_skeleton.routers.interactions",
    "tai42_skeleton.routers.login",
    "tai42_skeleton.routers.manifest",
    "tai42_skeleton.routers.marketplace",
    "tai42_skeleton.routers.metrics",
    "tai42_skeleton.routers.notifications",
    "tai42_skeleton.routers.observability",
    "tai42_skeleton.routers.presets",
    "tai42_skeleton.routers.resources",
    "tai42_skeleton.routers.schedules",
    "tai42_skeleton.routers.storage",
    "tai42_skeleton.routers.sub_mcp",
    "tai42_skeleton.routers.system_kinds",
    "tai42_skeleton.routers.templates",
    "tai42_skeleton.routers.tool_extensions",
    "tai42_skeleton.routers.tool_runs",
    "tai42_skeleton.routers.tools",
)
