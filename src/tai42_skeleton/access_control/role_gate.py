"""The per-tag ACCESS-LEVEL gate — Layer 2 of the (resource-x-action) model.

A role's editable grant is a map ``feature-tag → none|read|write``. This module holds
the PURE decision the enforcement gate and the capability projection SHARE (never two
divergent copies): given a request's resolved route (its ``tags`` + ``action``) and a
role's grant map, decide allow/deny and name the CAUSE of any denial. It also resolves
a concrete request ``(path, method)`` to its registered :class:`RouteMetadata` so the
gate can read the route's feature tags and action-class.

The base-tier ceiling (owner-scoping, the ``/api/auth`` control-plane gate, the viewer
read-only ceiling) is a SEPARATE term carried by the seeded jq base and enforced
alongside this one — this module answers ONLY the per-tag level question plus the
admin-only fence, and the enforcement site intersects the two (fail-closed).
"""

from __future__ import annotations

import re
from enum import Enum

from tai42_skeleton.access_control.path_canon import MalformedPathError, canonicalize_path
from tai42_skeleton.app.route_registry import (
    RouteMetadata,
    load_all_routes,
    method_to_action,
)

# A role's editable grant map: feature-group TAG name → its access level. An absent tag
# is treated as ``none`` (deny) — the default-deny posture the whole gate rests on.
RoleGrants = dict[str, str]

# The access levels, ranked so a higher level satisfies every lower action: ``write``
# satisfies read+write, ``read`` satisfies read, ``none`` satisfies nothing.
_LEVEL_RANK: dict[str, int] = {"none": 0, "read": 1, "write": 2}
_ACTION_RANK: dict[str, int] = {"read": 1, "write": 2}


class DenialCause(Enum):
    """The internal CAUSE of a per-request denial, attached to the denial for
    debugging/logging. The EXTERNAL response stays a generic 403 (no information leak);
    only the internal detail/log line names the cause."""

    HARD_FENCE = "hard-fence"
    LEVEL_MISS = "level-miss"
    SCOPE_MISS = "scope-miss"


def level_satisfies(level: str, action: str) -> bool:
    """Whether a role's ``level`` on a tag satisfies a request's derived ``action``:
    ``write`` satisfies ``read`` and ``write``; ``read`` satisfies ``read``; ``none``
    (or an unknown level) satisfies nothing. Fail-closed on an unknown level."""
    return _LEVEL_RANK.get(level, 0) >= _ACTION_RANK[action]


def grant_map_admits(meta: RouteMetadata, method: str, grants: RoleGrants) -> tuple[bool, DenialCause | None]:
    """The pure per-tag decision for a NON-ADMIN role over a resolved gated route.

    * A ``fenced``/``secret`` route is admin-only — no per-tag level opens it, so it is a
      hard-fence DENY.
    * Otherwise the route is grantable: it is admitted iff ANY of the route's feature
      tags carries a level that satisfies the method's derived action (a role granted a
      tag reaches every grantable route under it). An absent tag is level ``none``.

    Returns ``(allowed, cause)``; ``cause`` is ``None`` on an allow, else the denial
    cause for the caller to log."""
    if meta.action in ("fenced", "secret"):
        return False, DenialCause.HARD_FENCE
    derived = method_to_action(method)
    for tag in meta.tags:
        if level_satisfies(grants.get(tag, "none"), derived):
            return True, None
    return False, DenialCause.LEVEL_MISS


def effective_tag_levels(grants: RoleGrants, tags: list[str]) -> dict[str, str]:
    """The role's effective level on each feature ``tag`` (an absent tag → ``none``) —
    the shared answer the Studio nav, the effective-access view, and the gate read, so
    they can never drift. Fenced/secret routes are never grantable, so a tag that
    carries only fenced/secret routes still reads ``none`` here (the caller filters the
    grantable tag set)."""
    return {tag: grants.get(tag, "none") for tag in tags}


# -- concrete (path, method) → registered route resolution -------------------

# The route registry is import-populated and immutable at runtime (routes record only as
# router modules import at startup), so the concrete/templated index is built once and
# reused. A test that registers a new route calls :func:`reset_route_index` to rebuild.
_concrete_index: dict[tuple[str, str], RouteMetadata] | None = None
_templated_matchers: list[tuple[re.Pattern[str], frozenset[str], RouteMetadata]] | None = None

# A path-template segment (``{name}`` or ``{name:path}``) → its matching sub-pattern: a
# plain segment matches one path segment; a ``:path`` segment matches the rest greedily.
_TEMPLATE_SEGMENT = re.compile(r"\{([^}:]+)(:path)?\}")


def _template_to_regex(template: str) -> re.Pattern[str]:
    def _sub(match: re.Match[str]) -> str:
        return "(?:.+)" if match.group(2) else "(?:[^/]+)"

    pattern = _TEMPLATE_SEGMENT.sub(lambda m: _sub(m), re.escape(template).replace(r"\{", "{").replace(r"\}", "}"))
    return re.compile(f"^{pattern}$")


def _build_index() -> None:
    global _concrete_index, _templated_matchers
    concrete: dict[tuple[str, str], RouteMetadata] = {}
    templated: list[tuple[re.Pattern[str], frozenset[str], RouteMetadata]] = []
    # ``load_all_routes`` ensures the router modules are imported (offline) so the
    # registry is populated before the index builds — in a booted server they already
    # are; in a CLI/test process this triggers the offline import.
    for meta in load_all_routes():
        methods = frozenset(meta.methods)
        # Key/compile on the CANONICALIZED registered path so the index and the
        # canonicalized lookup in ``resolve_route_meta`` decide on the identical form —
        # a registered route can never fail to resolve through a shape mismatch.
        canonical = canonicalize_path(meta.path)
        if "{" in canonical:
            templated.append((_template_to_regex(canonical), methods, meta))
        else:
            for method in methods:
                concrete[canonical, method] = meta
    _concrete_index = concrete
    _templated_matchers = templated


def reset_route_index() -> None:
    """Drop the cached concrete/templated route index so it rebuilds against the current
    registry — for tests that register routes after the index was first built."""
    global _concrete_index, _templated_matchers
    _concrete_index = None
    _templated_matchers = None


def resolve_route_meta(path: str, method: str | None) -> RouteMetadata | None:
    """The registered :class:`RouteMetadata` a concrete request ``(path, method)``
    resolves to, or ``None`` when the path is not a registered route.

    ``None`` is NOT an allow: a path with no registered route is not a grantable gated
    route (it is the public SPA shell, an operational probe, or an unmapped path), and
    those are governed by the scope layer + the jq base — the per-tag pass simply does
    not act on them. A registered route always resolves here so its fence/level applies.
    A concrete path matching a templated route's pattern resolves to that route's
    metadata (feature tags + action-class). The path is canonicalized first so the match
    decides on the same form the verifier uses; a malformed path resolves to ``None``
    (denied fail-closed by the verifier/middleware, never opened here)."""
    if method is None:
        return None
    if _concrete_index is None or _templated_matchers is None:
        _build_index()
    assert _concrete_index is not None
    assert _templated_matchers is not None
    try:
        canonical = canonicalize_path(path)
    except MalformedPathError:
        return None
    exact = _concrete_index.get((canonical, method))
    if exact is not None:
        return exact
    for pattern, methods, meta in _templated_matchers:
        if method in methods and pattern.fullmatch(canonical):
            return meta
    return None
