"""The route action-class: the central ``method_to_action`` derivation, the registry's
per-route validation, and the boot audit (``route_action_violations`` /
``check_route_actions``) that fails on a missing/mismatched action-class.
"""

from __future__ import annotations

import dataclasses

import pytest

from tai42_skeleton.app.route_registry import (
    RouteRegistry,
    derive_route_action,
    method_to_action,
    route_action_violations,
)


async def _plain(request):  # pragma: no cover - never invoked, only introspected
    from starlette.responses import JSONResponse

    return JSONResponse({})


def test_method_to_action_maps_every_method() -> None:
    for method in ("GET", "HEAD", "OPTIONS", "get"):
        assert method_to_action(method) == "read"
    for method in ("POST", "PUT", "PATCH", "DELETE", "delete"):
        assert method_to_action(method) == "write"


def test_method_to_action_raises_on_unknown_method() -> None:
    with pytest.raises(ValueError, match="unclassifiable"):
        method_to_action("BREW")
    with pytest.raises(ValueError, match="unclassifiable"):
        method_to_action("")


def test_derive_route_action_is_write_if_any_write_method() -> None:
    assert derive_route_action(("GET",)) == "read"
    assert derive_route_action(("POST",)) == "write"
    assert derive_route_action(("GET", "POST")) == "write"


def _record(registry: RouteRegistry, *, methods, action=None, authed=True) -> None:
    registry.record(
        path="/api/thing",
        methods=methods,
        name="x",
        handler=_plain,
        summary="Thing",
        tags=["t"],
        authed=authed,
        request_model=None,
        response_model=None,
        action=action,
    )


def test_record_derives_action_for_public_route_when_omitted() -> None:
    # A PUBLIC route (authed=False) enforces no action, so an omitted action is derived
    # from the method for the spec rather than refused.
    registry = RouteRegistry()
    _record(registry, methods=["GET"], authed=False)
    (meta,) = registry.routes()
    assert meta.action == "read"


def test_record_authed_route_without_action_raises() -> None:
    # An AUTHED route that declares no action-class BOOT-FAILS at registration —
    # allow-by-omission would be a fail-open fence.
    registry = RouteRegistry()
    with pytest.raises(ValueError, match="no action-class"):
        _record(registry, methods=["GET"], authed=True)


def test_record_rejects_grantable_action_mismatched_with_method() -> None:
    registry = RouteRegistry()
    with pytest.raises(ValueError, match="derives"):
        _record(registry, methods=["POST"], action="read")


def test_record_accepts_fenced_and_secret_on_any_method() -> None:
    for action in ("fenced", "secret"):
        registry = RouteRegistry()
        _record(registry, methods=["GET"], action=action)
        (meta,) = registry.routes()
        assert meta.action == action


def test_record_rejects_fenced_or_secret_on_public_route() -> None:
    # A fence is enforced only in the authenticated path, so a fenced/secret class on a
    # PUBLIC (authed=False) route would silently open an admin-only door. Registration
    # refuses it, symmetric with the authed-without-action raise.
    for action in ("fenced", "secret"):
        registry = RouteRegistry()
        with pytest.raises(ValueError, match="public route"):
            _record(registry, methods=["GET"], action=action, authed=False)


def test_route_action_violations_flags_a_mismatched_gated_route() -> None:
    # A synthetic gated route whose stored action disagrees with its method: the boot
    # audit lists it (allow-by-omission / mis-declaration is dead).
    from tai42_skeleton.app import route_registry as rr

    rr.load_all_routes()  # populate the registry (import routers offline)
    meta = rr.route_registry.routes()[0]
    bad = dataclasses.replace(meta, path="/api/__bad__", methods=("POST",), action="read", authed=True)
    rr.route_registry._routes["/api/__bad__", ("POST",)] = bad
    try:
        violations = route_action_violations()
        assert any("/api/__bad__" in v for v in violations)
    finally:
        del rr.route_registry._routes["/api/__bad__", ("POST",)]


def test_route_action_violations_flags_an_unclassified_action() -> None:
    from tai42_skeleton.app import route_registry as rr

    rr.load_all_routes()  # populate the registry (import routers offline)
    meta = rr.route_registry.routes()[0]
    bad = dataclasses.replace(meta, path="/api/__none__", methods=("GET",), action=None, authed=True)  # type: ignore[arg-type]
    rr.route_registry._routes["/api/__none__", ("GET",)] = bad
    try:
        violations = route_action_violations()
        assert any("/api/__none__" in v for v in violations)
    finally:
        del rr.route_registry._routes["/api/__none__", ("GET",)]


async def test_check_route_actions_raises_on_a_bad_route() -> None:
    from tai42_skeleton.access_control.startup import check_route_actions
    from tai42_skeleton.app import route_registry as rr

    rr.load_all_routes()  # populate the registry (import routers offline)
    meta = rr.route_registry.routes()[0]
    bad = dataclasses.replace(meta, path="/api/__bad2__", methods=("DELETE",), action="read", authed=True)
    rr.route_registry._routes["/api/__bad2__", ("DELETE",)] = bad
    try:
        with pytest.raises(RuntimeError, match="action-class audit"):
            await check_route_actions()
    finally:
        del rr.route_registry._routes["/api/__bad2__", ("DELETE",)]


async def test_check_route_actions_passes_on_the_real_surface() -> None:
    # The real, correctly-classified route surface passes the audit.
    from tai42_skeleton.access_control.startup import check_route_actions

    await check_route_actions()
