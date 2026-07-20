"""Connectors router: providers/connections views, connect/reconnect/patch/
disconnect, and the oauth/complete decode flow (its first production caller)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import cast

import pytest
from starlette.requests import Request
from tai_contract.connectors.models import AuthHealthState
from tai_contract.connectors.service import (
    AliasInUseError,
    CompleteConnectResult,
    DisconnectResult,
    FlowOperation,
    NoAuthConnectResult,
    PatchResult,
    StartConnectResult,
)

import tai_skeleton.operations.connectors as conn_ops
import tai_skeleton.routers.connectors as router
from tai_skeleton.connectors.oauth import client as oauth_client
from tai_skeleton.connectors.oauth import state

# The mode-wrapped fan-out summary the service threads onto a mutating result; the
# router/operation must surface it verbatim in the connector's HTTP response.
_FANOUT = {"mode": "fleet", "op": "reload_config", "results": [{"origin": "serve-x", "outcome": "applied"}]}


def _req(body=None, **path_params) -> Request:
    async def _json():
        return body

    return cast(Request, SimpleNamespace(json=_json, path_params=path_params, query_params={}, headers={}))


def _data(resp):
    return json.loads(bytes(resp.body))


def _record(cid="c1"):
    return SimpleNamespace(
        connection_id=cid,
        provider_id="github",
        alias="work",
        kind="oauth",
        account_identity="me@x",
        enabled_sub_services=["repo"],
        granted_scopes=["repo"],
        auth_health_state=AuthHealthState.HEALTHY,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _provider():
    sub = SimpleNamespace(id="repo", display_name="Repos", description="d", scopes=["repo"])
    field = SimpleNamespace(key="host", label="Host", target="env", required=True, secret=False)
    return SimpleNamespace(
        id="github",
        display_name="GitHub",
        description="d",
        icon_url="/i.png",
        kind="oauth",
        origin="system",
        category="dev",
        sub_services={"repo": sub},
        config_fields=[field],
    )


@pytest.fixture
def wiring(monkeypatch):
    monkeypatch.setattr(router, "compute_redirect_uri", lambda r: "https://app/oauth-bridge.html")
    monkeypatch.setattr(router, "compute_deployment_origin", lambda r: "https://app")
    monkeypatch.setattr(conn_ops, "list_providers", lambda: [_provider()])
    return SimpleNamespace()


# -- Views -------------------------------------------------------------------


async def test_providers_view(wiring):
    resp = await router.providers(_req())
    p = _data(resp)["data"][0]
    assert p["id"] == "github"
    assert p["sub_services"][0]["id"] == "repo"
    # Reshaped wire model: per-sub scopes (not a scopes_summary string), and the
    # flat provider list carries no nested `connections`.
    assert p["sub_services"][0]["scopes"] == ["repo"]
    assert "scopes_summary" not in p["sub_services"][0]
    assert "connections" not in p
    assert p["config_fields"][0]["key"] == "host"


async def test_connections_list_excludes_secrets(wiring, monkeypatch):
    monkeypatch.setattr(conn_ops, "token_store", lambda: SimpleNamespace(list=_alist(["c1"])))
    monkeypatch.setattr(conn_ops, "load_record_or_none", _acall({"c1": _record()}))
    resp = await router.connections(_req())
    data = _data(resp)["data"]
    assert data["total"] == 1
    item = data["items"][0]
    assert item["connection_id"] == "c1"
    assert "access_token" not in item
    assert "config_values" not in item


async def test_get_connection_404(wiring, monkeypatch):
    monkeypatch.setattr(conn_ops, "load_record_or_none", _acall({}))
    resp = await router.get_connection(_req(connection_id="missing"))
    assert resp.status_code == 404


# -- Connect / reconnect / patch / disconnect --------------------------------


async def test_start_connect_oauth(wiring, monkeypatch):
    async def _start(**kw):
        assert kw["origin"] == "https://app"
        assert kw["redirect_uri"] == "https://app/oauth-bridge.html"
        return StartConnectResult(flow_id="f1", authorize_url="https://gh/auth")

    monkeypatch.setattr(router.connection_service, "start_connect", _start)
    resp = await router.start_connect(
        _req({"provider_id": "github", "alias": "work", "enabled_sub_services": ["repo"]})
    )
    assert _data(resp)["data"] == {"flow_id": "f1", "authorize_url": "https://gh/auth"}


async def test_start_connect_no_auth(wiring, monkeypatch):
    async def _start(**kw):
        return NoAuthConnectResult(connection_id="c9", added_manifest_entries=["e"], fanout=_FANOUT)

    monkeypatch.setattr(router.connection_service, "start_connect", _start)
    resp = await router.start_connect(_req({"provider_id": "p", "alias": "a", "enabled_sub_services": ["s"]}))
    body = _data(resp)["data"]
    assert body["connection_id"] == "c9"
    # The awaited fleet report of the manifest broadcast rides the no-auth response.
    assert body["fanout"] == _FANOUT


async def test_start_connect_validation_400(wiring):
    resp = await router.start_connect(_req({"provider_id": "", "alias": "a", "enabled_sub_services": ["s"]}))
    assert resp.status_code == 400


async def test_start_connect_alias_in_use_409(wiring, monkeypatch):
    async def _start(**kw):
        raise AliasInUseError("alias taken")

    monkeypatch.setattr(router.connection_service, "start_connect", _start)
    resp = await router.start_connect(_req({"provider_id": "p", "alias": "a", "enabled_sub_services": ["s"]}))
    assert resp.status_code == 409


async def test_start_connect_oauth_error_maps_to_400(wiring, monkeypatch):
    """An off-list Origin surfaces from the service as RedirectUriNotAllowedError
    (an OAuthError, not a ValueError); the router maps it to a clean 400, not a 500."""

    async def _start(**kw):
        raise oauth_client.RedirectUriNotAllowedError("origin off-list")

    monkeypatch.setattr(router.connection_service, "start_connect", _start)
    resp = await router.start_connect(_req({"provider_id": "p", "alias": "a", "enabled_sub_services": ["s"]}))
    assert resp.status_code == 400
    assert "error" in _data(resp)


async def test_reconnect_oauth_error_maps_to_400(wiring, monkeypatch):
    monkeypatch.setattr(conn_ops, "load_record_or_none", _acall({"c1": _record()}))

    async def _reconnect(**kw):
        raise oauth_client.RedirectUriNotAllowedError("origin off-list")

    monkeypatch.setattr(router.connection_service, "start_reconnect", _reconnect)
    resp = await router.reconnect(_req({"enabled_sub_services": ["repo"]}, connection_id="c1"))
    assert resp.status_code == 400
    assert "error" in _data(resp)


async def test_patch_oauth_error_maps_to_400(wiring, monkeypatch):
    monkeypatch.setattr(conn_ops, "load_record_or_none", _acall({"c1": _record()}))

    async def _patch(**kw):
        raise oauth_client.RedirectUriNotAllowedError("origin off-list")

    monkeypatch.setattr(router.connection_service, "patch_sub_services", _patch)
    resp = await router.patch_sub_services(_req({"enabled_sub_services": ["repo"]}, connection_id="c1"))
    assert resp.status_code == 400
    assert "error" in _data(resp)


async def test_patch_inline_carries_fanout(wiring, monkeypatch):
    monkeypatch.setattr(conn_ops, "load_record_or_none", _acall({"c1": _record()}))

    async def _patch(**kw):
        return PatchResult(
            connection_id="c1",
            enabled_sub_services=["repo", "issues"],
            consent_required=False,
            flow_id=None,
            authorize_url=None,
            added_manifest_entries=["github_issues_work"],
            removed_manifest_entries=[],
            fanout=_FANOUT,
        )

    monkeypatch.setattr(router.connection_service, "patch_sub_services", _patch)
    resp = await router.patch_sub_services(_req({"enabled_sub_services": ["repo", "issues"]}, connection_id="c1"))
    body = _data(resp)["data"]
    assert body["consent_required"] is False
    # The inline toggle mutated the manifest, so its fleet report rides the response.
    assert body["fanout"] == _FANOUT


async def test_patch_consent_only_carries_no_fanout(wiring, monkeypatch):
    monkeypatch.setattr(conn_ops, "load_record_or_none", _acall({"c1": _record()}))

    async def _patch(**kw):
        return PatchResult(
            connection_id="c1",
            enabled_sub_services=["repo"],
            consent_required=True,
            flow_id="f1",
            authorize_url="https://gh/auth",
            added_manifest_entries=[],
            removed_manifest_entries=[],
            fanout=None,
        )

    monkeypatch.setattr(router.connection_service, "patch_sub_services", _patch)
    resp = await router.patch_sub_services(_req({"enabled_sub_services": ["repo", "cal"]}, connection_id="c1"))
    body = _data(resp)["data"]
    assert body["consent_required"] is True
    # A consent-only toggle made no manifest change here ⇒ null fanout in the body.
    assert body["fanout"] is None


async def test_start_connect_malformed_body_400(wiring):
    """A body that fails the contract StartConnectRequest schema (missing the
    required enabled_sub_services) is a 400 in the {"error": ...} envelope, never a
    500 from an escaped pydantic ValidationError."""
    resp = await router.start_connect(_req({"provider_id": "p", "alias": "a"}))
    assert resp.status_code == 400
    assert "error" in _data(resp)


async def test_patch_malformed_body_400(wiring, monkeypatch):
    monkeypatch.setattr(conn_ops, "load_record_or_none", _acall({"c1": _record()}))
    resp = await router.patch_sub_services(_req({"enabled_sub_services": []}, connection_id="c1"))
    assert resp.status_code == 400
    assert "error" in _data(resp)


def _bad_json_req(**path_params) -> Request:
    async def _json():
        raise ValueError("not json")

    return cast(Request, SimpleNamespace(json=_json, path_params=path_params, query_params={}, headers={}))


async def test_start_connect_bad_json_and_non_dict_body_400(wiring):
    # The HTTP-edge extractor maps a malformed body to a 400 in the {"error": ...}
    # envelope (bad JSON, and a non-object body), never an escaped 422/500.
    assert (await router.start_connect(_bad_json_req())).status_code == 400
    resp = await router.start_connect(_req([1, 2]))
    assert resp.status_code == 400
    assert "must be a JSON object" in _data(resp)["error"]


async def test_reconnect_bad_json_and_malformed_body_400(wiring):
    assert (await router.reconnect(_bad_json_req(connection_id="c1"))).status_code == 400
    # A body failing StartReconnectRequest (missing enabled_sub_services) is a 400.
    resp = await router.reconnect(_req({}, connection_id="c1"))
    assert resp.status_code == 400
    assert "error" in _data(resp)


async def test_patch_bad_json_400(wiring):
    assert (await router.patch_sub_services(_bad_json_req(connection_id="c1"))).status_code == 400


async def test_oauth_complete_bad_json_400(wiring):
    resp = await router.oauth_complete(_bad_json_req())
    assert resp.status_code == 400
    assert "invalid JSON body" in _data(resp)["error"]


async def test_disconnect(wiring, monkeypatch):
    async def _disc(**kw):
        return DisconnectResult(
            connection_id="c1",
            upstream_revoke_outcome="success",
            upstream_revoke_status=200,
            removed_manifest_entries=[],
            fanout=_FANOUT,
        )

    monkeypatch.setattr(router.connection_service, "disconnect", _disc)
    resp = await router.disconnect(_req(connection_id="c1"))
    body = _data(resp)["data"]
    assert body["upstream_revoke_outcome"] == "success"
    # The awaited fleet report of the manifest broadcast rides the disconnect response.
    assert body["fanout"] == _FANOUT


async def test_disconnect_404(wiring, monkeypatch):
    # A genuinely-absent connection surfaces as ConnectionNotFoundError from the
    # service (which loads with include_expired), mapped to 404 by the route.
    async def _disc(**kw):
        raise router.connection_service.ConnectionNotFoundError(kw["connection_id"])

    monkeypatch.setattr(router.connection_service, "disconnect", _disc)
    resp = await router.disconnect(_req(connection_id="x"))
    assert resp.status_code == 404


async def test_disconnect_does_not_serving_filter_expired(wiring, monkeypatch):
    # The route must NOT gate on a serving-filtered pre-check: an EXPIRED connection
    # reads as absent through load_record_or_none, yet disconnect — which loads with
    # include_expired — still purges it. Modelled by making the serving-filtered load
    # return empty while the service succeeds; the route must return the purge result,
    # not a 404. Guards the end-to-end fix that a unit test on disconnect() alone misses.
    monkeypatch.setattr(conn_ops, "load_record_or_none", _acall({}))

    async def _disc(**kw):
        return DisconnectResult(
            connection_id="c1",
            upstream_revoke_outcome="success",
            upstream_revoke_status=200,
            removed_manifest_entries=[],
        )

    monkeypatch.setattr(router.connection_service, "disconnect", _disc)
    resp = await router.disconnect(_req(connection_id="c1"))
    assert resp.status_code == 200
    assert _data(resp)["data"]["upstream_revoke_outcome"] == "success"


# -- oauth/complete (decode) -------------------------------------------------


async def test_oauth_complete_cancelled(wiring):
    resp = await router.oauth_complete(_req({"error": "access_denied"}))
    assert _data(resp)["data"]["kind"] == "cancelled"


async def test_oauth_complete_missing_fields_400(wiring):
    assert (await router.oauth_complete(_req({"code": "x"}))).status_code == 400
    assert (await router.oauth_complete(_req({"state": "x"}))).status_code == 400


async def test_oauth_complete_state_invalid(wiring, monkeypatch):
    def _decode(s):
        raise state.StateInvalidError("bad")

    monkeypatch.setattr(router.state, "decode", _decode)
    resp = await router.oauth_complete(_req({"state": "bad", "code": "c"}))
    assert resp.status_code == 400
    assert _data(resp)["data"] == {"kind": "failed", "reason": "StateInvalid"}


async def test_oauth_complete_oauth_error(wiring, monkeypatch):
    monkeypatch.setattr(router.state, "decode", lambda s: state.DecodedState(flow_id="f1", origin="https://app"))

    async def _complete(**kw):
        raise oauth_client.OAuthError("exchange failed")

    monkeypatch.setattr(router.connection_service, "complete_connect", _complete)
    resp = await router.oauth_complete(_req({"state": "ok", "code": "c"}))
    assert resp.status_code == 400
    assert _data(resp)["data"]["reason"] == "OAuthError"


@pytest.mark.parametrize(
    "make_exc",
    [
        lambda: router.connection_service.AliasInUseError("dup alias"),
        lambda: router.connection_service.ConnectionNotFoundError("c1"),
        lambda: router.connection_service.ConcurrentConnectionUpdateError("c1"),
    ],
)
async def test_oauth_complete_completion_error_maps_to_failed(wiring, monkeypatch, make_exc):
    """A post-exchange completion failure (alias collision, vanished connection,
    CAS miss) is a 4xx discriminated {"kind": "failed"} body, not a raw 500."""
    monkeypatch.setattr(router.state, "decode", lambda s: state.DecodedState(flow_id="f1", origin="https://app"))
    exc = make_exc()

    async def _complete(**kw):
        raise exc

    monkeypatch.setattr(router.connection_service, "complete_connect", _complete)
    resp = await router.oauth_complete(_req({"state": "ok", "code": "c"}))
    assert resp.status_code == 400
    body = _data(resp)["data"]
    assert body["kind"] == "failed"
    assert body["reason"] == type(exc).__name__


async def test_oauth_complete_drops_redirect_uri_arg(wiring, monkeypatch):
    """oauth/complete no longer passes redirect_uri — it is read from the stored
    flow state inside complete_connect."""
    monkeypatch.setattr(router.state, "decode", lambda s: state.DecodedState(flow_id="f1", origin="https://app"))

    async def _complete(**kw):
        assert "redirect_uri" not in kw
        return CompleteConnectResult(
            connection_id="c1",
            return_url="/connectors",
            operation=FlowOperation.CONNECT,
            added_manifest_entries=[],
            removed_manifest_entries=[],
        )

    monkeypatch.setattr(router.connection_service, "complete_connect", _complete)
    resp = await router.oauth_complete(_req({"state": "ok", "code": "c"}))
    assert _data(resp)["data"]["kind"] == "success"


async def test_oauth_complete_success(wiring, monkeypatch):
    monkeypatch.setattr(router.state, "decode", lambda s: state.DecodedState(flow_id="f1", origin="https://app"))

    async def _complete(**kw):
        assert kw["flow_id"] == "f1"
        return CompleteConnectResult(
            connection_id="c1",
            return_url="/connectors",
            operation=FlowOperation.CONNECT,
            added_manifest_entries=[],
            removed_manifest_entries=[],
            fanout=_FANOUT,
        )

    monkeypatch.setattr(router.connection_service, "complete_connect", _complete)
    resp = await router.oauth_complete(_req({"state": "ok", "code": "c"}))
    body = _data(resp)["data"]
    # The oauth-complete success body gains the awaited fleet report of its manifest add.
    assert body == {
        "kind": "success",
        "connection_id": "c1",
        "return_url": "/connectors",
        "fanout": _FANOUT,
    }


# -- async helpers -----------------------------------------------------------


def _alist(items):
    async def _f():
        return items

    return _f


def _acall(mapping):
    async def _f(cid):
        return mapping.get(cid)

    return _f
