"""Op-level oracles for the connector operations.

The route oracles in ``tests/routers/test_connectors.py`` drive the reads and the
happy/error paths of the mutations through the adapter; these pin the branches the
round-trip route tests do not reach (the reconnect/patch unknown-connection 404s)
and the destructive projection.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from tai42_contract.connectors.models import AuthHealthState
from tai42_contract.manifest import ApiToolsConfig

from tai42_skeleton.operations import NotFoundError, OperationRegistry, operation_metadata_of
from tai42_skeleton.operations import connectors as conn_ops
from tai42_skeleton.operations.projection import project_operations


def _missing_loader():
    async def _load(cid):
        return None

    return _load


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


async def test_get_connection_returns_secret_free_view(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _load(cid):
        return _record(cid)

    monkeypatch.setattr(conn_ops, "load_record_or_none", _load)
    view = await conn_ops.get_connection(connection_id="c1")
    assert view["connection_id"] == "c1"
    assert "access_token" not in view
    assert "config_values" not in view


async def test_reconnect_returns_authorize_url(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _load(cid):
        return _record(cid)

    async def _reconnect(**kw):
        return SimpleNamespace(flow_id="f1", authorize_url="https://gh/auth")

    monkeypatch.setattr(conn_ops, "load_record_or_none", _load)
    monkeypatch.setattr(conn_ops.connection_service, "start_reconnect", _reconnect)
    result = await conn_ops.reconnect(
        connection_id="c1",
        enabled_sub_services=["repo"],
        return_url="/connectors",
        redirect_uri="https://app/oauth-bridge.html",
        origin="https://app",
    )
    assert result == {"flow_id": "f1", "authorize_url": "https://gh/auth"}


async def test_patch_sub_services_returns_result_view(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _load(cid):
        return _record(cid)

    async def _patch(**kw):
        return SimpleNamespace(
            connection_id="c1",
            enabled_sub_services=["repo", "issues"],
            consent_required=False,
            flow_id=None,
            authorize_url=None,
            added_manifest_entries=["issues"],
            removed_manifest_entries=[],
            fanout={"mode": "local-only"},
        )

    monkeypatch.setattr(conn_ops, "load_record_or_none", _load)
    monkeypatch.setattr(conn_ops.connection_service, "patch_sub_services", _patch)
    result = await conn_ops.patch_sub_services(
        connection_id="c1",
        enabled_sub_services=["repo", "issues"],
        return_url="/connectors",
        redirect_uri="https://app/oauth-bridge.html",
        origin="https://app",
    )
    assert result["connection_id"] == "c1"
    assert result["enabled_sub_services"] == ["repo", "issues"]
    assert result["consent_required"] is False
    # The patch that mutated the manifest threads the service's fleet report through.
    assert result["fanout"] == {"mode": "local-only"}


async def test_reconnect_unknown_connection_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(conn_ops, "load_record_or_none", _missing_loader())
    with pytest.raises(NotFoundError, match="connection not found"):
        await conn_ops.reconnect(
            connection_id="gone",
            enabled_sub_services=["repo"],
            return_url="/connectors",
            redirect_uri="https://app/oauth-bridge.html",
            origin="https://app",
        )


async def test_patch_sub_services_unknown_connection_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(conn_ops, "load_record_or_none", _missing_loader())
    with pytest.raises(NotFoundError, match="connection not found"):
        await conn_ops.patch_sub_services(
            connection_id="gone",
            enabled_sub_services=["repo"],
            return_url="/connectors",
            redirect_uri="https://app/oauth-bridge.html",
            origin="https://app",
        )


def test_connector_reads_project_and_mutations_carry_destructive_hint() -> None:
    # The reads (providers/connections/get_connection) are default-in; the mutations
    # are destructive (off the default surface, includable) and carry destructiveHint.
    reg = OperationRegistry()
    for op in (
        conn_ops.list_connector_providers,
        conn_ops.list_connections,
        conn_ops.get_connection,
        conn_ops.start_connect,
        conn_ops.disconnect,
        conn_ops.reconnect,
        conn_ops.patch_sub_services,
    ):
        reg.register(operation_metadata_of(op))

    class _Rec:
        def __init__(self) -> None:
            self.registered: dict[str, dict] = {}

        def tool(self, *, force, name, tags, annotations):
            self.registered[name] = {"annotations": annotations}
            return lambda fn: fn

    class _App:
        def __init__(self) -> None:
            self.tools = _Rec()

    app = _App()
    names = project_operations(app, ApiToolsConfig(expose_destructive=True), registry=reg)
    assert {"list_connector_providers", "list_connections", "get_connection"} <= set(names)
    assert app.tools.registered["list_connector_providers"]["annotations"] is None
    # start_connect / reconnect / patch_sub_services declare destructive=True.
    assert app.tools.registered["start_connect"]["annotations"].destructiveHint is True
    assert app.tools.registered["reconnect"]["annotations"].destructiveHint is True
    assert app.tools.registered["patch_sub_services"]["annotations"].destructiveHint is True
