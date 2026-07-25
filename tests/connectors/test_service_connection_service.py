"""Connection lifecycle: start/complete connect, reconnect, disconnect, patch."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest
from tai42_contract.connectors.providers import (
    McpServerDescriptor,
    ProviderDescriptor,
    SubServiceDescriptor,
)
from tai42_contract.connectors.service import (
    AliasInUseError,
    CompleteConnectResult,
    FlowOperation,
    NoAuthConnectResult,
    StartConnectResult,
)

import tai42_skeleton.connectors.service.connection_service as cs
from tai42_skeleton.connectors.oauth import client as oauth_client
from tai42_skeleton.connectors.oauth.client import TokenResponse
from tai42_skeleton.connectors.oauth.state import OAuthFlowState
from tai42_skeleton.connectors.oauth.state import decode as decode_state
from tai42_skeleton.connectors.service.connection_service import (
    ConcurrentConnectionUpdateError,
    ConnectionNotFoundError,
    _extract_account_identity,
    _scopes_for,
    _validate_config_values,
    _validate_return_url,
    complete_connect,
    disconnect,
    patch_sub_services,
    start_connect,
    start_reconnect,
)

from .conftest import (
    CID,
    make_noauth_http_descriptor,
    make_noauth_record,
    make_noauth_stdio_descriptor,
    make_oauth_descriptor,
    make_oauth_record,
)

REDIRECT = "https://app.example.com/oauth-bridge.html"
ORIGIN = "https://app.example.com"

# The mode-wrapped fan-out summary a real ApplyResult exposes on a single-worker
# deployment (see operations._broadcast.fleet_fanout). The fake pipeline returns it
# on every apply so a writer test can assert the connector result threads it through;
# the real per-origin/unreachable shapes are exercised in tests/config/test_service.py.
_FANOUT = {"mode": "local-only", "note": "no worker bus configured; only this worker reloaded"}


class _Applied:
    """The :class:`~tai42_skeleton.config.service.ApplyResult` stand-in the fake pipeline
    returns — only the ``fanout`` summary the connection service reads off it."""

    fanout = _FANOUT


class _FakeConfigService:
    """Stand-in for the manifest-mutation pipeline.

    ``apply_change`` runs the connection service's mutator against an in-memory
    PRESERVED manifest document — exactly as the real transaction hands it in —
    then records the managed titles that landed / left so a test can assert the
    pipeline both ran (``applies`` ⇒ validate + reload + broadcast) and applied the
    intended mutation. The real reload / fleet broadcast are exercised in
    ``tests/config/test_service.py``.
    """

    def __init__(self, events: dict[str, list]) -> None:
        self.doc: dict[str, Any] = {"mcp": []}
        self._events = events
        self.applies = 0

    def seed(self, *, descriptor, enabled_sub_services, alias, connection_id) -> None:
        """Pre-populate the document with a connection's managed entries so a later
        remove/toggle-off can be observed leaving."""
        from tai42_skeleton.connectors.service.manifest_writer import add_managed_entries

        add_managed_entries(
            self.doc,
            descriptor=descriptor,
            enabled_sub_services=enabled_sub_services,
            alias=alias,
            connection_id=connection_id,
        )

    async def apply_change(self, mutator):
        self.applies += 1
        before = {e["title"] for e in self.doc.get("mcp") or []}
        mutator(self.doc)
        after_titles = [e["title"] for e in self.doc.get("mcp") or []]
        after = set(after_titles)
        self._events["added"].extend(title for title in after_titles if title not in before)
        self._events["removed"].extend(sorted(before - after))
        # The connection service reads its added/removed titles from the mutator's
        # captured lists; it reads only ``fanout`` off this ApplyResult stand-in.
        return _Applied()


def _blob_for(cid: str) -> bytes:
    return f"blob:{cid}".encode()


class _FakeStore:
    """Record-keyed fake with real compare-and-set + durable alias-uniqueness
    semantics on ``put``.

    ``blobs`` tracks the currently-stored ciphertext per connection; a put with
    ``expected_blob`` commits only when it matches. ``aliases`` models the
    durable ``UNIQUE (provider_id, alias)`` constraint: a create-only insert
    colliding on it raises :class:`AliasInUseError`, the store's authority."""

    def __init__(self, records) -> None:
        self.records = records
        self.blobs: dict[str, bytes] = {}
        self.aliases: dict[tuple[str | None, str | None], str] = {}
        self.puts: list = []
        self.deleted: list = []
        # Connection ids whose session has lapsed: a default (serving) load reads
        # them as missing, only an include_expired cleanup load sees them.
        self.expired: set[str] = set()

    async def put(
        self,
        connection_id,
        blob,
        *,
        create_only=False,
        expected_blob=None,
        session_expires_at=None,
        provider_id=None,
        alias=None,
    ):
        if create_only:
            owner = self.aliases.get((provider_id, alias))
            if owner is not None and owner != connection_id:
                raise AliasInUseError(f"alias {alias!r} is already in use for provider {provider_id!r}")
            self.aliases[(provider_id, alias)] = connection_id
        if expected_blob is not None and self.blobs.get(connection_id, _blob_for(connection_id)) != expected_blob:
            return False
        self.puts.append((connection_id, create_only))
        self.blobs[connection_id] = blob
        return True

    async def delete(self, connection_id):
        self.deleted.append(connection_id)
        self.blobs.pop(connection_id, None)
        # The record dict backs the fake load path too, so a delete makes a
        # subsequent load raise ConnectionNotFoundError (mirrors the real store).
        self.records.pop(connection_id, None)
        for key, owner in list(self.aliases.items()):
            if owner == connection_id:
                del self.aliases[key]

    async def list(self):
        return list(self.records)


@pytest.fixture
def harness(monkeypatch, oauth_client_env):
    records: dict[str, object] = {}
    providers: dict[str, object] = {
        "acme": make_oauth_descriptor(),
        "widgets": make_noauth_stdio_descriptor(),
        "httpsvc": make_noauth_http_descriptor(),
    }
    store = _FakeStore(records)
    flows: dict[str, OAuthFlowState] = {}
    events = {"added": [], "removed": [], "revoked": [], "state_put": []}

    monkeypatch.setattr(cs, "token_store", lambda: store)
    monkeypatch.setattr(cs, "get_provider", lambda pid: providers[pid])

    async def fake_load(cid, *, include_expired=False):
        if cid not in records or (cid in store.expired and not include_expired):
            from tai42_skeleton.connectors.store.persistence import ConnectionNotFoundError

            raise ConnectionNotFoundError(cid)
        return records[cid]

    async def fake_load_with_blob(cid, *, include_expired=False):
        return await fake_load(cid, include_expired=include_expired), store.blobs.get(cid, _blob_for(cid))

    monkeypatch.setattr(cs, "load_record", fake_load)
    monkeypatch.setattr(cs, "load_record_with_blob", fake_load_with_blob)

    @asynccontextmanager
    async def fake_lock(cid):
        yield

    monkeypatch.setattr(cs, "connection_lock", fake_lock)

    # Every writer converges through ConfigService.apply_change; inject a fake
    # pipeline that runs the real manifest_writer mutator against an in-memory
    # document. from_app returns the SAME instance so the document persists across
    # a test's operations.
    manifest_service = _FakeConfigService(events)

    class _FakeConfigServiceFactory:
        @staticmethod
        def from_app() -> _FakeConfigService:
            return manifest_service

    monkeypatch.setattr(cs, "ConfigService", _FakeConfigServiceFactory)

    # OAuth state store: in-memory.
    async def fake_state_put(flow_state):
        flows[flow_state.flow_id] = flow_state
        events["state_put"].append(flow_state.flow_id)

    async def fake_state_get_delete(flow_id):
        return flows.pop(flow_id, None)

    monkeypatch.setattr(cs.state, "put", fake_state_put)
    monkeypatch.setattr(cs.state, "get_and_delete", fake_state_get_delete)

    return cs, records, store, flows, events, providers


def _token_response(
    *,
    access_token: str = "at",
    refresh_token: str | None = "rt",
    expires_at: datetime | None = None,
    granted_scopes: list[str] | None = None,
    raw: dict | None = None,
) -> TokenResponse:
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_at=expires_at if expires_at is not None else datetime.now(UTC) + timedelta(hours=1),
        granted_scopes=granted_scopes if granted_scopes is not None else ["mail.read", "mail.send"],
        raw=raw if raw is not None else {},
    )


# -- pure helpers ------------------------------------------------------------


def test_validate_return_url_accepts_path():
    assert _validate_return_url("/connectors?x=1") == "/connectors?x=1"


@pytest.mark.parametrize("bad", ["//evil.com", "http://x", "no-slash", "/\nbad"])
def test_validate_return_url_rejects_bad(bad):
    with pytest.raises(ValueError, match="same-origin path"):
        _validate_return_url(bad)


def test_validate_sub_services_empty_raises():
    from tai42_skeleton.connectors.service.connection_service import _validate_sub_services

    with pytest.raises(ValueError, match="non-empty"):
        _validate_sub_services(make_oauth_descriptor(), [])


def test_scopes_for_unions_and_sorts():
    desc = make_oauth_descriptor()
    assert _scopes_for(desc, ["mail", "cal"]) == ["cal.read", "mail.read", "mail.send"]


def test_validate_config_values_unknown_key():
    desc = make_noauth_stdio_descriptor()
    with pytest.raises(ValueError, match="unknown config values"):
        _validate_config_values(desc, {"nope": "x"})


def test_validate_config_values_missing_required():
    desc = make_noauth_stdio_descriptor()
    with pytest.raises(ValueError, match="missing required"):
        _validate_config_values(desc, {})


def test_extract_account_identity_none_without_id_token():
    assert _extract_account_identity({}) is None


def test_extract_account_identity_from_id_token():
    import base64
    import json

    claims = base64.urlsafe_b64encode(json.dumps({"email": "u@x.test"}).encode()).rstrip(b"=").decode()
    id_token = f"header.{claims}.sig"
    assert _extract_account_identity({"id_token": id_token}) == "u@x.test"


def test_extract_account_identity_wrong_part_count():
    assert _extract_account_identity({"id_token": "only.two"}) is None


def test_extract_account_identity_bad_payload():
    assert _extract_account_identity({"id_token": "h.!!!notb64json!!!.s"}) is None


# -- start_connect -----------------------------------------------------------


async def test_start_connect_unknown_provider(harness):
    with pytest.raises(ValueError, match="unknown provider"):
        await start_connect(
            provider_id="nope",
            alias="a",
            enabled_sub_services=["mail"],
            return_url="/x",
            redirect_uri=REDIRECT,
            origin=ORIGIN,
        )


async def test_start_connect_bad_alias(harness):
    with pytest.raises(ValueError, match="alias must be"):
        await start_connect(
            provider_id="acme",
            alias="Bad Alias!",
            enabled_sub_services=["mail"],
            return_url="/x",
            redirect_uri=REDIRECT,
            origin=ORIGIN,
        )


async def test_start_connect_unknown_sub_service(harness):
    with pytest.raises(ValueError, match="unknown sub-services"):
        await start_connect(
            provider_id="acme",
            alias="work",
            enabled_sub_services=["bogus"],
            return_url="/x",
            redirect_uri=REDIRECT,
            origin=ORIGIN,
        )


async def test_start_connect_oauth_returns_authorize_url(harness):
    result = await start_connect(
        provider_id="acme",
        alias="work",
        enabled_sub_services=["mail"],
        return_url="/connectors",
        redirect_uri=REDIRECT,
        origin=ORIGIN,
    )
    assert isinstance(result, StartConnectResult)
    assert "acme.test/authorize" in result.authorize_url
    _, _, _, _flows, events, _ = harness
    assert events["state_put"]  # flow persisted


async def test_start_connect_signs_origin_into_recoverable_state(harness):
    """The deployment origin is signed into the authorize-URL ``state`` so a
    callback routed through the OAuth bridge can decode it and bounce the code
    back here — decoding the emitted state must recover exactly that origin."""
    result = await start_connect(
        provider_id="acme",
        alias="work",
        enabled_sub_services=["mail"],
        return_url="/connectors",
        redirect_uri=REDIRECT,
        origin=ORIGIN,
    )
    assert isinstance(result, StartConnectResult)
    encoded_state = parse_qs(urlparse(result.authorize_url).query)["state"][0]
    decoded = decode_state(encoded_state)
    assert decoded.origin == ORIGIN
    assert decoded.flow_id == result.flow_id


async def test_start_connect_oauth_rejects_config_values(harness):
    with pytest.raises(ValueError, match="config_values are not accepted"):
        await start_connect(
            provider_id="acme",
            alias="work",
            enabled_sub_services=["mail"],
            config_values={"x": "y"},
            return_url="/x",
            redirect_uri=REDIRECT,
            origin=ORIGIN,
        )


async def test_start_connect_no_auth_alias_in_use(harness):
    """A no-auth connect creates immediately, so a duplicate (provider, alias)
    trips the store's durable uniqueness authority and surfaces AliasInUseError."""
    _, _, store, _, _, _ = harness
    store.aliases[("widgets", "main")] = "existing-cid"
    with pytest.raises(AliasInUseError):
        await start_connect(
            provider_id="widgets",
            alias="main",
            enabled_sub_services=["search"],
            config_values={"api_key": "k"},
            return_url="/x",
            redirect_uri=REDIRECT,
            origin=ORIGIN,
        )


async def test_complete_connect_alias_in_use(harness, monkeypatch):
    """Two concurrent oauth flows with the same alias both pass start_connect;
    the loser's create-only insert trips the durable uniqueness at completion."""
    _, _, store, flows, _, _ = harness
    store.aliases[("acme", "work")] = "existing-cid"
    flow = _flow()
    flows[flow.flow_id] = flow

    async def fake_exchange(**kwargs):
        return _token_response()

    monkeypatch.setattr(oauth_client, "exchange_code", fake_exchange)
    with pytest.raises(AliasInUseError):
        await complete_connect(flow_id=flow.flow_id, code="auth")


async def test_start_connect_no_auth_creates_immediately(harness):
    result = await start_connect(
        provider_id="widgets",
        alias="main",
        enabled_sub_services=["search"],
        config_values={"api_key": "k"},
        return_url="/x",
        redirect_uri=REDIRECT,
        origin=ORIGIN,
    )
    assert isinstance(result, NoAuthConnectResult)
    _, _, store, _, events, _ = harness
    assert store.puts
    assert store.puts[0][1] is True
    assert events["added"]
    # The manifest-add always crosses the pipeline, so the fleet report rides back.
    assert result.fanout == _FANOUT


async def test_start_connect_no_auth_ignores_off_list_origin(harness):
    """A no-auth connect has no redirect flow, so it is NOT gated on the redirect
    allow-list — an off-list Origin still creates the connection immediately."""
    result = await start_connect(
        provider_id="widgets",
        alias="main",
        enabled_sub_services=["search"],
        config_values={"api_key": "k"},
        return_url="/x",
        redirect_uri=REDIRECT,
        origin="https://evil.com",
    )
    assert isinstance(result, NoAuthConnectResult)
    _, _, store, _, events, _ = harness
    assert store.puts
    assert events["added"]


# -- start_reconnect ---------------------------------------------------------


async def test_start_reconnect_no_auth_rejected(harness):
    _, records, _, _, _, _ = harness
    records[CID] = make_noauth_record()
    with pytest.raises(ValueError, match="cannot be reconnected"):
        await start_reconnect(
            connection_id=CID,
            enabled_sub_services=["search"],
            return_url="/x",
            redirect_uri=REDIRECT,
            origin=ORIGIN,
        )


async def test_start_reconnect_oauth(harness):
    _, records, _, _, _, _ = harness
    records[CID] = make_oauth_record(alias="work")
    result = await start_reconnect(
        connection_id=CID,
        enabled_sub_services=["mail", "cal"],
        return_url="/connectors",
        redirect_uri=REDIRECT,
        origin=ORIGIN,
    )
    assert isinstance(result, StartConnectResult)


async def test_start_reconnect_provider_removed_raises_value_error(harness, monkeypatch):
    """A reconnect whose provider plugin was unregistered surfaces a typed
    ValueError the router maps to a 4xx, not a raw KeyError 500."""
    cs_mod, records, _, _, _, _ = harness
    records[CID] = make_oauth_record(connection_id=CID, alias="work")

    def _boom(pid):
        raise KeyError(pid)

    monkeypatch.setattr(cs_mod, "get_provider", _boom)
    with pytest.raises(ValueError, match="unknown provider"):
        await start_reconnect(
            connection_id=CID,
            enabled_sub_services=["mail"],
            return_url="/x",
            redirect_uri=REDIRECT,
            origin=ORIGIN,
        )


async def test_start_reconnect_rejects_off_list_origin(harness):
    """A spoofed Origin not on the redirect allow-list is rejected inside
    _start_flow before any flow state is persisted — reconnect is always an OAuth
    redirect, so nothing is written when the origin fails closed."""
    _, records, store, _flows, events, _ = harness
    records[CID] = make_oauth_record(connection_id=CID, alias="work")
    with pytest.raises(oauth_client.RedirectUriNotAllowedError):
        await start_reconnect(
            connection_id=CID,
            enabled_sub_services=["mail"],
            return_url="/x",
            redirect_uri=REDIRECT,
            origin="https://evil.com",
        )
    assert store.puts == []
    assert events["state_put"] == []


# -- complete_connect --------------------------------------------------------


def _flow(operation=FlowOperation.CONNECT, reconnect_cid=None, redirect_uri=REDIRECT):
    return OAuthFlowState(
        flow_id="ffffffff-ffff-4fff-8fff-ffffffffffff",
        provider_id="acme",
        alias="work",
        requested_scopes=["mail.read", "mail.send"],
        enabled_sub_services=["mail"],
        pkce_verifier="verifier",
        return_url="/connectors",
        redirect_uri=redirect_uri,
        operation=operation,
        reconnect_connection_id=reconnect_cid,
    )


async def test_complete_connect_unknown_flow(harness, monkeypatch):
    with pytest.raises(oauth_client.OAuthError, match="state mismatch"):
        await complete_connect(flow_id="no-such", code="c")


async def test_complete_connect_creates_record(harness, monkeypatch):
    _, _records, store, flows, events, _ = harness
    flow = _flow()
    flows[flow.flow_id] = flow

    async def fake_exchange(**kwargs):
        return _token_response()

    monkeypatch.setattr(oauth_client, "exchange_code", fake_exchange)
    result = await complete_connect(flow_id=flow.flow_id, code="auth")
    assert isinstance(result, CompleteConnectResult)
    assert result.operation == FlowOperation.CONNECT
    assert store.puts
    assert store.puts[0][1] is True
    assert events["added"]
    # The fresh-connect manifest-add always crosses the pipeline ⇒ fleet report.
    assert result.fanout == _FANOUT


async def test_require_refresh_token_true_for_connect_false_for_reconnect(harness, monkeypatch):
    """A fresh CONNECT demands the provider return a refresh_token; a RECONNECT /
    TOGGLE inherits the existing one, so the exchange must NOT require it."""
    _, records, _store, flows, _events, _ = harness
    seen: list[bool] = []

    async def fake_exchange(**kwargs):
        seen.append(kwargs["require_refresh_token"])
        return _token_response()

    monkeypatch.setattr(oauth_client, "exchange_code", fake_exchange)

    connect_flow = _flow(operation=FlowOperation.CONNECT)
    flows[connect_flow.flow_id] = connect_flow
    await complete_connect(flow_id=connect_flow.flow_id, code="auth")
    assert seen == [True]

    records[CID] = make_oauth_record(connection_id=CID, alias="work", enabled_sub_services=["mail"])
    reconnect_flow = _flow(operation=FlowOperation.RECONNECT, reconnect_cid=CID)
    flows[reconnect_flow.flow_id] = reconnect_flow
    await complete_connect(flow_id=reconnect_flow.flow_id, code="auth")
    assert seen == [True, False]


async def test_complete_connect_reconnect_replaces_tokens(harness, monkeypatch):
    cs_mod, records, _store, flows, events, providers = harness
    existing = make_oauth_record(
        connection_id=CID,
        alias="work",
        enabled_sub_services=["mail"],
        granted_scopes=["mail.read", "mail.send"],
    )
    records[CID] = existing
    # Seed the prior mail entry so the reconnect's toggle-off is observed leaving.
    cs_mod.ConfigService.from_app().seed(
        descriptor=providers["acme"], enabled_sub_services=["mail"], alias="work", connection_id=CID
    )
    flow = _flow(operation=FlowOperation.RECONNECT, reconnect_cid=CID)
    flow.enabled_sub_services = ["cal"]  # drop mail, add cal
    flows[flow.flow_id] = flow

    async def fake_exchange(**kwargs):
        return _token_response(access_token="new-at", granted_scopes=["cal.read"])

    monkeypatch.setattr(oauth_client, "exchange_code", fake_exchange)
    result = await complete_connect(flow_id=flow.flow_id, code="auth")
    assert result.connection_id == CID
    assert existing.access_token is not None
    assert existing.access_token.get_secret_value() == "new-at"
    assert existing.enabled_sub_services == ["cal"]
    # mail removed + cal added in ONE pipeline transaction (single mutator).
    assert result.removed_manifest_entries == ["acme_mail_work"]
    assert result.added_manifest_entries == ["acme_cal_work"]
    assert events["removed"] == ["acme_mail_work"]
    assert events["added"] == ["acme_cal_work"]
    # A single apply_change ⇒ one validate + reload + broadcast for remove+add.
    assert cs_mod.ConfigService.from_app().applies == 1
    # The remove+add mutated the manifest, so the fleet report rides back.
    assert result.fanout == _FANOUT


async def test_complete_reconnect_no_sub_service_delta_carries_no_fanout(harness, monkeypatch):
    """A reconnect that changes no sub-services rotates tokens only — no manifest
    mutation runs, so the result honestly carries no fleet report (``fanout`` is
    ``None``)."""
    cs_mod, records, _store, flows, _events, _providers = harness
    records[CID] = make_oauth_record(
        connection_id=CID,
        alias="work",
        enabled_sub_services=["mail"],
        granted_scopes=["mail.read", "mail.send"],
    )
    # The reconnect flow's enabled set equals the record's ⇒ no add/remove delta.
    flow = _flow(operation=FlowOperation.RECONNECT, reconnect_cid=CID)
    flows[flow.flow_id] = flow

    async def fake_exchange(**kwargs):
        return _token_response(access_token="new-at")

    monkeypatch.setattr(oauth_client, "exchange_code", fake_exchange)
    result = await complete_connect(flow_id=flow.flow_id, code="auth")

    assert result.added_manifest_entries == []
    assert result.removed_manifest_entries == []
    assert result.fanout is None
    assert cs_mod.ConfigService.from_app().applies == 0


async def test_reconnect_clears_refresh_cooldown(harness, monkeypatch):
    # An explicit reconnect restores fresh tokens + HEALTHY, so it must drop any
    # refresh-cooldown breaker a prior failing run armed.
    cs_mod, records, _store, flows, _events, _ = harness
    cleared: list[str] = []

    async def fake_clear(cid):
        cleared.append(cid)

    monkeypatch.setattr(cs_mod, "clear_refresh_cooldown", fake_clear)

    records[CID] = make_oauth_record(
        connection_id=CID,
        alias="work",
        enabled_sub_services=["mail"],
        granted_scopes=["mail.read", "mail.send"],
    )
    flow = _flow(operation=FlowOperation.RECONNECT, reconnect_cid=CID)
    flow.enabled_sub_services = ["mail"]
    flows[flow.flow_id] = flow

    async def fake_exchange(**kwargs):
        return _token_response(access_token="new-at")

    monkeypatch.setattr(oauth_client, "exchange_code", fake_exchange)
    await complete_connect(flow_id=flow.flow_id, code="auth")
    assert cleared == [CID]


async def test_complete_connect_revalidates_redirect_uri(harness, monkeypatch):
    """The stored redirect_uri is re-validated against the allow-list at token
    exchange, so a value that fell off the allow-list since authorize-start is
    rejected before any code is exchanged."""
    _, _, _, flows, _, _ = harness
    flow = _flow(redirect_uri="https://evil.example.net/oauth-bridge.html")
    flows[flow.flow_id] = flow

    async def fake_exchange(**kwargs):
        raise AssertionError("must not exchange against an off-list redirect_uri")

    monkeypatch.setattr(oauth_client, "exchange_code", fake_exchange)
    with pytest.raises(oauth_client.RedirectUriNotAllowedError):
        await complete_connect(flow_id=flow.flow_id, code="auth")


async def test_complete_connect_uses_stored_redirect_uri(harness, monkeypatch):
    """The exchange re-sends the flow-state redirect_uri byte-identically, never a
    value recomputed from the completion request."""
    _, _, _, flows, _, _ = harness
    flow = _flow(redirect_uri=REDIRECT)
    flows[flow.flow_id] = flow
    seen: dict[str, str] = {}

    async def fake_exchange(**kwargs):
        seen["redirect_uri"] = kwargs["redirect_uri"]
        return _token_response()

    monkeypatch.setattr(oauth_client, "exchange_code", fake_exchange)
    await complete_connect(flow_id=flow.flow_id, code="auth")
    assert seen["redirect_uri"] == REDIRECT


async def test_complete_connect_reconnect_without_cid_raises(harness, monkeypatch):
    _, _, _, flows, _, _ = harness
    flow = _flow(operation=FlowOperation.RECONNECT, reconnect_cid=None)
    flows[flow.flow_id] = flow

    async def fake_exchange(**kwargs):
        return _token_response()

    monkeypatch.setattr(oauth_client, "exchange_code", fake_exchange)
    with pytest.raises(oauth_client.OAuthError, match="requires reconnect_connection_id"):
        await complete_connect(flow_id=flow.flow_id, code="auth")


async def test_complete_connect_provider_removed_mid_flow(harness, monkeypatch):
    """A provider unregistered between authorize-start and completion surfaces as
    a typed OAuthError (the router maps it to a 4xx failed body, not a 500)."""
    cs_mod, _, _, flows, _, _ = harness
    flow = _flow()
    flows[flow.flow_id] = flow

    def _boom(pid):
        raise KeyError(pid)

    monkeypatch.setattr(cs_mod, "get_provider", _boom)
    with pytest.raises(oauth_client.OAuthError, match="no longer registered"):
        await complete_connect(flow_id=flow.flow_id, code="auth")


# -- Item C: post-exchange failure revoke policy -----------------------------


async def test_fresh_connect_failure_revokes_grant(harness, monkeypatch):
    """A fresh CONNECT whose create-only persist fails (alias collision) revokes
    the just-issued, unshared grant so the upstream consent is not orphaned."""
    _, _, store, flows, _, _ = harness
    store.aliases[("acme", "work")] = "existing-cid"  # collide the create-only insert
    flow = _flow()
    flows[flow.flow_id] = flow

    revoked: list[str] = []

    async def fake_exchange(**kwargs):
        return _token_response(refresh_token="fresh-rt")

    async def fake_revoke(*, descriptor, token):
        revoked.append(token)
        return oauth_client.RevokeOutcome(outcome="success", http_status=200)

    monkeypatch.setattr(oauth_client, "exchange_code", fake_exchange)
    monkeypatch.setattr(oauth_client, "revoke", fake_revoke)
    with pytest.raises(AliasInUseError):
        await complete_connect(flow_id=flow.flow_id, code="auth")
    assert revoked == ["fresh-rt"]


async def test_reconnect_cas_miss_does_not_revoke(harness, monkeypatch):
    """A reconnect completion that loses the CAS wrote nothing; the surviving
    connection keeps its (possibly shared, non-rotating) refresh token, so the
    reconnect path must NOT revoke — that would kill a live connection."""
    cs_mod, records, store, flows, _, _ = harness
    records[CID] = make_oauth_record(
        connection_id=CID,
        alias="work",
        enabled_sub_services=["mail"],
        granted_scopes=["mail.read", "mail.send"],
    )
    flow = _flow(operation=FlowOperation.RECONNECT, reconnect_cid=CID)
    flow.enabled_sub_services = ["mail"]
    flows[flow.flow_id] = flow

    orig_load = cs_mod.load_record_with_blob

    async def racing_load(cid):
        rec, blob = await orig_load(cid)
        store.blobs[cid] = b"rotated-by-peer"  # a peer writes between our load and persist
        return rec, blob

    monkeypatch.setattr(cs_mod, "load_record_with_blob", racing_load)

    revoked: list[str] = []

    async def fake_revoke(*, descriptor, token):
        revoked.append(token)
        return oauth_client.RevokeOutcome(outcome="success")

    async def fake_exchange(**kwargs):
        return _token_response(access_token="new-at")

    monkeypatch.setattr(oauth_client, "exchange_code", fake_exchange)
    monkeypatch.setattr(oauth_client, "revoke", fake_revoke)
    with pytest.raises(ConcurrentConnectionUpdateError):
        await complete_connect(flow_id=flow.flow_id, code="auth")
    assert revoked == []


# -- Item N: fresh-connect writes are locked ---------------------------------


@asynccontextmanager
async def _depth_lock(depth: dict[str, int], cid):
    depth["n"] += 1
    try:
        yield
    finally:
        depth["n"] -= 1


async def test_fresh_connect_persist_and_manifest_locked(harness, monkeypatch):
    """The fresh CONNECT persists + adds manifest entries INSIDE connection_lock,
    so a concurrent disconnect cannot strand the added entries."""
    cs_mod, _, store, flows, _, _ = harness
    flow = _flow()
    flows[flow.flow_id] = flow
    depth = {"n": 0}
    put_held: list[bool] = []
    add_held: list[bool] = []

    monkeypatch.setattr(cs_mod, "connection_lock", lambda cid: _depth_lock(depth, cid))

    orig_put = store.put

    async def spy_put(*a, **k):
        put_held.append(depth["n"] > 0)
        return await orig_put(*a, **k)

    store.put = spy_put
    real_add = cs_mod.manifest_writer.add_managed_entries

    def spy_add(document, **kwargs):
        add_held.append(depth["n"] > 0)
        return real_add(document, **kwargs)

    monkeypatch.setattr(cs_mod.manifest_writer, "add_managed_entries", spy_add)

    async def fake_exchange(**kwargs):
        return _token_response()

    monkeypatch.setattr(oauth_client, "exchange_code", fake_exchange)
    await complete_connect(flow_id=flow.flow_id, code="auth")
    assert put_held == [True]
    assert add_held == [True]


async def test_no_auth_connect_persist_and_manifest_locked(harness, monkeypatch):
    """The no-auth create path also persists + adds manifest entries under the
    connection lock."""
    cs_mod, _, store, _, _, _ = harness
    depth = {"n": 0}
    put_held: list[bool] = []
    add_held: list[bool] = []

    monkeypatch.setattr(cs_mod, "connection_lock", lambda cid: _depth_lock(depth, cid))

    orig_put = store.put

    async def spy_put(*a, **k):
        put_held.append(depth["n"] > 0)
        return await orig_put(*a, **k)

    store.put = spy_put
    real_add = cs_mod.manifest_writer.add_managed_entries

    def spy_add(document, **kwargs):
        add_held.append(depth["n"] > 0)
        return real_add(document, **kwargs)

    monkeypatch.setattr(cs_mod.manifest_writer, "add_managed_entries", spy_add)
    await start_connect(
        provider_id="widgets",
        alias="main",
        enabled_sub_services=["search"],
        config_values={"api_key": "k"},
        return_url="/x",
        redirect_uri=REDIRECT,
        origin=ORIGIN,
    )
    assert put_held == [True]
    assert add_held == [True]


# -- Item A: origin allow-list at flow start ---------------------------------


async def test_start_connect_rejects_off_list_origin(harness):
    """A spoofed Origin not on the redirect allow-list is rejected before any
    flow state is persisted (fail-closed)."""
    _, _, _, _flows, events, _ = harness
    with pytest.raises(oauth_client.RedirectUriNotAllowedError):
        await start_connect(
            provider_id="acme",
            alias="work",
            enabled_sub_services=["mail"],
            return_url="/x",
            redirect_uri=REDIRECT,
            origin="https://evil.com",
        )
    assert events["state_put"] == []


# -- disconnect --------------------------------------------------------------


async def test_disconnect_no_auth_skips_revoke(harness):
    cs_mod, records, store, _, events, providers = harness
    records[CID] = make_noauth_record()
    # Seed the connection's managed entry so disconnect is observed removing it.
    cs_mod.ConfigService.from_app().seed(
        descriptor=providers["widgets"], enabled_sub_services=["search"], alias="main", connection_id=CID
    )
    result = await disconnect(connection_id=CID)
    assert result.upstream_revoke_outcome == "skipped"
    assert CID in store.deleted
    # The managed entry left through the pipeline (validate + reload + broadcast).
    assert events["removed"] == ["widgets_search_main"]
    assert result.removed_manifest_entries == ["widgets_search_main"]
    assert cs_mod.ConfigService.from_app().doc["mcp"] == []
    # A disconnect always removes managed entries through the pipeline ⇒ fleet report.
    assert result.fanout == _FANOUT


async def test_disconnect_oauth_revokes(harness, monkeypatch):
    _, records, store, _, _events, _ = harness
    records[CID] = make_oauth_record(connection_id=CID)

    async def fake_revoke(*, descriptor, token):
        return oauth_client.RevokeOutcome(outcome="success", http_status=200)

    monkeypatch.setattr(oauth_client, "revoke", fake_revoke)
    result = await disconnect(connection_id=CID)
    assert result.upstream_revoke_outcome == "success"
    assert CID in store.deleted
    # The oauth disconnect removes managed entries through the pipeline ⇒ fleet report.
    assert result.fanout == _FANOUT


async def test_disconnect_with_failed_revoke_still_purges_locally(harness, monkeypatch):
    """A best-effort upstream revoke that reports failure (not a raise) must not
    block the local purge — the blob and manifest entries are removed regardless,
    and the failed outcome is surfaced on the result."""
    cs_mod, records, store, _, events, providers = harness
    records[CID] = make_oauth_record(connection_id=CID)
    cs_mod.ConfigService.from_app().seed(
        descriptor=providers["acme"], enabled_sub_services=["mail"], alias="work", connection_id=CID
    )

    async def fake_revoke(*, descriptor, token):
        return oauth_client.RevokeOutcome(outcome="failed", http_status=500)

    monkeypatch.setattr(oauth_client, "revoke", fake_revoke)
    result = await disconnect(connection_id=CID)
    assert result.upstream_revoke_outcome == "failed"
    assert result.upstream_revoke_status == 500
    assert CID in store.deleted
    assert events["removed"] == ["acme_mail_work"]


async def test_disconnect_provider_gone_purges_locally(harness, monkeypatch):
    """Item D: with the provider plugin unregistered, get_provider raises KeyError;
    disconnect skips the upstream revoke and still purges the blob + manifest so a
    retry is not wedged at 500 forever."""
    cs_mod, records, store, _, events, providers = harness
    records[CID] = make_oauth_record(connection_id=CID)
    # Seed before the provider disappears — removal keys off the manifest's own
    # managed back-reference, not a live provider lookup.
    cs_mod.ConfigService.from_app().seed(
        descriptor=providers["acme"], enabled_sub_services=["mail"], alias="work", connection_id=CID
    )

    def _boom(pid):
        raise KeyError(pid)

    monkeypatch.setattr(cs_mod, "get_provider", _boom)

    async def fake_revoke(*, descriptor, token):
        raise AssertionError("must not revoke when the provider is gone")

    monkeypatch.setattr(oauth_client, "revoke", fake_revoke)
    result = await disconnect(connection_id=CID)
    assert result.upstream_revoke_outcome == "skipped"
    assert result.upstream_revoke_status is None
    assert CID in store.deleted
    assert events["removed"] == ["acme_mail_work"]


async def test_disconnect_purges_expired_connection(harness, monkeypatch):
    """An EXPIRED connection must stay cleanable: its session has lapsed (a
    serving load reads it as missing), but disconnect loads with include_expired
    so the blob + manifest entries are still revoked and purged instead of
    stranded forever."""
    cs_mod, records, store, _, events, providers = harness
    records[CID] = make_oauth_record(connection_id=CID)
    store.expired.add(CID)
    cs_mod.ConfigService.from_app().seed(
        descriptor=providers["acme"], enabled_sub_services=["mail"], alias="work", connection_id=CID
    )

    # Sanity: the default serving load no longer surfaces the record.
    from tai42_skeleton.connectors.store.persistence import ConnectionNotFoundError

    with pytest.raises(ConnectionNotFoundError):
        await cs_mod.load_record(CID)

    async def fake_revoke(*, descriptor, token):
        return oauth_client.RevokeOutcome(outcome="success", http_status=200)

    monkeypatch.setattr(oauth_client, "revoke", fake_revoke)
    result = await disconnect(connection_id=CID)
    assert result.upstream_revoke_outcome == "success"
    assert CID in store.deleted
    assert events["removed"] == ["acme_mail_work"]


async def test_disconnect_under_lock_blocks_patch_from_stranding(harness, monkeypatch):
    """M13: disconnect holds the connection lock across its manifest removal +
    record delete, so an in-flight patch (whose manifest add is now also under
    the lock) cannot slip its add in after the record is gone. The patch queued
    behind the lock reloads a deleted record and raises — stranding nothing."""
    cs_mod, records, store, _, events, _ = harness
    records[CID] = make_oauth_record(
        connection_id=CID,
        enabled_sub_services=["mail"],
        granted_scopes=["mail.read", "mail.send", "cal.read"],
    )

    real_lock = asyncio.Lock()
    gate = asyncio.Event()

    @asynccontextmanager
    async def serializing_lock(cid):
        async with real_lock:
            yield

    monkeypatch.setattr(cs_mod, "connection_lock", serializing_lock)

    async def fake_revoke(*, descriptor, token):
        await gate.wait()  # park disconnect INSIDE the lock until released
        return oauth_client.RevokeOutcome(outcome="success", http_status=200)

    monkeypatch.setattr(oauth_client, "revoke", fake_revoke)

    disc = asyncio.ensure_future(disconnect(connection_id=CID))
    await asyncio.sleep(0)  # let disconnect acquire the lock and park in revoke
    patch_task = asyncio.ensure_future(
        patch_sub_services(
            connection_id=CID, desired=["mail", "cal"], return_url="/x", redirect_uri=REDIRECT, origin=ORIGIN
        )
    )
    await asyncio.sleep(0)  # patch now blocks on the lock held by disconnect
    gate.set()  # release disconnect → it removes entries + deletes, frees the lock
    await disc

    # patch then acquires the lock, reloads the now-deleted record and raises —
    # never running its manifest add, so nothing is stranded.
    with pytest.raises(ConnectionNotFoundError):
        await patch_task
    assert CID in store.deleted
    assert events["added"] == []


async def test_start_connect_rejects_websocket_sub_service(harness):
    """A websocket sub-service probes healthy but every managed call raises, so a
    Connect for it is rejected at validation (fail-loud) before any OAuth flow."""
    from tai42_contract.connectors.providers import (
        McpServerDescriptor,
        ProviderDescriptor,
        SubServiceDescriptor,
    )

    _, _, _, _, _, providers = harness
    providers["wsprov"] = ProviderDescriptor(
        id="wsprov",
        display_name="WS",
        icon_url="https://ws.test/icon.png",
        kind="none",
        origin="system",
        category="data",
        sub_services={
            "live": SubServiceDescriptor(
                id="live",
                display_name="Live",
                mcp_server=McpServerDescriptor(type="websocket", url="wss://ws.test/mcp"),
            ),
        },
    )
    with pytest.raises(ValueError, match="transport 'websocket'"):
        await start_connect(
            provider_id="wsprov",
            alias="live",
            enabled_sub_services=["live"],
            config_values={},
            return_url="/x",
            redirect_uri=REDIRECT,
            origin=ORIGIN,
        )


# -- patch_sub_services ------------------------------------------------------


async def test_patch_unchanged_raises(harness):
    _, records, _, _, _, _ = harness
    records[CID] = make_oauth_record(connection_id=CID, enabled_sub_services=["mail"])
    with pytest.raises(ValueError, match="unchanged"):
        await patch_sub_services(
            connection_id=CID,
            desired=["mail"],
            return_url="/x",
            redirect_uri=REDIRECT,
            origin=ORIGIN,
        )


async def test_patch_inline_add_when_scopes_granted(harness):
    _, records, _, _, events, _ = harness
    # cal scope already granted → inline add, no consent.
    records[CID] = make_oauth_record(
        connection_id=CID,
        enabled_sub_services=["mail"],
        granted_scopes=["mail.read", "mail.send", "cal.read"],
    )
    result = await patch_sub_services(
        connection_id=CID,
        desired=["mail", "cal"],
        return_url="/x",
        redirect_uri=REDIRECT,
        origin=ORIGIN,
    )
    assert result.consent_required is False
    assert "cal" in result.enabled_sub_services
    assert events["added"]
    # The inline add mutated the manifest through the pipeline ⇒ fleet report.
    assert result.fanout == _FANOUT


async def test_patch_inline_add_ignores_off_list_origin(harness):
    """An inline-only PATCH (cal's scope already granted → no consent fork, no
    redirect flow) is NOT gated on the redirect allow-list — an off-list Origin
    still commits the inline change."""
    _, records, store, _, events, _ = harness
    records[CID] = make_oauth_record(
        connection_id=CID,
        enabled_sub_services=["mail"],
        granted_scopes=["mail.read", "mail.send", "cal.read"],
    )
    result = await patch_sub_services(
        connection_id=CID,
        desired=["mail", "cal"],
        return_url="/x",
        redirect_uri=REDIRECT,
        origin="https://evil.com",
    )
    assert result.consent_required is False
    assert "cal" in result.enabled_sub_services
    assert store.puts
    assert events["added"]


async def test_patch_removal_toggles_off(harness):
    cs_mod, records, _, _, events, providers = harness
    records[CID] = make_oauth_record(
        connection_id=CID,
        enabled_sub_services=["mail", "cal"],
        granted_scopes=["mail.read", "mail.send", "cal.read"],
    )
    cs_mod.ConfigService.from_app().seed(
        descriptor=providers["acme"], enabled_sub_services=["mail", "cal"], alias="work", connection_id=CID
    )
    result = await patch_sub_services(
        connection_id=CID,
        desired=["mail"],
        return_url="/x",
        redirect_uri=REDIRECT,
        origin=ORIGIN,
    )
    assert result.consent_required is False
    assert result.enabled_sub_services == ["mail"]
    assert events["removed"] == ["acme_cal_work"]
    assert result.removed_manifest_entries == ["acme_cal_work"]
    # The toggle-off mutated the manifest through the pipeline ⇒ fleet report.
    assert result.fanout == _FANOUT


async def test_patch_interleaved_writers_loser_raises_instead_of_clobbering(harness, monkeypatch):
    """Two writers race the same connection with the lock unavailable (the fake
    lock never serialises, mirroring the fail-open Redis-outage posture): the
    writer whose persist runs second loses the compare-and-set, raises
    ConcurrentConnectionUpdateError, and leaves the winner's record untouched."""
    cs_mod, records, store, _, events, _ = harness
    records[CID] = make_oauth_record(
        connection_id=CID,
        enabled_sub_services=["mail"],
        granted_scopes=["mail.read", "mail.send", "cal.read"],
    )

    loser_loaded = asyncio.Event()
    loser_may_persist = asyncio.Event()
    loads = {"count": 0}
    unpaused_load = cs_mod.load_record_with_blob

    async def gated_load(cid):
        result = await unpaused_load(cid)
        loads["count"] += 1
        if loads["count"] == 1:
            # First writer pauses between its load and its persist so the
            # second writer can commit in that window.
            loser_loaded.set()
            await loser_may_persist.wait()
        return result

    monkeypatch.setattr(cs_mod, "load_record_with_blob", gated_load)

    loser = asyncio.ensure_future(
        patch_sub_services(
            connection_id=CID,
            desired=["mail", "cal"],
            return_url="/x",
            redirect_uri=REDIRECT,
            origin=ORIGIN,
        )
    )
    await loser_loaded.wait()

    winner = await patch_sub_services(
        connection_id=CID,
        desired=["cal"],
        return_url="/x",
        redirect_uri=REDIRECT,
        origin=ORIGIN,
    )
    assert winner.enabled_sub_services == ["cal"]
    winner_blob = store.blobs[CID]
    events_after_winner = (list(events["added"]), list(events["removed"]))

    loser_may_persist.set()
    with pytest.raises(ConcurrentConnectionUpdateError, match="re-read the connection and retry"):
        await loser

    # The loser wrote nothing: the winner's blob is intact and no further
    # manifest reconciliation ran.
    assert store.blobs[CID] == winner_blob
    assert (events["added"], events["removed"]) == (events_after_winner[0], events_after_winner[1])


async def test_patch_off_list_origin_persists_nothing(harness):
    """A consent-requiring PATCH (cal's scope is not yet granted, so it forks an
    OAuth flow) with an off-list Origin fails closed BEFORE the inline persist,
    manifest write, or flow start — the store CAS and manifest events stay
    untouched, so no partial sub-service change leaks. The typed
    RedirectUriNotAllowedError (an OAuthError) surfaces for the router to map to 400."""
    _, records, store, _flows, events, _ = harness
    records[CID] = make_oauth_record(
        connection_id=CID,
        enabled_sub_services=["mail"],
        granted_scopes=["mail.read", "mail.send"],
    )
    with pytest.raises(oauth_client.RedirectUriNotAllowedError):
        await patch_sub_services(
            connection_id=CID,
            desired=["mail", "cal"],
            return_url="/x",
            redirect_uri=REDIRECT,
            origin="https://evil.com",
        )
    assert store.puts == []
    assert events["added"] == []
    assert events["removed"] == []
    assert events["state_put"] == []


async def test_patch_provider_removed_raises_value_error(harness, monkeypatch):
    """A PATCH whose provider plugin was unregistered surfaces a typed ValueError
    the router maps to a 4xx, not a raw KeyError 500."""
    cs_mod, records, _, _, _, _ = harness
    records[CID] = make_oauth_record(connection_id=CID, enabled_sub_services=["mail"])

    def _boom(pid):
        raise KeyError(pid)

    monkeypatch.setattr(cs_mod, "get_provider", _boom)
    with pytest.raises(ValueError, match="unknown provider"):
        await patch_sub_services(
            connection_id=CID,
            desired=["mail", "cal"],
            return_url="/x",
            redirect_uri=REDIRECT,
            origin=ORIGIN,
        )


async def test_patch_consent_fork_when_scopes_missing(harness):
    cs_mod, records, _, _flows, _events, _ = harness
    # cal scope NOT granted → consent flow.
    records[CID] = make_oauth_record(
        connection_id=CID,
        enabled_sub_services=["mail"],
        granted_scopes=["mail.read", "mail.send"],
    )
    result = await patch_sub_services(
        connection_id=CID,
        desired=["mail", "cal"],
        return_url="/connectors",
        redirect_uri=REDIRECT,
        origin=ORIGIN,
    )
    assert result.consent_required is True
    assert result.flow_id is not None
    assert result.authorize_url is not None
    # A consent-only toggle makes no manifest change here (the fork's own completion
    # reports its broadcast), so this response honestly carries no fleet report.
    assert result.fanout is None
    assert cs_mod.ConfigService.from_app().applies == 0


def _noauth_multi_descriptor() -> ProviderDescriptor:
    """A no-auth provider with two sub-services, one declaring scopes."""
    return ProviderDescriptor(
        id="noauthmulti",
        display_name="NoAuthMulti",
        icon_url="https://noauthmulti.test/icon.png",
        kind="none",
        origin="system",
        category="data",
        sub_services={
            "main": SubServiceDescriptor(
                id="main",
                display_name="Main",
                mcp_server=McpServerDescriptor(type="http", url="https://noauthmulti.test/mcp/main"),
            ),
            "extra": SubServiceDescriptor(
                id="extra",
                display_name="Extra",
                scopes=["extra.scope"],
                mcp_server=McpServerDescriptor(type="http", url="https://noauthmulti.test/mcp/extra"),
            ),
        },
    )


async def test_patch_no_auth_toggles_on_inline_ignoring_scopes(harness):
    """A no-auth connection has no OAuth consent flow, so toggling a sub-service ON
    is always inline — even when that sub-service declares scopes the record never
    granted (the consent-fork path is reserved for kind == "oauth")."""
    _, records, _, _flows, events, providers = harness
    providers["noauthmulti"] = _noauth_multi_descriptor()
    records[CID] = make_noauth_record(
        connection_id=CID,
        provider_id="noauthmulti",
        enabled_sub_services=["main"],
    )
    result = await patch_sub_services(
        connection_id=CID,
        desired=["main", "extra"],
        return_url="/x",
        redirect_uri=REDIRECT,
        origin=ORIGIN,
    )
    assert result.consent_required is False
    assert result.flow_id is None
    assert "extra" in result.enabled_sub_services
    assert events["added"]
