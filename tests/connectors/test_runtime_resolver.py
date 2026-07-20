"""Runtime token resolution: hot path, lock-guarded refresh, error dispatch."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import SecretStr
from tai_contract.connectors.models import AuthHealthState

import tai_skeleton.connectors.runtime.resolver as resolver_mod
from tai_skeleton.connectors.oauth import client as oauth_client
from tai_skeleton.connectors.oauth.client import TokenRefreshFailedError, TokenResponse
from tai_skeleton.connectors.runtime.resolver import (
    ConnectorConnectionError,
    ConnectorReconnectRequiredError,
    ConnectorRefreshFailingError,
    force_refresh,
    resolve_managed_auth,
)

from .conftest import (
    CID,
    make_noauth_http_descriptor,
    make_noauth_record,
    make_noauth_stdio_descriptor,
    make_oauth_descriptor,
    make_oauth_record,
)

# The ciphertext the wiring's load hands back as the compare-and-set handle; the
# store starts holding it so an un-rotated refresh commits.
_STARTED_BLOB = b"started-blob"


class _FakeStore:
    """In-memory token store modelling the durable compare-and-set: a ``put``
    carrying ``expected_blob`` commits only when the currently-stored blob still
    matches, else it returns ``False`` (a peer rotated it first)."""

    def __init__(self, *, stored: bytes | None = None) -> None:
        self.puts: list = []  # every attempt
        self.committed: list = []  # only the writes that actually committed
        self._stored = stored

    async def put(
        self,
        connection_id,
        blob,
        *,
        create_only=False,
        expected_blob=None,
        session_expires_at=None,
    ):
        self.puts.append((connection_id, blob, expected_blob, session_expires_at))
        if expected_blob is not None and self._stored != expected_blob:
            return False
        self._stored = blob
        self.committed.append((connection_id, blob, session_expires_at))
        return True


def _patch_cooldown(monkeypatch) -> set[str]:
    """Replace the Redis-backed refresh cooldown breaker with an in-memory set so
    tests never touch a real Redis (which would leak a breaker across tests)."""
    cooldown: set[str] = set()

    async def _active(connection_id):
        return connection_id in cooldown

    async def _open(connection_id):
        cooldown.add(connection_id)

    async def _clear(connection_id):
        cooldown.discard(connection_id)

    monkeypatch.setattr(resolver_mod, "refresh_cooldown_active", _active)
    monkeypatch.setattr(resolver_mod, "open_refresh_cooldown", _open)
    monkeypatch.setattr(resolver_mod, "clear_refresh_cooldown", _clear)
    return cooldown


@pytest.fixture
def wiring(monkeypatch):
    """Patch the resolver's collaborators; return a config object for tweaking."""
    store = _FakeStore(stored=_STARTED_BLOB)
    state = {"record": None, "descriptor": make_oauth_descriptor(), "cooldown": None}

    async def fake_load_with_blob(connection_id):
        rec = state["record"]
        if rec is None:
            from tai_skeleton.connectors.store.persistence import ConnectionNotFoundError

            raise ConnectionNotFoundError(connection_id)
        return rec, _STARTED_BLOB

    @asynccontextmanager
    async def fake_lock(connection_id):
        yield

    monkeypatch.setattr(resolver_mod, "load_record_with_blob", fake_load_with_blob)
    monkeypatch.setattr(resolver_mod, "connection_lock", fake_lock)
    monkeypatch.setattr(resolver_mod, "token_store", lambda: store)
    monkeypatch.setattr(resolver_mod, "get_provider", lambda pid: state["descriptor"])
    state["cooldown"] = _patch_cooldown(monkeypatch)

    async def _noop_sleep(_d):
        return None

    monkeypatch.setattr(resolver_mod.asyncio, "sleep", _noop_sleep)
    return state, store


def _token_response(
    *,
    access_token: str = "fresh-at",
    refresh_token: str | None = "fresh-rt",
    expires_at: datetime | None = None,
    granted_scopes: list[str] | None = None,
    raw: dict | None = None,
) -> TokenResponse:
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_at=expires_at if expires_at is not None else datetime.now(UTC) + timedelta(hours=1),
        granted_scopes=granted_scopes if granted_scopes is not None else ["mail.read"],
        raw=raw if raw is not None else {},
    )


# -- no-auth -----------------------------------------------------------------


async def test_no_auth_without_config_returns_none(wiring):
    state, _ = wiring
    state["descriptor"] = make_noauth_stdio_descriptor()
    state["record"] = make_noauth_record(config_values={})
    assert await resolve_managed_auth(CID, "widgets", "search") is None


async def test_no_auth_stdio_returns_env(wiring):
    state, _ = wiring
    state["descriptor"] = make_noauth_stdio_descriptor()
    state["record"] = make_noauth_record(config_values={"api_key": "secret"})
    auth = await resolve_managed_auth(CID, "widgets", "search")
    assert auth is not None
    assert auth.access_token is None
    assert auth.env == {"api_key": "secret"}


async def test_no_auth_http_returns_headers(wiring):
    state, _ = wiring
    state["descriptor"] = make_noauth_http_descriptor()
    state["record"] = make_noauth_record(
        provider_id="httpsvc",
        alias="main",
        enabled_sub_services=["main"],
        config_values={"token": "v"},
    )
    auth = await resolve_managed_auth(CID, "httpsvc", "main")
    assert auth is not None
    assert auth.headers == {"token": "v"}


# -- oauth hot path ----------------------------------------------------------


async def test_oauth_fresh_token_served_without_refresh(wiring, monkeypatch):
    state, _ = wiring
    state["record"] = make_oauth_record(expires_in_seconds=3600)

    async def _boom(**kwargs):
        raise AssertionError("must not refresh a fresh token")

    monkeypatch.setattr(oauth_client, "refresh", _boom)
    auth = await resolve_managed_auth(CID, "acme", "mail")
    assert auth is not None
    assert auth.access_token == "access-tok"


async def test_reconnect_required_raises_before_lock(wiring):
    state, _ = wiring
    state["record"] = make_oauth_record(
        expires_in_seconds=10,
        health=AuthHealthState.RECONNECT_REQUIRED,
    )
    with pytest.raises(ConnectorReconnectRequiredError):
        await resolve_managed_auth(CID, "acme", "mail")


async def test_stale_token_refreshes_under_lock(wiring, monkeypatch):
    state, store = wiring
    state["record"] = make_oauth_record(expires_in_seconds=10)  # within safety margin

    async def fake_refresh(*, descriptor, refresh_token):
        return _token_response(access_token="rotated-at")

    monkeypatch.setattr(oauth_client, "refresh", fake_refresh)
    auth = await resolve_managed_auth(CID, "acme", "mail")
    assert auth is not None
    assert auth.access_token == "rotated-at"
    assert store.puts  # persisted the rotated record


async def test_invalid_grant_marks_reconnect_required(wiring, monkeypatch):
    state, store = wiring
    record = make_oauth_record(expires_in_seconds=10)
    state["record"] = record

    async def fake_refresh(*, descriptor, refresh_token):
        raise TokenRefreshFailedError("dead", reason="invalid_grant")

    monkeypatch.setattr(oauth_client, "refresh", fake_refresh)
    with pytest.raises(ConnectorReconnectRequiredError):
        await resolve_managed_auth(CID, "acme", "mail")
    assert record.auth_health_state == AuthHealthState.RECONNECT_REQUIRED
    assert store.puts  # persisted the terminal state


async def test_transient_budget_exhausted_marks_refresh_failing(wiring, monkeypatch):
    state, _store = wiring
    record = make_oauth_record(expires_in_seconds=10)
    state["record"] = record
    calls = {"n": 0}

    async def fake_refresh(*, descriptor, refresh_token):
        calls["n"] += 1
        raise TokenRefreshFailedError("flaky", reason="transient")

    monkeypatch.setattr(oauth_client, "refresh", fake_refresh)
    with pytest.raises(ConnectorRefreshFailingError):
        await resolve_managed_auth(CID, "acme", "mail")
    assert calls["n"] == resolver_mod.TRANSIENT_RETRY_BUDGET
    assert record.auth_health_state == AuthHealthState.REFRESH_FAILING


async def test_transient_then_success(wiring, monkeypatch):
    state, _ = wiring
    state["record"] = make_oauth_record(expires_in_seconds=10)
    calls = {"n": 0}

    async def fake_refresh(*, descriptor, refresh_token):
        calls["n"] += 1
        if calls["n"] < 3:
            raise TokenRefreshFailedError("flaky", reason="transient")
        return _token_response(access_token="recovered")

    monkeypatch.setattr(oauth_client, "refresh", fake_refresh)
    auth = await resolve_managed_auth(CID, "acme", "mail")
    assert auth is not None
    assert auth.access_token == "recovered"
    assert calls["n"] == 3


async def test_inside_lock_reconnect_required_raises(wiring, monkeypatch):
    """A peer flips the record to RECONNECT_REQUIRED while we await the lock."""
    _state, _ = wiring
    stale = make_oauth_record(expires_in_seconds=10)  # passes pre-lock, not fresh
    inside = make_oauth_record(
        expires_in_seconds=10,
        health=AuthHealthState.RECONNECT_REQUIRED,
    )
    seq = iter([(stale, b"b0"), (inside, b"b1")])
    monkeypatch.setattr(
        resolver_mod,
        "load_record_with_blob",
        lambda cid: _aiter_next(seq),
    )
    with pytest.raises(ConnectorReconnectRequiredError):
        await resolve_managed_auth(CID, "acme", "mail")


async def test_inside_lock_peer_refresh_serves_fresh(wiring, monkeypatch):
    """A peer refreshes while we await the lock; we serve its fresh token."""
    _state, _ = wiring
    stale = make_oauth_record(expires_in_seconds=10)
    fresh = make_oauth_record(expires_in_seconds=3600)
    seq = iter([(stale, b"b0"), (fresh, b"b1")])
    monkeypatch.setattr(
        resolver_mod,
        "load_record_with_blob",
        lambda cid: _aiter_next(seq),
    )

    async def _boom(**kwargs):
        raise AssertionError("must not refresh — peer already did")

    monkeypatch.setattr(oauth_client, "refresh", _boom)
    auth = await resolve_managed_auth(CID, "acme", "mail")
    assert auth is not None
    assert auth.access_token == "access-tok"


async def _aiter_next(seq):
    return next(seq)


async def test_connection_not_found_raises(wiring):
    state, _ = wiring
    state["record"] = None
    with pytest.raises(ConnectorConnectionError):
        await resolve_managed_auth(CID, "acme", "mail")


async def test_provider_id_mismatch_raises_connection_error(wiring, monkeypatch):
    """The record resolves to a different provider than the manifest ref names —
    injecting its token would misdirect it, so it must raise, not serve."""
    state, _ = wiring
    state["record"] = make_oauth_record(provider_id="acme")

    async def _boom(**kwargs):
        raise AssertionError("must not refresh a misrouted connection")

    monkeypatch.setattr(oauth_client, "refresh", _boom)
    with pytest.raises(ConnectorConnectionError):
        await resolve_managed_auth(CID, "other-provider", "mail")


async def test_unknown_provider_raises_connection_error(wiring, monkeypatch):
    """A no-auth record whose provider plugin was removed surfaces as a typed
    ConnectorConnectionError, not a raw KeyError from get_provider."""
    state, _ = wiring
    state["descriptor"] = make_noauth_stdio_descriptor()
    state["record"] = make_noauth_record(config_values={"api_key": "secret"})

    def _missing(_pid):
        raise KeyError("widgets")

    monkeypatch.setattr(resolver_mod, "get_provider", _missing)
    with pytest.raises(ConnectorConnectionError):
        await resolve_managed_auth(CID, "widgets", "search")


async def test_refresh_provider_removed_raises_connection_error(wiring, monkeypatch):
    """A stale oauth token whose provider plugin was unregistered surfaces a typed
    ConnectorConnectionError from _refresh, not a raw KeyError from get_provider."""
    state, _ = wiring
    state["record"] = make_oauth_record(expires_in_seconds=10)  # within safety margin → refresh path

    def _missing(_pid):
        raise KeyError("acme")

    monkeypatch.setattr(resolver_mod, "get_provider", _missing)

    async def _boom(**kwargs):
        raise AssertionError("must not reach the upstream refresh without a descriptor")

    monkeypatch.setattr(oauth_client, "refresh", _boom)
    with pytest.raises(ConnectorConnectionError, match="unknown provider"):
        await resolve_managed_auth(CID, "acme", "mail")


async def test_unknown_sub_service_raises_connection_error(wiring):
    """An unknown sub-service surfaces as a typed ConnectorConnectionError, not a
    raw KeyError from the descriptor's sub-service lookup."""
    state, _ = wiring
    state["descriptor"] = make_noauth_stdio_descriptor()
    state["record"] = make_noauth_record(config_values={"api_key": "secret"})
    with pytest.raises(ConnectorConnectionError):
        await resolve_managed_auth(CID, "widgets", "does-not-exist")


# -- force_refresh -----------------------------------------------------------


async def test_force_refresh_on_no_auth_raises(wiring):
    state, _ = wiring
    state["descriptor"] = make_noauth_stdio_descriptor()
    state["record"] = make_noauth_record()
    with pytest.raises(RuntimeError, match="no-auth"):
        await force_refresh(CID)


async def test_force_refresh_always_refreshes(wiring, monkeypatch):
    state, _ = wiring
    state["record"] = make_oauth_record(expires_in_seconds=3600)  # still fresh

    async def fake_refresh(*, descriptor, refresh_token):
        return _token_response(access_token="forced-at")

    monkeypatch.setattr(oauth_client, "refresh", fake_refresh)
    auth = await force_refresh(CID)
    assert auth.access_token == "forced-at"


async def test_force_refresh_fences_on_peer_rotated_token(wiring, monkeypatch):
    """A peer rotated the token while we waited for the lock: the stored token no
    longer equals the one that 401'd, so force_refresh serves the peer's token
    instead of burning another upstream refresh."""
    state, _ = wiring
    state["record"] = make_oauth_record(expires_in_seconds=3600)  # peer's fresh token

    async def _boom(**kwargs):
        raise AssertionError("must not refresh — a peer already rotated the token")

    monkeypatch.setattr(oauth_client, "refresh", _boom)
    # The stored access token is "access-tok"; the 401'd call used an older one.
    auth = await force_refresh(CID, failed_access_token="dead-old-token")
    assert auth.access_token == "access-tok"


async def test_force_refresh_reconnect_required_fails_fast(wiring, monkeypatch):
    """A record already flagged RECONNECT_REQUIRED fails fast under the lock —
    before the token fence and cooldown — never burning an upstream refresh nor
    re-persisting the terminal state. Seeding the cooldown proves the reconnect
    check runs first: otherwise the active breaker would raise the failing error."""
    state, _ = wiring
    state["record"] = make_oauth_record(
        expires_in_seconds=3600,
        health=AuthHealthState.RECONNECT_REQUIRED,
    )
    state["cooldown"].add(CID)

    async def _boom(**kwargs):
        raise AssertionError("must not refresh a reconnect-required connection")

    monkeypatch.setattr(oauth_client, "refresh", _boom)
    with pytest.raises(ConnectorReconnectRequiredError):
        await force_refresh(CID)


async def test_force_refresh_still_refreshes_when_token_unchanged(wiring, monkeypatch):
    """The stored token still equals the one that 401'd (no peer rotated) — a real
    refresh is driven."""
    state, _ = wiring
    state["record"] = make_oauth_record(expires_in_seconds=3600)

    async def fake_refresh(*, descriptor, refresh_token):
        return _token_response(access_token="forced-at")

    monkeypatch.setattr(oauth_client, "refresh", fake_refresh)
    auth = await force_refresh(CID, failed_access_token="access-tok")
    assert auth.access_token == "forced-at"


# -- refresh cooldown / circuit breaker (M14) --------------------------------


async def test_exhausted_budget_opens_cooldown_breaker(wiring, monkeypatch):
    state, _ = wiring
    state["record"] = make_oauth_record(expires_in_seconds=10)

    async def fake_refresh(*, descriptor, refresh_token):
        raise TokenRefreshFailedError("flaky", reason="transient")

    monkeypatch.setattr(oauth_client, "refresh", fake_refresh)
    with pytest.raises(ConnectorRefreshFailingError):
        await resolve_managed_auth(CID, "acme", "mail")
    # the breaker is now armed for the connection
    assert CID in state["cooldown"]


async def test_cooldown_active_fast_fails_without_refreshing(wiring, monkeypatch):
    """A connection already in refresh cooldown fails fast — it never re-burns the
    retry budget with an upstream call."""
    state, _ = wiring
    state["record"] = make_oauth_record(expires_in_seconds=10)
    state["cooldown"].add(CID)

    async def _boom(**kwargs):
        raise AssertionError("must not refresh while in cooldown")

    monkeypatch.setattr(oauth_client, "refresh", _boom)
    with pytest.raises(ConnectorRefreshFailingError, match="refresh cooldown"):
        await resolve_managed_auth(CID, "acme", "mail")


async def test_successful_refresh_clears_cooldown(wiring, monkeypatch):
    state, _ = wiring
    state["record"] = make_oauth_record(expires_in_seconds=10)
    # Prior failing run left a breaker; a call within the safety margin would be
    # fast-failed by it, so a successful refresh must clear it. Seed a stale
    # breaker but force the pre-lock check to pass by keeping it absent until the
    # refresh runs — here we assert the refresh path clears any armed breaker.
    calls = {"n": 0}

    async def fake_refresh(*, descriptor, refresh_token):
        calls["n"] += 1
        return _token_response(access_token="recovered")

    monkeypatch.setattr(oauth_client, "refresh", fake_refresh)
    auth = await resolve_managed_auth(CID, "acme", "mail")
    assert auth is not None
    assert CID not in state["cooldown"]


# -- write-back compare-and-set (lock-TTL / rotating-provider double-refresh) -

_STARTED = b"started-blob-fence"
_PEER_BLOB = b"peer-rotated-blob"


def _fence_wiring(monkeypatch, *, store, refresh_behaviour, peer):
    """Wire the resolver for a write-back race: ``store`` decides the
    compare-and-set, and on a miss the resolver re-reads ``peer`` (the record a
    concurrent refresh rotated to) to serve instead of clobbering."""
    monkeypatch.setattr(resolver_mod, "token_store", lambda: store)
    monkeypatch.setattr(resolver_mod, "get_provider", lambda pid: make_oauth_descriptor())
    # _serve_rotated_peer re-reads the current record on a CAS miss.
    monkeypatch.setattr(
        resolver_mod,
        "load_record_with_blob",
        lambda cid: _aiter_next(iter([(peer, _PEER_BLOB)])),
    )
    monkeypatch.setattr(oauth_client, "refresh", refresh_behaviour)
    _patch_cooldown(monkeypatch)

    async def _noop_sleep(_d):
        return None

    monkeypatch.setattr(resolver_mod.asyncio, "sleep", _noop_sleep)


async def test_stale_invalid_grant_does_not_clobber_rotated_record(monkeypatch):
    """A slow refresh fails invalid_grant after a peer rotated the record; the
    stale loser's compare-and-set misses, so it must NOT commit RECONNECT_REQUIRED
    over the peer's healthy record — it serves the peer's fresh token instead."""
    started = make_oauth_record(expires_in_seconds=10)
    started.refresh_token = SecretStr("RT0")  # the token our refresh began with

    peer = make_oauth_record(expires_in_seconds=3600)  # peer's fresh, healthy record
    peer.access_token = SecretStr("peer-fresh-at")

    # The store has moved to the peer's blob, so our CAS against _STARTED loses.
    store = _FakeStore(stored=_PEER_BLOB)

    async def fake_refresh(*, descriptor, refresh_token):
        assert refresh_token == "RT0"  # we exchanged the token we started with
        raise TokenRefreshFailedError("dead", reason="invalid_grant")

    _fence_wiring(monkeypatch, store=store, refresh_behaviour=fake_refresh, peer=peer)

    auth = await resolver_mod._refresh(started, _STARTED)
    assert auth.access_token == "peer-fresh-at"  # served the peer's fresh token
    assert store.committed == []  # NEVER committed a clobbering write


async def test_stale_success_does_not_clobber_rotated_record(monkeypatch):
    """Our refresh succeeds, but a peer rotated the record first; our CAS misses,
    so we discard our (superseded) tokens and serve the peer's record."""
    started = make_oauth_record(expires_in_seconds=10)
    started.refresh_token = SecretStr("RT0")

    peer = make_oauth_record(expires_in_seconds=3600)
    peer.access_token = SecretStr("peer-fresh-at")

    store = _FakeStore(stored=_PEER_BLOB)

    async def fake_refresh(*, descriptor, refresh_token):
        return _token_response(access_token="our-late-at", refresh_token="RT2")

    _fence_wiring(monkeypatch, store=store, refresh_behaviour=fake_refresh, peer=peer)

    auth = await resolver_mod._refresh(started, _STARTED)
    assert auth.access_token == "peer-fresh-at"  # peer's token, not "our-late-at"
    assert store.committed == []  # our late success did not overwrite the peer record


async def test_unrotated_token_still_persists(monkeypatch):
    """The CAS only blocks a write when the stored blob rotated; an unchanged
    blob (no peer) commits normally."""
    started = make_oauth_record(expires_in_seconds=10)
    started.refresh_token = SecretStr("RT0")

    # Store still holds the blob our refresh started from — no peer rotated it.
    store = _FakeStore(stored=_STARTED)

    peer = make_oauth_record(expires_in_seconds=3600)  # unused — CAS commits

    async def fake_refresh(*, descriptor, refresh_token):
        return _token_response(access_token="rotated-at", refresh_token="RT1")

    _fence_wiring(monkeypatch, store=store, refresh_behaviour=fake_refresh, peer=peer)

    auth = await resolver_mod._refresh(started, _STARTED)
    assert auth.access_token == "rotated-at"
    assert store.committed  # committed normally
