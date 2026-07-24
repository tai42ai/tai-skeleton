"""The management provisioning surface — cross-backend ORCHESTRATION.

Every policy op delegates to the Postgres store (the ``pg`` fake); the api-key
IDENTITY record is owned by a stub :class:`ApiKeyIdentityProvider`; and the live
context + version counter live on the ``FakeRedis``. These tests cover the
fail-closed mint/revoke order, the ``_UNSET`` PUT-edit split across store and
provider, the tokens-payload merge, and the version bump — not the store SQL
(that is ``test_store.py``).
"""

from __future__ import annotations

import pytest
from tai42_contract.access_control import KEY_FINGERPRINT_CLAIM, OWNER_USER_ID_CLAIM, registry
from tai42_contract.access_control.identity import ApiKeyIdentityProvider, AuthIdentity, IdentityProvider
from tai42_kit.settings import reset_all_settings

from tai42_skeleton.access_control import management
from tai42_skeleton.access_control.settings import access_control_settings

from .conftest import FakeAccessControlPg, FakeRedis, make_client_ctx

S = access_control_settings()


class _SpyProvider(ApiKeyIdentityProvider):
    """In-memory api-key identity provider: models the record store the real
    ``tai42-identity-redis`` plugin owns, so the orchestration can be driven without
    a plugin."""

    def __init__(self) -> None:
        self.identities: dict[str, str] = {}
        self.provision_calls: list[str] = []
        self.provision_owners: dict[str, str | None] = {}
        self.revoke_calls: list[str] = []
        self.description_calls: list[tuple[str, str]] = []

    async def validate_token(self, token: str):  # pragma: no cover - unused here
        return None

    async def provision(self, user_id: str, description: str, *, owner_user_id: str | None = None) -> str:
        self.provision_calls.append(user_id)
        self.provision_owners[user_id] = owner_user_id
        self.identities[user_id] = description
        return f"sk-{user_id}"

    async def revoke(self, user_id: str) -> bool:
        self.revoke_calls.append(user_id)
        return self.identities.pop(user_id, None) is not None

    async def update_description(self, user_id: str, description: str) -> bool:
        self.description_calls.append((user_id, description))
        if user_id not in self.identities:
            return False
        self.identities[user_id] = description
        return True

    async def list_identities(self) -> list[tuple[str, str]]:
        return list(self.identities.items())


class _ValidatorProvider(IdentityProvider):
    """A non-mintable provider (validates tokens only); never provisions keys."""

    async def validate_token(self, token: str) -> AuthIdentity | None:  # pragma: no cover - unused here
        return None


@pytest.fixture
def provider() -> _SpyProvider:
    """Register a spy provider as ``"redis"`` (the default ``auth_providers`` entry); the
    autouse registry-isolation fixture restores the real registration afterwards."""
    spy = _SpyProvider()
    registry._REGISTRY["redis"] = lambda _settings: spy
    return spy


@pytest.fixture
def redis(monkeypatch) -> FakeRedis:
    """The AC Redis backing the live-context delete + the version counter."""
    fake = FakeRedis(strings={})
    monkeypatch.setattr(management, "client_ctx", make_client_ctx(fake))
    return fake


def _ctx_key(user_id: str) -> str:
    return f"{S.context_prefix}{user_id}"


# -- mint --------------------------------------------------------------------


async def test_mint_provisions_key_writes_policy_and_writes_no_context(
    pg: FakeAccessControlPg, provider: _SpyProvider, redis: FakeRedis
) -> None:
    await management.add_url_to_scope("scope-a", "/a")
    raw_key, body, key_fingerprint = await management.add_user_api_key("u1", "desc", ["scope-a"])
    assert raw_key == "sk-u1"
    # 1. provider owns the identity record; 2. policy row in PG. Mint writes NO
    # context — the live-context hash is created by the first counter write, so an
    # absent hash is the correct empty starting state.
    assert provider.identities == {"u1": "desc"}
    assert pg.policy("u1")["scopes"] == ["scope-a"]
    assert body["scopes"] == ["scope-a"]
    # Every mint stamps a fresh per-mint fingerprint into policy_data, returned alongside.
    assert key_fingerprint
    assert body["policy_data"][KEY_FINGERPRINT_CLAIM] == key_fingerprint
    assert _ctx_key("u1") not in redis._hashes


async def test_mint_rejects_unknown_scope_before_provisioning(
    pg: FakeAccessControlPg, provider: _SpyProvider, redis: FakeRedis
) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        await management.add_user_api_key("u1", "desc", ["ghost"])
    # The pre-check fires BEFORE the provider mints anything — no half-provisioned key.
    assert provider.provision_calls == []
    assert pg.policy("u1") is None


async def test_mint_accepts_universal_wildcard_scope(
    pg: FakeAccessControlPg, provider: _SpyProvider, redis: FakeRedis
) -> None:
    # "*" names no routed scope, so the typo guard must NOT 400 a wildcard mint: the
    # policy stores ["*"] and the mint still stamps a fresh per-mint fingerprint.
    raw_key, body, key_fingerprint = await management.add_user_api_key("u1", "desc", ["*"])
    assert raw_key == "sk-u1"
    assert pg.policy("u1")["scopes"] == ["*"]
    assert body["scopes"] == ["*"]
    assert key_fingerprint
    assert body["policy_data"][KEY_FINGERPRINT_CLAIM] == key_fingerprint


async def test_mint_still_rejects_a_typo_scope_alongside_the_wildcard(
    pg: FakeAccessControlPg, provider: _SpyProvider, redis: FakeRedis
) -> None:
    # The wildcard skip does not disable the guard: a typo naming no route still raises.
    with pytest.raises(ValueError, match="does not exist"):
        await management.add_user_api_key("u1", "desc", ["*", "ghost"])
    assert provider.provision_calls == []
    assert pg.policy("u1") is None


async def test_mint_rejects_duplicate_user_before_provisioning(
    pg: FakeAccessControlPg, provider: _SpyProvider, redis: FakeRedis
) -> None:
    pg.add_policy("u1", scopes=[])
    with pytest.raises(ValueError, match="already in use"):
        await management.add_user_api_key("u1", "desc", [])
    assert provider.provision_calls == []


async def test_mint_step2_failure_raises_and_leaves_key_policy_less(
    pg: FakeAccessControlPg, provider: _SpyProvider, redis: FakeRedis
) -> None:
    # A policy-write failure AFTER the key is provisioned raises loudly, naming the
    # revoke-then-remint recovery. The key exists but is denied everything.
    pg.fault = ("INSERT INTO access_control_policies", RuntimeError("pg down"))
    with pytest.raises(RuntimeError, match="revoke_api_key"):
        await management.add_user_api_key("u1", "desc", [])
    assert provider.identities == {"u1": "desc"}  # key provisioned
    assert pg.policy("u1") is None  # but no policy


async def test_revoke_then_remint_recovery_succeeds(
    pg: FakeAccessControlPg, provider: _SpyProvider, redis: FakeRedis
) -> None:
    # Step-2 failure leaves the key policy-less; the documented recovery is
    # revoke-then-remint, which then succeeds with a fresh key.
    pg.fault = ("INSERT INTO access_control_policies", RuntimeError("pg down"))
    with pytest.raises(RuntimeError):
        await management.add_user_api_key("u1", "desc", [])
    pg.fault = None
    assert await management.revoke_api_key("u1") is True
    raw_key, _body, _fingerprint = await management.add_user_api_key("u1", "desc", [])
    assert raw_key == "sk-u1"
    assert pg.policy("u1") is not None


async def test_revoke_then_remint_mints_a_fresh_fingerprint(
    pg: FakeAccessControlPg, provider: _SpyProvider, redis: FakeRedis
) -> None:
    # A revoke deletes the policy row, fingerprint and all, so a remint of the same
    # user_id writes a brand-new one that no old binding can match.
    _raw, _body, first = await management.add_user_api_key("u1", "desc", [])
    assert await management.revoke_api_key("u1") is True
    _raw2, body2, second = await management.add_user_api_key("u1", "desc", [])
    assert first != second
    assert body2["policy_data"][KEY_FINGERPRINT_CLAIM] == second


async def test_plain_remint_of_same_user_raises_duplicate(
    pg: FakeAccessControlPg, provider: _SpyProvider, redis: FakeRedis
) -> None:
    await management.add_user_api_key("u1", "desc", [])
    # NOT idempotent: a plain retry hits the duplicate-user guard.
    with pytest.raises(ValueError, match="already in use"):
        await management.add_user_api_key("u1", "desc", [])


# -- first-mintable resolution + owner threading -----------------------------


async def test_first_mintable_provider_chosen(pg: FakeAccessControlPg, redis: FakeRedis, monkeypatch) -> None:
    # A validator-only provider first in the chain is skipped; the second, mintable
    # provider is chosen to provision.
    spy = _SpyProvider()
    registry._REGISTRY["validator"] = lambda _s: _ValidatorProvider()
    registry._REGISTRY["mintable"] = lambda _s: spy
    monkeypatch.setenv("ACCESS_CONTROL_AUTH_PROVIDERS", '["validator", "mintable"]')
    reset_all_settings()
    try:
        raw_key, _body, _fingerprint = await management.add_user_api_key("u1", "desc", [])
        assert raw_key == "sk-u1"
        assert spy.provision_calls == ["u1"]
    finally:
        reset_all_settings()


async def test_no_mintable_provider_raises_typeerror(pg: FakeAccessControlPg, monkeypatch) -> None:
    registry._REGISTRY["validator"] = lambda _s: _ValidatorProvider()
    monkeypatch.setenv("ACCESS_CONTROL_AUTH_PROVIDERS", '["validator"]')
    reset_all_settings()
    try:
        with pytest.raises(TypeError, match="no configured identity provider"):
            await management.add_user_api_key("u1", "desc", [])
    finally:
        reset_all_settings()


async def test_provider_capabilities_reports_mintability(monkeypatch) -> None:
    registry._REGISTRY["validator"] = lambda _s: _ValidatorProvider()
    registry._REGISTRY["mintable"] = lambda _s: _SpyProvider()
    monkeypatch.setenv("ACCESS_CONTROL_AUTH_PROVIDERS", '["validator", "mintable"]')
    reset_all_settings()
    try:
        assert management.provider_capabilities() == [("validator", False), ("mintable", True)]
    finally:
        reset_all_settings()


async def test_owner_threaded_to_provision_and_written_to_policy(
    pg: FakeAccessControlPg, provider: _SpyProvider, redis: FakeRedis
) -> None:
    _raw, body, _fingerprint = await management.add_user_api_key("k1", "desc", [], owner_user_id="owner-1")
    # Owner reaches the provider (identity-claim home) as a keyword arg ...
    assert provider.provision_owners["k1"] == "owner-1"
    # ... and is dual-homed into the committed policy_data (management/listing home).
    assert body["policy_data"][OWNER_USER_ID_CLAIM] == "owner-1"
    assert pg.policy("k1")["policy_data"][OWNER_USER_ID_CLAIM] == "owner-1"


async def test_ownerless_mint_writes_no_owner_claim(
    pg: FakeAccessControlPg, provider: _SpyProvider, redis: FakeRedis
) -> None:
    _raw, body, _fingerprint = await management.add_user_api_key("k1", "desc", [])
    assert provider.provision_owners["k1"] is None
    assert OWNER_USER_ID_CLAIM not in body["policy_data"]


# -- revoke ------------------------------------------------------------------


async def test_revoke_kills_policy_first_then_context_then_key(
    pg: FakeAccessControlPg, provider: _SpyProvider, redis: FakeRedis
) -> None:
    await management.add_user_api_key("u1", "desc", [])
    # An external metering writer accrued live counters into the context hash.
    redis._hashes[_ctx_key("u1")] = {"used": "9"}
    assert await management.revoke_api_key("u1") is True
    # Policy row gone (PG), key record gone (provider), context hash deleted (Redis).
    assert pg.policy("u1") is None
    assert provider.identities == {}
    assert _ctx_key("u1") not in redis._hashes


async def test_revoke_bumps_the_policy_version_with_the_policy_delete(
    pg: FakeAccessControlPg, provider: _SpyProvider, redis: FakeRedis, monkeypatch
) -> None:
    # The policy cache is keyed on (user_id, version), so a warm slot keeps serving the
    # revoked key until the version moves: the bump must ride with the policy-row delete.
    settings = management._settings()
    await management.add_user_api_key("u1", "desc", [])
    before = redis._strings.get(settings.policy_version_key)

    async def _boom(*_a, **_k):
        raise RuntimeError("redis down")

    monkeypatch.setattr(redis, "delete", _boom)
    with pytest.raises(RuntimeError, match="redis down"):
        await management.revoke_api_key("u1")

    assert pg.policy("u1") is None
    assert int(redis._strings[settings.policy_version_key]) > int(before or 0)


async def test_revoke_unknown_user_returns_false(
    pg: FakeAccessControlPg, provider: _SpyProvider, redis: FakeRedis
) -> None:
    assert await management.revoke_api_key("missing") is False


async def test_revoke_leaves_a_policy_row_that_was_never_a_key_untouched(
    pg: FakeAccessControlPg, provider: _SpyProvider, redis: FakeRedis
) -> None:
    # Role assignment provisions policy rows this surface does not own; the identity
    # record is the only existence signal of a MINTED key.
    pg.add_policy("account-user", scopes=["hooks"])
    assert await management.revoke_api_key("account-user") is False
    assert pg.policy("account-user") is not None


async def test_revoke_failed_policy_delete_raises_leaving_the_key_authority_less(
    pg: FakeAccessControlPg, provider: _SpyProvider, redis: FakeRedis
) -> None:
    await management.add_user_api_key("u1", "desc", [])
    pg.fault = ("DELETE FROM access_control_policies", RuntimeError("pg down"))
    with pytest.raises(RuntimeError, match="pg down"):
        await management.revoke_api_key("u1")
    # The authority a fire runs on dies FIRST, so a failure there leaves the key untouched
    # rather than an identity-less policy row no retry can reach.
    assert provider.identities == {"u1": "desc"}
    assert pg.policy("u1") is not None


async def test_revoke_retry_after_a_failed_policy_delete_completes(
    pg: FakeAccessControlPg, provider: _SpyProvider, redis: FakeRedis
) -> None:
    # The identity record (the existence signal) is deleted LAST, so a retry finishes the
    # revocation instead of answering 404 over a row that still carries fire authority.
    await management.add_user_api_key("u1", "desc", [])
    pg.fault = ("DELETE FROM access_control_policies", RuntimeError("pg down"))
    with pytest.raises(RuntimeError, match="pg down"):
        await management.revoke_api_key("u1")

    pg.fault = None
    assert await management.revoke_api_key("u1") is True
    assert pg.policy("u1") is None
    assert provider.identities == {}


async def test_revoke_failed_context_delete_raises(
    pg: FakeAccessControlPg, provider: _SpyProvider, monkeypatch
) -> None:
    plain = FakeRedis(strings={})
    monkeypatch.setattr(management, "client_ctx", make_client_ctx(plain))
    await management.add_user_api_key("u1", "desc", [])
    # Now make the context delete fail: an orphaned context hash would corrupt a
    # future remint, so the failure must surface loudly.
    broken = FakeRedis(hashes={_ctx_key("u1"): {"used": "1"}})

    async def _boom(*_a, **_k):
        raise RuntimeError("redis down")

    broken.delete = _boom  # type: ignore[method-assign]
    monkeypatch.setattr(management, "client_ctx", make_client_ctx(broken))
    with pytest.raises(RuntimeError, match="redis down"):
        await management.revoke_api_key("u1")
    # The identity record outlives the fault, so the retry reaches the context delete
    # instead of leaving a hash that would corrupt the next remint of the same id.
    assert provider.identities == {"u1": "desc"}
    monkeypatch.setattr(management, "client_ctx", make_client_ctx(plain))
    plain._hashes[_ctx_key("u1")] = {"used": "1"}
    assert await management.revoke_api_key("u1") is True
    assert _ctx_key("u1") not in plain._hashes
    assert provider.identities == {}


async def test_remint_after_revoke_starts_with_fresh_context(
    pg: FakeAccessControlPg, provider: _SpyProvider, redis: FakeRedis
) -> None:
    await management.add_user_api_key("u1", "desc", [])
    redis._hashes[_ctx_key("u1")] = {"used": "99"}  # simulate accrued counters
    await management.revoke_api_key("u1")
    assert _ctx_key("u1") not in redis._hashes
    await management.add_user_api_key("u1", "desc", [])
    # A remint of the reused id starts fresh: no seed, no hash — an absent hash is
    # the empty live view, never inheriting the dead key's counters.
    assert _ctx_key("u1") not in redis._hashes


# -- edit (PUT split across provider + store) --------------------------------


async def test_edit_splits_description_to_provider_and_policy_to_store(
    pg: FakeAccessControlPg, provider: _SpyProvider, redis: FakeRedis
) -> None:
    await management.add_url_to_scope("scope-b", "/b")
    await management.add_user_api_key("u1", "old-desc", [])
    updated = await management.edit_user_payload("u1", description="new-desc", scopes=["scope-b"])
    assert updated is not None
    assert updated["scopes"] == ["scope-b"]  # policy → store
    assert provider.identities["u1"] == "new-desc"  # description → provider
    assert provider.description_calls == [("u1", "new-desc")]


async def test_edit_description_only_leaves_policy_untouched(
    pg: FakeAccessControlPg, provider: _SpyProvider, redis: FakeRedis
) -> None:
    _raw, _body, fingerprint = await management.add_user_api_key("u1", "old", [], policy_data={"k": 1}, condition=".c")
    updated = await management.edit_user_payload("u1", description="new")
    assert updated is not None
    # A description-only edit preserves every stored policy field (returns them),
    # including the per-mint fingerprint the mint stamped into policy_data.
    assert updated == {
        "scopes": [],
        "policy_data": {"k": 1, KEY_FINGERPRINT_CLAIM: fingerprint},
        "condition": ".c",
        "condition_id": None,
        "condition_kwargs": None,
    }
    assert provider.identities["u1"] == "new"


async def test_edit_scopes_only_never_touches_provider(
    pg: FakeAccessControlPg, provider: _SpyProvider, redis: FakeRedis
) -> None:
    await management.add_url_to_scope("scope-b", "/b")
    await management.add_user_api_key("u1", "desc", [])
    await management.edit_user_payload("u1", scopes=["scope-b"])
    # description left _UNSET → the provider is never called.
    assert provider.description_calls == []


async def test_edit_explicit_null_clears_policy_fields(
    pg: FakeAccessControlPg, provider: _SpyProvider, redis: FakeRedis
) -> None:
    _raw, _body, fingerprint = await management.add_user_api_key("u1", "desc", [], policy_data={"k": 1}, condition=".c")
    updated = await management.edit_user_payload("u1", policy_data=None, condition=None)
    assert updated is not None
    # The explicit clear drops the caller's policy_data but preserves the server-owned,
    # immutable per-mint fingerprint (an edit does not remint the key).
    assert updated["policy_data"] == {KEY_FINGERPRINT_CLAIM: fingerprint}
    assert updated["condition"] is None


async def test_edit_unknown_user_returns_none_without_provider_call(
    pg: FakeAccessControlPg, provider: _SpyProvider, redis: FakeRedis
) -> None:
    assert await management.edit_user_payload("missing", description="x") is None
    assert provider.description_calls == []


async def test_edit_rejects_unknown_scope(pg: FakeAccessControlPg, provider: _SpyProvider, redis: FakeRedis) -> None:
    await management.add_user_api_key("u1", "desc", [])
    with pytest.raises(ValueError, match="does not exist"):
        await management.edit_user_payload("u1", scopes=["ghost"])


async def test_edit_missing_identity_while_policy_exists_raises(
    pg: FakeAccessControlPg, provider: _SpyProvider, redis: FakeRedis
) -> None:
    # A policy row with no matching identity record is a genuine inconsistency: a
    # supplied description that the provider cannot find raises loudly.
    pg.add_policy("u1", scopes=[])
    with pytest.raises(RuntimeError, match="identity record for user"):
        await management.edit_user_payload("u1", description="x")


# -- tokens payload (identity + policy merge) --------------------------------


async def test_tokens_payload_merges_identity_and_policy(
    pg: FakeAccessControlPg, provider: _SpyProvider, redis: FakeRedis
) -> None:
    _raw, _body, fingerprint = await management.add_user_api_key("u1", "desc", [])
    pg.policy("u1")["scopes"] = ["scope-a"]
    payload = await management.get_all_existing_tokens_payload()
    assert payload == [
        {
            "user_id": "u1",
            "description": "desc",
            "scopes": ["scope-a"],
            "policy_data": {KEY_FINGERPRINT_CLAIM: fingerprint},
            "condition": None,
            "condition_id": None,
            "condition_kwargs": None,
        }
    ]


async def test_tokens_payload_skips_reserved_and_falsy(
    pg: FakeAccessControlPg, provider: _SpyProvider, redis: FakeRedis
) -> None:
    provider.identities = {"__root__": "root", "": "blank", "u1": "desc"}
    payload = await management.get_all_existing_tokens_payload()
    assert [p["user_id"] for p in payload] == ["u1"]


async def test_tokens_payload_empty_on_validator_only_deployment(pg: FakeAccessControlPg, monkeypatch) -> None:
    # A validator-only chain has no mint-capable provider, so there are no api-keys to
    # enumerate: the payload is empty.
    registry._REGISTRY["validator"] = lambda _s: _ValidatorProvider()
    monkeypatch.setenv("ACCESS_CONTROL_AUTH_PROVIDERS", '["validator"]')
    reset_all_settings()
    try:
        assert await management.get_all_existing_tokens_payload() == []
    finally:
        reset_all_settings()


# -- version bump ------------------------------------------------------------


async def test_bump_policy_version_increments(
    pg: FakeAccessControlPg, provider: _SpyProvider, redis: FakeRedis
) -> None:
    assert await management.bump_policy_version() == 1
    assert await management.bump_policy_version() == 2


async def test_failed_version_bump_raises(pg: FakeAccessControlPg, provider: _SpyProvider, monkeypatch) -> None:
    fake = FakeRedis(strings={})

    async def _boom(*_a, **_k):
        raise RuntimeError("redis down")

    fake.incr = _boom  # type: ignore[method-assign]
    monkeypatch.setattr(management, "client_ctx", make_client_ctx(fake))
    with pytest.raises(RuntimeError, match="redis down"):
        await management.bump_policy_version()
