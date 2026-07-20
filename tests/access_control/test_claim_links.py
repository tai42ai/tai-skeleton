"""One-time claim links: create → exchange round-trip, single-use burn, TTL expiry,
the ownership matrix, the invalid-key / ttl-ceiling 400s, the revoked-between 404,
the concurrent-burn race, and the no-secret-in-logs pin.

The verifier chain resolves through a fake ``ApiKeyIdentityProvider`` registered as
``redis`` (the default ``auth_providers`` name); the store rides the AC conftest's
``FakeRedis`` behind the module's ``client_ctx`` seam.
"""

from __future__ import annotations

import asyncio
import logging

import pytest
from tai_contract.access_control import OWNER_USER_ID_CLAIM, registry
from tai_contract.access_control.identity import ApiKeyIdentityProvider, AuthIdentity
from tai_kit.utils.data.string_util import hash_api_key

from tai_skeleton.access_control import claim_links
from tai_skeleton.access_control.claim_links import (
    ClaimLinkError,
    create_claim_link,
    exchange_claim_token,
)
from tai_skeleton.access_control.settings import access_control_settings

from .conftest import FakeRedis, make_client_ctx


class _FakeProvider(ApiKeyIdentityProvider):
    """A validator over an in-memory ``token -> AuthIdentity`` map. Subclasses
    ``ApiKeyIdentityProvider`` so the verifier PRESERVES the owner claim (it strips the
    claim only for non-mint providers). ``forget`` models a revoke during the TTL
    window."""

    def __init__(self, identities: dict[str, AuthIdentity]) -> None:
        self._identities = dict(identities)

    async def validate_token(self, token: str) -> AuthIdentity | None:
        return self._identities.get(token)

    def forget(self, token: str) -> None:
        self._identities.pop(token, None)

    async def provision(
        self, user_id: str, description: str, *, owner_user_id: str | None = None
    ) -> str:  # pragma: no cover - unused
        raise NotImplementedError

    async def revoke(self, user_id: str) -> bool:  # pragma: no cover - unused
        raise NotImplementedError

    async def update_description(self, user_id: str, description: str) -> bool:  # pragma: no cover - unused
        raise NotImplementedError

    async def list_identities(self) -> list[tuple[str, str]]:  # pragma: no cover - unused
        return []


def _identity(user_id: str, owner: str | None = None) -> AuthIdentity:
    claims = {OWNER_USER_ID_CLAIM: owner} if owner is not None else {}
    return AuthIdentity(user_id=user_id, claims=claims)


@pytest.fixture
def redis(monkeypatch: pytest.MonkeyPatch) -> FakeRedis:
    fake = FakeRedis(strings={})
    monkeypatch.setattr(claim_links, "client_ctx", make_client_ctx(fake))
    return fake


@pytest.fixture
def provider() -> _FakeProvider:
    prov = _FakeProvider(
        {
            "sk-alice": _identity("alice"),  # alice's own key
            "sk-dev": _identity("dev1", owner="alice"),  # a key alice minted (owned by alice)
            "sk-bob": _identity("bob"),  # someone else's key
        }
    )
    registry._REGISTRY["redis"] = lambda _settings: prov
    return prov


# -- happy path + single-use burn --------------------------------------------


async def test_create_then_exchange_round_trip(redis: FakeRedis, provider: _FakeProvider) -> None:
    result = await create_claim_link(
        api_key="sk-alice", caller_id="alice", caller_is_admin=False, caller_owner_claim=None, ttl_seconds=None
    )
    token = result["token"]
    assert token.startswith("clm-")
    assert result["claim_path"] == f"/login#claim={token}"
    assert "expires_at" in result
    # The record is stored by HASH, never by the raw token.
    assert f"{access_control_settings().claim_prefix}{hash_api_key(token)}" in redis._strings

    exchanged = await exchange_claim_token(token)
    assert exchanged == {"token": "sk-alice", "user_id": "alice"}


async def test_second_exchange_is_404(redis: FakeRedis, provider: _FakeProvider) -> None:
    result = await create_claim_link(
        api_key="sk-alice", caller_id="alice", caller_is_admin=True, caller_owner_claim=None, ttl_seconds=None
    )
    await exchange_claim_token(result["token"])
    with pytest.raises(ClaimLinkError) as exc:
        await exchange_claim_token(result["token"])
    assert exc.value.status == 404
    assert exc.value.message == "unknown or already used claim token"


async def test_unknown_token_is_404(redis: FakeRedis, provider: _FakeProvider) -> None:
    with pytest.raises(ClaimLinkError) as exc:
        await exchange_claim_token("clm-never-existed")
    assert exc.value.status == 404


async def test_ttl_expiry_is_404(redis: FakeRedis, provider: _FakeProvider) -> None:
    result = await create_claim_link(
        api_key="sk-alice", caller_id="alice", caller_is_admin=True, caller_owner_claim=None, ttl_seconds=120
    )
    redis.advance(121)
    with pytest.raises(ClaimLinkError) as exc:
        await exchange_claim_token(result["token"])
    assert exc.value.status == 404


# -- ownership matrix --------------------------------------------------------


async def test_admin_may_claim_link_any_key(redis: FakeRedis, provider: _FakeProvider) -> None:
    result = await create_claim_link(
        api_key="sk-bob", caller_id="admin", caller_is_admin=True, caller_owner_claim=None, ttl_seconds=None
    )
    assert (await exchange_claim_token(result["token"]))["user_id"] == "bob"


async def test_non_admin_may_claim_link_own_key(redis: FakeRedis, provider: _FakeProvider) -> None:
    # Case (c): the resolved identity IS the caller.
    result = await create_claim_link(
        api_key="sk-alice", caller_id="alice", caller_is_admin=False, caller_owner_claim=None, ttl_seconds=None
    )
    assert (await exchange_claim_token(result["token"]))["user_id"] == "alice"


async def test_non_admin_may_claim_link_a_key_it_minted(redis: FakeRedis, provider: _FakeProvider) -> None:
    # Case (b): the resolved key's owner claim is the caller.
    result = await create_claim_link(
        api_key="sk-dev", caller_id="alice", caller_is_admin=False, caller_owner_claim=None, ttl_seconds=None
    )
    assert (await exchange_claim_token(result["token"]))["user_id"] == "dev1"


async def test_non_admin_may_not_claim_link_someone_elses_key(redis: FakeRedis, provider: _FakeProvider) -> None:
    with pytest.raises(ClaimLinkError) as exc:
        await create_claim_link(
            api_key="sk-bob", caller_id="alice", caller_is_admin=False, caller_owner_claim=None, ttl_seconds=None
        )
    assert exc.value.status == 403


async def test_owned_caller_may_claim_link_only_its_own_key(redis: FakeRedis, provider: _FakeProvider) -> None:
    # An owned caller (its own credential carries an owner claim) may do case (c) only.
    result = await create_claim_link(
        api_key="sk-alice", caller_id="alice", caller_is_admin=False, caller_owner_claim="root", ttl_seconds=None
    )
    assert (await exchange_claim_token(result["token"]))["user_id"] == "alice"

    # Even a key whose owner claim equals the caller (case (b)) is refused for an owned
    # caller — it can never have minted a key, so this can only be someone else's device.
    with pytest.raises(ClaimLinkError) as exc:
        await create_claim_link(
            api_key="sk-dev", caller_id="alice", caller_is_admin=False, caller_owner_claim="root", ttl_seconds=None
        )
    assert exc.value.status == 403


# -- invalid key + ttl ceiling -----------------------------------------------


async def test_invalid_key_is_400(redis: FakeRedis, provider: _FakeProvider) -> None:
    with pytest.raises(ClaimLinkError) as exc:
        await create_claim_link(
            api_key="sk-garbage", caller_id="alice", caller_is_admin=True, caller_owner_claim=None, ttl_seconds=None
        )
    assert exc.value.status == 400
    assert exc.value.message == "not a valid API key"


async def test_ttl_above_ceiling_is_400(redis: FakeRedis, provider: _FakeProvider) -> None:
    ceiling = access_control_settings().claim_link_max_ttl_seconds
    with pytest.raises(ClaimLinkError) as exc:
        await create_claim_link(
            api_key="sk-alice",
            caller_id="alice",
            caller_is_admin=True,
            caller_owner_claim=None,
            ttl_seconds=ceiling + 1,
        )
    assert exc.value.status == 400
    # The diagnostic names BOTH the requested ttl and the ceiling it exceeded.
    assert str(ceiling + 1) in exc.value.message
    assert str(ceiling) in exc.value.message


async def test_non_positive_ttl_is_400(redis: FakeRedis, provider: _FakeProvider) -> None:
    with pytest.raises(ClaimLinkError) as exc:
        await create_claim_link(
            api_key="sk-alice", caller_id="alice", caller_is_admin=True, caller_owner_claim=None, ttl_seconds=0
        )
    assert exc.value.status == 400


# -- revoked between create and exchange -------------------------------------


async def test_key_revoked_between_creation_and_exchange_is_404_and_burns(
    redis: FakeRedis, provider: _FakeProvider
) -> None:
    result = await create_claim_link(
        api_key="sk-alice", caller_id="alice", caller_is_admin=True, caller_owner_claim=None, ttl_seconds=None
    )
    provider.forget("sk-alice")  # the underlying key is revoked during the TTL window
    with pytest.raises(ClaimLinkError) as exc:
        await exchange_claim_token(result["token"])
    assert exc.value.status == 404
    # The record is burned regardless (GETDEL ran before re-validation).
    assert f"{access_control_settings().claim_prefix}{hash_api_key(result['token'])}" not in redis._strings


# -- concurrent burn: exactly one winner -------------------------------------


async def test_concurrent_exchange_has_exactly_one_winner(redis: FakeRedis, provider: _FakeProvider) -> None:
    result = await create_claim_link(
        api_key="sk-alice", caller_id="alice", caller_is_admin=True, caller_owner_claim=None, ttl_seconds=None
    )
    outcomes = await asyncio.gather(
        exchange_claim_token(result["token"]),
        exchange_claim_token(result["token"]),
        return_exceptions=True,
    )
    winners = [o for o in outcomes if isinstance(o, dict)]
    losers = [o for o in outcomes if isinstance(o, ClaimLinkError)]
    assert len(winners) == 1
    assert winners[0] == {"token": "sk-alice", "user_id": "alice"}
    assert len(losers) == 1
    assert losers[0].status == 404


# -- NX collision on mint: retry once, then fail loud ------------------------


async def test_mint_retries_once_on_nx_collision(
    redis: FakeRedis, provider: _FakeProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The first minted token collides on its NX set; the mint must retry with a fresh
    # token and succeed rather than fail.
    tokens = iter(["collide", "fresh"])
    monkeypatch.setattr(claim_links.secrets, "token_urlsafe", lambda _n: next(tokens))
    prefix = access_control_settings().claim_prefix
    redis._strings[f"{prefix}{hash_api_key('clm-collide')}"] = "occupied"

    result = await create_claim_link(
        api_key="sk-alice", caller_id="alice", caller_is_admin=True, caller_owner_claim=None, ttl_seconds=None
    )
    assert result["token"] == "clm-fresh"
    assert (await exchange_claim_token("clm-fresh"))["user_id"] == "alice"


async def test_mint_raises_loudly_when_both_tokens_collide(
    redis: FakeRedis, provider: _FakeProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Both attempts mint the same already-occupied token, so the NX set collides twice —
    # the mint refuses to loop and raises loudly rather than returning a bad token.
    monkeypatch.setattr(claim_links.secrets, "token_urlsafe", lambda _n: "dup")
    prefix = access_control_settings().claim_prefix
    redis._strings[f"{prefix}{hash_api_key('clm-dup')}"] = "occupied"

    with pytest.raises(RuntimeError, match="collided twice"):
        await create_claim_link(
            api_key="sk-alice", caller_id="alice", caller_is_admin=True, caller_owner_claim=None, ttl_seconds=None
        )


# -- no secret in logs -------------------------------------------------------


async def test_no_token_or_key_in_logs(redis: FakeRedis, provider: _FakeProvider, caplog) -> None:
    with caplog.at_level(logging.INFO, logger="tai_skeleton.access_control.claim_links"):
        result = await create_claim_link(
            api_key="sk-alice", caller_id="alice", caller_is_admin=True, caller_owner_claim=None, ttl_seconds=None
        )
        await exchange_claim_token(result["token"])
    # Positive control: the expected SAFE log line IS captured, so the negative
    # assertions below cannot pass vacuously if log capture ever breaks (logger
    # renamed, level raised, logging removed).
    assert "claim link created by alice" in caplog.text
    assert result["token"] not in caplog.text
    assert "sk-alice" not in caplog.text
