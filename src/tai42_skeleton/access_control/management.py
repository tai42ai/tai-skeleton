"""Provisioning surface over the access-control policy store.

The auth gate (``policy``/``verifier`` + the identity provider) is the read side:
it resolves an inbound key to an identity, loads that identity's policy, and maps a
request path to a scope id. This module is the write side — the CRUD an operator
UI drives to mint keys, edit policies, and wire routes to scopes.

Three backends are orchestrated here, each owning a distinct slice of state:

- The POLICY RULES (scopes, route/pattern mappings, per-user policy bodies) live in
  Postgres — the :class:`~tai42_skeleton.access_control.store.PostgresAccessControlStore`
  this module delegates every policy op to. It is the ONLY policy store.
- The api-key IDENTITY record (key hash → ``{user_id, description}`` plus its
  ``user_id`` → hash reverse lookup) is OWNED by the active identity provider (the
  ``tai42-identity-redis`` plugin for ``auth_providers=["redis"]``), reached ONLY through
  the ``ApiKeyIdentityProvider`` API (``provision``/``revoke``/``update_description``/
  ``list_identities``) resolved through the module-level registry the runtime auth
  adapter uses — this module never imports the plugin nor touches ``ac:key:*``.
- The ``ac:context:{user_id}`` per-user LIVE COUNTERS are a plain Redis HASH on the
  AC Redis, created by the first counter write (external metering writers use
  ``HSET``/``HINCRBY`` with JSON-encoded values) and deleted at revoke here; the
  plain-Redis ``ac:policy_version`` cache-buster is bumped here after every mutation.

Backend errors are never swallowed — they propagate so a failed provisioning op is
loud. The mint/revoke orchestration is fail-closed (identity record first on mint,
key first on revoke); a failure after the identity step raises loudly rather than
leaving a silent orphan.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from tai42_contract.access_control import OWNER_USER_ID_CLAIM
from tai42_contract.access_control.identity import ApiKeyIdentityProvider, IdentityProvider
from tai42_contract.access_control.registry import get_identity_provider_factory
from tai42_kit.clients import client_ctx
from tai42_kit.clients.impl.redis import RedisClient

from tai42_skeleton.access_control.settings import AccessControlSettings, access_control_settings
from tai42_skeleton.access_control.store import access_control_store
from tai42_skeleton.utils.redis_typing import awaited

# Identities that are infrastructure, not provisioned end-user keys, and so are
# omitted from the enumerated tokens payload.
_RESERVED_USER_IDS = frozenset({"__root__"})


class _Unset(Enum):
    """Sentinel marking an ``edit_user_payload`` argument the caller did not
    supply. A partial edit leaves such a field at its stored value; only fields
    given an explicit value (including ``None``/``{}``/``""``) are overwritten, so
    editing one field never silently clears another."""

    UNSET = "unset"


_UNSET = _Unset.UNSET


def _settings() -> AccessControlSettings:
    return access_control_settings()


def _resolve_provider(name: str) -> IdentityProvider:
    """Build the provider named ``name`` through the module-level registry — the SAME
    path the runtime auth adapter uses (never a direct import of the plugin). An
    unregistered name raises LOUDLY (``KeyError`` out of the registry)."""
    return get_identity_provider_factory(name)(_settings())


def _identity_provider() -> ApiKeyIdentityProvider:
    """Resolve the FIRST mint-capable identity provider in the configured chain.

    Walks ``auth_providers`` in order and returns the first that implements
    ``ApiKeyIdentityProvider`` — the provider owns the identity record; this surface
    orchestrates it for the record half and delegates the policy half to the PG store.

    An unregistered provider name raises LOUDLY (``KeyError`` out of the registry) — a
    management op must never run when access control resolves no identity provider.
    When NO configured provider is mint-capable (a validator-only deployment), raise
    ``TypeError`` naming the chain — the loud behavior the capabilities route surfaces
    ahead of a mint attempt rather than a raw 500 at mint time."""
    s = _settings()
    for name in s.auth_providers:
        provider = _resolve_provider(name)
        if isinstance(provider, ApiKeyIdentityProvider):
            return provider
    raise TypeError(
        f"no configured identity provider {s.auth_providers!r} implements ApiKeyIdentityProvider; "
        "the api-key provisioning surface requires a key-minting provider"
    )


def provider_capabilities() -> list[tuple[str, bool]]:
    """Each configured provider as ``(name, mintable)`` — ``mintable`` iff it
    implements ``ApiKeyIdentityProvider``. The capabilities route surfaces this so a
    validator-only deployment disables its mint UI instead of erroring at mint time."""
    return [(name, isinstance(_resolve_provider(name), ApiKeyIdentityProvider)) for name in _settings().auth_providers]


def _context_key(s: AccessControlSettings, user_id: str) -> str:
    return f"{s.context_prefix}{user_id}"


# -- scope / route reads (delegated to the PG store) -------------------------


async def get_all_existing_scopes() -> dict[str, str]:
    """Every non-public route mapping as ``{url: scope_id}``."""
    return await access_control_store().get_all_existing_scopes()


async def get_all_route_mappings() -> dict[str, str]:
    """Every route mapping as ``{url: value}``, INCLUDING public routes whose value
    is the public marker (which ``get_all_existing_scopes`` filters out). The full,
    faithful set a backup needs so an explicit public mapping round-trips instead of
    silently reverting to protected on restore."""
    return await access_control_store().get_all_route_mappings()


async def get_all_existing_patterns() -> dict[str, str]:
    """Every dynamic route's ``{url: pattern}`` mapping."""
    return await access_control_store().get_all_existing_patterns()


async def get_all_existing_tokens_payload() -> list[dict[str, Any]]:
    """Every provisioned key's identity merged with its policy — an ORCHESTRATION,
    not a single store read.

    The identities (``user_id``/``description``) come from the active provider's
    ``list_identities`` enumeration (it owns the identity records); each is merged
    with its policy read from the PG store. Infrastructure identities (falsy or
    reserved ``user_id``) are skipped; policy fields win on merge. Key material is
    never included — only the stored ``user_id``/``description`` plus policy fields.

    Ownership rides the policy merge: the mint path dual-homes the owner claim into
    ``policy_data`` under ``OWNER_USER_ID_CLAIM`` (its management/listing home), so an
    owned key's ``owner_user_id`` surfaces here for the route layer to filter on.

    Returns ``[]`` on a validator-only deployment (no mint-capable provider in the
    chain): with no key-minting provider there are no provisioned api-keys, so the
    empty enumeration is the accurate answer rather than an error.
    """
    # No mint-capable provider means no provisioned api-keys to enumerate — the accurate
    # empty answer, so this stays clear of ``_identity_provider()`` (which raises when no
    # provider mints). The genuine mint path (``add_user_api_key``) still raises loudly.
    if not any(mintable for _name, mintable in provider_capabilities()):
        return []
    provider = _identity_provider()
    store = access_control_store()
    identities = await provider.list_identities()
    payload: list[dict[str, Any]] = []
    for user_id, description in identities:
        if not user_id or user_id in _RESERVED_USER_IDS:
            continue
        merged: dict[str, Any] = {"user_id": user_id, "description": description}
        policy = await store.get_policy_body(user_id)
        if policy:
            merged.update(policy)
        payload.append(merged)
    return payload


# -- scope / route mutations (delegated to the PG store) ---------------------


async def add_url_to_scope(scope_id: str, url: str, pattern: str | None = None) -> None:
    """Map ``url`` to ``scope_id`` (optionally with a dynamic ``pattern``)."""
    await access_control_store().add_url_to_scope(scope_id, url, pattern)


async def remove_url_from_scope(url: str) -> tuple[bool, list[tuple[str, dict[str, Any]]]]:
    """Unmap ``url``, cascading its scope out of every token policy when the scope
    loses its last url. Returns ``(existed, [(user_id, committed_body), …])``."""
    return await access_control_store().remove_url_from_scope(url)


async def remove_scope(scope_id: str) -> tuple[int, list[tuple[str, dict[str, Any]]]]:
    """Delete a scope, stripping it from every token policy and deleting its routes.
    Returns ``(deleted_count, [(user_id, committed_body), …])``. Removing the public
    marker raises ``ValueError``."""
    return await access_control_store().remove_scope(scope_id)


async def get_public_route_pins() -> list[str]:
    """The sorted urls pinned to the public marker."""
    return await access_control_store().get_public_route_pins()


async def pin_route_public(url: str, pattern: str | None = None) -> None:
    """Pin ``url`` public (optionally with a dynamic ``pattern``), re-pointing it off
    any prior scope. The dedicated public-pin writer — the marker never routes through
    ``add_url_to_scope``."""
    await access_control_store().pin_route_public(url, pattern)


async def unpin_public_route(url: str) -> bool:
    """Unpin a public ``url``. Returns ``False`` when it was not pinned public."""
    return await access_control_store().unpin_public_route(url)


async def get_policy_body(user_id: str) -> dict[str, Any] | None:
    """The full policy record enforcement serves for ``user_id``, or ``None``."""
    return await access_control_store().get_policy_body(user_id)


async def restore_policy_body(user_id: str, body: dict[str, Any]) -> dict[str, Any] | None:
    """Write a prior policy ``body`` back as the enforced policy — the store side of
    a version rollback. Returns the restored body, or ``None`` if ``user_id`` is not
    provisioned (a falsy sentinel the route's 404 guard tests)."""
    return await access_control_store().restore_policy_body(user_id, body)


# -- key mint / revoke / edit (cross-backend orchestration) ------------------


async def add_user_api_key(
    user_id: str,
    description: str,
    scopes: list[str],
    policy_data: dict[str, Any] | None = None,
    condition: str | None = None,
    condition_id: str | None = None,
    condition_kwargs: dict[str, Any] | None = None,
    owner_user_id: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Provision a new key for ``user_id`` and return ``(raw_sk_key, committed_body)``
    — the raw ``sk-…`` (surfaced to the caller exactly once) and the exact policy
    body committed to Postgres (so the caller records it as durable version history
    without re-reading the store).

    When ``owner_user_id`` is given the key is an OWNED key: the owner claim is
    DUAL-HOMED at this single mint — the provider persists it on the identity record
    (the ENFORCEMENT source, arriving on every request) and it is also written into the
    committed ``policy_data`` under ``OWNER_USER_ID_CLAIM`` (the MANAGEMENT/listing
    source, readable from the store the tokens-payload merge already performs). Both
    homes are written only by this mint path; ``None`` mints an ownerless machine key.

    ORCHESTRATES the backends in a FAIL-CLOSED order. Raises ``ValueError`` if the
    user id is already provisioned or if any requested scope does not exist (has no
    url mapping) — both checked BEFORE any write, so a user error never leaves a
    half-provisioned key. Then:

    1. the provider's ``provision`` writes the identity/key record FIRST — the key
       authenticates but GRANTS NOTHING until the policy below exists;
    2. the policy row is written to the PG store.

    The per-user live-context hash ``ac:context:{user_id}`` needs NO seed — a Redis
    hash is created by its first counter write, and an absent hash reads as an empty
    live context via ``HGETALL``.

    A failure in step 2 RAISES loudly: the key exists but is denied everything, so
    the documented recovery is ``revoke_api_key(user_id)`` then a fresh mint (the
    mint is NOT idempotent — a plain retry hits the duplicate guard).
    """
    provider = _identity_provider()
    store = access_control_store()

    # Pre-checks with NO side effect, so a duplicate user or an unknown scope raises
    # before the provider mints anything (never a half-provisioned key).
    if await store.policy_exists(user_id):
        raise ValueError(f"user id {user_id!r} is already in use")
    if scopes:
        valid = set((await store.get_all_existing_scopes()).values())
        for scope in scopes:
            if scope not in valid:
                raise ValueError(f"scope {scope!r} does not exist or has no urls assigned")

    # 1. Identity record FIRST (fail-closed order) — the provider owns it. The owner
    #    claim rides the record so it surfaces in AuthIdentity.claims on every request.
    raw_key = await provider.provision(user_id, description, owner_user_id=owner_user_id)

    # Second owner home: the policy_data copy the management/listing surface reads.
    if owner_user_id is not None:
        policy_data = {**(policy_data or {}), OWNER_USER_ID_CLAIM: owner_user_id}

    try:
        # 2. Policy row. The live-context hash needs no seed — it is created by the
        #    first counter write and an absent hash reads as an empty context.
        body = await store.create_policy(user_id, scopes, policy_data, condition, condition_id, condition_kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"api key for user {user_id!r} was provisioned but its policy write failed; the key "
            f"authenticates but is denied everything — recover with revoke_api_key({user_id!r}) then re-mint"
        ) from exc
    return raw_key, body


async def edit_user_payload(
    user_id: str,
    description: str | _Unset = _UNSET,
    scopes: list[str] | _Unset = _UNSET,
    policy_data: dict[str, Any] | None | _Unset = _UNSET,
    condition: str | None | _Unset = _UNSET,
    condition_id: str | None | _Unset = _UNSET,
    condition_kwargs: dict[str, Any] | None | _Unset = _UNSET,
) -> dict[str, Any] | None:
    """Partially update an existing key's description and policy in place (never
    rotates the key). Only the arguments the caller actually supplies are written;
    an argument left at its ``_UNSET`` default preserves the stored value, so editing
    one field never clears another. A field supplied as ``None``/``{}``/``""`` is
    written verbatim (an explicit clear).

    The edit SPLITS by field across the backends: ``description`` — whose single home
    is the identity record — goes to the provider's ``update_description``; the policy
    fields go to the PG store. The ``_UNSET`` partial-edit semantics are resolved at
    this boundary, so neither the store nor the provider ever sees ``_UNSET``.

    Returns the committed policy body on success, or ``None`` if ``user_id`` is not
    provisioned (a falsy sentinel the route's 404 guard tests). Raises ``ValueError``
    if any supplied scope does not exist.
    """
    provider = _identity_provider()
    store = access_control_store()

    # Resolve _UNSET to the fields actually supplied; the store sees only those.
    updates: dict[str, Any] = {}
    if not isinstance(scopes, _Unset):
        updates["scopes"] = scopes
    if not isinstance(policy_data, _Unset):
        updates["policy_data"] = policy_data
    if not isinstance(condition, _Unset):
        updates["condition"] = condition
    if not isinstance(condition_id, _Unset):
        updates["condition_id"] = condition_id
    if not isinstance(condition_kwargs, _Unset):
        updates["condition_kwargs"] = condition_kwargs

    policy = await store.update_policy_fields(user_id, updates)
    # No policy row → not provisioned. The description edit below is never attempted
    # for a user with no policy (its single existence signal this surface can read).
    if policy is None:
        return None

    # Description edit → the provider (its single home). Only a supplied description
    # reaches the provider; ``update_description`` returning ``False`` for a user
    # whose policy we just wrote means the identity record is missing while the
    # policy exists — a genuine inconsistency, so raise loudly rather than silently.
    if not isinstance(description, _Unset) and not await provider.update_description(user_id, description):
        raise RuntimeError(
            f"identity record for user {user_id!r} is missing while its policy exists — cannot update the description"
        )
    return policy


async def revoke_api_key(user_id: str) -> bool:
    """Delete a provisioned key and all of its records. Returns ``False`` if
    ``user_id`` is not provisioned.

    ORCHESTRATES the backends in FAIL-CLOSED order: (1) the provider's ``revoke``
    deletes the identity/key record FIRST — authentication stops immediately,
    regardless of the policy rows; then (2) the PG policy row is deleted and (3) the
    live-context hash ``ac:context:{user_id}`` is deleted. A failure in step 2/3
    RAISES loudly (never a silent orphan): an orphaned policy denies nothing once the
    key is dead, but the failure must still surface, and an orphaned context hash
    would corrupt a future remint of the same ``user_id`` by leaving it the dead
    key's usage/quota counters."""
    s = _settings()
    provider = _identity_provider()
    store = access_control_store()

    # 1. Kill the KEY first (provider) — the next request with the deleted key fails
    #    to authenticate. An unknown user is a clean False the route maps to a 404.
    if not await provider.revoke(user_id):
        return False

    # 2. Policy row, then 3. the live-context hash. Both MUST clear — a failed delete
    #    raises loudly rather than leaving a silent orphan.
    await store.delete_policy(user_id)
    async with client_ctx(RedisClient, s.redis) as r:
        await awaited(r.delete(_context_key(s, user_id)))
    return True


async def bump_policy_version() -> int:
    """Increment the policy-version counter (plain Redis), forcing a cross-worker
    policy cache miss on the next read. Called after any scope/policy mutation. A
    failed bump RAISES loudly — it is never swallowed."""
    s = _settings()
    async with client_ctx(RedisClient, s.redis) as r:
        return await awaited(r.incr(s.policy_version_key))
