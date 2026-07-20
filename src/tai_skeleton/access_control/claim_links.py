"""One-time claim links — the QR-onboarding backend.

A claim link carries a raw API key from a device that holds it to a device that does
not (a phone scanning a QR, a second terminal), WITHOUT the key ever touching a log or
a query string. Two halves:

- :func:`create_claim_link` (authed) resolves a submitted key through the SAME verifier
  chain the gate uses, applies the ownership rule, and writes a single-use record at
  ``ac:claim:<sha256(token)>`` with ``SET ... EX <ttl> NX``. The response returns the
  claim TOKEN once and a fragment-carrier PATH (``/login#claim=<token>``) — the token
  rides the URL FRAGMENT (never a query param, never a header echo), so it never reaches
  a server access log.
- :func:`exchange_claim_token` (public) burns the record with an atomic ``GETDEL`` — the
  first caller wins, every other caller (used / unknown / expired) gets the SAME 404 —
  re-validates the stored key through the verifier, and hands the raw key back once.

Security posture:

- The record is the SINGLE at-rest home of a raw key in the whole system. The raw token
  is NEVER stored; the record is findable only by ``sha256(token)`` — the identity
  store's own posture. There is NO reverse index, NO listing, and NO revoke-claim route:
  a claim is killed by waiting out the TTL or by revoking the underlying key (the
  exchange re-validation then refuses to hand out a revoked credential).
- Nothing ever logs the raw token or the raw key. Creation logs the caller + the target
  ``user_id``; exchange logs a hash PREFIX + the outcome.
- ``exchange`` guarantees the handed-out key is not REVOKED (it re-resolves through
  ``validate_token``). It does NOT re-run the owner-disable / attenuation checks that
  live in ``backend.authenticate`` — a key whose OWNER was disabled during the TTL
  window still exchanges but is correctly POWERLESS at first use (the gate 403s it via
  owner-death attenuation on every request). The guarantee is not-REVOKED, not
  owner-alive.

Every error RAISES (fail closed) — there is no fallback that could hand out or bury a
credential silently.
"""

from __future__ import annotations

import json
import logging
import secrets
from datetime import UTC, datetime, timedelta

from tai_contract.access_control import OWNER_USER_ID_CLAIM
from tai_contract.access_control.identity import IdentityProvider
from tai_contract.access_control.registry import get_identity_provider_factory
from tai_kit.clients import client_ctx
from tai_kit.clients.impl.redis import RedisClient
from tai_kit.utils.data.string_util import hash_api_key

from tai_skeleton.access_control.settings import AccessControlSettings, access_control_settings
from tai_skeleton.access_control.verifier import AccessControlVerifier
from tai_skeleton.utils.redis_typing import awaited

logger = logging.getLogger(__name__)

_INVALID_KEY_MESSAGE = "not a valid API key"
_OWNERSHIP_MESSAGE = "may only create claim links for keys you own or your own key"
# The single, deliberately-indistinguishable exchange miss: used / unknown / expired all
# answer this so the public surface leaks no oracle distinguishing "never existed" from
# "already claimed" (the server-side log DOES distinguish what it can, by hash prefix).
_UNKNOWN_TOKEN_MESSAGE = "unknown or already used claim token"

# The number of hex chars of ``sha256(token)`` a log line names — enough to correlate a
# creation with its exchange in the log stream, far too few to be the credential.
_LOG_HASH_PREFIX = 12


class ClaimLinkError(Exception):
    """A typed claim-link failure carrying the HTTP status the adapters map it to.

    ``status`` is 400 (an unresolvable submitted key), 403 (a key that is not the
    caller's to share), or 404 (the uniform exchange miss). The adapters translate it
    to the matching operation error; the message is the operator-facing text."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def _verifier(settings: AccessControlSettings) -> AccessControlVerifier:
    """The verifier chain the gate itself resolves credentials through, built from the
    SAME module-level identity-provider registry (never a parallel provider
    construction): each configured provider is resolved through
    ``get_identity_provider_factory`` in order, first-match-wins. An unknown provider
    name raises loudly out of the factory on first use."""

    def _factories() -> list[IdentityProvider]:
        return [get_identity_provider_factory(name)(settings) for name in settings.auth_providers]

    return AccessControlVerifier(settings, provider_factories=_factories)


def _record_key(settings: AccessControlSettings, token: str) -> str:
    return f"{settings.claim_prefix}{hash_api_key(token)}"


async def create_claim_link(
    *,
    api_key: str,
    caller_id: str | None,
    caller_is_admin: bool,
    caller_owner_claim: str | None,
    ttl_seconds: int | None,
) -> dict:
    """Mint a one-time claim link for ``api_key`` and return its carrier.

    The submitted key is resolved through the gate's verifier chain: an unresolvable key
    is a 400 (never a dead QR). The resolved identity must satisfy the ownership rule
    — (a) the caller is admin, (b) the resolved key was minted by the caller (its owner
    claim is the caller), or (c) the resolved identity IS the caller (its own key) —
    else a 403. (An authenticated caller CAN thus distinguish a live key from garbage
    here; this adds no capability the ``/api/auth/me`` carve-out does not already grant a
    caller holding a candidate key. The uniform-404 no-oracle rule is the EXCHANGE
    surface's, not creation's.)

    The record — ``{"api_key", "user_id", "created_by"}`` — is written at
    ``ac:claim:<sha256(token)>`` with ``SET ... EX <ttl> NX`` (the raw token is never
    stored). ``ttl_seconds`` defaults to the settings default and is capped at the
    settings ceiling; an over-cap request is a loud 400 (never a silent clamp). Returns
    ``{"claim_path": "/login#claim=<token>", "token": <token>, "expires_at": <iso8601>}``
    — the token rides the URL fragment, and the server returns a PATH, not an absolute
    URL (it does not know the public origin behind a reverse proxy)."""
    settings = access_control_settings()
    ttl = _resolve_ttl(settings, ttl_seconds)

    access_token = await _verifier(settings).verify_token(api_key)
    if access_token is None:
        raise ClaimLinkError(400, _INVALID_KEY_MESSAGE)

    resolved_user_id = access_token.client_id
    resolved_owner = access_token.claims.get(OWNER_USER_ID_CLAIM)
    is_own_key = resolved_user_id == caller_id
    is_key_the_caller_minted = resolved_owner is not None and resolved_owner == caller_id

    # An OWNED caller (its own credential carries an owner claim) can neither mint keys
    # nor be admin, so it may ONLY move its own credential between its own devices —
    # case (c). State that confinement explicitly rather than leaning on the cases being
    # structurally unreachable for it.
    if caller_owner_claim is not None and not is_own_key:
        raise ClaimLinkError(403, _OWNERSHIP_MESSAGE)
    if not (caller_is_admin or is_key_the_caller_minted or is_own_key):
        raise ClaimLinkError(403, _OWNERSHIP_MESSAGE)

    record = json.dumps({"api_key": api_key, "user_id": resolved_user_id, "created_by": caller_id})
    token = await _write_record(settings, record, ttl)

    expires_at = (datetime.now(UTC) + timedelta(seconds=ttl)).isoformat()
    # Never the token or the key — the caller and the target identity only.
    logger.info("access_control: claim link created by %s for user %s (ttl=%ss)", caller_id, resolved_user_id, ttl)
    return {"claim_path": f"/login#claim={token}", "token": token, "expires_at": expires_at}


def _resolve_ttl(settings: AccessControlSettings, ttl_seconds: int | None) -> int:
    """The effective ttl: the settings default when unset, else the request value —
    which must be a positive integer no greater than the settings ceiling. An out-of-
    range request is a loud 400 naming both numbers, never a silent clamp."""
    if ttl_seconds is None:
        return settings.claim_link_ttl_seconds
    if ttl_seconds <= 0:
        raise ClaimLinkError(400, f"ttl_seconds ({ttl_seconds}) must be a positive integer")
    if ttl_seconds > settings.claim_link_max_ttl_seconds:
        raise ClaimLinkError(
            400, f"ttl_seconds ({ttl_seconds}) exceeds the maximum {settings.claim_link_max_ttl_seconds}"
        )
    return ttl_seconds


async def _write_record(settings: AccessControlSettings, record: str, ttl: int) -> str:
    """Write ``record`` under a freshly minted token with ``SET ... EX ttl NX`` and
    return the token. An ``NX`` collision (astronomically rare — 256 bits of entropy)
    retries ONCE with a new token, then raises rather than looping."""
    async with client_ctx(RedisClient, settings.redis) as r:
        for _attempt in range(2):
            token = f"clm-{secrets.token_urlsafe(32)}"
            if await awaited(r.set(_record_key(settings, token), record, ex=ttl, nx=True)):
                return token
    raise RuntimeError("access_control: claim token collided twice on mint; refusing to retry further")


async def exchange_claim_token(token: str) -> dict:
    """Burn the claim record for ``token`` and return the raw key it carried.

    The record is popped with an atomic ``GETDEL`` — the first caller wins, and a
    concurrent second caller, a replay of a used token, an unknown token, and an expired
    token ALL answer the same 404 (no oracle). The stored key is then re-validated
    through the verifier chain: a key REVOKED since the link was created answers the same
    404, with the record already gone — a revoked credential is never handed out. The
    guarantee is not-REVOKED, not owner-alive (owner-death is enforced by the gate on
    every request, not here). Returns ``{"token": <raw key>, "user_id": ...}`` — the
    ``loginResult`` wire shape."""
    settings = access_control_settings()
    token_hash = hash_api_key(token)
    async with client_ctx(RedisClient, settings.redis) as r:
        raw = await awaited(r.getdel(f"{settings.claim_prefix}{token_hash}"))
    if raw is None:
        logger.info("access_control: claim exchange miss for token hash %s", token_hash[:_LOG_HASH_PREFIX])
        raise ClaimLinkError(404, _UNKNOWN_TOKEN_MESSAGE)

    record = json.loads(raw)
    api_key = record["api_key"]

    # Re-validate the stored key: a key revoked during the TTL window must not be handed
    # out. The record is already burned, so the caller gets the uniform 404 either way.
    if await _verifier(settings).verify_token(api_key) is None:
        logger.info(
            "access_control: claim exchange refused a revoked key (token hash %s)", token_hash[:_LOG_HASH_PREFIX]
        )
        raise ClaimLinkError(404, _UNKNOWN_TOKEN_MESSAGE)

    logger.info(
        "access_control: claim exchanged for user %s (token hash %s)",
        record["user_id"],
        token_hash[:_LOG_HASH_PREFIX],
    )
    return {"token": api_key, "user_id": record["user_id"]}
