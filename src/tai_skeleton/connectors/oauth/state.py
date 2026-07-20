"""OAuth ``state`` for the Connectors flow.

Two parts:

- An HMAC-signed envelope carrying the single-use ``flow_id`` AND the originating
  deployment origin. The signature is the CSRF guard: the callback re-derives the
  HMAC and rejects a tampered or forged ``state``. Wire format::

      base64url(json({"f": flow_id, "o": origin})) + "." + hex(hmac_sha256(payload_b64, key))

  The HMAC is over the base64url payload string (not the decoded JSON) so the
  verifier never re-serialises JSON. The origin rides in the readable base64
  payload (not behind the key) precisely so a central OAuth bridge — a different
  deployment holding no HMAC key — can read it to route the code back to the
  originating deployment; the bridge treats it as UNTRUSTED and allow-lists it
  before bouncing, while the destination re-verifies the HMAC on
  :func:`decode`. With ``CONNECTORS_OAUTH_BRIDGE_URL`` unset the provider
  redirects to this deployment directly and the signed origin is simply its own.

- A redis-backed transient store for the in-flight flow record (TTL 600s).
  Single-use: :func:`get_and_delete` removes the key on the call that returns
  it, so a replayed callback produces no second token write.
"""

from __future__ import annotations

import base64
import hmac
import json
import logging
from hashlib import sha256

from pydantic import BaseModel
from tai_contract.connectors.service import FlowOperation
from tai_kit.clients import client_ctx
from tai_kit.clients.impl.redis import RedisClient

from tai_skeleton.connectors.settings import (
    connector_crypto_secrets,
    connector_store_settings,
)

logger = logging.getLogger(__name__)

_KEY_PREFIX = "connectors:flow:"
_TTL_SECONDS = 600


# -- State envelope ----------------------------------------------------------


class StateInvalidError(ValueError):
    """The ``state`` envelope is malformed or tampered. Reported to clients as a
    generic mismatch; the reason is logged only."""


class DecodedState(BaseModel):
    """The verified contents of an OAuth ``state`` envelope: the single-use
    ``flow_id`` and the originating deployment ``origin`` the code came from."""

    flow_id: str
    origin: str


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def encode(*, flow_id: str, origin: str) -> str:
    """Encode ``flow_id`` + originating deployment ``origin`` into a signed
    envelope. Fails loudly if either is empty or if
    ``CONNECTORS_STATE_HMAC_KEY`` is not configured."""
    if not flow_id:
        raise ValueError("flow_id must be non-empty")
    if not origin:
        raise ValueError("origin must be non-empty")
    key = connector_crypto_secrets().require_state_hmac_key_bytes()
    payload_json = json.dumps(
        {"f": flow_id, "o": origin},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    payload_b64 = _b64url_encode(payload_json)
    tag = hmac.new(key, payload_b64.encode("ascii"), sha256).hexdigest()
    return f"{payload_b64}.{tag}"


def decode(state: str) -> DecodedState:
    """Return the verified :class:`DecodedState` after HMAC verification. Raises
    :class:`StateInvalidError` on every failure, logging a short reason code."""
    if not isinstance(state, str) or not state:
        raise StateInvalidError("state must be a non-empty string")

    parts = state.split(".")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        logger.warning("connectors: state rejected (bad_shape)")
        raise StateInvalidError("state envelope has wrong shape")
    payload_b64, tag_hex = parts

    key = connector_crypto_secrets().require_state_hmac_key_bytes()
    expected = hmac.new(key, payload_b64.encode("ascii"), sha256).hexdigest()
    if not hmac.compare_digest(expected, tag_hex):
        logger.warning("connectors: state rejected (bad_hmac)")
        raise StateInvalidError("state HMAC mismatch")

    try:
        obj = json.loads(_b64url_decode(payload_b64))
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning("connectors: state rejected (bad_payload)")
        raise StateInvalidError("state payload is not decodable") from exc
    if not isinstance(obj, dict):
        logger.warning("connectors: state rejected (non_object)")
        raise StateInvalidError("state payload is not a JSON object")

    flow_id = obj.get("f")
    if not isinstance(flow_id, str) or not flow_id:
        logger.warning("connectors: state rejected (missing_flow_id)")
        raise StateInvalidError("state payload missing flow_id")
    origin = obj.get("o")
    if not isinstance(origin, str) or not origin:
        logger.warning("connectors: state rejected (missing_origin)")
        raise StateInvalidError("state payload missing origin")
    return DecodedState(flow_id=flow_id, origin=origin)


# -- Flow state store --------------------------------------------------------


class OAuthFlowState(BaseModel):
    flow_id: str
    provider_id: str
    alias: str
    requested_scopes: list[str]
    enabled_sub_services: list[str]
    pkce_verifier: str
    return_url: str
    # The provider redirect URI validated at authorize-start. Stored so the token
    # exchange re-sends it byte-identically (RFC 6749) instead of recomputing it
    # from the completion request's Origin header.
    redirect_uri: str
    operation: FlowOperation
    reconnect_connection_id: str | None = None


def _key(flow_id: str) -> str:
    return f"{_KEY_PREFIX}{flow_id}"


async def put(state: OAuthFlowState) -> None:
    """Write a flow record with the standard TTL. ``flow_id`` is a fresh uuid4,
    so the key is always new."""
    async with client_ctx(RedisClient, connector_store_settings().redis) as redis:
        await redis.set(_key(state.flow_id), state.model_dump_json(), ex=_TTL_SECONDS)


async def get_and_delete(flow_id: str) -> OAuthFlowState | None:
    """Pop the flow record (single-use). Returns ``None`` if expired or absent."""
    async with client_ctx(RedisClient, connector_store_settings().redis) as redis:
        pipe = redis.pipeline()
        pipe.get(_key(flow_id))
        pipe.delete(_key(flow_id))
        raw, _ = await pipe.execute()
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return OAuthFlowState.model_validate_json(raw)
