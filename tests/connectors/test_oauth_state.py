"""OAuth ``state`` envelope (HMAC sign/verify) + single-use redis flow store."""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest
from tai_contract.connectors.service import FlowOperation

import tai_skeleton.connectors.oauth.state as state_mod
from tai_skeleton.connectors.oauth import state
from tai_skeleton.connectors.oauth.state import OAuthFlowState, StateInvalidError

FLOW_ID = "33333333-3333-4333-8333-333333333333"
ORIGIN = "https://studio.example.com"


# -- Envelope sign / verify --------------------------------------------------


def test_encode_decode_round_trip():
    envelope = state.encode(flow_id=FLOW_ID, origin=ORIGIN)
    assert envelope.count(".") == 1
    decoded = state.decode(envelope)
    assert decoded.flow_id == FLOW_ID
    assert decoded.origin == ORIGIN


def test_encode_rejects_empty_flow_id():
    with pytest.raises(ValueError, match="flow_id must be non-empty"):
        state.encode(flow_id="", origin=ORIGIN)


def test_encode_rejects_empty_origin():
    with pytest.raises(ValueError, match="origin must be non-empty"):
        state.encode(flow_id=FLOW_ID, origin="")


def test_decode_rejects_non_string():
    with pytest.raises(StateInvalidError):
        state.decode(None)  # type: ignore[arg-type]


def test_decode_rejects_empty():
    with pytest.raises(StateInvalidError):
        state.decode("")


@pytest.mark.parametrize("bad", ["nopayload", "a.b.c", ".tag", "payload.", "."])
def test_decode_rejects_bad_shape(bad):
    with pytest.raises(StateInvalidError, match="wrong shape"):
        state.decode(bad)


def test_decode_rejects_tampered_hmac():
    envelope = state.encode(flow_id=FLOW_ID, origin=ORIGIN)
    payload_b64, tag = envelope.split(".")
    # flip the last hex char of the tag
    tampered_tag = tag[:-1] + ("0" if tag[-1] != "0" else "1")
    with pytest.raises(StateInvalidError, match="HMAC mismatch"):
        state.decode(f"{payload_b64}.{tampered_tag}")


def test_decode_rejects_tampered_payload():
    """Mutating the payload invalidates the HMAC."""
    envelope = state.encode(flow_id=FLOW_ID, origin=ORIGIN)
    payload_b64, tag = envelope.split(".")
    forged_payload = payload_b64[:-1] + ("A" if payload_b64[-1] != "A" else "B")
    with pytest.raises(StateInvalidError, match="HMAC mismatch"):
        state.decode(f"{forged_payload}.{tag}")


def test_decode_rejects_non_object_payload(monkeypatch):
    """A validly-signed but non-dict JSON payload is rejected."""
    import hmac
    from hashlib import sha256

    from tai_skeleton.connectors.settings import connector_crypto_secrets

    key = connector_crypto_secrets().require_state_hmac_key_bytes()
    payload_b64 = state_mod._b64url_encode(b"[1, 2, 3]")
    tag = hmac.new(key, payload_b64.encode("ascii"), sha256).hexdigest()
    with pytest.raises(StateInvalidError, match="not a JSON object"):
        state.decode(f"{payload_b64}.{tag}")


def test_decode_rejects_undecodable_payload():
    """A signed payload whose bytes are not valid JSON is rejected."""
    import hmac
    from hashlib import sha256

    from tai_skeleton.connectors.settings import connector_crypto_secrets

    key = connector_crypto_secrets().require_state_hmac_key_bytes()
    payload_b64 = state_mod._b64url_encode(b"\xff\xfe not json")
    tag = hmac.new(key, payload_b64.encode("ascii"), sha256).hexdigest()
    with pytest.raises(StateInvalidError, match="not decodable"):
        state.decode(f"{payload_b64}.{tag}")


def test_decode_rejects_missing_flow_id():
    import hmac
    from hashlib import sha256

    from tai_skeleton.connectors.settings import connector_crypto_secrets

    key = connector_crypto_secrets().require_state_hmac_key_bytes()
    payload_b64 = state_mod._b64url_encode(b'{"x": "y"}')
    tag = hmac.new(key, payload_b64.encode("ascii"), sha256).hexdigest()
    with pytest.raises(StateInvalidError, match="missing flow_id"):
        state.decode(f"{payload_b64}.{tag}")


def test_decode_rejects_missing_origin():
    """A validly-signed envelope with flow_id but no origin is rejected."""
    import hmac
    import json
    from hashlib import sha256

    from tai_skeleton.connectors.settings import connector_crypto_secrets

    key = connector_crypto_secrets().require_state_hmac_key_bytes()
    payload_json = json.dumps({"f": FLOW_ID}, separators=(",", ":"), sort_keys=True).encode("ascii")
    payload_b64 = state_mod._b64url_encode(payload_json)
    tag = hmac.new(key, payload_b64.encode("ascii"), sha256).hexdigest()
    with pytest.raises(StateInvalidError, match="missing origin"):
        state.decode(f"{payload_b64}.{tag}")


def test_b64url_round_trip_with_padding():
    for raw in [b"", b"a", b"ab", b"abc", b"abcd"]:
        enc = state_mod._b64url_encode(raw)
        assert "=" not in enc
        assert state_mod._b64url_decode(enc) == raw


# -- Redis flow store --------------------------------------------------------


class _FakeRedisPipeline:
    def __init__(self, store: dict[str, str]) -> None:
        self._store = store
        self._ops: list[tuple[str, str]] = []

    def get(self, key):
        self._ops.append(("get", key))

    def delete(self, key):
        self._ops.append(("delete", key))

    async def execute(self):
        results = []
        for op, key in self._ops:
            if op == "get":
                results.append(self._store.get(key))
            else:
                results.append(1 if self._store.pop(key, None) is not None else 0)
        return results


class _FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.set_calls: list[tuple[str, str, int | None]] = []

    async def set(self, key: str, value: str, ex: int | None = None):
        self.set_calls.append((key, value, ex))
        self.store[key] = value

    def pipeline(self):
        return _FakeRedisPipeline(self.store)


@pytest.fixture
def fake_redis(monkeypatch):
    redis = _FakeRedis()

    @asynccontextmanager
    async def fake_client_ctx(client_cls, settings=None, **kwargs):
        yield redis

    monkeypatch.setattr(state_mod, "client_ctx", fake_client_ctx)
    return redis


def _flow_state() -> OAuthFlowState:
    return OAuthFlowState(
        flow_id=FLOW_ID,
        provider_id="acme",
        alias="work",
        requested_scopes=["mail.read"],
        enabled_sub_services=["mail"],
        pkce_verifier="verifier",
        return_url="/connectors",
        redirect_uri="https://app.example.com/oauth-bridge.html",
        operation=FlowOperation.CONNECT,
    )


async def test_put_writes_with_ttl(fake_redis):
    await state.put(_flow_state())
    key, value, ex = fake_redis.set_calls[-1]
    assert key == f"connectors:flow:{FLOW_ID}"
    assert ex == state_mod._TTL_SECONDS
    assert "acme" in value


async def test_get_and_delete_round_trip(fake_redis):
    await state.put(_flow_state())
    loaded = await state.get_and_delete(FLOW_ID)
    assert loaded is not None
    assert loaded.flow_id == FLOW_ID
    assert loaded.provider_id == "acme"
    # The authorize-time redirect_uri survives the round-trip so the token
    # exchange can re-send it byte-identically (RFC 6749).
    assert loaded.redirect_uri == "https://app.example.com/oauth-bridge.html"


async def test_get_and_delete_is_single_use(fake_redis):
    """A replayed callback finds nothing on the second call."""
    await state.put(_flow_state())
    assert await state.get_and_delete(FLOW_ID) is not None
    assert await state.get_and_delete(FLOW_ID) is None


async def test_get_and_delete_missing_returns_none(fake_redis):
    assert await state.get_and_delete("no-such-flow") is None


async def test_get_and_delete_decodes_bytes(fake_redis, monkeypatch):
    """A redis client returning bytes is decoded before validation."""
    await state.put(_flow_state())
    # Re-store the value as bytes to mimic decode_responses=False.
    key = f"connectors:flow:{FLOW_ID}"
    fake_redis.store[key] = fake_redis.store[key].encode("utf-8")
    loaded = await state.get_and_delete(FLOW_ID)
    assert loaded is not None
    assert loaded.flow_id == FLOW_ID
