"""Single decrypt-and-parse path for ConnectionRecord blobs."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from cryptography.exceptions import InvalidTag
from tai_contract.connectors.models import ConnectionRecord

from tai_skeleton.connectors.oauth import crypto
from tai_skeleton.connectors.store import persistence
from tai_skeleton.connectors.store.persistence import (
    ConnectionNotFoundError,
    load_record,
    load_record_or_none,
    load_record_with_blob,
    session_expires_at_for,
)

from .conftest import CID, make_oauth_record


class _FakeStore:
    def __init__(self, blob: bytes | None, *, expired: bool = False) -> None:
        self._blob = blob
        # When ``expired`` the record exists but reads as missing unless the
        # caller opts into ``include_expired`` — models the real store's
        # ``session_expires_at`` filter (serving reads hide it, cleanup sees it).
        self._expired = expired

    async def get(self, connection_id: str, *, include_expired: bool = False) -> bytes | None:
        if self._expired and not include_expired:
            return None
        return self._blob


@pytest.fixture
def install_store(monkeypatch):
    def _install(blob: bytes | None, *, expired: bool = False):
        monkeypatch.setattr(persistence, "token_store", lambda: _FakeStore(blob, expired=expired))

    return _install


def _encrypted_record() -> tuple[bytes, ConnectionRecord]:
    record = make_oauth_record()
    blob = crypto.encrypt(record.to_storage_json().encode("utf-8"), connection_id=CID)
    return blob, record


def test_session_expires_at_for_caps_with_ttl(monkeypatch):
    # Inject a known TTL rather than depend on the ambient prod default.
    from tai_kit.settings import reset_all_settings

    monkeypatch.setenv("CONNECTORS_MAX_SESSION_TTL", "3600")  # 1 hour, in seconds
    reset_all_settings()
    record = make_oauth_record()
    before = datetime.now(UTC)
    out = session_expires_at_for(record)
    delta = out - before
    assert timedelta(seconds=3600) <= delta <= timedelta(seconds=3605)


async def test_load_record_round_trip(install_store):
    blob, record = _encrypted_record()
    install_store(blob)
    loaded = await load_record(CID)
    assert loaded.connection_id == record.connection_id
    assert loaded.access_token is not None
    assert loaded.access_token.get_secret_value() == "access-tok"


async def test_load_record_with_blob_returns_record_and_ciphertext(install_store):
    blob, record = _encrypted_record()
    install_store(blob)
    loaded, raw = await load_record_with_blob(CID)
    assert loaded.connection_id == record.connection_id
    # the raw ciphertext is returned verbatim — it is the compare-and-set handle
    assert raw == blob


async def test_load_record_with_blob_missing_raises(install_store):
    install_store(None)
    with pytest.raises(ConnectionNotFoundError):
        await load_record_with_blob(CID)


async def test_load_record_missing_raises(install_store):
    install_store(None)
    with pytest.raises(ConnectionNotFoundError):
        await load_record(CID)


async def test_load_record_expired_missing_by_default_but_loadable_for_cleanup(install_store):
    # An expired connection reads as missing on the default (serving) path, but
    # the cleanup load (include_expired=True) still returns it so disconnect can
    # purge it. This is the whole point of the flag: an expired session must not
    # be served, yet must remain disconnectable.
    blob, record = _encrypted_record()
    install_store(blob, expired=True)

    with pytest.raises(ConnectionNotFoundError):
        await load_record(CID)

    loaded = await load_record(CID, include_expired=True)
    assert loaded.connection_id == record.connection_id
    loaded_with_blob, raw = await load_record_with_blob(CID, include_expired=True)
    assert loaded_with_blob.connection_id == record.connection_id
    assert raw == blob


async def test_load_record_decrypt_failure_reraises(install_store):
    # A valid 0x01 format version, then a garbage nonce+ciphertext: past the
    # length guard, a real tag mismatch — surfaced loudly as InvalidTag.
    install_store(b"\x01" + b"\x00" * 39)
    with pytest.raises(InvalidTag):
        await load_record(CID)


async def test_load_record_or_none_missing(install_store):
    install_store(None)
    assert await load_record_or_none(CID) is None


async def test_load_record_or_none_round_trip(install_store):
    blob, _ = _encrypted_record()
    install_store(blob)
    rec = await load_record_or_none(CID)
    assert rec is not None
    assert rec.connection_id == CID


async def test_load_record_or_none_unreadable_returns_none(install_store):
    install_store(b"\x00" * 40)
    assert await load_record_or_none(CID) is None
