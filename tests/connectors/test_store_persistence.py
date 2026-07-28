"""Single decrypt-and-parse path for ConnectionRecord blobs."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta

import pytest
from cryptography.exceptions import InvalidTag
from pydantic import ValidationError
from tai42_contract.connectors.models import ConnectionRecord
from tai42_kit.settings import reset_all_settings

from tai42_skeleton.connectors.oauth import crypto
from tai42_skeleton.connectors.oauth.crypto import ConnectorEncryptionConfigError
from tai42_skeleton.connectors.store import persistence
from tai42_skeleton.connectors.store.persistence import (
    ConnectionNotFoundError,
    load_record,
    load_record_or_none,
    load_record_with_blob,
    session_expires_at_for,
)

from .conftest import CID, CID2, make_oauth_record


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


class _FakeListingStore:
    """The store as ``list_connections`` reads it: the ids it enumerates."""

    def __init__(self, ids: list[str]) -> None:
        self._ids = ids

    async def list(self) -> list[str]:
        return self._ids


@pytest.fixture
def install_store(monkeypatch):
    def _install(blob: bytes | None, *, expired: bool = False):
        monkeypatch.setattr(persistence, "token_store", lambda: _FakeStore(blob, expired=expired))

    return _install


def _encrypted_record() -> tuple[bytes, ConnectionRecord]:
    record = make_oauth_record()
    blob = crypto.encrypt(record.to_storage_json().encode("utf-8"), connection_id=CID)
    return blob, record


def _encrypted_shape_drift() -> bytes:
    """A blob that decrypts cleanly but whose plaintext is no longer a
    ``ConnectionRecord``: the stored JSON with the required ``kind`` field dropped."""
    stored = json.loads(make_oauth_record().to_storage_json())
    del stored["kind"]
    return crypto.encrypt(json.dumps(stored).encode("utf-8"), connection_id=CID)


def _assert_logged_server_side(caplog: pytest.LogCaptureFixture, detail: str) -> None:
    """Exactly one ERROR record from the persistence logger, carrying the caught
    exception, with ``detail`` present in the formatted log. Formatting the record renders
    its own message and appends the attached traceback, so ``detail`` is pinned to THAT
    record rather than to anything else the capture happens to hold."""
    errors = [r for r in caplog.records if r.levelno == logging.ERROR and r.name == persistence.logger.name]
    assert len(errors) == 1
    assert errors[0].exc_info is not None
    assert detail in logging.Formatter().format(errors[0])


def _assert_warned_server_side(caplog: pytest.LogCaptureFixture, *details: str) -> None:
    """Exactly one WARNING record from the persistence logger, carrying the caught
    exception, with every ``details`` entry present in the formatted log. Formatting the
    record renders its own message and appends the attached traceback, so each detail is
    pinned to THAT record rather than to anything else the capture happens to hold."""
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and r.name == persistence.logger.name]
    assert len(warnings) == 1
    assert warnings[0].exc_info is not None
    formatted = logging.Formatter().format(warnings[0])
    for detail in details:
        assert detail in formatted


def test_session_expires_at_for_caps_with_ttl(monkeypatch):
    # Inject a known TTL rather than depend on the ambient prod default.
    from tai42_kit.settings import reset_all_settings

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


async def test_load_record_decrypt_failure_reraises(install_store, caplog):
    # A valid 0x01 format version, then a garbage nonce+ciphertext: past the
    # length guard, a real tag mismatch — surfaced loudly as InvalidTag, and recorded
    # server-side first: one ERROR naming the connection, with the traceback attached, so
    # a wrong CONNECTORS_KEK or a corrupted blob is an operator-visible incident and not
    # only an exception in the caller's face.
    install_store(b"\x01" + b"\x00" * 39)
    with caplog.at_level(logging.ERROR, logger=persistence.logger.name), pytest.raises(InvalidTag):
        await load_record(CID)
    # The log call uses %r, so the record carries the id's repr.
    _assert_logged_server_side(caplog, repr(CID))


async def test_load_record_shape_drift_raises(install_store):
    # The blob decrypts cleanly and only the PARSE fails, because the stored JSON no
    # longer matches the current ``ConnectionRecord`` shape. The list projection skips
    # such a row; the single-record path must not — it raises so the caller sees the
    # drift instead of being handed a record that could not be rebuilt.
    install_store(_encrypted_shape_drift())
    with pytest.raises(ValidationError):
        await load_record(CID)


async def test_load_record_broken_kek_config_raises_unlogged(install_store, monkeypatch, caplog):
    # A missing CONNECTORS_KEK fails EVERY blob, so it is a deployment fault and not this
    # blob's. It propagates — and, unlike a decrypt failure, files no per-blob ERROR, so
    # the incident is not misattributed to whichever connection was read first.
    blob, _ = _encrypted_record()
    install_store(blob)
    monkeypatch.delenv("CONNECTORS_KEK", raising=False)
    reset_all_settings()

    with (
        caplog.at_level(logging.ERROR, logger=persistence.logger.name),
        pytest.raises(ConnectorEncryptionConfigError),
    ):
        await load_record(CID)

    assert [r for r in caplog.records if r.name == persistence.logger.name] == []


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


async def test_load_record_or_none_shape_drift_returns_none(install_store, caplog):
    # Shape drift: the blob decrypts cleanly and only the PARSE fails, because the stored
    # JSON no longer matches the current ``ConnectionRecord`` shape. That is one bad row,
    # not a deployment fault, so the projection skips it instead of poisoning the whole
    # list — and the skip is recorded server-side as one WARNING naming the connection,
    # with the validation error attached, so the drift is visible rather than a silent gap.
    install_store(_encrypted_shape_drift())
    with caplog.at_level(logging.WARNING, logger=persistence.logger.name):
        assert await load_record_or_none(CID) is None
    # The log call uses %r, so the record carries the id's repr; the attached traceback
    # names the field the stored JSON dropped.
    _assert_warned_server_side(caplog, repr(CID), "kind")


async def test_load_record_or_none_broken_kek_config_raises(install_store, monkeypatch, caplog):
    # A missing CONNECTORS_KEK is not a property of this blob — it fails EVERY row.
    # Tolerated per row, a list projection would answer "no connections" behind one
    # warning per row and pass a deployment fault off as empty data, so the config
    # error propagates and nothing is logged as an unreadable row.
    blob, _ = _encrypted_record()
    install_store(blob)
    monkeypatch.delenv("CONNECTORS_KEK", raising=False)
    reset_all_settings()

    with (
        caplog.at_level(logging.WARNING, logger=persistence.logger.name),
        pytest.raises(ConnectorEncryptionConfigError),
    ):
        await load_record_or_none(CID)

    assert [r for r in caplog.records if r.name == persistence.logger.name] == []


async def test_list_connections_projection_raises_on_broken_kek_config(install_store, monkeypatch):
    # The projection the tolerance exists for: with the KEK gone it must fail loudly
    # rather than render an empty connection list over a deployment that has two.
    from tai42_skeleton.operations import connectors as conn_ops

    blob, _ = _encrypted_record()
    install_store(blob)
    monkeypatch.setattr(conn_ops, "token_store", lambda: _FakeListingStore([CID, CID2]))
    monkeypatch.delenv("CONNECTORS_KEK", raising=False)
    reset_all_settings()

    with pytest.raises(ConnectorEncryptionConfigError):
        await conn_ops.list_connections()
