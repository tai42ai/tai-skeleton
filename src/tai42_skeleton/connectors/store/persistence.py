"""Single decrypt-and-parse path for ConnectionRecord blobs.

The one canonical helper for the ``store.get`` -> ``crypto.decrypt`` ->
``ConnectionRecord.model_validate_json`` sequence, so every caller shares one
error-handling contract instead of duplicating it.

:func:`load_record` raises :class:`ConnectionNotFoundError` on a missing blob and
re-raises (after logging ERROR) on a decrypt failure. :func:`load_record_or_none`
returns ``None`` on either, for list / projection paths that must keep going.

A broken ``CONNECTORS_KEK`` config is neither: it is not a property of any one blob,
so both helpers let it propagate — unlogged in :func:`load_record`, so the deployment
fault is not filed against one blob, and unswallowed in :func:`load_record_or_none`,
so it does not empty every projection.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from tai42_contract.connectors.models import ConnectionRecord

from tai42_skeleton.connectors.oauth import crypto
from tai42_skeleton.connectors.settings import connector_engine_config
from tai42_skeleton.connectors.store import token_store

logger = logging.getLogger(__name__)


def session_expires_at_for(record: ConnectionRecord) -> datetime:
    """The effective session expiry, used as the Redis cache TTL and persisted
    as the ``session_expires_at`` column.

    Capped by ``CONNECTORS_MAX_SESSION_TTL``. The TTL resets on every write, so
    a regularly-used connection behaves like an inactivity window.
    """
    return datetime.now(UTC) + connector_engine_config().max_session_ttl


class ConnectionNotFoundError(KeyError):
    """Missing connection; routers surface as 404.

    Defined here (not in ``connection_service``) so the persistence helper has
    no upward dependency on the much-larger service module.
    """


async def load_record_with_blob(
    connection_id: str,
    *,
    include_expired: bool = False,
) -> tuple[ConnectionRecord, bytes]:
    """Decrypt + parse, returning the record AND the raw ciphertext it loaded
    from. Raises :class:`ConnectionNotFoundError` on missing.

    The blob is the compare-and-set handle for a refresh write-back: the resolver
    captures it before a slow upstream refresh and passes it back to
    ``store.put(expected_blob=...)`` so the durable store commits only when no
    peer rotated the record meanwhile.

    ``include_expired`` is forwarded to ``store.get``: left ``False`` for every
    serving read (an expired session reads as missing), set ``True`` ONLY by the
    cleanup path (disconnect) so an expired connection stays loadable-to-purge.

    Decrypt failures log at ERROR (a ``CONNECTORS_KEK`` that is not the one this blob
    was written with, or a corrupted blob, are operator-visible incidents) and re-raise
    the underlying exception unchanged. A missing or malformed ``CONNECTORS_KEK`` is
    excluded: it raises :class:`ConnectorEncryptionConfigError` for EVERY blob, so it
    propagates unlogged here rather than blaming one blob for a deployment fault.
    """
    blob = await token_store().get(connection_id, include_expired=include_expired)
    if blob is None:
        raise ConnectionNotFoundError(connection_id)
    try:
        plain = crypto.decrypt(blob, connection_id=connection_id)
    except crypto.ConnectorEncryptionConfigError:
        raise
    except Exception:
        logger.error(
            "connectors: connection blob %r failed to decrypt — wrong CONNECTORS_KEK for this blob or a corrupted blob",
            connection_id,
            exc_info=True,
        )
        raise
    return ConnectionRecord.model_validate_json(plain), blob


async def load_record(connection_id: str, *, include_expired: bool = False) -> ConnectionRecord:
    """Decrypt + parse. Raises :class:`ConnectionNotFoundError` on missing.

    Thin projection of :func:`load_record_with_blob` for the read paths that do
    not need the compare-and-set handle. ``include_expired`` is forwarded — left
    ``False`` for serving reads, set ``True`` ONLY by disconnect's cleanup load.
    """
    record, _ = await load_record_with_blob(connection_id, include_expired=include_expired)
    return record


async def load_record_or_none(connection_id: str) -> ConnectionRecord | None:
    """Return ``None`` on missing OR unreadable, and raise on a broken KEK config.

    For list-projection paths where one bad row must not poison the whole
    response. Unreadable blobs log at WARNING.

    Three cases reach the ``except``, and only two of them are one bad row:

    * a decrypt failure for this blob — the KEK is not the one it was written
      with, or the ciphertext is corrupt;
    * shape drift — the blob decrypts but its JSON does not match the current
      ``ConnectionRecord`` shape (e.g. one missing a required field like kind);
    * a missing or malformed ``CONNECTORS_KEK``, which is not about this blob at
      all: it raises :class:`ConnectorEncryptionConfigError` for EVERY row, so
      tolerating it would answer "no connections" behind a warning per row and
      pass a deployment fault off as empty data.

    The first two skip the one bad row so it never poisons a whole
    list/projection; the config fault propagates. The single-record
    :func:`load_record` path raises loudly on all three.
    """
    blob = await token_store().get(connection_id)
    if blob is None:
        return None
    try:
        plain = crypto.decrypt(blob, connection_id=connection_id)
        return ConnectionRecord.model_validate_json(plain)
    except crypto.ConnectorEncryptionConfigError:
        raise
    except Exception:
        logger.warning(
            "connectors: skipping unreadable connection %r",
            connection_id,
            exc_info=True,
        )
        return None
