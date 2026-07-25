"""Single decrypt-and-parse path for ConnectionRecord blobs.

The one canonical helper for the ``store.get`` -> ``crypto.decrypt`` ->
``ConnectionRecord.model_validate_json`` sequence, so every caller shares one
error-handling contract instead of duplicating it.

:func:`load_record` raises :class:`ConnectionNotFoundError` on a missing blob and
re-raises (after logging ERROR) on a decrypt failure. :func:`load_record_or_none`
returns ``None`` on either, for list / projection paths that must keep going.
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

    Decrypt failures log at ERROR (KEK rotation / on-disk corruption are
    operator-visible incidents) and re-raise the underlying exception unchanged.
    """
    blob = await token_store().get(connection_id, include_expired=include_expired)
    if blob is None:
        raise ConnectionNotFoundError(connection_id)
    try:
        plain = crypto.decrypt(blob, connection_id=connection_id)
    except Exception:
        logger.error(
            "connectors: connection blob %s failed to decrypt — possible KEK rotation issue or on-disk corruption",
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
    """Return ``None`` on missing OR unreadable.

    For list-projection paths where one bad row must not poison the whole
    response. Unreadable blobs log at WARNING.
    """
    blob = await token_store().get(connection_id)
    if blob is None:
        return None
    try:
        plain = crypto.decrypt(blob, connection_id=connection_id)
        return ConnectionRecord.model_validate_json(plain)
    except Exception:
        # Unreadable = decrypt failure OR a blob whose JSON does not match the
        # current ConnectionRecord shape (e.g. one missing a required field like
        # kind). Skip the one bad row so it never poisons a whole list/projection;
        # the single-record load_record path still raises loudly.
        logger.warning(
            "connectors: skipping unreadable connection %s",
            connection_id,
            exc_info=True,
        )
        return None
