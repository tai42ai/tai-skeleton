"""Trigger links — minted, token-bearing PUBLIC capability URLs that fire a hook
topic.

A trigger link resolves a raw token to a hook TOPIC (plus optional per-link
``tool_kwargs``) and lets whoever holds the URL dispatch the topic's registered
hooks — the QR-on-a-wall backend. Three STRING keys per link, all built ONLY by
the :class:`HooksSettings` key helpers (the literal key strings live nowhere else):

- a RECORD key (keyed by ``sha256(token)``) — the record JSON; the resolver's
  lookup key. The RAW token is never stored — a link's QR is unrecoverable by design.
- a NAME-index key (keyed by the link name) — its value is the token hash; the
  revocation/list index the operator holds once the token is gone (revoke is by NAME).
- a TOMBSTONE key (keyed by ``sha256(token)``) — the permanent revocation marker.

Both the record and name keys are written in ONE Lua script (a ``MULTI`` cannot
branch on the name key's existence, so it could strand the loser's record key on a
name collision); revoke and restore are each ONE Lua script too, so no record ever
dies without its tombstone and no live record survives unindexed. Every miss at
the door answers the SAME uniform 404 (unknown / expired / revoked / tombstoned /
verifier-bound / in-memory are deliberately indistinguishable to the caller); the
server log distinguishes what it can by hash prefix. Every error RAISES (fail
closed) — a store or verifier-lookup failure is a 500, never a soft dispatch.

Trigger links REQUIRE the Redis hooks backend: an in-memory hooks deployment would
make a supposedly durable public URL per-worker, restart-volatile state — a
correctness lie — so the CRUD refuses loudly with a 501 and the resolver answers
the uniform 404.
"""

from __future__ import annotations

import json
import logging
import math
import re
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator
from tai42_kit.clients import client_ctx
from tai42_kit.clients.impl.redis import RedisClient
from tai42_kit.utils.data.string_util import hash_api_key

from tai42_skeleton.hooks.cache import get_hooks_manager
from tai42_skeleton.hooks.managers.in_memory_hooks_manager import InMemoryHooksManager
from tai42_skeleton.hooks.settings import HooksSettings
from tai42_skeleton.utils.redis_typing import awaited, eval_script

logger = logging.getLogger(__name__)

# The single uniform door miss — unknown / expired / revoked / tombstoned /
# verifier-bound / in-memory all answer this, so the public surface leaks no oracle
# distinguishing them (the server log records the real cause by hash prefix).
_UNKNOWN_OR_EXPIRED = "unknown or expired trigger link"

# The loud in-memory refusal for the CRUD seams (the resolver stays the uniform 404).
_IN_MEMORY_REFUSAL = "trigger links require the redis hooks backend"

# The name is the revocation handle AND rides a URL path segment AND a CLI argument:
# it must address cleanly through the DELETE route, a proxy's dot-segment
# normalization, and a shell option parser. So the first character is never ``-``
# (a leading-dash name parses as a CLI option) and the name carries at least one
# ``[A-Za-z0-9_]`` word character (a pure-dot/pure-dash name normalizes or parses
# away). Max 64 chars. Never sanitized/mutated — a violation is rejected loudly.
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.][A-Za-z0-9_.-]{0,63}\Z")
_NAME_WORD_CHAR = re.compile(r"[A-Za-z0-9_]")
_NAME_RULE = (
    "a trigger link name must match ^[A-Za-z0-9_.-]{1,64}$, start with a non-'-' character, "
    "and carry at least one letter, digit, or underscore"
)

# A stored token hash is a lowercase sha256 hexdigest.
_HEX64 = re.compile(r"^[0-9a-f]{64}\Z")

# The upper PHYSICAL expiry bound (not a product ceiling — there is none): a conservative
# cap comfortably inside both the ``expires_at`` datetime arithmetic and Redis's EX range,
# so a larger ttl is surfaced as a loud 400 rather than a raw store error deeper down.
_MAX_TTL_SECONDS = 10**10

# The number of hex chars of the token hash a log line names — enough to correlate a
# creation/list/resolve/revoke in the log stream, far too few to be the credential.
_LOG_HASH_PREFIX = 12

# One atomic pair-write. KEYS[1]=name key. ARGV = rec_prefix, token_hash, record_json,
# ttl (0 = permanent, >0 = seconds). The name key present ⇒ write NOTHING ⇒ signal
# taken; absent ⇒ write both (both EX ttl when timed), so a timed link's pair expires
# together and Redis erases it with no cleanup job.
_CREATE_LUA = """
-- trigger:create:atomic
local name_key = KEYS[1]
local rec_prefix, token_hash, record_json, ttl = ARGV[1], ARGV[2], ARGV[3], tonumber(ARGV[4])
if redis.call('EXISTS', name_key) == 1 then
  return 0
end
local rec_key = rec_prefix .. token_hash
if ttl > 0 then
  redis.call('SET', name_key, token_hash, 'EX', ttl)
  redis.call('SET', rec_key, record_json, 'EX', ttl)
else
  redis.call('SET', name_key, token_hash)
  redis.call('SET', rec_key, record_json)
end
return 1
"""

# One atomic revoke. KEYS[1]=name key. ARGV = rec_prefix, tomb_prefix. Reads the
# name key's CURRENT hash IN-SCRIPT, DELs that record key + the name key, and writes
# the permanent tombstone — so a revoke racing a same-name re-create cannot orphan
# the new record key. Returns the hash on success, false on a missing name.
_REVOKE_LUA = """
-- trigger:revoke:atomic
local name_key = KEYS[1]
local rec_prefix, tomb_prefix = ARGV[1], ARGV[2]
local token_hash = redis.call('GET', name_key)
if not token_hash then
  return false
end
redis.call('DEL', rec_prefix .. token_hash)
redis.call('DEL', name_key)
redis.call('SET', tomb_prefix .. token_hash, '1')
return token_hash
"""

# One atomic restore. KEYS[1]=name key. ARGV = rec_prefix, tomb_prefix, token_hash,
# record_json, ttl (0 = permanent, >0 = seconds). A tombstoned hash is refused
# in-script (a revoke racing the restore must not slip a live pair in behind a fresh
# tombstone); the incoming hash already live under a DIFFERENT name is refused (one
# name per hash); a name pointing at a DIFFERENT hash is displacement — its record
# is DEL'd (no tombstone: displacement is not revocation). Returns one of
# skipped_tombstoned / hash_conflict / updated / created.
_RESTORE_LUA = """
-- trigger:restore:atomic
local name_key = KEYS[1]
local rec_prefix, tomb_prefix, token_hash, record_json, ttl =
  ARGV[1], ARGV[2], ARGV[3], ARGV[4], tonumber(ARGV[5])
if redis.call('EXISTS', tomb_prefix .. token_hash) == 1 then
  return 'skipped_tombstoned'
end
local rec_key = rec_prefix .. token_hash
local current = redis.call('GET', name_key)
local function write_pair()
  if ttl > 0 then
    redis.call('SET', name_key, token_hash, 'EX', ttl)
    redis.call('SET', rec_key, record_json, 'EX', ttl)
  else
    redis.call('SET', name_key, token_hash)
    redis.call('SET', rec_key, record_json)
  end
end
if current == token_hash then
  write_pair()
  return 'updated'
end
if redis.call('EXISTS', rec_key) == 1 then
  return 'hash_conflict'
end
if current then
  redis.call('DEL', rec_prefix .. current)
  write_pair()
  return 'updated'
end
write_pair()
return 'created'
"""


class TriggerLinkError(Exception):
    """A typed trigger-link failure carrying the HTTP status the adapters map it to.

    ``status`` is 400 (an invalid ttl/name/params or verifier-bound topic), 404
    (the uniform door/revoke miss), 409 (a taken explicit name), or 501 (the
    in-memory-mode refusal). The message is the operator-facing text."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class _TriggerRecord(BaseModel):
    """The stored record body, validated on restore so a malformed backup never
    revives a live URL that 500s at every fire."""

    model_config = ConfigDict(extra="forbid")

    name: str
    topic: str
    tool_kwargs: dict[str, Any] | None = None
    created_by: str | None = None
    created_at: str
    expires_at: str | None = None

    @field_validator("topic")
    @classmethod
    def _topic_non_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("topic must be a non-empty string")
        return value

    @field_validator("created_at")
    @classmethod
    def _created_at_parseable(cls, value: str) -> str:
        datetime.fromisoformat(value)
        return value


def _redis_manager() -> Any:
    """The Redis hooks manager, or a loud 501 when the deployment is in-memory."""
    manager = get_hooks_manager()
    if isinstance(manager, InMemoryHooksManager):
        raise TriggerLinkError(501, _IN_MEMORY_REFUSAL)
    return manager


def _validate_name(name: str) -> None:
    if not (_NAME_PATTERN.match(name) and _NAME_WORD_CHAR.search(name)):
        raise TriggerLinkError(400, _NAME_RULE)


def _validate_ttl(ttl_seconds: int | None) -> int:
    """The Lua ``ttl`` arg: 0 for a permanent link, else the positive seconds. A
    non-positive or over-physical-bound ttl is a loud 400, never a silent clamp."""
    if ttl_seconds is None:
        return 0
    if ttl_seconds <= 0:
        raise TriggerLinkError(
            400, f"ttl_seconds ({ttl_seconds}) must be a positive integer or null for a permanent link"
        )
    if ttl_seconds > _MAX_TTL_SECONDS:
        raise TriggerLinkError(
            400, f"ttl_seconds ({ttl_seconds}) exceeds the store's enforced expiry bound {_MAX_TTL_SECONDS}"
        )
    return ttl_seconds


async def _verifier_bound(manager: Any, topic: str) -> bool:
    """Whether ``topic`` carries a webhook-verifier binding. A lookup ERROR
    propagates (a 500) — never treated as unbound, which would create/fire on a
    verified topic."""
    return await manager.get_topic_verifier(topic) is not None


async def create_trigger_link(
    *,
    topic: str,
    name: str | None,
    ttl_seconds: int | None,
    tool_kwargs: dict[str, Any] | None,
    created_by: str | None,
) -> dict:
    """Mint a trigger link for ``topic`` and return its carrier.

    ``ttl_seconds`` is the creator's explicit choice (``null`` permanent, positive
    timed); a verifier-bound topic is refused; ``tool_kwargs`` is stored
    verbatim and merged last at each fire. Returns
    ``{"name", "trigger_path", "token", "topic", "expires_at"}`` — the ONLY place
    the raw token ever appears."""
    manager = _redis_manager()
    settings: HooksSettings = manager.settings

    ttl = _validate_ttl(ttl_seconds)

    if await _verifier_bound(manager, topic):
        raise TriggerLinkError(
            400, "topic has a webhook verifier binding; trigger links are refused for verified topics"
        )

    if tool_kwargs is not None and not isinstance(tool_kwargs, dict):
        raise TriggerLinkError(400, "tool_kwargs must be a JSON object or null")

    explicit_name = name is not None
    if explicit_name:
        assert name is not None
        _validate_name(name)

    now = datetime.now(UTC)
    expires_at = (now + timedelta(seconds=ttl)).isoformat() if ttl > 0 else None

    async with client_ctx(RedisClient, settings.redis) as r:
        for _attempt in range(2):
            link_name = name if explicit_name else _default_name()
            token = f"trg-{secrets.token_urlsafe(32)}"
            token_hash = hash_api_key(token)
            record = json.dumps(
                {
                    "name": link_name,
                    "topic": topic,
                    "tool_kwargs": tool_kwargs,
                    "created_by": created_by,
                    "created_at": now.isoformat(),
                    "expires_at": expires_at,
                }
            )
            created = await eval_script(
                r,
                _CREATE_LUA,
                1,
                settings.trigger_name_key(link_name),
                settings.trigger_record_key_prefix,
                token_hash,
                record,
                str(ttl),
            )
            if created:
                logger.info(
                    "hooks: trigger link created by %s name=%s topic=%s ttl=%s",
                    created_by,
                    link_name,
                    topic,
                    ttl if ttl > 0 else "permanent",
                )
                return {
                    "name": link_name,
                    "trigger_path": f"/trigger/{token}",
                    "token": token,
                    "topic": topic,
                    "expires_at": expires_at,
                }
            # Name taken: an EXPLICIT name is the caller's unactionable conflict;
            # a GENERATED name retries ONCE with fresh entropy, then raises.
            if explicit_name:
                raise TriggerLinkError(409, "trigger link name already exists")
    raise TriggerLinkError(409, "generated trigger link name collided twice on mint; refusing to retry further")


def _default_name() -> str:
    return f"trg-link-{secrets.token_hex(4)}"


async def list_trigger_links() -> dict:
    """Every live trigger link's record plus its hash PREFIX (never a raw token,
    none is stored). A name key whose record is absent is a PERMANENT orphan (a
    corrupt hand-edited backup) — logged at WARNING and skipped, not left invisibly
    409-squatting its name; a name key whose value is nil (expired between SCAN and
    MGET) is a pure TTL race, skipped silently."""
    manager = _redis_manager()
    settings: HooksSettings = manager.settings
    prefix = settings.trigger_name_key_prefix

    items: list[dict] = []
    async with client_ctx(RedisClient, settings.redis) as r:
        name_keys = await _scan_all(r, settings.trigger_name_scan_pattern())
        if not name_keys:
            return {"items": [], "total": 0}
        hashes = await awaited(r.mget(name_keys))
        pairs = [(nk, h) for nk, h in zip(name_keys, hashes, strict=True) if h is not None]
        if not pairs:
            return {"items": [], "total": 0}
        record_keys = [settings.trigger_record_key(_as_str(h)) for _nk, h in pairs]
        records = await awaited(r.mget(record_keys))
        for (name_key, token_hash), record_raw in zip(pairs, records, strict=True):
            name = _as_str(name_key).removeprefix(prefix)
            if record_raw is None:
                logger.warning("hooks: trigger link name %r has no record (permanent orphan); skipping", name)
                continue
            record = json.loads(_as_str(record_raw))
            record["token_hash_prefix"] = _as_str(token_hash)[:_LOG_HASH_PREFIX]
            items.append(record)
    return {"items": items, "total": len(items)}


async def revoke_trigger_link(name: str) -> None:
    """Revoke a link by name — DEL the record + name keys and write the permanent
    tombstone in ONE atomic script (so a revoke racing a same-name re-create cannot
    orphan a live record key). A missing name is a loud 404."""
    manager = _redis_manager()
    settings: HooksSettings = manager.settings
    async with client_ctx(RedisClient, settings.redis) as r:
        token_hash = await eval_script(
            r,
            _REVOKE_LUA,
            1,
            settings.trigger_name_key(name),
            settings.trigger_record_key_prefix,
            settings.trigger_tomb_key_prefix,
        )
    if not token_hash:
        raise TriggerLinkError(404, "unknown trigger link")
    logger.info("hooks: trigger link revoked name=%s hash=%s", name, _as_str(token_hash)[:_LOG_HASH_PREFIX])


async def resolve_trigger_token(token: str) -> tuple[str, dict | None]:
    """Resolve a raw token to ``(topic, tool_kwargs)`` for a fire — multi-use, NO
    burn. ONE ``MGET`` of record + tombstone: a record miss OR a tombstone
    present ⇒ the uniform 404 (a tombstoned hash is dead at the door itself, not
    only at backup import). A corrupt stored record raises (a 500, nothing
    dispatched). The verifier binding is re-checked: a topic verified after the
    link was minted answers the SAME 404 + a server log naming the cause."""
    manager = get_hooks_manager()
    if isinstance(manager, InMemoryHooksManager):
        # Trigger links cannot exist in-memory; the CRUD refuses them, so a resolve
        # here is a miss like any other — logged with the token hash prefix and the
        # true cause, matching every other resolve outcome.
        logger.info(
            "hooks: trigger resolve miss hash=%s cause=in-memory-backend",
            hash_api_key(token)[:_LOG_HASH_PREFIX],
        )
        raise TriggerLinkError(404, _UNKNOWN_OR_EXPIRED)

    settings: HooksSettings = manager.settings
    token_hash = hash_api_key(token)
    hash_prefix = token_hash[:_LOG_HASH_PREFIX]
    async with client_ctx(RedisClient, settings.redis) as r:
        record_raw, tomb_raw = await awaited(
            r.mget([settings.trigger_record_key(token_hash), settings.trigger_tomb_key(token_hash)])
        )
    if tomb_raw is not None:
        logger.info("hooks: trigger resolve miss hash=%s cause=tombstoned", hash_prefix)
        raise TriggerLinkError(404, _UNKNOWN_OR_EXPIRED)
    if record_raw is None:
        logger.info("hooks: trigger resolve miss hash=%s cause=unknown-or-expired", hash_prefix)
        raise TriggerLinkError(404, _UNKNOWN_OR_EXPIRED)

    record = json.loads(_as_str(record_raw))
    topic = record["topic"]
    if await _verifier_bound(manager, topic):
        logger.info("hooks: trigger resolve miss hash=%s cause=verifier-bound", hash_prefix)
        raise TriggerLinkError(404, _UNKNOWN_OR_EXPIRED)

    logger.info("hooks: trigger resolve hit hash=%s outcome=accepted", hash_prefix)
    return topic, record.get("tool_kwargs")


# -- Backup seams — the section module calls ONLY these -----------------


async def export_trigger_links() -> dict:
    """The trigger-link records (each with its FULL token hash) plus the tombstone
    hashes, for the ``webhooks`` backup section. The full hash rides here by
    construction (hash-not-token); the list route keeps returning only the prefix.
    On an in-memory deployment the store provably holds none, so this returns
    truthfully empty rather than refusing (hooks export is unaffected)."""
    manager = get_hooks_manager()
    if isinstance(manager, InMemoryHooksManager):
        return {"trigger_links": [], "tombstones": []}

    settings: HooksSettings = manager.settings
    name_prefix = settings.trigger_name_key_prefix
    tomb_prefix = settings.trigger_tomb_key_prefix

    links: list[dict] = []
    async with client_ctx(RedisClient, settings.redis) as r:
        name_keys = await _scan_all(r, settings.trigger_name_scan_pattern())
        if name_keys:
            hashes = await awaited(r.mget(name_keys))
            pairs = [(nk, h) for nk, h in zip(name_keys, hashes, strict=True) if h is not None]
            if pairs:
                record_keys = [settings.trigger_record_key(_as_str(h)) for _nk, h in pairs]
                records = await awaited(r.mget(record_keys))
                for (name_key, token_hash), record_raw in zip(pairs, records, strict=True):
                    if record_raw is None:
                        name = _as_str(name_key).removeprefix(name_prefix)
                        logger.warning("hooks: trigger link name %r has no record; excluded from export", name)
                        continue
                    links.append(
                        {
                            "name": _as_str(name_key).removeprefix(name_prefix),
                            "token_hash": _as_str(token_hash),
                            "record": json.loads(_as_str(record_raw)),
                        }
                    )
        tomb_keys = await _scan_all(r, settings.trigger_tomb_scan_pattern())
        tombstones = [_as_str(tk).removeprefix(tomb_prefix) for tk in tomb_keys]
    return {"trigger_links": links, "tombstones": tombstones}


async def bound_hashes_by_name() -> dict[str, str]:
    """Every ``name:*`` index binding on the store — ``name`` -> token hash —
    INCLUDING orphans (a name key whose ``rec:*`` record is gone). This is the
    "what hash is bound under ANY name" view the import duplicate-hash refusal needs:
    revoke reads a name key's hash and DELs that record whether or not the record
    still exists, so an orphaned binding is authoritative. If import ignored orphans,
    a NEW name binding an already-orphaned hash would slip past the refusal, and later
    revoking the orphan would destroy the new name's live record. A name key nil
    between SCAN and MGET (a pure TTL race) is skipped. An in-memory deployment holds
    none, so this returns truthfully empty."""
    manager = get_hooks_manager()
    if isinstance(manager, InMemoryHooksManager):
        return {}

    settings: HooksSettings = manager.settings
    prefix = settings.trigger_name_key_prefix

    bindings: dict[str, str] = {}
    async with client_ctx(RedisClient, settings.redis) as r:
        name_keys = await _scan_all(r, settings.trigger_name_scan_pattern())
        if not name_keys:
            return bindings
        hashes = await awaited(r.mget(name_keys))
        for name_key, token_hash in zip(name_keys, hashes, strict=True):
            if token_hash is None:
                continue
            bindings[_as_str(name_key).removeprefix(prefix)] = _as_str(token_hash)
    return bindings


async def restore_trigger_link(*, name: str, token_hash: str, record: dict) -> str:
    """Restore one exported record atomically. Refuses malformed triples loudly
    (name mismatch, pattern-violating name, non-hex hash, unparseable ``expires_at``,
    a record body failing full model validation); refuses a tombstoned hash
    (``skipped_tombstoned``) and a hash already live under a different name; skips an
    already-expired record (``skipped_expired``). Returns one of
    created / updated / skipped_tombstoned / skipped_expired."""
    manager = _redis_manager()
    settings: HooksSettings = manager.settings

    if not isinstance(name, str):
        raise TriggerLinkError(400, "trigger link name must be a string")
    if not isinstance(token_hash, str):
        raise TriggerLinkError(400, "token_hash must be a string")
    _validate_name(name)
    if not _HEX64.match(token_hash):
        raise TriggerLinkError(400, "token_hash must be a 64-character lowercase sha256 hexdigest")
    try:
        model = _TriggerRecord.model_validate(record)
    except ValidationError as exc:
        raise TriggerLinkError(400, f"invalid trigger link record: {exc}") from exc
    if model.name != name:
        raise TriggerLinkError(400, f"record name {model.name!r} does not match its index name {name!r}")

    ttl = _remaining_ttl(model.expires_at)
    if ttl is _EXPIRED:
        return "skipped_expired"
    assert isinstance(ttl, int)

    async with client_ctx(RedisClient, settings.redis) as r:
        result = await eval_script(
            r,
            _RESTORE_LUA,
            1,
            settings.trigger_name_key(name),
            settings.trigger_record_key_prefix,
            settings.trigger_tomb_key_prefix,
            token_hash,
            json.dumps(record),
            str(ttl),
        )
    outcome = _as_str(result)
    if outcome == "hash_conflict":
        raise TriggerLinkError(400, f"token hash for {name!r} is already live under a different name")
    return outcome


async def restore_tombstone(token_hash: str) -> None:
    """Restore a tombstone marker (idempotent). Refuses loudly in-memory and on a
    non-hex hash — an imported tombstone is a permanent kill switch, never written
    for garbage."""
    manager = _redis_manager()
    settings: HooksSettings = manager.settings
    if not isinstance(token_hash, str):
        raise TriggerLinkError(400, "token_hash must be a string")
    if not _HEX64.match(token_hash):
        raise TriggerLinkError(400, "token_hash must be a 64-character lowercase sha256 hexdigest")
    async with client_ctx(RedisClient, settings.redis) as r:
        await awaited(r.set(settings.trigger_tomb_key(token_hash), "1"))


# -- internals ----------------------------------------------------------------


class _Expired:
    """A distinct sentinel so a computed ``ttl`` of "already expired" never collides
    with a real positive ttl."""


_EXPIRED = _Expired()


def _remaining_ttl(expires_at: str | None) -> int | _Expired:
    """The Lua ``ttl`` arg for a restore: 0 for a permanent link; ``_EXPIRED`` when
    nothing remains; else the whole seconds remaining, CEILed so a sub-second
    remainder restores with EX 1 (never EX 0, a Redis error). An unparseable
    ``expires_at`` is a loud, typed refusal."""
    if expires_at is None:
        return 0
    try:
        deadline = datetime.fromisoformat(expires_at)
    except ValueError as exc:
        raise TriggerLinkError(400, f"record expires_at {expires_at!r} is not a parseable timestamp") from exc
    remaining = (deadline - datetime.now(UTC)).total_seconds()
    if remaining <= 0:
        return _EXPIRED
    return max(1, math.ceil(remaining))


async def _scan_all(r: Any, pattern: str) -> list[str]:
    """Every key matching ``pattern`` once, across ALL SCAN pages (a first-page-only
    cursor bug would silently drop live links from the management surface and from
    backups; SCAN may also return a key more than once under a concurrent rehash, so
    duplicates are collapsed while first-seen order is preserved)."""
    keys: list[str] = []
    cursor = 0
    while True:
        cursor, batch = await awaited(r.scan(cursor, match=pattern, count=100))
        keys.extend(_as_str(k) for k in batch)
        if cursor == 0:
            break
    return list(dict.fromkeys(keys))


def _as_str(value: Any) -> str:
    """Redis may hand back ``bytes`` or ``str`` depending on decode settings; the key
    and record strings are ascii-safe, so normalize to ``str``."""
    return value.decode() if isinstance(value, bytes) else value
