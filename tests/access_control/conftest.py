"""Fakes for the access_control feature.

Two stand-ins, no real backends:

* :class:`FakeRedis` — the kit ``RedisClient`` surface the auth gate touches: the
  plain-Redis ``get``/``incr`` policy-version counter and the ``hgetall``/``hset``
  live-context hashes. Yielded through the ``client_ctx`` seam via
  :func:`make_client_ctx`.
* :class:`FakeAccessControlPg` — a stateful in-memory Postgres modelling the two
  policy tables (``access_control_policies`` + ``access_control_routes``), mirroring
  ``tests/versioning/conftest.py``'s ``FakeVersioningPg``: it interprets the store's
  SQL by normalized prefix, snapshots/restores the tables on a transaction rollback,
  enforces the ``UNIQUE`` constraints, and takes an injectable ``fault``. Yielded via
  :func:`make_pg_ctx`.
"""

from __future__ import annotations

import asyncio
import copy
import fnmatch
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest
from psycopg.errors import UniqueViolation
from redis.exceptions import WatchError
from tai42_kit.clients.impl.postgres import Json, PostgresClient


class FakeRedis:
    """Matches exactly the redis operations the access_control modules call."""

    def __init__(
        self,
        *,
        strings: dict | None = None,
        hashes: dict | None = None,
        sets: dict | None = None,
        raise_get: Exception | None = None,
        raise_hgetall: Exception | None = None,
    ) -> None:
        self._strings = strings or {}
        self._hashes = hashes or {}
        self._sets: dict[str, set[str]] = sets or {}
        self._raise_get = raise_get
        self._raise_hgetall = raise_hgetall

        # TTL modelling for the claim-link store's ``SET ... EX`` / ``GETDEL``: a
        # per-key deadline against a test-driven clock (``advance`` moves it forward),
        # so ``EX`` is honored deterministically without wall-clock sleeps.
        self._expiry: dict[str, float] = {}
        self._now: float = 0.0

        # Optimistic-lock bookkeeping for the WATCH/MULTI pipeline below. Every
        # write bumps a global revision and stamps it on the touched key; a
        # ``watch`` snapshots those stamps and ``execute`` aborts (WatchError) when
        # a watched key's stamp moved. ``force_conflict_key`` and
        # ``execute_hooks`` let a test drive that abort deterministically, and
        # ``watch_conflicts`` counts the aborts a retry recovered from.
        self._rev = 0
        self._key_rev: dict[str, int] = {}
        self.force_conflict_key: str | None = None
        self.execute_hooks: list[Callable[[], Awaitable[None]]] = []
        self.watch_conflicts = 0

    def _touch(self, key: str) -> None:
        self._rev += 1
        self._key_rev[key] = self._rev

    def advance(self, seconds: float) -> None:
        """Move the fake clock forward, so a key past its ``EX`` deadline reads as
        expired — the claim-link TTL time-travel helper."""
        self._now += seconds

    def _purge_if_expired(self, key: str) -> None:
        deadline = self._expiry.get(key)
        if deadline is not None and self._now >= deadline:
            self._strings.pop(key, None)
            self._expiry.pop(key, None)

    def pipeline(self, transaction: bool = True, shard_hint: str | None = None) -> _FakePipeline:
        return _FakePipeline(self)

    async def get(self, key):
        if self._raise_get is not None:
            raise self._raise_get
        self._purge_if_expired(key)
        value = self._strings.get(key)
        # A real network round-trip suspends after the read; yield here so two concurrent
        # callers doing a non-atomic read-then-write (``get`` then ``delete``) BOTH
        # complete their read before either deletes — modelling the check-then-act race.
        # ``getdel`` deliberately does NOT yield (it is one atomic step), so an atomic
        # single-use burn still admits exactly one winner.
        await asyncio.sleep(0)
        return value

    async def getdel(self, key):
        """Atomic read-and-delete — the claim-link single-use burn. Returns the value
        and removes the key, or ``None`` when absent/expired."""
        self._purge_if_expired(key)
        value = self._strings.pop(key, None)
        if value is not None:
            self._expiry.pop(key, None)
            self._touch(key)
        return value

    async def hgetall(self, key):
        if self._raise_hgetall is not None:
            raise self._raise_hgetall
        return dict(self._hashes.get(key, {}))

    # -- write / scan surface (the provisioning module's mutations) ----------

    async def set(self, key, value, *, ex=None, nx=False):
        # ``nx``: only write when the key is absent (claim-link create), returning
        # ``None`` on a collision as redis does. ``ex``: an expiry in seconds against the
        # fake clock.
        self._purge_if_expired(key)
        if nx and key in self._strings:
            return None
        self._strings[key] = str(value)
        if ex is not None:
            self._expiry[key] = self._now + ex
        else:
            self._expiry.pop(key, None)
        self._touch(key)
        return True

    async def mget(self, keys):
        return [self._strings.get(k) for k in keys]

    async def incr(self, key):
        value = int(self._strings.get(key, "0")) + 1
        self._strings[key] = str(value)
        self._touch(key)
        return value

    async def delete(self, *keys):
        removed = 0
        for key in keys:
            present = any(key in store for store in (self._strings, self._hashes, self._sets))
            self._strings.pop(key, None)
            self._hashes.pop(key, None)
            self._sets.pop(key, None)
            self._expiry.pop(key, None)
            # Redis marks a key modified (invalidating a WATCH) only when DEL
            # actually removes something; a DEL of an absent key dirties nothing.
            if present:
                removed += 1
                self._touch(key)
        return removed

    async def hset(self, key, mapping):
        existing = self._hashes.setdefault(key, {})
        stored = {field: str(value) for field, value in mapping.items()}
        new_fields = sum(1 for field in stored if field not in existing)
        existing.update(stored)
        self._touch(key)
        return new_fields

    async def hget(self, key, field):
        return self._hashes.get(key, {}).get(field)

    async def hdel(self, key, *fields):
        h = self._hashes.get(key, {})
        removed = sum(1 for f in fields if h.pop(f, None) is not None)
        # Redis marks a key modified (invalidating a WATCH) only when HDEL actually
        # removes a field; an HDEL of an absent field is a no-op that must NOT bump
        # the key's watch revision, or an unrelated writer would spuriously abort a
        # concurrent transaction watching this key.
        if removed:
            self._touch(key)
        return removed

    # -- set surface (per-scope ``scope_urls`` aggregate) --------------------
    #
    # SADD/SREM only bump the key's watch revision when they actually change the
    # set — matching redis, which marks a key dirty (and so invalidates a WATCH)
    # only on a real mutation. An empty set is deleted, as redis does, so an
    # existing ``scope_urls`` key always holds at least one url.

    async def sadd(self, key, *members):
        members = {str(m) for m in members}
        s = self._sets.setdefault(key, set())
        added = len(members - s)
        if added:
            s |= members
            self._touch(key)
        return added

    async def srem(self, key, *members):
        s = self._sets.get(key)
        if not s:
            return 0
        removed = len(s & {str(m) for m in members})
        if removed:
            s -= {str(m) for m in members}
            if not s:
                del self._sets[key]
            self._touch(key)
        return removed

    async def smembers(self, key):
        return set(self._sets.get(key, set()))

    async def scard(self, key):
        return len(self._sets.get(key, set()))

    async def sismember(self, key, member):
        return str(member) in self._sets.get(key, set())

    async def scan_iter(self, match):
        # Redis SCAN matches keys across every value type with GLOB semantics, so
        # ``fnmatch`` is the faithful stand-in: a metacharacter mid-``match`` (``*``,
        # ``?``, ``[``) matches exactly as redis would, not merely as a prefix.
        seen: set[str] = set()
        for store in (self._strings, self._hashes, self._sets):
            for key in list(store):
                if key not in seen and fnmatch.fnmatchcase(key, match):
                    seen.add(key)
                    yield key


class _FakePipeline:
    """A faithful in-memory stand-in for a redis-py async WATCH/MULTI pipeline.

    Before ``multi()`` the pipeline is in immediate mode: read commands hit the
    store now and return a value. ``multi()`` opens the transactional block, after
    which write commands are QUEUED (returning the pipeline) and applied together
    only on ``execute()``. ``execute`` first fires any test-injected hook and the
    forced-conflict switch, then aborts with ``WatchError`` if a watched key's
    revision moved since it was watched; otherwise it applies the queue in order
    and returns each command's result. Every path resets the pipeline so the same
    object is reusable across optimistic-lock retries, exactly as redis-py does.
    """

    def __init__(self, fake: FakeRedis) -> None:
        self._fake = fake
        self._watched: dict[str, int] = {}
        self._in_multi = False
        self._queue: list[tuple[str, tuple]] = []

    async def __aenter__(self) -> _FakePipeline:
        return self

    async def __aexit__(self, *exc) -> bool:
        await self.reset()
        return False

    async def reset(self) -> None:
        self._watched = {}
        self._in_multi = False
        self._queue = []

    async def watch(self, *names: str) -> bool:
        if self._in_multi:
            raise RuntimeError("Cannot issue a WATCH after a MULTI")
        for name in names:
            self._watched[name] = self._fake._key_rev.get(name, 0)
        return True

    def multi(self) -> None:
        self._in_multi = True

    # -- immediate reads (before MULTI) --------------------------------------

    async def get(self, key):
        return await self._fake.get(key)

    async def hget(self, key, field):
        return await self._fake.hget(key, field)

    async def hgetall(self, key):
        return await self._fake.hgetall(key)

    async def mget(self, keys):
        return await self._fake.mget(keys)

    async def smembers(self, key):
        return await self._fake.smembers(key)

    async def scard(self, key):
        return await self._fake.scard(key)

    async def sismember(self, key, member):
        return await self._fake.sismember(key, member)

    # -- queued writes (after MULTI) -----------------------------------------

    def set(self, key, value) -> _FakePipeline:
        self._queue.append(("set", (key, value)))
        return self

    def delete(self, *keys) -> _FakePipeline:
        self._queue.append(("delete", keys))
        return self

    def hset(self, key, mapping) -> _FakePipeline:
        self._queue.append(("hset", (key, mapping)))
        return self

    def hdel(self, key, *fields) -> _FakePipeline:
        self._queue.append(("hdel", (key, fields)))
        return self

    def sadd(self, key, *members) -> _FakePipeline:
        self._queue.append(("sadd", (key, members)))
        return self

    def srem(self, key, *members) -> _FakePipeline:
        self._queue.append(("srem", (key, members)))
        return self

    async def execute(self, raise_on_error: bool = True) -> list:
        try:
            if self._fake.execute_hooks:
                await self._fake.execute_hooks.pop(0)()
            if self._fake.force_conflict_key is not None:
                self._fake._touch(self._fake.force_conflict_key)
            for key, snapshot in self._watched.items():
                if self._fake._key_rev.get(key, 0) != snapshot:
                    self._fake.watch_conflicts += 1
                    raise WatchError("watched key changed")
            results: list = []
            for op, args in self._queue:
                results.append(await self._apply(op, args))
            return results
        finally:
            await self.reset()

    async def _apply(self, op: str, args: tuple):
        if op == "set":
            return await self._fake.set(*args)
        if op == "delete":
            return await self._fake.delete(*args)
        if op == "hset":
            return await self._fake.hset(*args)
        if op == "hdel":
            key, fields = args
            return await self._fake.hdel(key, *fields)
        if op == "sadd":
            key, members = args
            return await self._fake.sadd(key, *members)
        if op == "srem":
            key, members = args
            return await self._fake.srem(key, *members)
        raise AssertionError(f"unhandled queued op {op!r}")


def make_client_ctx(fake: FakeRedis):
    """A drop-in for ``tai42_kit.clients.client_ctx`` yielding ``fake`` for any
    client class, ignoring the settings/pool/fresh arguments."""

    @asynccontextmanager
    async def _ctx(client_cls, settings=None, *, fresh=False, **kwargs):
        yield fake

    return _ctx


@pytest.fixture
def make_ctx():
    return make_client_ctx


@pytest.fixture(autouse=True)
def _isolate_identity_registry():
    """Snapshot + restore the module-level identity-provider registry around each
    test, so a test that registers (or clears) a provider never leaks into the
    next. The skeleton's interim ``redis`` registration (registered at the provider
    module's import) is captured by the baseline snapshot and restored, so tests
    that build the real ``AuthAdapter`` still resolve ``auth_providers=["redis"]``."""
    from tai42_contract.access_control import registry

    saved = dict(registry._REGISTRY)
    try:
        yield
    finally:
        registry._REGISTRY.clear()
        registry._REGISTRY.update(saved)


class _FakeResourceManager:
    """Renders a condition/expr by returning the inline ``content`` unchanged
    (the auth gate's policy condition is inline jq), recording each call."""

    def __init__(self) -> None:
        self.calls: list = []

    async def render_by_id_or_content(self, *, content, template_id, kwargs):
        self.calls.append((content, template_id, kwargs))
        return content


class _FakeStorage:
    """The ``tai42_app.storage`` facet: exposes the template manager, matching the
    real ``AppStorage`` shape the auth backend reaches through."""

    def __init__(self) -> None:
        self.resource_manager = _FakeResourceManager()


class _FakeApp:
    """Minimal ``tai42_app`` impl exposing the members the auth backend reaches."""

    def __init__(self) -> None:
        self.storage = _FakeStorage()

    def effective_router_modules(self) -> None:
        # Not a started router deployment: the shared route importer reads this to
        # choose its enumeration universe, and ``None`` selects the whole-package
        # universe (the offline default) rather than a curated served set.
        return None


@pytest.fixture
def bound_app():
    """Bind a fake app onto the global ``tai42_app`` handle for the test, then
    restore the unbound state so other suites still see the loud pre-bind error."""
    from tai42_contract.app import tai42_app

    app = _FakeApp()
    tai42_app.bind(app)
    try:
        yield app
    finally:
        tai42_app.bind(None)


# -- Fake Postgres for the access-control policy store -----------------------


_POLICY_USER_UNIQUE = "access_control_policies_user_id_unique"


def _unwrap(value: Any) -> Any:
    return value.obj if isinstance(value, Json) else value


class _PolicyUserViolation(UniqueViolation):
    """A UniqueViolation carrying the ``user_id`` unique-constraint name, as psycopg
    reports a duplicate policy row (``diag`` is a read-only property, so — like
    ``FakeVersioningPg`` — the name is a class attribute)."""

    diag: Any = SimpleNamespace(constraint_name=_POLICY_USER_UNIQUE)


class _PgTxn:
    """Snapshot-and-restore savepoint: rolls the tables back on any exception."""

    def __init__(self, pg: FakeAccessControlPg) -> None:
        self._pg = pg
        self._snapshot: tuple[list[dict], list[dict]] | None = None

    async def __aenter__(self) -> _PgTxn:
        self._snapshot = (copy.deepcopy(self._pg.policies), copy.deepcopy(self._pg.routes))
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None and self._snapshot is not None:
            self._pg.policies, self._pg.routes = self._snapshot
        return False


class _PgCursor:
    def __init__(self, pg: FakeAccessControlPg) -> None:
        self._pg = pg
        self.rowcount = 0
        self._one: Any = None
        self._all: list = []

    async def __aenter__(self) -> _PgCursor:
        return self

    async def __aexit__(self, *exc) -> bool:
        return False

    async def execute(self, sql: str, params: tuple = ()) -> None:
        norm = " ".join(sql.split())
        pg = self._pg
        pg.executed.append(norm)
        if pg.fault is not None and norm.startswith(pg.fault[0]):
            raise pg.fault[1]
        self._one = None
        self._all = []
        self.rowcount = 0

        if norm.startswith("SELECT COUNT(*) FROM access_control_policies WHERE policy_data"):
            pointer_key, role_name = params
            count = sum(1 for p in pg.policies if (p.get("policy_data") or {}).get(pointer_key) == role_name)
            self._one = (count,)
        elif norm.startswith("INSERT INTO access_control_routes"):
            url, scope_id, pattern = params
            existing = next((r for r in pg.routes if r["url"] == url), None)
            if existing is not None:
                existing["scope_id"] = scope_id
                existing["pattern"] = pattern
            else:
                pg.routes.append({"id": pg.next_route_id(), "url": url, "scope_id": scope_id, "pattern": pattern})
        elif norm.startswith("INSERT INTO access_control_policies"):
            user_id, scopes, policy_data, condition, condition_id, condition_kwargs = params
            if any(p["user_id"] == user_id for p in pg.policies):
                raise _PolicyUserViolation()
            pg.policies.append(
                {
                    "id": pg.next_policy_id(),
                    "user_id": user_id,
                    "scopes": list(scopes),
                    "policy_data": _unwrap(policy_data),
                    "condition": condition,
                    "condition_id": condition_id,
                    "condition_kwargs": _unwrap(condition_kwargs),
                }
            )
        elif norm.startswith("DELETE FROM access_control_routes WHERE scope_id"):
            (scope_id,) = params
            before = len(pg.routes)
            pg.routes = [r for r in pg.routes if r["scope_id"] != scope_id]
            self.rowcount = before - len(pg.routes)
        elif norm.startswith("DELETE FROM access_control_routes WHERE url"):
            (url,) = params
            before = len(pg.routes)
            pg.routes = [r for r in pg.routes if r["url"] != url]
            self.rowcount = before - len(pg.routes)
        elif norm.startswith("DELETE FROM access_control_policies WHERE user_id"):
            (user_id,) = params
            before = len(pg.policies)
            pg.policies = [p for p in pg.policies if p["user_id"] != user_id]
            self.rowcount = before - len(pg.policies)
        elif norm.startswith("UPDATE access_control_policies SET scopes = array_remove"):
            # Strip the scope and RETURN the committed (post-strip) bodies, mirroring
            # the store's single UPDATE ... array_remove ... RETURNING.
            scope_id, scope_id2 = params
            affected = []
            for p in pg.policies:
                if scope_id2 in p["scopes"]:
                    p["scopes"] = [s for s in p["scopes"] if s != scope_id]
                    affected.append(
                        (
                            p["user_id"],
                            list(p["scopes"]),
                            p["policy_data"],
                            p["condition"],
                            p["condition_id"],
                            p["condition_kwargs"],
                        )
                    )
            self.rowcount = len(affected)
            self._all = affected
        elif norm.startswith("UPDATE access_control_policies SET scopes = %s"):
            scopes, policy_data, condition, condition_id, condition_kwargs, user_id = params
            for p in pg.policies:
                if p["user_id"] == user_id:
                    p["scopes"] = list(scopes)
                    p["policy_data"] = _unwrap(policy_data)
                    p["condition"] = condition
                    p["condition_id"] = condition_id
                    p["condition_kwargs"] = _unwrap(condition_kwargs)
                    self.rowcount = 1
        elif norm.startswith("SELECT url, scope_id FROM access_control_routes"):
            if "<>" in norm:
                (public,) = params
                self._all = [(r["url"], r["scope_id"]) for r in pg.routes if r["scope_id"] != public]
            else:
                self._all = [(r["url"], r["scope_id"]) for r in sorted(pg.routes, key=lambda r: r["url"])]
        elif norm.startswith("SELECT url FROM access_control_routes WHERE scope_id"):
            (scope_id,) = params
            # Emulate the real query's ORDER BY faithfully: sort only when the SQL
            # asks for it, so a dropped ORDER BY surfaces instead of being masked.
            matched = [r for r in pg.routes if r["scope_id"] == scope_id]
            if "ORDER BY url" in norm:
                matched = sorted(matched, key=lambda r: r["url"])
            self._all = [(r["url"],) for r in matched]
        elif norm.startswith("SELECT url, pattern FROM access_control_routes"):
            self._all = [
                (r["url"], r["pattern"]) for r in sorted(pg.routes, key=lambda r: r["url"]) if r["pattern"] is not None
            ]
        elif norm.startswith("SELECT pattern, url FROM access_control_routes"):
            self._all = [(r["pattern"], r["url"]) for r in pg.routes if r["pattern"] is not None]
        elif norm.startswith("SELECT scope_id FROM access_control_routes WHERE url"):
            (url,) = params
            row = next((r for r in pg.routes if r["url"] == url), None)
            self._one = (row["scope_id"],) if row is not None else None
        elif norm.startswith("SELECT 1 FROM access_control_routes WHERE scope_id"):
            (scope_id,) = params
            self._one = (1,) if any(r["scope_id"] == scope_id for r in pg.routes) else None
        elif norm.startswith("SELECT scope_id FROM access_control_routes WHERE scope_id = ANY"):
            # The grant paths' lock-and-validate query. FOR SHARE is a no-op in this
            # single-threaded fake (it cannot model row locks); the scope-validity
            # filter is what the fake exercises.
            scopes, public = params
            self._all = [(r["scope_id"],) for r in pg.routes if r["scope_id"] in scopes and r["scope_id"] != public]
        elif norm.startswith("SELECT scopes, policy_data, condition, condition_id, condition_kwargs"):
            (user_id,) = params
            p = next((p for p in pg.policies if p["user_id"] == user_id), None)
            self._one = (
                None
                if p is None
                else (
                    list(p["scopes"]),
                    p["policy_data"],
                    p["condition"],
                    p["condition_id"],
                    p["condition_kwargs"],
                )
            )
        elif norm.startswith("SELECT 1 FROM access_control_policies WHERE user_id"):
            (user_id,) = params
            self._one = (1,) if any(p["user_id"] == user_id for p in pg.policies) else None
        else:
            raise AssertionError(f"unhandled SQL in fake pg: {norm!r}")

    async def fetchone(self) -> Any:
        return self._one

    async def fetchall(self) -> list:
        return self._all


class _PgConn:
    def __init__(self, pg: FakeAccessControlPg) -> None:
        self._pg = pg

    async def __aenter__(self) -> _PgConn:
        return self

    async def __aexit__(self, *exc) -> bool:
        return False

    def cursor(self) -> _PgCursor:
        return _PgCursor(self._pg)

    def transaction(self) -> _PgTxn:
        return _PgTxn(self._pg)


class FakeAccessControlPg:
    """In-memory stand-in for the two access-control policy tables."""

    def __init__(self) -> None:
        self.policies: list[dict] = []
        self.routes: list[dict] = []
        self.executed: list[str] = []
        self.fault: tuple[str, Exception] | None = None
        self._policy_seq = 0
        self._route_seq = 0

    def connection(self) -> _PgConn:
        return _PgConn(self)

    def next_policy_id(self) -> int:
        self._policy_seq += 1
        return self._policy_seq

    def next_route_id(self) -> int:
        self._route_seq += 1
        return self._route_seq

    # -- seeding / assertion helpers -----------------------------------------

    def add_route(self, url: str, scope_id: str, pattern: str | None = None) -> None:
        self.routes.append({"id": self.next_route_id(), "url": url, "scope_id": scope_id, "pattern": pattern})

    def add_policy(self, user_id: str, scopes: list[str] | None = None, **fields: Any) -> None:
        body = {
            "scopes": list(scopes or []),
            "policy_data": {},
            "condition": None,
            "condition_id": None,
            "condition_kwargs": None,
        }
        body.update(fields)
        self.policies.append({"id": self.next_policy_id(), "user_id": user_id, **body})

    # ``policy``/``policy_body``/``route`` return ``Any`` (not ``dict | None``) so a
    # test can both compare to ``None`` and subscript the result without a narrowing
    # dance — they are assertion helpers over known-present rows.
    def policy(self, user_id: str) -> Any:
        return next((p for p in self.policies if p["user_id"] == user_id), None)

    def policy_body(self, user_id: str) -> Any:
        """The clean policy body (no ``id``/``user_id``), matching what the store's
        ``get_policy_body`` returns — for asserting the enforced record's contents."""
        p = self.policy(user_id)
        if p is None:
            return None
        return {k: p[k] for k in ("scopes", "policy_data", "condition", "condition_id", "condition_kwargs")}

    def route(self, url: str) -> Any:
        return next((r for r in self.routes if r["url"] == url), None)

    def scope_urls(self, scope_id: str) -> set[str]:
        return {r["url"] for r in self.routes if r["scope_id"] == scope_id}


def make_pg_ctx(fake: FakeAccessControlPg):
    """A drop-in for ``client_ctx`` yielding ``fake`` for the PostgresClient."""

    @asynccontextmanager
    async def _ctx(client_cls, settings=None, **kwargs):
        if client_cls is not PostgresClient:
            raise AssertionError(f"unexpected client_cls in fake pg: {client_cls!r}")
        yield fake

    return _ctx


@pytest.fixture
def pg(monkeypatch) -> FakeAccessControlPg:
    """A fake Postgres wired over the store module's ``client_ctx`` seam."""
    import tai42_skeleton.access_control.store as store_module

    fake = FakeAccessControlPg()
    monkeypatch.setattr(store_module, "client_ctx", make_pg_ctx(fake))
    return fake
