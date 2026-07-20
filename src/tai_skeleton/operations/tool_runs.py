"""Background tool-run operations — submit, get-by-id, and per-tool list.

The synchronous ``POST /api/run-tool`` door holds one request open for the whole
tool call; a dropped connection loses the result even though the tool finished
server-side. These operations detach the run from the request:

* ``submit_run`` — takes ``tool_name`` + ``arguments`` (the same field shape the
  sync door's parser enforces), returns ``{"run_id": ...}`` at once (the route
  answers ``202``) and executes the tool as an in-process background task through
  the SAME ``tai_app.tools.run_tool`` seam the sync door uses — with
  ``offload_sync`` set, so a blocking sync tool runs on a worker thread and cannot
  starve the supervisor's liveness refresh. A "run any tool by name" door, so it
  is a tier-1 meta-executor (never projected to the MCP surface, like
  ``run_tool``).
* ``get_run`` — the run record ``{run_id, tool_name, status, started_at,
  finished_at?, result?, error?}``; an unknown/expired id is a loud 404.
  ``status ∈ running | succeeded | failed | lost``.
* ``list_tool_runs`` — the recent runs for one tool (id, tool name, status,
  timestamps only — never ``result``/``error``), newest first, from a per-tool
  ZSET trimmed to ``ToolRunsSettings.recent_runs_limit``.

Per-identity isolation: a run records the OWNING identity of its submitter (always
the caller's OWN id — each key is its own island, never sharing its owner's or a
sibling owned key's slice) and is indexed under a per-identity
``recent:{user_id}:{tool_name}`` window in addition to the shared per-tool window. A
restricted caller reads and prunes only its own per-identity window (complete within
its own bound, never truncated by other identities' volume) and may GET only a run it
owns — another identity's run id is a loud ``403`` (never a ``404``: the run exists,
it is simply not the caller's). An unrestricted caller keeps the full view over the
shared window.

A supervisor wraps each run: it refreshes a per-run liveness key while the tool
runs, writes the terminal record when the tool returns or raises (``succeeded``
+ result, or ``failed`` + the caught error string — the error becomes visible
record data, never swallowed), and in ``finally`` cancels the liveness refresher.
``lost`` is computed-and-persisted one way: the FIRST read of a record still
``running`` whose liveness key has expired writes ``status: lost`` (a dead
process never wrote its terminal record, so it cannot later flip to succeeded).
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field
from tai_contract.app import tai_app
from tai_kit.clients import client_ctx
from tai_kit.clients.impl.redis import RedisClient

from tai_skeleton.access_control.user import request_identity
from tai_skeleton.operations import BadRequestError, ForbiddenError, NotFoundError, UnavailableError, operation
from tai_skeleton.routers.tool_runs_settings import ToolRunsSettings, tool_runs_settings

logger = logging.getLogger(__name__)

# The spawned supervisor tasks are held here so the event loop keeps a strong
# reference (``asyncio`` only holds a weak one) — a dropped task would be
# garbage-collected mid-run. Each task removes itself on completion.
_SUPERVISORS: set[asyncio.Task[None]] = set()

# Per-worker count of in-flight background runs, enforced against
# ``max_concurrent_runs``. The submit door increments it synchronously before
# creating a record; each supervisor's done-callback decrements it. Exact on the
# single event loop (no interleaving between the capacity check and the
# increment).
_ACTIVE_RUNS: int = 0

_RUNNING = "running"
_SUCCEEDED = "succeeded"
_FAILED = "failed"
_LOST = "lost"

# Atomic compare-and-set terminal write. ``lost`` is one-way: a reader that finds
# a still-``running`` record whose liveness key has expired writes ``lost``, and
# the supervisor's own terminal write (``succeeded``/``failed``) must never
# overwrite it — nor may a late ``lost`` clobber a terminal the supervisor just
# wrote. Both writes go through this script, which transitions the record ONLY
# while its stored ``status`` is still ``running``, making the read-decide-write
# one atomic server-side step (a Python read-then-write across two awaits could
# interleave with the other writer).
#   KEYS[1] = run_key
#   ARGV[1] = record TTL (seconds); ARGV[2..] = HSET field/value pairs
_TERMINAL_CAS_LUA = """
-- tool_runs:terminal-cas
if redis.call('HGET', KEYS[1], 'status') ~= 'running' then return 0 end
redis.call('HSET', KEYS[1], unpack(ARGV, 2))
redis.call('EXPIRE', KEYS[1], ARGV[1])
return 1
"""


def _now() -> datetime:
    return datetime.now(UTC)


class ToolRunSubmission(BaseModel):
    """A background tool-run submission: the ``tool_name`` and its keyword
    ``arguments``. Mirrors the shape ``read_tool_call`` enforces at runtime."""

    tool_name: str = Field(min_length=1, description="Registered tool name.")
    arguments: dict[str, object] = Field(default_factory=dict, description="Tool keyword arguments.")


# -- Redis store -------------------------------------------------------------


class ToolRunStore:
    """Every tool-run key shape and the read/write operations behind one class.

    Operations take the redis client as an argument; each caller opens it from
    the tool-runs settings via ``client_ctx(RedisClient, settings.redis)``. Loud
    by contract — no swallowed errors, no silent fallback."""

    def __init__(self, key_prefix: str) -> None:
        self._p = key_prefix

    # -- key shapes ----------------------------------------------------------

    def run_key(self, run_id: str) -> str:
        return f"{self._p}run:{run_id}"

    def liveness_key(self, run_id: str) -> str:
        return f"{self._p}live:{run_id}"

    def recent_key(self, tool_name: str, user_id: str | None = None) -> str:
        """The recent-runs index key for ``tool_name``. With ``user_id`` given, the
        PER-IDENTITY index ``recent:{user_id}:{tool_name}`` a restricted caller reads
        (its own complete window); without it, the shared ``recent:{tool_name}`` index
        an unrestricted caller reads."""
        if user_id is None:
            return f"{self._p}recent:{tool_name}"
        return f"{self._p}recent:{user_id}:{tool_name}"

    # -- writes --------------------------------------------------------------

    async def create_run(
        self,
        r: Any,
        run_id: str,
        tool_name: str,
        started_at: str,
        score: float,
        settings: ToolRunsSettings,
        user_id: str | None = None,
    ) -> None:
        """Persist a new ``running`` record, prime its liveness key, and index it in
        the tool's recent-runs ZSET — trimming the ZSET to the newest
        ``recent_runs_limit`` members. The record hash and the index both carry the
        record TTL so a tool that stops being run eventually drops its index.

        ``user_id`` is the OWNING identity of the run — always the caller's own id
        (each key is its own island). When present it is stamped onto
        the record AND the run id is also pushed onto the per-identity index
        ``recent:{user_id}:{tool_name}`` (its own bound/TTL, mirroring the shared
        index), so a restricted caller's list reads a complete window that other
        identities' volume can never truncate. A caller with no bound identity (gate
        off) leaves ``user_id`` absent and writes only the shared index.

        None of the writes branches on a prior result, so they are all issued in ONE
        pipeline (a single round trip) rather than sequentially."""
        run_key = self.run_key(run_id)
        recent_key = self.recent_key(tool_name)
        record: dict[str, str] = {"tool_name": tool_name, "status": _RUNNING, "started_at": started_at}
        if user_id is not None:
            record["user_id"] = user_id
        pipe = r.pipeline()
        pipe.hset(run_key, mapping=record)
        pipe.expire(run_key, settings.result_ttl_seconds)
        pipe.set(self.liveness_key(run_id), "1", ex=settings.liveness_ttl_seconds)
        pipe.zadd(recent_key, {run_id: score})
        # Trim to the newest N: rank 0..-(limit+1) is every member older than the
        # newest ``limit`` (lowest-scored first), removed in one call.
        pipe.zremrangebyrank(recent_key, 0, -(settings.recent_runs_limit + 1))
        pipe.expire(recent_key, settings.result_ttl_seconds)
        if user_id is not None:
            # The per-identity index mirrors the shared index's shape exactly (same
            # bound, same TTL) so a restricted caller's own window stays complete.
            user_key = self.recent_key(tool_name, user_id)
            pipe.zadd(user_key, {run_id: score})
            pipe.zremrangebyrank(user_key, 0, -(settings.recent_runs_limit + 1))
            pipe.expire(user_key, settings.result_ttl_seconds)
        await pipe.execute()

    async def refresh_liveness(self, r: Any, run_id: str, ttl: int) -> None:
        await r.set(self.liveness_key(run_id), "1", ex=ttl)

    async def mark_terminal_if_running(self, r: Any, run_id: str, fields: dict[str, str], ttl: int) -> bool:
        """Compare-and-set terminal write: write ``fields`` (the new ``status`` +
        ``finished_at`` and any ``result``/``error``) and refresh the record TTL,
        but ONLY while the stored ``status`` is still ``running`` — enforcing the
        one-way ``lost`` invariant atomically (see ``_TERMINAL_CAS_LUA``). Returns
        ``True`` when this call performed the transition, ``False`` when the record
        was no longer ``running`` (another writer reached a terminal state first)."""
        flat: list[Any] = []
        for field, value in fields.items():
            flat.extend((field, value))
        written = await r.eval(_TERMINAL_CAS_LUA, 1, self.run_key(run_id), ttl, *flat)
        return bool(written)

    # -- reads ---------------------------------------------------------------

    async def get_run(self, r: Any, run_id: str) -> dict[str, str] | None:
        record = await r.hgetall(self.run_key(run_id))
        return record or None

    async def get_runs(self, r: Any, run_ids: list[str]) -> list[dict[str, str] | None]:
        """Batch ``HGETALL`` for many run ids in ONE pipeline, aligned to the input
        order; a vanished record maps to ``None`` (no per-id N+1)."""
        if not run_ids:
            return []
        pipe = r.pipeline()
        for run_id in run_ids:
            pipe.hgetall(self.run_key(run_id))
        return [record or None for record in await pipe.execute()]

    async def liveness_present(self, r: Any, run_id: str) -> bool:
        return await r.get(self.liveness_key(run_id)) is not None

    async def liveness_present_many(self, r: Any, run_ids: list[str]) -> list[bool]:
        """Batch liveness-key ``GET`` for many run ids in ONE pipeline, aligned to
        the input order; each entry is ``True`` when that run's liveness key is
        present."""
        if not run_ids:
            return []
        pipe = r.pipeline()
        for run_id in run_ids:
            pipe.get(self.liveness_key(run_id))
        return [value is not None for value in await pipe.execute()]

    async def recent_run_ids(self, r: Any, tool_name: str, limit: int, user_id: str | None = None) -> list[str]:
        # Highest score (most recent start) first. With ``user_id`` given, reads the
        # caller's per-identity index; without it, the shared index.
        return await r.zrevrange(self.recent_key(tool_name, user_id), 0, limit - 1)

    async def prune_recent(self, r: Any, tool_name: str, run_id: str, user_id: str | None = None) -> None:
        # Prune the SAME index the list read: the per-identity index for a restricted
        # caller (``user_id`` given), the shared index otherwise — so an expired entry
        # is never pruned from the wrong index.
        await r.zrem(self.recent_key(tool_name, user_id), run_id)


# -- supervisor --------------------------------------------------------------


def _spawn_supervisor(run_id: str, tool_name: str, arguments: dict[str, Any]) -> None:
    task = asyncio.create_task(_supervise(run_id, tool_name, arguments))
    _SUPERVISORS.add(task)
    task.add_done_callback(lambda t: _on_supervisor_done(t, run_id, tool_name))


def _on_supervisor_done(task: asyncio.Task[None], run_id: str, tool_name: str) -> None:
    """Done-callback for a supervisor task: drop the strong reference AND surface a
    failure at completion time.

    The supervisor's inner ``try`` persists a tool that raises as a ``failed``
    record, but a failure BEFORE it (e.g. the ``client_ctx`` enter raising because
    Redis died after submit) escapes that guard — asyncio would then report it only
    via the nondeterministic 'never retrieved' message at GC. Logging it here with
    the ``run_id``/``tool_name`` makes it a timely, attributable signal. A
    cancellation (test teardown / shutdown) is the normal stop and stays silent."""
    global _ACTIVE_RUNS
    _SUPERVISORS.discard(task)
    # Release the concurrency slot the submit door reserved for this run. Every
    # spawned supervisor reaches this callback exactly once, so the count returns
    # to the submit door's increment.
    _ACTIVE_RUNS -= 1
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("tool-run %s (%s) supervisor task failed", run_id, tool_name, exc_info=exc)


async def _refresh_liveness_loop(r: Any, store: ToolRunStore, run_id: str, settings: ToolRunsSettings) -> None:
    """Re-set the run's liveness key every ``liveness_ttl_seconds / 3`` — a
    constant cadence — so a live run (including a slow sync tool offloaded to a
    thread, which leaves the loop free to run this task) never looks ``lost``.

    A transient failure of a single refresh is logged and the loop CONTINUES: one
    failed ``SET`` must never stop the refresher, or a still-``running`` run would
    lose liveness while alive and be wrongly reconciled to ``lost``."""
    cadence = settings.liveness_ttl_seconds / 3
    while True:
        try:
            await store.refresh_liveness(r, run_id, settings.liveness_ttl_seconds)
        except Exception:
            # Loud, not silent: log and keep refreshing on the next cadence rather
            # than letting one failed SET permanently kill the refresher.
            logger.warning("tool-run %s liveness refresh failed; retrying next cadence", run_id, exc_info=True)
        await asyncio.sleep(cadence)


async def _supervise(run_id: str, tool_name: str, arguments: dict[str, Any]) -> None:
    settings = tool_runs_settings()
    store = ToolRunStore(settings.key_prefix)
    async with client_ctx(RedisClient, settings.redis) as r:
        refresher = asyncio.create_task(_refresh_liveness_loop(r, store, run_id, settings))
        try:
            try:
                result = await tai_app.tools.run_tool(tool_name, arguments, offload_sync=True)
                # ``run_tool`` already json-normalizes the body; a residual dumps
                # failure surfaces as a ``failed`` record rather than a lost run.
                result_json = json.dumps(result)
            except asyncio.CancelledError:
                # Server shutdown cancelled this run mid-flight. Record it as
                # ``failed`` through the same one-way CAS the normal path uses (so a
                # record already reconciled to ``lost`` is never overwritten), then
                # re-raise so the cancellation propagates to the drain handler. Safe
                # to await here: the drain cancels each task exactly once, then waits.
                fields = {
                    "status": _FAILED,
                    "finished_at": _now().isoformat(),
                    "error": "server shutdown before the tool-run completed",
                }
                persisted = await store.mark_terminal_if_running(r, run_id, fields, settings.result_ttl_seconds)
                if not persisted:
                    logger.warning(
                        "tool-run %s (%s) was cancelled at shutdown but the record was already "
                        "reconciled to lost; terminal write skipped (one-way lost)",
                        run_id,
                        tool_name,
                    )
                raise
            except Exception as exc:
                # No open request to propagate to — persist the raised error as
                # record data so the requester reads it. Logged too; never dropped.
                logger.exception("tool-run %s (%s) failed", run_id, tool_name)
                fields = {"status": _FAILED, "finished_at": _now().isoformat(), "error": str(exc)}
            else:
                fields = {"status": _SUCCEEDED, "finished_at": _now().isoformat(), "result": result_json}
            # Gate the terminal write on the record still being ``running`` so it
            # can never overwrite a ``lost`` a reader already wrote (one-way lost).
            persisted = await store.mark_terminal_if_running(r, run_id, fields, settings.result_ttl_seconds)
            if not persisted:
                logger.warning(
                    "tool-run %s (%s) finished as %s but the record was already reconciled to lost; "
                    "terminal write skipped (one-way lost)",
                    run_id,
                    tool_name,
                    fields["status"],
                )
        finally:
            refresher.cancel()
            with suppress(asyncio.CancelledError):
                await refresher


@tai_app.lifecycle.on_shutdown
async def _drain_supervisors() -> None:
    """Cancel every in-flight supervisor at shutdown and wait, bounded, for each to
    write its terminal ``failed``/shutdown record.

    Shutdown handlers run BEFORE ``_teardown_resources`` closes the pooled clients,
    so a cancelled supervisor still has a live Redis to write through. The wait is
    bounded by ``shutdown_drain_seconds`` so teardown always proceeds; a supervisor
    that misses the window is logged loudly and its run reconciles to ``lost`` later
    — an explicit, logged recovery, never a silent one."""
    tasks = [t for t in _SUPERVISORS if not t.done()]
    if not tasks:
        return
    for t in tasks:
        t.cancel()
    _done, pending = await asyncio.wait(tasks, timeout=tool_runs_settings().shutdown_drain_seconds)
    if pending:
        logger.error(
            "tool-runs shutdown: %d supervisor(s) did not finish their terminal write within "
            "TAI_TOOL_RUNS_SHUTDOWN_DRAIN_SECONDS; those runs will reconcile to lost",
            len(pending),
        )


async def _reconcile_lost_with_liveness(
    r: Any, store: ToolRunStore, run_id: str, record: dict[str, str], liveness_present: bool, ttl: int
) -> dict[str, str]:
    """Persist ``lost`` one way when a still-``running`` ``record`` has lost its
    liveness key (``liveness_present`` is ``False``) — a dead supervisor's
    ``finally`` never wrote a terminal record. The write is a compare-and-set
    gated on ``running`` (``mark_terminal_if_running``): should the supervisor's
    own terminal write land between this reader's GET and the CAS, the CAS is
    rejected and the real terminal record is re-read rather than reporting a stale
    ``lost``. A live run keeps its liveness key, so it is never reconciled."""
    if record.get("status") != _RUNNING or liveness_present:
        return record
    finished_at = _now().isoformat()
    if await store.mark_terminal_if_running(r, run_id, {"status": _LOST, "finished_at": finished_at}, ttl):
        return {**record, "status": _LOST, "finished_at": finished_at}
    # The supervisor reached a terminal state first — reflect the real record.
    return await store.get_run(r, run_id) or record


async def _reconcile_lost(r: Any, store: ToolRunStore, run_id: str, record: dict[str, str], ttl: int) -> dict[str, str]:
    """Single-record ``lost`` reconciliation for the GET-by-id door: read the run's
    liveness only while it is still ``running`` (a terminal record is never
    reconciled), then apply ``_reconcile_lost_with_liveness``."""
    liveness_present = record.get("status") == _RUNNING and await store.liveness_present(r, run_id)
    return await _reconcile_lost_with_liveness(r, store, run_id, record, liveness_present, ttl)


# -- response views ----------------------------------------------------------


def _run_view(run_id: str, record: dict[str, str]) -> dict[str, Any]:
    """The full GET view; ``result`` is parsed back from its stored JSON."""
    view: dict[str, Any] = {
        "run_id": run_id,
        "tool_name": record["tool_name"],
        "status": record["status"],
        "started_at": record["started_at"],
    }
    if "finished_at" in record:
        view["finished_at"] = record["finished_at"]
    if "result" in record:
        view["result"] = json.loads(record["result"])
    if "error" in record:
        view["error"] = record["error"]
    return view


def _list_view(run_id: str, record: dict[str, str]) -> dict[str, Any]:
    """The list view — id/tool name/status/timestamps only, never ``result``/``error``."""
    view: dict[str, Any] = {
        "run_id": run_id,
        "tool_name": record["tool_name"],
        "status": record["status"],
        "started_at": record["started_at"],
    }
    if "finished_at" in record:
        view["finished_at"] = record["finished_at"]
    return view


# -- operations --------------------------------------------------------------


@operation(
    name="submit_run",
    summary="Submit a tool for background execution",
    tags=["tool-runs"],
    destructive=True,
    reload_gated=True,
    meta_executor=True,
    errors=[BadRequestError, NotFoundError, UnavailableError],
    request_model=ToolRunSubmission,
)
async def submit_run(tool_name: str, arguments: dict[str, object]) -> dict:
    """Submit a tool for background execution — returns ``202 {run_id}`` at once
    and runs the tool through the same seam the sync door uses."""
    settings = tool_runs_settings()
    store = ToolRunStore(settings.key_prefix)

    # Resolve the name against the live registry BEFORE creating a record: an
    # unknown tool is a loud 404 up front, so a typo'd name never earns a
    # ``running`` record that the supervisor would only later fail — keeping a
    # real runtime failure distinguishable from a bad request and the store clean.
    tools = await tai_app.tools.get_tools()
    if tool_name not in tools:
        raise NotFoundError(f"unknown tool: {tool_name}")

    # Per-worker concurrency cap: check + increment are synchronous (no await
    # between them) so two concurrent submits cannot both pass the check before
    # either reserves its slot. The slot is released by the supervisor's
    # done-callback, or by the ``except`` below if the record write fails.
    global _ACTIVE_RUNS
    if settings.max_concurrent_runs <= _ACTIVE_RUNS:
        raise UnavailableError(
            f"tool-run capacity reached ({settings.max_concurrent_runs} concurrent runs); "
            "retry later or raise TAI_TOOL_RUNS_MAX_CONCURRENT_RUNS"
        )
    _ACTIVE_RUNS += 1

    # The owning identity of this run is always the caller's own id — each key is its
    # own island. Stamped on the record and used to build the per-identity index. An
    # unauthenticated caller leaves it None, so only the shared index is written.
    user_id, _restricted = request_identity()
    owning_identity = user_id

    run_id = secrets.token_urlsafe(16)
    started = _now()
    try:
        async with client_ctx(RedisClient, settings.redis) as r:
            await store.create_run(
                r, run_id, tool_name, started.isoformat(), started.timestamp(), settings, user_id=owning_identity
            )
        _spawn_supervisor(run_id, tool_name, arguments)
    except Exception:
        # The record never became a live run (no supervisor owns the slot), so
        # return the reserved slot here and re-raise loudly.
        _ACTIVE_RUNS -= 1
        raise
    return {"run_id": run_id}


@operation(
    name="get_run",
    summary="Get a background tool run's status and result",
    tags=["tool-runs"],
    errors=[ForbiddenError, NotFoundError],
)
async def get_run(run_id: str) -> dict:
    settings = tool_runs_settings()
    store = ToolRunStore(settings.key_prefix)
    _user_id, restricted = request_identity()
    async with client_ctx(RedisClient, settings.redis) as r:
        record = await store.get_run(r, run_id)
        if record is None:
            raise NotFoundError(f"run {run_id!r} not found")
        # A restricted caller may read only a run owned by its OWN identity. A mismatch
        # (including a run whose ``user_id`` is absent — owned by no identity) is a
        # loud ``403`` naming the denial, NEVER a ``404`` — the run exists under an id
        # namespace of unguessable tokens, so an honest denial leaks nothing
        # actionable while a ``404`` would lie about existence.
        if restricted is not None and record.get("user_id") != restricted:
            raise ForbiddenError("run belongs to another identity")
        record = await _reconcile_lost(r, store, run_id, record, settings.result_ttl_seconds)
    return _run_view(run_id, record)


@operation(
    name="list_tool_runs",
    summary="List background tool runs for a tool",
    tags=["tool-runs"],
    errors=[BadRequestError],
)
async def list_tool_runs(tool_name: str) -> list[dict]:
    """List the recent runs for one tool, newest first.

    A restricted caller reads its OWN per-identity index (complete within its own
    bound — never truncated by other identities' volume that saturates the shared
    window); an unrestricted caller reads the shared index unchanged. An empty list
    is the honest answer to "my runs of this tool" — this filters a collection to the
    caller's own slice (distinct from GET-by-id, which raises ``403`` for a NAMED run
    owned by another identity)."""
    settings = tool_runs_settings()
    store = ToolRunStore(settings.key_prefix)
    _user_id, restricted = request_identity()
    entries: list[dict[str, Any]] = []
    async with client_ctx(RedisClient, settings.redis) as r:
        run_ids = await store.recent_run_ids(r, tool_name, settings.recent_runs_limit, user_id=restricted)
        # One pipeline for every record hash — no per-id N+1 of HGETALLs.
        records = await store.get_runs(r, run_ids)
        present: list[tuple[str, dict[str, str]]] = []
        for run_id, record in zip(run_ids, records, strict=True):
            if record is None:
                # The record hash expired out from under the index — prune the
                # phantom from the SAME index that was read (per-identity for a
                # restricted caller, shared otherwise) so the list doesn't carry a
                # vanished run.
                await store.prune_recent(r, tool_name, run_id, user_id=restricted)
                continue
            present.append((run_id, record))
        # One pipeline for the liveness keys of only the still-``running`` subset;
        # a terminal record is never reconciled and needs no liveness read.
        running_ids = [run_id for run_id, record in present if record.get("status") == _RUNNING]
        liveness = dict(zip(running_ids, await store.liveness_present_many(r, running_ids), strict=True))
        for run_id, record in present:
            record = await _reconcile_lost_with_liveness(
                r, store, run_id, record, liveness.get(run_id, True), settings.result_ttl_seconds
            )
            entries.append(_list_view(run_id, record))
    return entries
