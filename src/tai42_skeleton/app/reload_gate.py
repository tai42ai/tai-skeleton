"""The process-wide reload gate — the mutual-exclusion seam between an admin
config reload and the live serving surface.

A reload (`admin.reload_config` / `admin.reload_mcp` / the failed-MCP re-probe)
is heavy and synchronous: it re-reads env, resets settings caches, re-imports
the manifest modules and re-probes every MCP server. Running it inline on the
serving loop freezes health checks, MCP keepalives and in-flight SSE streams for
its whole duration. The gate replaces that freeze:

* :meth:`ReloadGate.run` holds an :class:`asyncio.Lock` and awaits the heavy sync
  body on a worker thread (`asyncio.to_thread`), so the serving loop keeps
  running while the reload executes;
* while the lock is held, routes that dispatch a run or mutate the live
  tool/agent registries answer :meth:`reject_response` (HTTP 503, retriable), and
  the FastMCP session surface rejects tool calls the same way. Health, liveness
  and read-only GETs keep answering.

The lock is owned by the serving loop. A single process runs a single serving
loop, so the lock is created once (on first use, or explicitly via
:meth:`bind_to_running_loop` at lifespan startup) and bound to that loop for the
process lifetime; an off-serving-loop caller reaches the gate by scheduling a
coroutine onto the serving loop (``asyncio.run_coroutine_threadsafe``), never by
acquiring the lock from a foreign loop. The raw lock is exposed via :attr:`lock`
for an async-side holder (a re-probe pass that must block reloads while it runs).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TypeVar

from starlette.responses import JSONResponse

T = TypeVar("T")

# The single retriable-rejection message, shared by the HTTP 503 body and the
# FastMCP session-tool rejection so a client sees one consistent surface.
REJECT_MESSAGE = "reloading — the server is applying a config reload; retry shortly"


class ReloadGate:
    """Process-wide reload lock plus its retriable-rejection response.

    Instantiated once per process (the module-level :data:`reload_gate`
    singleton). The lock is bound to the serving loop; because a process runs a
    single serving loop it is created once and reused for the process lifetime.
    """

    def __init__(self) -> None:
        self._lock: asyncio.Lock | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def _serving_lock(self) -> asyncio.Lock:
        """The lock bound to the running serving loop, created on first use.

        A process runs a single serving loop, so this binds exactly once. It
        rebinds only if the running loop differs from the bound one — the loop
        was replaced (there is never a second live loop in the same process to
        share the lock with), which keeps the process-wide singleton correct
        across a loop swap rather than raising the cross-loop ``RuntimeError`` an
        ``asyncio.Lock`` would.
        """
        loop = asyncio.get_running_loop()
        if self._lock is None or self._loop is not loop:
            self._lock = asyncio.Lock()
            self._loop = loop
        return self._lock

    def bind_to_running_loop(self) -> None:
        """Bind the lock to the current running loop — the lifespan-startup hook so
        the gate is owned by the serving loop from the first request rather than
        lazily on the first reload."""
        self._serving_lock()

    @property
    def lock(self) -> asyncio.Lock:
        """The raw serving-loop lock, for an async-side holder that must block
        reloads while it runs (``async with reload_gate.lock: ...``)."""
        return self._serving_lock()

    @property
    def locked(self) -> bool:
        """Whether a reload is in progress. ``locked()`` reads a plain flag and
        never touches the loop, so a route entry check is safe before the lock has
        ever been bound (no reload has run yet)."""
        return self._lock is not None and self._lock.locked()

    async def run(self, fn: Callable[[], T]) -> T:
        """Hold the reload lock and await ``fn`` on a worker thread.

        The heavy reload bodies stay synchronous and unchanged; only the acquire
        and the thread offload live here, so the serving loop keeps running while
        the reload executes.
        """
        async with self._serving_lock():
            return await asyncio.to_thread(fn)

    def reject_response(self) -> JSONResponse:
        """The retriable 503 a gated route returns while a reload holds the lock —
        explicit and loud so a client/CLI can branch on ``reloading``."""
        return JSONResponse(
            {"error": REJECT_MESSAGE, "reloading": True},
            status_code=503,
            headers={"Retry-After": "5"},
        )


# The one process-wide gate. Routers import this singleton and check
# ``reload_gate.locked`` at entry; reload callers await ``reload_gate.run(...)``.
reload_gate = ReloadGate()
