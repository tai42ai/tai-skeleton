import asyncio
import logging
import re
import threading
from collections.abc import AsyncIterator
from concurrent.futures import CancelledError, Future
from contextlib import asynccontextmanager, suppress

from fastmcp import FastMCP
from fastmcp.server.http import create_sse_app
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.types import ASGIApp
from tai_contract.sub_mcp import RouteConfig

from tai_skeleton.middleware.body_limit import BodyLimitMiddleware
from tai_skeleton.sub_mcp.store import get_sub_mcp_store

logger = logging.getLogger(__name__)

ROOT_PREFIX = "/app"

# A slug is dispatched as ONE path segment, so it must be a single lowercase-safe
# segment: a ``/`` (or a trailing newline) would mint a route that is unreachable
# and undeletable. Validated here at the core so EVERY caller — the HTTP router
# AND the backup-restore path — is checked, not just the route. ``\Z`` (not ``$``)
# anchors the true end of string so a trailing ``\n`` cannot slip through.
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*\Z")

# The transports the dispatcher + build path support end to end. Owned here at the
# core (where the build path consumes it) and validated at registration so an
# unknown transport is rejected loudly rather than silently built as ``http``; the
# HTTP router imports this same set for its request-shape 400.
_VALID_TRANSPORTS = ("http", "sse", "stdio")


def validate_registration(slug: str, transport: str) -> None:
    """Reject a malformed slug or an unknown transport loudly.

    The single validation the router core and the durable write service (which
    validates BEFORE its store write, so invalid input never persists) share, so a
    slug that would mint an unreachable/undeletable route, or a transport the build
    path cannot serve, is caught in exactly one place.
    """
    if not _SLUG_RE.match(slug):
        raise ValueError(f"slug {slug!r} must match {_SLUG_RE.pattern} (one lowercase path segment)")
    if transport not in _VALID_TRANSPORTS:
        raise ValueError(f"transport {transport!r} must be one of {_VALID_TRANSPORTS}")


class _SubAppLifespan:
    """Runs a sub-app's fastmcp lifespan (enter AND exit) inside ONE dedicated task.

    fastmcp's server sets a ``_current_server`` ContextVar with a token when its
    lifespan is entered and ``token.reset()``s it when the lifespan exits. A
    ContextVar token can only be reset in the SAME ``contextvars.Context`` it was
    set in, and every asyncio task runs in its own copied context. A sub-app
    lifespan is entered LAZILY on the first request (that request task's context)
    but torn down LATER on a ``DELETE`` (a different request task's context, even
    on the same loop); resetting the token from that different context raises
    ``ValueError: <Token ...> was created in a different Context``, which surfaces
    as ``RuntimeError: Unexpected ASGI message 'http.response.start' sent, after
    response already completed`` and a 500 on the DELETE.

    Holding the whole ``async with sub_app.lifespan(...)`` open inside one
    long-lived task keeps enter and exit in the SAME context: the task enters the
    lifespan, parks until asked to stop, then exits the lifespan — all in its own
    context, so the token reset is always valid. Requests are served from other
    tasks against the same session-manager task group; only the enter/exit contexts
    live in the dedicated task. The task is created on, and ``aclose()`` is awaited
    on, the router's owner loop (the same loop that serves the sub-app), keeping the
    lifespan enter and close on that one loop.
    """

    def __init__(self, sub_app: ASGIApp) -> None:
        self._sub_app = sub_app
        self._started = asyncio.Event()
        self._stop = asyncio.Event()
        # Holds an enter failure (raised by start) OR a teardown failure (raised by
        # aclose): the run task stores it here and re-raises so the awaiter sees it.
        self._error: BaseException | None = None
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Enter the lifespan in a dedicated task and block until it is live.

        Raises if the lifespan enter failed, by which point the run task has already
        unwound the half-opened lifespan — so a caller need not clean up after a
        failed ``start()``. A cancellation while waiting for the enter (e.g. the
        request driving a lazy build is cancelled) cancels and reaps the dedicated
        task before re-raising, so its half-opened lifespan is unwound and the task
        is never orphaned — the caller holds no reference to reap it once ``start()``
        raises."""
        self._task = asyncio.create_task(self._run())
        try:
            await self._started.wait()
        except BaseException:
            # Cancelled (or otherwise interrupted) before the lifespan came live:
            # the dedicated task is not tied to this awaiter's cancellation, so cancel
            # and await it here to unwind its (possibly half-opened) lifespan rather
            # than leaking it parked on _stop forever, then re-raise.
            self._task.cancel()
            with suppress(BaseException):
                await self._task
            raise
        if self._error is not None:
            # Enter failed; the run task has already finished unwinding. Awaiting it
            # re-raises the failure and consumes the task's exception so it is not
            # reported as never-retrieved.
            await self._task

    async def _run(self) -> None:
        try:
            async with self._sub_app.lifespan(self._sub_app):
                self._started.set()
                await self._stop.wait()
        except BaseException as exc:
            # An enter failure (before _started is set) OR a teardown failure. Record
            # it, unblock a start() still waiting on enter, and re-raise so the task
            # carries the exception for aclose()/start() to surface.
            self._error = exc
            self._started.set()
            raise

    async def aclose(self) -> None:
        """Signal the lifespan task to exit and await it, re-raising any teardown
        failure. Idempotent — a second call just re-observes the finished task."""
        if self._task is None:
            return
        self._stop.set()
        # Await the run task so the lifespan's ``__aexit__`` (its ContextVar reset)
        # runs to completion in the task/context that entered it. A teardown failure
        # re-raises here (the run task carries it), preserving the "a failed teardown
        # is never reported as clean" contract the callers rely on.
        await self._task


class SubMcpAppRouter:
    """Per-worker cache of the durable sub-MCP registration store.

    The store (``tai_skeleton.sub_mcp.store``) is the source of truth for every
    ``slug -> RouteConfig`` binding; this router holds an in-process cache of it,
    rehydrated at boot/reload and repopulated on demand from the store when an
    unknown slug is dispatched (the cross-worker read path). Sub-MCP routing state
    is uvicorn-worker-scoped: only these HTTP/MCP workers serve ``/app/{slug}``, so
    only they rehydrate/serve routes; backend workers do not participate and the
    worker bus is not the propagation mechanism. Documented residual: a slug
    DELETED on a sibling stays served by workers that already built it until their
    next reload (a stale-positive; the store-backed ``GET`` list is already correct).
    """

    def __init__(self, app):
        self._app = app
        self._routes: dict[str, RouteConfig] = {}
        # stdio sub-apps have no ASGI surface, so a slug can cache as None and the
        # caller serves a 404 for it.
        self._server_cache: dict[str, ASGIApp | None] = {}
        self._app_exit_stacks: dict[str, _SubAppLifespan] = {}
        # A globally-monotonic build generation and the per-slug token stamped from
        # it on every register/replace. The token is captured with the config at the
        # start of a build and re-checked at the end: if it changed, a concurrent
        # replace superseded the config the build ran against, so the just-built app
        # is discarded rather than cached (see _get_or_build_app). GLOBAL monotonicity
        # (a fresh int per register/replace across the whole router, never reset to a
        # per-slug base) is what makes a token unrepeatable even across a reset()-then-
        # re-register — a plain per-slug counter would collide there and cache a stale
        # build. Both are guarded by the same _state_lock discipline as the dicts.
        self._generation: int = 0
        self._route_generations: dict[str, int] = {}
        # A loop-agnostic threading.Lock guards every mutation of the dicts +
        # generation counter above and every read that spans more than one dict
        # operation. A single
        # atomic ``dict.get`` on the dispatch fast path may read lock-free (a
        # stale hit or miss just yields a clean 404). The reload path drives
        # register/unregister/reset from a throwaway loop while _get_or_build_app
        # serves requests on the owner loop, so the guard must NOT be an
        # asyncio.Lock (which binds to one loop and raises when contended
        # cross-loop). It is only ever held for quick, synchronous dict work,
        # never across an await.
        self._state_lock = threading.Lock()
        # Serializes the async build of a sub-app so two owner-loop requests can't
        # build the same slug at once. Only _get_or_build_app (owner loop) takes
        # it, so it stays single-loop and safe as an asyncio.Lock.
        self._build_lock = asyncio.Lock()
        # The loop the router's lifespan runs on — the same loop that lazily
        # builds each sub-app and enters its lifespan. A sub-app's exit stack MUST
        # be closed on this loop, even when the reload path drives register /
        # reset from a throwaway loop, or the sub-app's task group is torn down
        # cross-loop.
        self._owner_loop: asyncio.AbstractEventLoop | None = None

    @property
    def root_prefix(self):
        return ROOT_PREFIX

    @property
    def routes(self) -> dict[str, RouteConfig]:
        return self._routes

    async def register_sub_mcp_app(self, slug: str, tools: list[str], transport: str = "http"):
        # Validate the slug shape + transport at the core so every caller is guarded
        # (the HTTP router maps this to a 400; the backup-restore path records it as
        # a per-section error). A malformed slug is rejected loudly here rather than
        # minting a route that the dispatcher can never reach or delete.
        validate_registration(slug, transport)
        # Reloading a live slug: drop its route/cache and grab its lifespan stack
        # under the state lock, then close the stale stack OUTSIDE the lock (the
        # close awaits and must not run under the non-async threading lock). Stamp a
        # fresh globally-monotonic generation token so an in-flight build of the
        # prior config detects it was superseded and does not cache a stale app.
        with self._state_lock:
            stale = self._pop_slug_locked(slug) if (slug in self._routes or slug in self._server_cache) else None
            if stale is not None or slug in self._routes:
                logger.info(f"Reloading MCP app: {slug}")
            self._routes[slug] = RouteConfig(tools=tools, transport=transport)
            self._generation += 1
            self._route_generations[slug] = self._generation
        if stale is not None:
            await self._aclose_slug_stack(slug, stale)

    async def unregister_sub_mcp_app(self, slug: str):
        with self._state_lock:
            stale = self._pop_slug_locked(slug)
        if stale is not None:
            await self._aclose_slug_stack(slug, stale)

    def _pop_slug_locked(self, slug: str) -> _SubAppLifespan | None:
        """Remove a slug's route + cached app + generation token and return its
        lifespan runner (or None) for the caller to close off-lock. Caller holds
        ``_state_lock``.

        Dropping the generation token here means an in-flight build whose captured
        token no longer matches (the token is gone) treats the slug as vanished and
        discards its build. The stack is popped here, so a close failure below can
        never leave a re-closable/leaked entry behind — the entry is already gone."""
        self._routes.pop(slug, None)
        self._server_cache.pop(slug, None)
        self._route_generations.pop(slug, None)
        return self._app_exit_stacks.pop(slug, None)

    async def _aclose_slug_stack(self, slug: str, stack: _SubAppLifespan) -> None:
        try:
            await self._aclose_on_owner(slug, stack)
        except Exception as e:
            # Log for the shutdown trail, then re-raise: a sub-app whose teardown
            # failed on the inline (owner-loop) branch must never be reported as
            # cleanly removed. The cross-loop branch never raises here — it schedules
            # the close and surfaces any failure through its done-callback instead.
            logger.error(f"Error shutting down MCP app {slug}: {e}")
            raise

    async def _aclose_on_owner(self, slug: str, stack: _SubAppLifespan) -> None:
        """Close a per-slug lifespan stack on the loop that entered it.

        The stack's lifespan (and its task group) was entered on the owner loop, so
        closing it must happen there or the task group is torn down cross-loop. Two
        cases:

        * On the owner loop (or with no live owner loop — a direct test): close
          inline and let any failure RAISE, so a caller that reports a slug removed
          (the HTTP ``DELETE`` path) never reports a failed teardown as clean.
        * On a DIFFERENT loop while the owner loop is live: the reload path drives
          register/unregister from a throwaway loop while the owner loop's thread is
          PARKED in ``_run_blocking`` waiting on that same reload. A block-wait on the
          owner loop's future would deadlock — the parked owner loop can never run the
          scheduled close. So schedule the close onto the owner loop and surface any
          failure through a logged done-callback, WITHOUT waiting on the future.
        """
        loop = self._owner_loop
        try:
            current = asyncio.get_running_loop()
        except RuntimeError:
            current = None
        if loop is not None and loop.is_running() and loop is not current:
            future = asyncio.run_coroutine_threadsafe(stack.aclose(), loop)
            future.add_done_callback(lambda f, s=slug: self._log_teardown_result(s, f))
        else:
            await stack.aclose()

    def reset(self) -> None:
        """Drop all routes + cached sub-apps and tear down their lifespans.

        Called on every start()/reload so a re-init stops serving sub-apps from
        the previous generation. Each stale sub-app's lifespan is closed on the
        loop it was entered on, and reset never block-waits a running loop: when a
        loop is live the close is SCHEDULED onto it — via ``create_task`` when that
        is reset's own loop, via ``run_coroutine_threadsafe`` when it is a different
        running loop — with a logged completion callback, so a teardown failure
        still surfaces rather than being lost. Only when no loop is live at all is
        the close driven inline on a one-shot loop.
        """
        with self._state_lock:
            self._routes.clear()
            self._server_cache.clear()
            # Bump the global generation and drop every per-slug token: an in-flight
            # build whose captured token no longer matches discards its stale build.
            self._generation += 1
            self._route_generations.clear()
            stacks = list(self._app_exit_stacks.items())
            self._app_exit_stacks.clear()

        owner = self._owner_loop
        try:
            current = asyncio.get_running_loop()
        except RuntimeError:
            current = None
        # A stack was entered on the owner loop; close it there. Fall back to the
        # current running loop when no owner loop is live (e.g. a direct test).
        close_loop = owner if (owner is not None and owner.is_running()) else current

        errors: list[Exception] = []
        for slug, stack in stacks:
            try:
                if close_loop is None:
                    # No running loop at all — safe to drive a one-shot loop.
                    asyncio.run(stack.aclose())
                elif close_loop is current:
                    # On the close loop's own thread: it can't block-wait for its
                    # own coroutine, so schedule it and surface any failure via a
                    # logged completion callback.
                    task = close_loop.create_task(stack.aclose())
                    task.add_done_callback(lambda t, s=slug: self._log_teardown_result(s, t))
                else:
                    # Owner loop is a different running loop: SCHEDULE the close onto
                    # it and never block-wait — a ``.result()`` here would deadlock if
                    # that loop is parked. Failures surface via the logged callback.
                    future = asyncio.run_coroutine_threadsafe(stack.aclose(), close_loop)
                    future.add_done_callback(lambda f, s=slug: self._log_teardown_result(s, f))
            except Exception as e:
                logger.error("Error tearing down sub-MCP app %s on reset: %s", slug, e)
                errors.append(e)
        if errors:
            raise ExceptionGroup("sub-MCP reset teardown failures", errors)

    @staticmethod
    def _log_teardown_result(slug: str, task: asyncio.Task | Future) -> None:
        # Shared done-callback for a scheduled (never awaited) teardown, on both an
        # ``asyncio.Task`` (reset's same-loop schedule) and a
        # ``concurrent.futures.Future`` (a cross-loop ``run_coroutine_threadsafe``
        # schedule). Both expose ``exception()`` returning the failure or ``None``,
        # so a teardown failure is surfaced loudly rather than lost. A cancelled
        # teardown returns quietly — and the two carriers raise DIFFERENT, unrelated
        # cancellation types (a Task raises ``asyncio.CancelledError``, a Future
        # ``concurrent.futures.CancelledError``), so both are caught here.
        try:
            exc = task.exception()
        except (CancelledError, asyncio.CancelledError):
            return
        if exc is not None:
            logger.error("Error tearing down sub-MCP app %s: %s", slug, exc)

    async def _build_sub_app(self, slug: str, config: RouteConfig) -> tuple[ASGIApp | None, _SubAppLifespan | None]:
        """Build the ASGI sub-app for ``slug`` from the CAPTURED ``config`` and enter
        its lifespan, returning ``(sub_app, lifespan)``.

        Pure build — it reads NO router state (the caller captured ``config`` with
        the generation token and decides, after the build, whether to record or
        discard it). ``stdio`` has no ASGI surface, so it returns ``(None, None)``;
        the caller caches that ``None`` for the slug. The lifespan is entered inside a
        dedicated task (see :class:`_SubAppLifespan`) so its fastmcp ``_current_server``
        ContextVar token is reset in the SAME context at teardown; on a lifespan-enter
        failure that task has already unwound it and ``start()`` re-raises.
        """
        mcp = FastMCP(f"{slug}-MCP")

        # Load tools
        for name in config.tools:
            mcp.add_tool(await self._app.tools.get_tool(name))

        # Tool-edge authorization: each sub-MCP mount builds its OWN FastMCP with
        # no fastmcp middleware, so the main server's ``AuthzMiddleware`` never
        # reaches it — re-add it here (via ``add_middleware``, a fastmcp
        # ``on_call_tool`` middleware, NOT the Starlette ``sub_middleware`` list
        # below, where it would never fire) so a projected op reached through the
        # sub-mcp mount is authorized identically to the main edge.
        from tai_skeleton.authz.middleware import AuthzMiddleware

        mcp.add_middleware(AuthzMiddleware(self._app))

        if config.transport == "stdio":
            return None, None

        # Each sub-app is a full Starlette app with its OWN ServerErrorMiddleware, so
        # the base app's body-size cap (outside the mount) cannot convert an over-cap
        # escape on a sub-MCP route into a 413 — that inner error handler would commit
        # a 500 first. Give every sub-app the cap inside its own stack, as the base app
        # does, so a streamed over-cap body on /app/{slug}/... answers 413 too.
        sub_middleware = [Middleware(BodyLimitMiddleware)]
        if config.transport == "sse":
            clean_root = self.root_prefix.rstrip("/")
            base_path = f"{clean_root}/{slug}"
            sub_app = create_sse_app(
                mcp, message_path=f"{base_path}/messages", sse_path=f"{base_path}/sse", middleware=sub_middleware
            )
        else:
            # Serve the streamable-HTTP endpoint at the sub-app ROOT. The
            # dispatcher rewrites an ``/app/{slug}`` request to sub-app scope path
            # ``/`` (see ``__call__``), so the endpoint must live at ``/`` — the
            # FastMCP default of ``/mcp`` would leave the rewritten ``/`` request
            # 404ing against the sub-app.
            sub_app = mcp.http_app(path="/", middleware=sub_middleware)

        app_lifespan = _SubAppLifespan(sub_app)
        # start() enters the lifespan in its dedicated task and blocks until it is
        # live, re-raising (with the lifespan already unwound) on an enter failure, so
        # a raised build leaves nothing half-open for the caller to clean up.
        await app_lifespan.start()
        return sub_app, app_lifespan

    async def _get_or_build_app(self, slug: str):
        # Fast path under the loop-agnostic state lock — no await, no cross-loop
        # hazard. A present cache KEY means "built" even when its value is None
        # (stdio has no ASGI surface), so membership distinguishes cached-None
        # from not-yet-built without a sentinel.
        with self._state_lock:
            if slug in self._server_cache:
                return self._server_cache[slug]
            known = slug in self._routes

        if not known:
            # Cross-worker read path: this worker never registered the slug,
            # but a sibling may have persisted it durably. Consult the store — we are
            # on the owner loop, so awaiting is fine. This costs one Redis ``HGET``
            # per unknown-slug request (same class as serving the 404) and NOTHING on
            # the known-slug fast path above, which never reaches here. ``None`` → the
            # 404 below; found → bind it into this worker's router and fall through to
            # the build path (register directly, NOT through the write service — the
            # store is already the source of this config, so re-writing it is wrong).
            # Residual: a slug DELETED on a sibling stays served by workers that
            # already built it until their next reload (stale-positive); the
            # store-backed ``GET`` list is already correct. ``reset()`` clears this
            # cache — this fallback + rehydrate repopulate it.
            config = await get_sub_mcp_store().get_route(slug)
            if config is None:
                return None
            await self.register_sub_mcp_app(slug, config.tools, config.transport)

        # Build path: serialize concurrent builds of the same slug on the owner
        # loop. _build_lock is only ever taken here (single loop), so it stays a
        # safe asyncio.Lock; route/cache mutation stays under _state_lock.
        #
        # Retry loop: capture the slug's generation token with its
        # config, build against that captured config, then cache the result ONLY if
        # the token still matches. A concurrent REPLACE bumps the token, so a build
        # that ran against a superseded config is discarded (its stack closed) and
        # retried against the newest registration. A slug stays registered across a
        # REPLACE (only its config changes), so matching on mere slug-membership would
        # wrongly accept a stale build — the token match is what distinguishes a
        # superseded config. The loop is UNBOUNDED by design:
        # each retry serves the newest registration, so a client hammering REPLACE on
        # a slug delays first-time builds — its own and any other not-yet-cached slug,
        # since ``_build_lock`` serializes builds router-wide. Cached slugs and the
        # register/unregister/dispatch/reset paths never take this lock, so they are
        # unaffected; the delay is self-limited to the REPLACE rate.
        async with self._build_lock:
            while True:
                with self._state_lock:
                    if slug in self._server_cache:
                        return self._server_cache[slug]
                    config = self._routes.get(slug)
                    if config is None:
                        return None
                    gen = self._route_generations[slug]

                sub_app, stack = await self._build_sub_app(slug, config)

                with self._state_lock:
                    current_gen = self._route_generations.get(slug)
                    if current_gen == gen:
                        # The registration this build ran against is still live.
                        if stack is not None:
                            self._app_exit_stacks[slug] = stack
                        self._server_cache[slug] = sub_app
                        return sub_app
                    vanished = current_gen is None

                # Stale build: a concurrent replace (token changed) or unregister/reset
                # (token gone) superseded the captured config. Close the just-built
                # stack inline on the owner loop (we are on it), then either give up
                # (the slug vanished) or retry against the newest registration.
                if stack is not None:
                    await self._aclose_on_owner(slug, stack)
                if vanished:
                    return None

    @asynccontextmanager
    async def lifespan(self, app: Starlette) -> AsyncIterator[None]:
        # Record the loop that owns every lazily-built sub-app lifespan so a
        # reload driven from a throwaway loop closes them here instead.
        self._owner_loop = asyncio.get_running_loop()
        try:
            yield
        finally:
            self._owner_loop = None
            # Close every sub-app still registered at shutdown, collecting
            # failures instead of letting the first one mask the rest. Grab the
            # stacks under the state lock; close them off-lock on this (owner) loop.
            with self._state_lock:
                stacks = list(self._app_exit_stacks.items())
                self._app_exit_stacks.clear()
            errors: list[Exception] = []
            for _slug, stack in stacks:
                try:
                    await stack.aclose()
                except Exception as e:
                    errors.append(e)
            if errors:
                raise ExceptionGroup("sub-MCP app shutdown failures", errors)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            # A websocket routed under the mount is closed explicitly (policy code
            # 1008) with a log line rather than returned with no ASGI message —
            # which would surface as a hung handshake / app error. Other
            # unrouteable scope types (lifespan) are ignored.
            if scope["type"] == "websocket":
                logger.warning("sub-MCP router received an unsupported websocket scope at %s", scope.get("path"))
                await send({"type": "websocket.close", "code": 1008})
            return

        # 1. PARSE PATH & FIND SLUG
        original_path = scope["path"]
        mount_clean = self.root_prefix.rstrip("/")

        # Check if we are actually under the mount path. Segment-exact: "/apple"
        # is NOT under "/app".
        under_mount = original_path == mount_clean or original_path.startswith(mount_clean + "/")
        path_suffix = original_path if not under_mount else original_path[len(mount_clean) :]

        # Extract slug from suffix: "/slug/..."
        parts = path_suffix.strip("/").split("/")
        if not parts or not parts[0]:
            await send({"type": "http.response.start", "status": 404, "headers": []})
            await send({"type": "http.response.body", "body": b"Missing Slug"})
            return

        slug = parts[0]

        # 2. GET CONFIG & SERVER
        sub_server = await self._get_or_build_app(slug)
        if not sub_server:
            await send({"type": "http.response.start", "status": 404, "headers": []})
            await send({"type": "http.response.body", "body": b"Unknown Route"})
            return

        # A concurrent unregister may have dropped the route between the build
        # and here; a clean 404 beats a KeyError.
        config = self._routes.get(slug)
        if config is None:
            await send({"type": "http.response.start", "status": 404, "headers": []})
            await send({"type": "http.response.body", "body": b"Unknown Route"})
            return

        # 3. ROUTE DISPATCH (PREPARE SCOPE)
        sub_scope = scope.copy()
        slug_prefix = f"{mount_clean}/{slug}"

        if config.transport == "sse":
            if not under_mount:
                sub_scope["path"] = f"{mount_clean}{original_path}"
            else:
                sub_scope["path"] = original_path

            sub_scope["root_path"] = ""
        else:
            if original_path.startswith(slug_prefix):
                new_path = original_path[len(slug_prefix) :]
                sub_scope["path"] = new_path if new_path else "/"
                # Under the real mount, ``scope["root_path"]`` already carries the
                # router's own mount ("/app"); only the slug segment is appended.
                # Re-adding the full "/app/<slug>" prefix would yield
                # "/app/app/<slug>" and break the sub-app's url_for / redirects.
                sub_scope["root_path"] = scope.get("root_path", "").rstrip("/") + f"/{slug}"
            else:
                sub_scope["path"] = "/"

        # 4. DISPATCH
        # fastmcp installs its auth middleware chain (token-verify + policy
        # resolution) app-level, wrapping the whole Starlette app INCLUDING this
        # mount, so an unauthenticated request is already denied upstream before it
        # reaches here — ``sub_scope`` carries the resolved ``scope["user"]`` /
        # ``scope["auth"]``. Dispatch straight into the sub-server on the rewritten
        # scope; re-applying the chain per request would verify twice.
        await sub_server(sub_scope, receive, send)
