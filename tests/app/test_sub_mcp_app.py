"""``SubMcpAppRouter`` coverage: route register/reload/unregister + exit-stack
teardown, the build/cache seam, and the ASGI ``__call__`` dispatch (slug parse,
404s, http/sse scope rewriting, and the auth-middleware gate).

The per-slug sub-server build is the FastMCP seam — for the dispatch tests it is
replaced with a recording fake ASGI app, so the router's routing logic is
exercised without standing up a real FastMCP server. Auth is enforced app-level
upstream (fastmcp wraps the whole app including this mount), so an integration
test pins the denial through the real ``ResourceGuardMiddleware`` wrapping a
finalized, mounted app — the app-level chain that guards the mount.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, cast

import pytest
from starlette.applications import Starlette
from starlette.types import ASGIApp
from tai42_contract.sub_mcp import RouteConfig

from tai42_skeleton.app import sub_mcp_app as sub_mcp_app_module
from tai42_skeleton.app.sub_mcp_app import ROOT_PREFIX, SubMcpAppRouter, _SubAppLifespan


class _FakeLifespan(_SubAppLifespan):
    """A ``_SubAppLifespan`` stand-in for the teardown tests.

    The router stores a real ``_SubAppLifespan`` per built slug and, on
    unregister/reset/shutdown, invokes only its ``aclose()``. This fake skips the
    real sub-app + dedicated-task machinery and drives just that one method: its
    ``aclose()`` runs an optional async callback (to record that — and on which
    loop — teardown ran) and then optionally re-raises, so a failing teardown can
    be exercised. It is genuinely a ``_SubAppLifespan`` (subclass), so it satisfies
    the ``dict[str, _SubAppLifespan]`` registry contract without weakening it.
    """

    def __init__(
        self,
        *,
        on_close: Callable[[], Awaitable[None]] | None = None,
        error: BaseException | None = None,
    ) -> None:
        self._on_close = on_close
        self._error_to_raise = error

    async def aclose(self) -> None:
        if self._on_close is not None:
            await self._on_close()
        if self._error_to_raise is not None:
            raise self._error_to_raise


class _FakeApp:
    """A minimal owning-app stand-in for the router. It carries a ``fastmcp``
    escape hatch (unused by dispatch — auth is enforced app-level upstream) and,
    for the real-build tests, the process app supplies ``tools``."""

    def __init__(self):
        self.fastmcp = SimpleNamespace()


class _FakeSub:
    """A recording ASGI sub-server stand-in."""

    def __init__(self):
        self.scopes: list = []

    async def __call__(self, scope, receive, send):
        self.scopes.append(scope)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})


async def _drive(asgi, scope):
    sent: list = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    await asgi(scope, receive, send)
    return sent


def _router() -> SubMcpAppRouter:
    return SubMcpAppRouter(app=_FakeApp())


# -- route management ---------------------------------------------------------


async def test_root_prefix_and_routes_properties():
    r = _router()
    assert r.root_prefix == ROOT_PREFIX
    assert r.routes == {}
    await r.register_sub_mcp_app("svc", ["t1"], transport="http")
    assert r.routes["svc"] == RouteConfig(tools=["t1"], transport="http")


async def test_register_existing_slug_reloads():
    r = _router()
    await r.register_sub_mcp_app("svc", ["a"])
    await r.register_sub_mcp_app("svc", ["b"])  # reload path -> unregister first
    assert r.routes["svc"].tools == ["b"]


@pytest.mark.parametrize("bad", ["bad/slug", "Bad", "svc\n", "-lead", "", "café"])
async def test_register_rejects_malformed_slug(bad):
    # The slug is validated at the core, so EVERY caller (the HTTP router AND the
    # backup-restore path) is guarded: a malformed slug — one that would mint an
    # unreachable, undeletable route — is rejected loudly here, not silently minted.
    r = _router()
    with pytest.raises(ValueError, match="must match"):
        await r.register_sub_mcp_app(bad, ["t1"])
    assert r.routes == {}


@pytest.mark.parametrize("bad", ["ws", "grpc", "HTTP", ""])
async def test_register_rejects_unknown_transport(bad):
    # Transport is validated at the core so an unknown transport is rejected
    # loudly instead of silently built as ``http``, guarding the backup-restore
    # path as well as the HTTP router.
    r = _router()
    with pytest.raises(ValueError, match="transport"):
        await r.register_sub_mcp_app("svc", ["t1"], transport=bad)
    assert r.routes == {}


async def test_unregister_clears_route_cache_and_exit_stack():
    r = _router()
    await r.register_sub_mcp_app("svc", [])
    r._server_cache["svc"] = None

    closed = {"ok": False}

    async def _mark() -> None:
        closed["ok"] = True

    r._app_exit_stacks["svc"] = _FakeLifespan(on_close=_mark)
    await r.unregister_sub_mcp_app("svc")
    assert "svc" not in r.routes
    assert "svc" not in r._server_cache
    assert "svc" not in r._app_exit_stacks
    assert closed["ok"] is True


async def test_unregister_reraises_exit_stack_error_after_cleanup():
    r = _router()

    r._app_exit_stacks["svc"] = _FakeLifespan(error=RuntimeError("teardown boom"))
    # A failing close must surface loudly — never report the sub-app removed
    # while it leaked...
    with pytest.raises(RuntimeError, match="teardown boom"):
        await r.unregister_sub_mcp_app("svc")
    # ...but the entry is still dropped, so a retry cannot re-close it.
    assert "svc" not in r._app_exit_stacks


async def test_reset_clears_routes_cache_and_closes_stacks():
    r = _router()
    await r.register_sub_mcp_app("svc", [])
    r._server_cache["svc"] = cast(ASGIApp, _FakeSub())

    closed = {"fired": False}

    async def _on_close() -> None:
        closed["fired"] = True

    r._app_exit_stacks["svc"] = _FakeLifespan(on_close=_on_close)

    r.reset()

    # Routes + cache + stack registry are cleared so stale sub-apps stop serving.
    assert r.routes == {}
    assert r._server_cache == {}
    assert r._app_exit_stacks == {}
    # The stale sub-app's lifespan teardown was scheduled on the running loop.
    await asyncio.sleep(0)
    assert closed["fired"] is True


async def test_start_resets_stale_sub_app_routes():
    # start()/update() must reset the router so a sub-app from the previous
    # generation stops serving after a re-init.
    from tai42_skeleton.app.instance import app
    from tai42_skeleton.manifest import Manifest

    manifest = Manifest.model_validate({})

    async def run():
        async with app.app_context(manifest):
            router = cast(SubMcpAppRouter, app.sub_app.mcp_sub_app_router)
            await router.register_sub_mcp_app("svc", [], transport="stdio")
            assert "svc" in router.routes
            app._update(manifest)
            assert "svc" not in router.routes

    await run()


async def test_route_mutation_runs_off_the_build_loop():
    # register/unregister mutate state under a loop-agnostic threading.Lock, not
    # the loop-bound _build_lock, so a reload driven from a throwaway loop can't
    # raise "asyncio.Lock ... is bound to a different event loop".
    r = _router()
    await r.register_sub_mcp_app("a", [], transport="stdio")
    # Build "a" on THIS loop so _build_lock binds to it.
    assert await r._get_or_build_app("a") is None

    captured: list[BaseException] = []

    def off_loop():
        async def ops():
            await r.register_sub_mcp_app("b", [], transport="stdio")
            await r.unregister_sub_mcp_app("a")

        try:
            asyncio.run(ops())
        except BaseException as exc:  # a cross-loop _build_lock acquire would land here
            captured.append(exc)

    thread = threading.Thread(target=off_loop)
    thread.start()
    thread.join()

    assert not captured, f"route mutation raised off the build loop: {captured}"
    assert "b" in r.routes
    assert "a" not in r.routes


async def test_build_stdio_returns_none_and_caches():
    r = _router()
    await r.register_sub_mcp_app("stdio_svc", [], transport="stdio")
    built = await r._get_or_build_app("stdio_svc")
    assert built is None
    assert r._server_cache["stdio_svc"] is None


async def test_get_or_build_unknown_slug_is_none():
    r = _router()
    assert await r._get_or_build_app("nope") is None


# -- ASGI dispatch ------------------------------------------------------------


async def test_websocket_scope_is_closed_explicitly():
    r = _router()
    # A websocket routed under the mount is closed with policy code 1008 (a log
    # line accompanies it), never left hanging with no ASGI message.
    sent = await _drive(r, {"type": "websocket", "path": "/app/svc"})
    assert sent == [{"type": "websocket.close", "code": 1008}]


async def test_lifespan_scope_is_ignored():
    r = _router()
    # An unrouteable non-http, non-websocket scope is ignored (no send calls).
    sent = await _drive(r, {"type": "lifespan"})
    assert sent == []


async def test_missing_slug_is_404():
    r = _router()
    sent = await _drive(r, {"type": "http", "path": "/app/"})
    assert sent[0]["status"] == 404
    assert sent[1]["body"] == b"Missing Slug"


async def test_unknown_route_is_404():
    r = _router()
    sent = await _drive(r, {"type": "http", "path": "/app/ghost/x"})
    assert sent[0]["status"] == 404
    assert sent[1]["body"] == b"Unknown Route"


async def _register_with_fake_sub(r, slug, transport):
    fake = _FakeSub()

    async def _build(s, config):
        return fake, _FakeLifespan()

    r._build_sub_app = _build
    await r.register_sub_mcp_app(slug, [], transport=transport)
    return fake


async def test_http_dispatch_rewrites_path_under_mount():
    r = _router()
    fake = await _register_with_fake_sub(r, "svc", "http")
    # Under the real mount, starlette hands the router the FULL path plus a
    # root_path already carrying the router's own "/app" mount. Only the slug
    # segment is appended — the sub-app's root_path must be "/app/svc", never a
    # doubled "/app/app/svc".
    sent = await _drive(r, {"type": "http", "path": "/app/svc/foo", "root_path": "/app"})
    assert sent[0]["status"] == 200
    # Path stripped of the slug prefix; root_path carries the slug mount.
    assert fake.scopes[0]["path"] == "/foo"
    assert fake.scopes[0]["root_path"] == "/app/svc"


async def test_http_dispatch_path_not_under_mount_falls_back_to_root():
    r = _router()
    fake = await _register_with_fake_sub(r, "svc", "http")
    # Path lacks the mount prefix -> slug parsed from the bare suffix, sub path "/".
    await _drive(r, {"type": "http", "path": "/svc"})
    assert fake.scopes[0]["path"] == "/"


async def test_sse_dispatch_keeps_path_under_mount():
    r = _router()
    fake = await _register_with_fake_sub(r, "sse_svc", "sse")
    await _drive(r, {"type": "http", "path": "/app/sse_svc/sse"})
    assert fake.scopes[0]["path"] == "/app/sse_svc/sse"
    assert fake.scopes[0]["root_path"] == ""


async def test_sse_dispatch_path_not_under_mount_is_prefixed():
    r = _router()
    fake = await _register_with_fake_sub(r, "sse_svc", "sse")
    await _drive(r, {"type": "http", "path": "/sse_svc/sse"})
    assert fake.scopes[0]["path"] == "/app/sse_svc/sse"


async def test_unauthenticated_request_is_denied_by_the_real_resource_guard():
    # fastmcp installs its auth chain APP-LEVEL, wrapping the whole app including
    # the /app mount, so an unauthenticated /app/<slug>/... request is denied
    # upstream before it reaches the router's dispatch — the router relies on the
    # app-level chain that guards the mount. Drive the REAL
    # ``ResourceGuardMiddleware`` (not a hand-built stub): mount the router the way
    # ``HttpSurface.finalize`` does, wrap the genuine guard around the whole mount,
    # and assert the denial runs through the real auth chain, plus that an
    # authenticated request reaches the sub-server (the mount is live).
    from starlette.authentication import AuthCredentials
    from starlette.testclient import TestClient

    from tai42_skeleton.access_control.middleware import ResourceGuardMiddleware

    r = _router()
    await _register_with_fake_sub(r, "svc", "http")

    class _FakeVerifier:
        # Every /app route resolves to a protected resource, so the guard's auth
        # decision (not route config) governs the denial.
        async def resolve_resource_ids(self, path, method=None, *, policy_version=None):
            return ["protected-res"]

    class _Authenticate:
        # Stand in for the upstream AuthenticationMiddleware: populate an
        # authenticated user + wildcard scopes ONLY when a credential is present, so
        # the real ResourceGuardMiddleware below makes the genuine allow/deny call.
        def __init__(self, app):
            self._app = app

        async def __call__(self, scope, receive, send):
            if scope["type"] == "http" and dict(scope["headers"]).get(b"authorization"):
                scope["user"] = SimpleNamespace(
                    is_authenticated=True, token=SimpleNamespace(client_id="tester", scopes=["*"], claims={})
                )
                scope["auth"] = AuthCredentials(["*"])
            await self._app(scope, receive, send)

    # Mount the router the way HttpSurface.finalize does, then wrap the app-level
    # auth chain (identity shim → real resource guard) around the whole mount.
    inner = Starlette()
    inner.mount(ROOT_PREFIX, r)
    guarded = ResourceGuardMiddleware(inner, verifier=cast(Any, _FakeVerifier()), public_resource_id="public-res")
    client = TestClient(_Authenticate(guarded))

    # No credential: the real ResourceGuardMiddleware denies (401) before the
    # request ever reaches the router's dispatch.
    assert client.get("/app/svc/foo").status_code == 401
    # With a credential: identity resolves, the guard authorizes, the mount serves.
    assert client.get("/app/svc/foo", headers={"Authorization": "Bearer t"}).status_code == 200


async def test_router_lifespan_enters_and_exits():
    r = _router()
    # A callback on a registered per-slug stack fires only when the lifespan
    # exits, so its flag makes setup-open / teardown-closed observable.
    closed = {"fired": False}

    async def _on_close() -> None:
        closed["fired"] = True

    r._app_exit_stacks["slug-a"] = _FakeLifespan(on_close=_on_close)

    # lifespan ignores its app argument; _FakeApp cannot structurally match Starlette.
    async with r.lifespan(cast("Starlette", _FakeApp())):
        assert closed["fired"] is False  # still open inside the context
    assert closed["fired"] is True  # exit closed the registered sub-app stack
    assert r._app_exit_stacks == {}


async def test_real_build_http_sse_stdio_and_cache():
    # Drive the real ``_build_sub_app`` against the process app so ``get_tool``
    # returns a genuine FastMCP tool, covering the http/sse/stdio build branches
    # and the build-cache reuse path.
    from tai42_skeleton.app.instance import app
    from tai42_skeleton.manifest import Manifest

    manifest = Manifest.model_validate(
        {"tools": [{"title": "fxt", "module": "tests.app._fixtures.tools_a", "include": ["greet"]}]}
    )

    async def run():
        async with app.app_context(manifest):
            router = cast(SubMcpAppRouter, app.sub_app.mcp_sub_app_router)
            async with router.lifespan(cast("Starlette", None)):
                await router.register_sub_mcp_app("http_svc", ["greet"], transport="http")
                built = await router._get_or_build_app("http_svc")
                assert built is not None
                # Second fetch reuses the cached build (no rebuild).
                assert await router._get_or_build_app("http_svc") is built

                await router.register_sub_mcp_app("sse_svc", ["greet"], transport="sse")
                assert await router._get_or_build_app("sse_svc") is not None

                await router.register_sub_mcp_app("stdio_svc", ["greet"], transport="stdio")
                assert await router._get_or_build_app("stdio_svc") is None

    await run()


async def test_sub_app_carries_the_body_size_cap_inside_its_own_error_handler():
    # Each sub-app is a full Starlette app with its OWN ServerErrorMiddleware, so the
    # base app's body-size cap (outside the mount) cannot convert an over-cap escape on
    # a sub-MCP route into a 413 — that inner error handler would commit a 500 first.
    # _build_sub_app injects BodyLimitMiddleware into each sub-app's own Starlette stack
    # (inside its ServerErrorMiddleware), so /app/{slug}/... answers 413 too. Reverting
    # that injection leaves the cap absent from the sub-app's user middleware here. No
    # tools are needed — the build produces a real Starlette app regardless.
    from tai42_skeleton.app.instance import app
    from tai42_skeleton.manifest import Manifest
    from tai42_skeleton.middleware.body_limit import BodyLimitMiddleware

    async def run():
        async with app.app_context(Manifest.model_validate({})):
            router = cast(SubMcpAppRouter, app.sub_app.mcp_sub_app_router)
            async with router.lifespan(cast("Starlette", None)):
                for transport in ("http", "sse"):
                    slug = f"{transport}_svc"
                    await router.register_sub_mcp_app(slug, [], transport=transport)
                    sub_app = await router._get_or_build_app(slug)
                    assert sub_app is not None
                    assert any(m.cls is BodyLimitMiddleware for m in sub_app.user_middleware), transport

    await run()


# -- generation-token build race + cross-worker store fallback -----------------


class _FakeBuiltApp:
    """A marked fake ASGI app remembering which tools it was built from and whether
    its recording lifespan is currently open."""

    def __init__(self, tools: list[str]) -> None:
        self.tools = list(tools)
        self.lifespan_open = False
        self.lifespan_entered = False
        # Records the ``path`` the builder passed to ``http_app`` (the fake FastMCP
        # stamps it after construction), so a test can pin that the builder mounts
        # the streamable-HTTP endpoint at the sub-app root ("/").
        self.http_path: str | None = None

    def lifespan(self, app):
        @asynccontextmanager
        async def _cm(_a):
            self.lifespan_open = True
            self.lifespan_entered = True
            try:
                yield
            finally:
                self.lifespan_open = False

        return _cm(app)

    async def __call__(self, scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": "|".join(self.tools).encode()})


def _install_fake_fastmcp(monkeypatch) -> list[_FakeBuiltApp]:
    """Swap the FastMCP seam for a recording fake and return the list of every
    ASGI app it builds (in build order), so a test can assert the stale build's
    lifespan was closed."""
    built: list[_FakeBuiltApp] = []

    class _FakeFastMCP:
        def __init__(self, name: str) -> None:
            self.name = name
            self._tools: list[str] = []
            self._middleware: list[object] = []

        def add_tool(self, tool) -> None:
            self._tools.append(tool)

        def add_middleware(self, middleware) -> None:
            self._middleware.append(middleware)

        def http_app(self, path=None, middleware=None):
            # Mirror FastMCP.http_app's signature: the builder passes ``path="/"``
            # so the streamable-HTTP endpoint sits at the sub-app root (matching the
            # dispatcher's ``/`` rewrite).
            app = _FakeBuiltApp(self._tools)
            app.http_path = path
            built.append(app)
            return app

    monkeypatch.setattr(sub_mcp_app_module, "FastMCP", _FakeFastMCP)
    return built


def _gated_tools(gate: asyncio.Event, gated_name: str):
    """A fake ``tools`` whose ``get_tool`` parks on ``gate`` for one tool name (the
    build that must be superseded), passing every other name straight through."""

    async def get_tool(name: str) -> str:
        if name == gated_name:
            await gate.wait()
        return name

    return SimpleNamespace(get_tool=get_tool)


async def test_replace_during_build_serves_newest_and_closes_stale(monkeypatch):
    # A REPLACE that lands while a build is parked must make the build discard the
    # stale app (its lifespan closed) and retry, serving the NEWEST config — never
    # cache the app built from the superseded config.
    built_apps = _install_fake_fastmcp(monkeypatch)
    gate = asyncio.Event()
    r = _router()
    r._owner_loop = asyncio.get_running_loop()
    r._app.tools = _gated_tools(gate, "old")

    await r.register_sub_mcp_app("svc", ["old"], transport="http")
    build_task = asyncio.create_task(r._get_or_build_app("svc"))
    # Let the build reach the parked get_tool("old").
    for _ in range(5):
        await asyncio.sleep(0)

    # REPLACE the live slug while the build is parked.
    await r.register_sub_mcp_app("svc", ["new"], transport="http")
    gate.set()
    built = await build_task

    assert isinstance(built, _FakeBuiltApp)
    assert built.tools == ["new"]  # serves the NEWEST config, not ["old"]
    assert r._server_cache["svc"] is built
    assert built.lifespan_open is True  # the served app's lifespan is live
    # Two apps were built (stale ["old"], then ["new"]); the stale one's lifespan was
    # entered and then CLOSED rather than cached.
    stale = built_apps[0]
    assert stale.tools == ["old"]
    assert stale.lifespan_entered is True
    assert stale.lifespan_open is False
    # Dispatch agrees with GET-visible routes: the new app serves.
    sent = await _drive(r, {"type": "http", "path": "/app/svc/x", "root_path": "/app"})
    assert sent[1]["body"] == b"new"
    assert r.routes["svc"].tools == ["new"]


async def test_unregister_during_build_returns_none_and_closes_orphan(monkeypatch):
    # A slug unregistered mid-build yields None (a 404) and the just-built orphan
    # stack is torn down, driven by the generation token.
    _install_fake_fastmcp(monkeypatch)
    gate = asyncio.Event()
    r = _router()
    r._owner_loop = asyncio.get_running_loop()
    r._app.tools = _gated_tools(gate, "old")

    await r.register_sub_mcp_app("svc", ["old"], transport="http")
    build_task = asyncio.create_task(r._get_or_build_app("svc"))
    for _ in range(5):
        await asyncio.sleep(0)

    await r.unregister_sub_mcp_app("svc")
    gate.set()
    built = await build_task

    assert built is None
    assert "svc" not in r._server_cache
    assert "svc" not in r._app_exit_stacks


async def test_reset_then_reregister_during_build_does_not_cache_stale(monkeypatch):
    # The global-monotonic property: a reset()-then-re-register while a build is
    # parked must NOT let the stale build cache. A plain per-slug counter would
    # reset to the same token and wrongly match here.
    _install_fake_fastmcp(monkeypatch)
    gate = asyncio.Event()
    r = _router()
    r._owner_loop = asyncio.get_running_loop()
    r._app.tools = _gated_tools(gate, "old")

    await r.register_sub_mcp_app("svc", ["old"], transport="http")
    build_task = asyncio.create_task(r._get_or_build_app("svc"))
    for _ in range(5):
        await asyncio.sleep(0)

    r.reset()
    await r.register_sub_mcp_app("svc", ["new"], transport="http")
    gate.set()
    built = await build_task

    assert isinstance(built, _FakeBuiltApp)
    assert built.tools == ["new"]
    assert r._server_cache["svc"] is built


async def test_known_slug_fast_path_never_touches_the_store(monkeypatch):
    # The store fallback must fire ONLY for an unknown slug; the known-slug fast
    # path serves entirely from the local cache without a store read.
    _install_fake_fastmcp(monkeypatch)
    r = _router()
    r._owner_loop = asyncio.get_running_loop()
    r._app.tools = SimpleNamespace(get_tool=_passthrough_get_tool)

    class _CountingStore:
        def __init__(self) -> None:
            self.get_calls = 0

        async def get_route(self, slug):
            self.get_calls += 1
            return None

    counting = _CountingStore()
    monkeypatch.setattr(sub_mcp_app_module, "get_sub_mcp_store", lambda: counting)

    await r.register_sub_mcp_app("svc", ["t"], transport="http")
    built = await r._get_or_build_app("svc")  # build + cache
    assert built is not None
    assert await r._get_or_build_app("svc") is built  # cached fetch
    assert counting.get_calls == 0  # never consulted the store for a known slug


async def test_unknown_slug_hydrates_from_the_store(monkeypatch):
    # A slug this worker never registered but a sibling persisted is served by
    # consulting the store on the owner loop, then binding + building it locally.
    _install_fake_fastmcp(monkeypatch)
    r = _router()
    r._owner_loop = asyncio.get_running_loop()
    r._app.tools = SimpleNamespace(get_tool=_passthrough_get_tool)

    class _Store:
        def __init__(self) -> None:
            self.get_calls = 0

        async def get_route(self, slug):
            self.get_calls += 1
            return RouteConfig(tools=["remote"], transport="http") if slug == "remote" else None

    store = _Store()
    monkeypatch.setattr(sub_mcp_app_module, "get_sub_mcp_store", lambda: store)

    built = await r._get_or_build_app("remote")
    assert isinstance(built, _FakeBuiltApp)
    assert built.tools == ["remote"]
    assert r.routes["remote"].tools == ["remote"]  # bound into the local router
    assert store.get_calls == 1

    # An unknown slug missing from the store is a clean None (404), one store read.
    assert await r._get_or_build_app("ghost") is None
    assert store.get_calls == 2


async def _passthrough_get_tool(name: str) -> str:
    return name


async def test_concurrent_build_waiter_returns_the_cached_app(monkeypatch):
    # Two owner-loop requests build the same slug: the first builds under _build_lock
    # while the second waits; when the second acquires the lock it sees the cache hit
    # (the loop-top double-check) and returns the SAME app rather than rebuilding.
    built_apps = _install_fake_fastmcp(monkeypatch)
    gate = asyncio.Event()
    r = _router()
    r._owner_loop = asyncio.get_running_loop()
    r._app.tools = _gated_tools(gate, "t")

    await r.register_sub_mcp_app("svc", ["t"], transport="http")
    first = asyncio.create_task(r._get_or_build_app("svc"))
    for _ in range(5):
        await asyncio.sleep(0)
    second = asyncio.create_task(r._get_or_build_app("svc"))
    for _ in range(5):
        await asyncio.sleep(0)

    gate.set()
    a, b = await asyncio.gather(first, second)
    assert a is b  # the waiter returned the cached build
    assert len(built_apps) == 1  # only one app was ever built


async def test_build_lock_waiter_returns_none_when_slug_unregistered(monkeypatch):
    # A slug unregistered while a build holds _build_lock: the waiting request, on
    # acquiring the lock, finds no config at the loop top and returns None (404).
    _install_fake_fastmcp(monkeypatch)
    gate = asyncio.Event()
    r = _router()
    r._owner_loop = asyncio.get_running_loop()
    r._app.tools = _gated_tools(gate, "t")

    await r.register_sub_mcp_app("svc", ["t"], transport="http")
    first = asyncio.create_task(r._get_or_build_app("svc"))
    for _ in range(5):
        await asyncio.sleep(0)
    second = asyncio.create_task(r._get_or_build_app("svc"))
    for _ in range(5):
        await asyncio.sleep(0)

    await r.unregister_sub_mcp_app("svc")
    gate.set()
    a, b = await asyncio.gather(first, second)
    assert a is None  # the build's slug vanished
    assert b is None  # the waiter found no config on acquiring the lock


async def test_build_propagates_a_lifespan_enter_failure(monkeypatch):
    # A lifespan-enter failure during the build is raised loudly (never a silently
    # half-built app), and the just-opened stack is closed.
    class _BadLifespanApp(_FakeBuiltApp):
        def lifespan(self, app):
            @asynccontextmanager
            async def _cm(_a):
                raise RuntimeError("lifespan boom")
                yield  # pragma: no cover

            return _cm(app)

    class _BadFastMCP:
        def __init__(self, name: str) -> None:
            self._tools: list[str] = []

        def add_tool(self, tool) -> None:
            self._tools.append(tool)

        def add_middleware(self, middleware) -> None:
            pass

        def http_app(self, path=None, middleware=None):
            return _BadLifespanApp(self._tools)

    monkeypatch.setattr(sub_mcp_app_module, "FastMCP", _BadFastMCP)
    r = _router()
    r._owner_loop = asyncio.get_running_loop()
    r._app.tools = SimpleNamespace(get_tool=_passthrough_get_tool)

    await r.register_sub_mcp_app("svc", ["t"], transport="http")
    with pytest.raises(RuntimeError, match="lifespan boom"):
        await r._get_or_build_app("svc")


def test_log_teardown_result_ignores_a_cancelled_future():
    # A cancelled scheduled teardown must not raise inside the done-callback (its
    # ``exception()`` would raise ``CancelledError``) — it is a no-op, no log.
    fut: concurrent.futures.Future = concurrent.futures.Future()
    assert fut.cancel()
    # Does not raise.
    SubMcpAppRouter._log_teardown_result("svc", fut)


async def test_log_teardown_result_ignores_a_cancelled_task(caplog):
    # The reset same-loop branch schedules via ``create_task``, so the done-callback
    # also fires on an ``asyncio.Task``. A cancelled Task's ``exception()`` raises
    # ``asyncio.CancelledError`` — an UNRELATED class to the Future's
    # ``concurrent.futures.CancelledError`` (``issubclass`` is False), so it must be
    # caught in its own right or it escapes the callback.
    async def _park() -> None:
        await asyncio.Event().wait()

    task = asyncio.create_task(_park())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()

    with caplog.at_level(logging.ERROR, logger="tai42_skeleton.app.sub_mcp_app"):
        # Does not raise ``asyncio.CancelledError`` out of the callback.
        SubMcpAppRouter._log_teardown_result("svc", task)
    assert caplog.text == ""  # a cancelled teardown returns quietly, no ERROR


# -- cross-loop teardown never block-waits on a parked owner loop -------------


def test_cross_loop_close_does_not_deadlock_on_a_parked_owner_loop(caplog):
    # A teardown scheduled onto the owner loop while that loop's
    # thread is PARKED in _run_blocking must not block-wait on it (a cross-loop
    # block-wait would deadlock the reload forever). The register call completes
    # despite the parked owner; the stale stack's close lands on the owner loop once
    # it is unparked, and a close failure is surfaced via the logged done-callback.
    r = SubMcpAppRouter(app=_FakeApp())

    owner_loop = asyncio.new_event_loop()
    owner_ready = threading.Event()

    def _run_owner():
        asyncio.set_event_loop(owner_loop)
        owner_loop.call_soon(owner_ready.set)
        owner_loop.run_forever()

    owner_thread = threading.Thread(target=_run_owner, daemon=True)
    owner_thread.start()
    owner_ready.wait()
    r._owner_loop = owner_loop

    # Register the slug on the owner loop, then record a stale stack whose close
    # RAISES — so the done-callback's error path is exercised — and which records the
    # loop it ran on.
    asyncio.run_coroutine_threadsafe(r.register_sub_mcp_app("svc", [], transport="stdio"), owner_loop).result()
    closed_on: dict[str, Any] = {}
    close_ran = threading.Event()

    async def _record_and_raise() -> None:
        closed_on["loop"] = asyncio.get_running_loop()
        close_ran.set()

    r._app_exit_stacks["svc"] = _FakeLifespan(on_close=_record_and_raise, error=RuntimeError("teardown boom"))

    # Park the owner loop's thread inside a call_soon callback blocked on a
    # concurrent.futures result — the shape _run_blocking's executor.submit().result()
    # imposes on the owner thread.
    unpark = concurrent.futures.Future()
    parked = threading.Event()

    def _park() -> None:
        parked.set()
        unpark.result()

    owner_loop.call_soon_threadsafe(_park)
    parked.wait()

    # From a throwaway loop on another thread, drive a REPLACE-register: the stale
    # stack's close must be SCHEDULED onto the owner loop, not block-waited.
    register_done = threading.Event()
    register_err: dict[str, BaseException] = {}

    def _foreign() -> None:
        try:
            asyncio.run(r.register_sub_mcp_app("svc", [], transport="stdio"))
        except BaseException as exc:  # a cross-loop block-wait would hang, not raise
            register_err["exc"] = exc
        finally:
            register_done.set()

    foreign_thread = threading.Thread(target=_foreign, daemon=True)
    foreign_thread.start()

    assert register_done.wait(timeout=5.0), "register block-waited on the parked owner loop (deadlock)"
    assert "exc" not in register_err
    # The close cannot have run yet — the owner loop is still parked.
    assert not close_ran.is_set()

    with caplog.at_level(logging.ERROR, logger="tai42_skeleton.app.sub_mcp_app"):
        unpark.set_result(None)  # release the owner loop
        assert close_ran.wait(timeout=5.0)
        # Let the done-callback fire on the owner loop.
        drained = threading.Event()
        owner_loop.call_soon_threadsafe(drained.set)
        drained.wait(timeout=5.0)

    assert closed_on["loop"] is owner_loop  # closed on the loop that entered it
    assert "teardown boom" in caplog.text  # the failure was surfaced, not lost

    owner_loop.call_soon_threadsafe(owner_loop.stop)
    owner_thread.join(timeout=5.0)
    owner_loop.close()


def test_reset_does_not_deadlock_on_a_parked_owner_loop(caplog):
    # reset() driven from a FOREIGN thread while the owner loop's
    # thread is PARKED must SCHEDULE each stale stack's close onto the owner loop and
    # never block-wait on it. A cross-loop .result() here would deadlock — the parked
    # owner loop can never run the scheduled close. reset() returns promptly; the
    # close lands once the owner loop unparks, and a failure surfaces via the logged
    # done-callback.
    r = SubMcpAppRouter(app=_FakeApp())

    owner_loop = asyncio.new_event_loop()
    owner_ready = threading.Event()

    def _run_owner():
        asyncio.set_event_loop(owner_loop)
        owner_loop.call_soon(owner_ready.set)
        owner_loop.run_forever()

    owner_thread = threading.Thread(target=_run_owner, daemon=True)
    owner_thread.start()
    owner_ready.wait()
    r._owner_loop = owner_loop

    # Register the slug on the owner loop, then record a stale stack whose close
    # RAISES (so the done-callback's error path is exercised) and which records the
    # loop it ran on.
    asyncio.run_coroutine_threadsafe(r.register_sub_mcp_app("svc", [], transport="stdio"), owner_loop).result()
    closed_on: dict[str, Any] = {}
    close_ran = threading.Event()

    async def _record_and_raise() -> None:
        closed_on["loop"] = asyncio.get_running_loop()
        close_ran.set()

    r._app_exit_stacks["svc"] = _FakeLifespan(on_close=_record_and_raise, error=RuntimeError("teardown boom"))

    # Park the owner loop's thread inside a call_soon callback blocked on a
    # concurrent.futures result — the shape _run_blocking imposes on the owner thread.
    unpark = concurrent.futures.Future()
    parked = threading.Event()

    def _park() -> None:
        parked.set()
        unpark.result()

    owner_loop.call_soon_threadsafe(_park)
    parked.wait()

    # From a foreign thread, drive reset(): the stale stack's close must be SCHEDULED
    # onto the owner loop, not block-waited.
    reset_done = threading.Event()
    reset_err: dict[str, BaseException] = {}

    def _foreign() -> None:
        try:
            r.reset()
        except BaseException as exc:  # a cross-loop block-wait would hang, not raise
            reset_err["exc"] = exc
        finally:
            reset_done.set()

    foreign_thread = threading.Thread(target=_foreign, daemon=True)
    foreign_thread.start()

    assert reset_done.wait(timeout=5.0), "reset() block-waited on the parked owner loop (deadlock)"
    assert "exc" not in reset_err
    # The close cannot have run yet — the owner loop is still parked.
    assert not close_ran.is_set()

    with caplog.at_level(logging.ERROR, logger="tai42_skeleton.app.sub_mcp_app"):
        unpark.set_result(None)  # release the owner loop
        assert close_ran.wait(timeout=5.0)
        # Let the done-callback fire on the owner loop.
        drained = threading.Event()
        owner_loop.call_soon_threadsafe(drained.set)
        drained.wait(timeout=5.0)

    assert closed_on["loop"] is owner_loop  # closed on the loop that entered it
    assert "teardown boom" in caplog.text  # the failure was surfaced, not lost

    owner_loop.call_soon_threadsafe(owner_loop.stop)
    owner_thread.join(timeout=5.0)
    owner_loop.close()
