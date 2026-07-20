"""``TaiMCP`` server-surface coverage.

Split between:
* a throwaway (unstarted) ``TaiMCP`` for the thin FastMCP forwarders, the
  registration decorators, the property accessors, and the "not started" guards
  (constructing one does NOT bind the global ``tai_app``); and
* the process ``app`` driven through ``app_context`` with fixture manifests for
  the tool/extension/agent/toolkit registration paths, ``run_tool``,
  ``get_client_tools``, and the MCP-tool binding.

The app's public surface is the 13 facet namespaces (``app.tools``,
``app.backends``, ...); the per-feature impl members are private. Tests drive the
app through the facets and patch the private impl members directly.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp.tools.function_tool import FunctionTool
from starlette.types import ASGIApp
from tai_contract.agent import Agent
from tai_contract.extensions import ExtensionKind
from tai_contract.manifest import MCPConfig, TaiMCPConfig
from tai_contract.storage import Storage

from tai_skeleton.app import server as server_module
from tai_skeleton.app.instance import app
from tai_skeleton.app.server import TaiMCP
from tai_skeleton.manifest import Manifest
from tai_skeleton.middleware.body_limit import BodyLimitMiddleware
from tai_skeleton.middleware.rate_limit import RateLimitMiddleware
from tai_skeleton.monitoring import registry as monitoring_registry
from tai_skeleton.tools.binding import ToolBinding

if TYPE_CHECKING:
    from tai_skeleton.template import ResourceManager


class _Storage(Storage):
    """Minimal concrete ``Storage`` used as a real ``type[Storage]`` sentinel."""

    async def load(self, path: str) -> str: ...
    async def list(self) -> list[str]: ...
    async def upload(self, path: str, content: str) -> None: ...
    async def delete(self, path: str) -> None: ...
    async def delete_dir(self, path: str) -> None: ...


def _fresh() -> TaiMCP:
    # Constructing a TaiMCP does not bind the global handle, so it is a safe
    # throwaway for surface tests that don't need a started app.
    return TaiMCP(name="srv-under-test", version="9.9")


@pytest.fixture(autouse=True)
def _clean_server():
    """Clear the singleton FastMCP server's tools around each test — it outlives
    one ``app_context``, so a tool a prior test bound would collide with a
    started-app test's bind under ``on_duplicate="error"``."""

    async def _clear() -> None:
        provider = app._fast_mcp.local_provider
        for tool in list(await provider.list_tools()):
            provider.remove_tool(tool.name)

    asyncio.run(_clear())
    yield
    asyncio.run(_clear())


# -- properties / not-started guards ------------------------------------------


def test_basic_properties():
    a = _fresh()
    # Server metadata is read through the raw FastMCP escape hatch, not a
    # per-method delegate on the app class.
    assert a.fastmcp is a._fast_mcp
    assert a.fastmcp.name == "srv-under-test"
    assert a.fastmcp.version == "9.9"
    assert a.fastmcp.instructions is None
    assert a.fastmcp.auth is None
    assert a.config.config_manager is a._config_manager
    assert a.sub_app.mcp_sub_app_router is a._mcp_sub_app_router
    assert a.backends.backend is None


def test_fastmcp_accessor_returns_live_server_and_registers():
    """The ``fastmcp`` accessor is the live server, and registrations made
    THROUGH it (prompt, middleware) are live on that same server — the
    ungoverned escape hatch the facets don't wrap."""
    a = _fresh()
    assert a.fastmcp is a._fast_mcp

    @a.fastmcp.prompt
    def greet() -> str:
        """A prompt."""
        return "hi"

    prompts = asyncio.run(a.fastmcp.list_prompts())
    assert "greet" in {p.name for p in prompts}

    from fastmcp.server.middleware import Middleware

    class _Mw(Middleware):
        pass

    mw = _Mw()
    a.fastmcp.add_middleware(mw)
    assert mw in a.fastmcp.middleware


def test_deleted_metadata_delegates_are_gone():
    """Grep-gate as an assertion: the pure metadata passthroughs were deleted
    from the server class — the raw server is reachable only via ``fastmcp``."""
    for gone in ("auth", "settings", "name", "version", "instructions"):
        assert gone not in vars(TaiMCP), f"delegate {gone!r} must not remain on TaiMCP"


def test_live_manifest_and_tool_title_require_start():
    a = _fresh()
    with pytest.raises(RuntimeError, match="not started"):
        _ = a.admin.live_manifest
    with pytest.raises(RuntimeError, match="not started"):
        a.tools.tool_title(lambda: None)
    with pytest.raises(RuntimeError, match="not started"):
        a._mcp_tools(TaiMCPConfig(title="x", config=MCPConfig(type="http", url="http://x")), [])


def test_get_agent_missing_raises():
    with pytest.raises(RuntimeError, match="No such agent"):
        _fresh().agents.get_agent("nope")


# -- FastMCP forwarders -------------------------------------------------------


def test_run_and_run_async_and_custom_route_forward():
    a = _fresh()
    a._fast_mcp = MagicMock()
    a._fast_mcp.run_async = AsyncMock()

    a.run("stdio", show_banner=False, x=1)
    a._fast_mcp.run.assert_called_once_with("stdio", False, x=1)

    asyncio.run(a.run_async("http", show_banner=True, y=2))
    a._fast_mcp.run_async.assert_awaited_once_with("http", True, y=2)

    a.http.custom_route("/p", ["GET"], name="n", include_in_schema=False, summary="P", tags=["t"], response_model=None)
    a._fast_mcp.custom_route.assert_called_once_with("/p", ["GET"], "n", False)


def test_run_backend_requires_configured_backend():
    a = _fresh()
    with pytest.raises(RuntimeError, match="not configured"):
        asyncio.run(a.run_backend(["arg"]))

    launched: list = []

    class _Backend:
        async def launch(self, args):
            launched.append(args)

    a._backend_holder._backend = _Backend()  # pyright: ignore[reportAttributeAccessIssue]
    asyncio.run(a.run_backend(["arg"]))
    assert launched == [["arg"]]


def test_finalize_app_setup_mounts_router_and_applies_middleware():
    a = _fresh()

    class _PassThrough:
        def __init__(self, app: ASGIApp) -> None:
            self.inner = app

        async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
            await self.inner(scope, receive, send)

    a.http.middleware(_PassThrough)
    base = MagicMock()
    out = a._http_surface.finalize(base)
    # The sub-app router is mounted, then the finalize middleware stack wraps the
    # app. finalize applies only the always-on outer RateLimitMiddleware (which
    # rejects before the app is entered); the body-size cap is NOT a finalize
    # wrapper — it is injected into the base app's own Starlette stack (inside that
    # app's ServerErrorMiddleware; see TaiMCP._with_body_limit). So this test's
    # _PassThrough wraps RateLimit, which wraps the mounted base.
    router = a.sub_app.mcp_sub_app_router
    base.mount.assert_called_once_with(router.root_prefix, router)
    assert isinstance(out, _PassThrough)
    assert isinstance(out.inner, RateLimitMiddleware)
    assert out.inner.app is base
    # The lifespan-bearing base app is recorded on the outermost wrapper so the
    # worker can enter its FastMCP lifespan even though a middleware wrapper
    # exposes none.
    assert cast(Any, out).mcp_lifespan_app is base


def test_sse_app_builds_and_finalizes():
    a = _fresh()
    sentinel = MagicMock()
    with patch.object(server_module, "create_sse_app", return_value=sentinel) as mk:
        result = a.sse_app()
    mk.assert_called_once()
    # The SSE app must carry the server's auth so /sse + /messages are gated when
    # access control is on (mirrors http_app); an unauthenticated SSE surface
    # would bypass it.
    assert mk.call_args.kwargs["auth"] is a._fast_mcp.auth
    # The body-size cap is passed into the base app's OWN middleware list (inside its
    # ServerErrorMiddleware) as the first entry, so an over-cap escape becomes a 413
    # before any error handler can commit a 500.
    assert mk.call_args.kwargs["middleware"][0].cls is BodyLimitMiddleware
    # finalize mounts the sub-app router on the base, then wraps only the always-on
    # outer RateLimitMiddleware around it.
    sentinel.mount.assert_called_once()
    assert isinstance(result, RateLimitMiddleware)
    assert result.app is sentinel


def test_http_app_builds_and_finalizes():
    a = _fresh()
    sentinel = MagicMock()
    a._fast_mcp = MagicMock()
    a._fast_mcp.http_app.return_value = sentinel
    result = a.http_app(path="/mcp", transport="http")
    a._fast_mcp.http_app.assert_called_once()
    # The body-size cap is injected into the base app's own middleware list (inside
    # ServerErrorMiddleware) as the first entry.
    assert a._fast_mcp.http_app.call_args.kwargs["middleware"][0].cls is BodyLimitMiddleware
    sentinel.mount.assert_called_once()
    # finalize wraps only the always-on outer RateLimitMiddleware around the base.
    assert isinstance(result, RateLimitMiddleware)
    assert result.app is sentinel


def test_remove_tool_forwards():
    a = _fresh()
    a._fast_mcp = MagicMock()
    a.tools.remove_tool("t")
    a._fast_mcp.local_provider.remove_tool.assert_called_once_with("t")


# -- registration decorators (no manifest needed) -----------------------------


def test_middleware_decorator_class_and_factory():
    a = _fresh()

    @a.http.middleware
    class _DirectMw:
        pass

    @a.http.middleware(option=1)
    class _FactoryMw:
        pass

    classes = [mw.cls for mw in a._http_surface._middlewares.values()]
    assert _DirectMw in classes
    assert _FactoryMw in classes


def test_rate_limit_middleware_registered_at_construction():
    # The public-door flood limiter is always on: registered in __init__, not left
    # to a manifest opt-in an operator could forget.
    a = _fresh()
    classes = [mw.cls for mw in a._http_surface._middlewares.values()]
    assert RateLimitMiddleware in classes


def test_middleware_direct_call_with_options():
    a = _fresh()

    class _Mw:
        pass

    # Direct-call form with options: the class is returned unchanged and the
    # recorded Middleware entry carries the option.
    assert a.http.middleware(_Mw, option=1) is _Mw
    entry = next(mw for mw in a._http_surface._middlewares.values() if mw.cls is _Mw)
    assert entry.kwargs == {"option": 1}


def test_register_backend_direct_and_decorator():
    a = _fresh()

    class _B1:
        pass

    a.backends.register_backend(_B1)
    assert isinstance(a.backends.backend, _B1)

    class _B2:
        pass

    # Decorator form: register_backend(cls) is exactly what ``@`` applies, and it
    # returns the class unchanged. Keeping _B2 as the class name (instead of
    # rebinding it via ``@``) preserves its ``type`` for the isinstance check.
    assert a.backends.register_backend(_B2) is _B2
    assert isinstance(a.backends.backend, _B2)


def test_register_monitoring_builds_and_installs():
    a = _fresh()
    built = MagicMock()
    with patch.object(monitoring_registry, "init_monitoring") as init:

        @a.monitoring.register_monitoring
        def _builder():
            return built

    init.assert_called_once_with(built)


def test_register_storage_forwards_to_registry():
    a = _fresh()
    a._storage_registry = MagicMock()
    a.storage.register_storage(_Storage)
    a._storage_registry.register_storage.assert_called_once_with(_Storage)


def test_register_connector_forwards_to_engine_registry():
    a = _fresh()
    with patch("tai_skeleton.connectors.providers.registry.register_connector") as reg:
        a.connectors.register_connector("descriptor")  # pyright: ignore[reportArgumentType]
    reg.assert_called_once_with("descriptor")


def test_token_store_property_forwards():
    a = _fresh()
    with patch("tai_skeleton.connectors.store.token_store", return_value="the-store"):
        assert a.connectors.token_store == "the-store"


def test_resource_manager_caches_instance():
    a = _fresh()
    # A string sentinel stands in for a built ResourceManager: the property must
    # return the cached object unchanged, so the concrete class is not needed.
    a._resource_manager_cache = cast("ResourceManager", "cached")
    assert a.storage.resource_manager == "cached"

    b = _fresh()
    b._storage_registry = MagicMock()
    with patch.object(server_module, "ResourceManager", return_value="built") as rm:
        assert b.storage.resource_manager == "built"
    rm.assert_called_once_with(b._storage_registry.provider)


def test_extension_decorator_noops_registration_into_registry():
    a = _fresh()

    @a.extensions.extension(kind=ExtensionKind.WRAPPER, name="x")
    def _ext(func, name, desc):
        return func

    assert a._extension_registry.get_extension("x") is _ext
    assert a._extension_registry.get_kind("x") is ExtensionKind.WRAPPER


def test_extension_decorator_direct_callable_form():
    a = _fresh()

    def _ext(func, name, desc):
        return func

    # Passing the callable positionally still registers it (default name).
    a.extensions.extension(_ext, kind=ExtensionKind.TRANSFORMER)
    assert a._extension_registry.get_extension("_ext") is _ext


def test_tool_decorator_noop_without_manifest():
    a = _fresh()  # _manifest is None

    @a.tools.tool
    def fn():
        return 1

    # No manifest -> the decorator returns the function unbound.
    assert fn() == 1


def test_agent_decorator_noop_without_manifest():
    a = _fresh()

    @a.agents.agent("x")
    class _A(Agent):
        async def run(self, **kwargs: Any) -> Any:
            return None

    with pytest.raises(RuntimeError, match="No such agent"):
        a.agents.get_agent("x")


def test_normalized_name_prefixes_and_dedupes():
    assert ToolBinding.normalized_name("srv", "Look-Up") == "srv_look_up"
    # Already-prefixed names are left as-is.
    assert ToolBinding.normalized_name("srv", "srv_tool") == "srv_tool"
    # No prefix -> bare normalized name.
    assert ToolBinding.normalized_name("", "Tool") == "tool"


def test_normalized_name_mixed_case_prefix_no_double_prefix():
    # A mixed-case prefix normalizes to lowercase before the prefix check, so
    # the result is a single lowercase-prefixed name.
    assert ToolBinding.normalized_name("GitHub", "list_repos") == "github_list_repos"
    # A name already carrying the lowercased prefix is left as-is — never a
    # "GitHub_github_" style double prefix.
    assert ToolBinding.normalized_name("GitHub", "github_list_repos") == "github_list_repos"


def test_tool_registry_info_forwarders():
    a = _fresh()
    a._tool_registry = MagicMock()
    a.tools.register_tool_info("i", [["ext"]])
    a._tool_registry.register_tool.assert_called_once_with("i", [["ext"]])
    a.tools.unregister_tool_info("i")
    a._tool_registry.unregister_tool.assert_called_once_with("i")
    a.tools.unregister_tool_base("b")
    a._tool_registry.unregister_tool_base.assert_called_once_with("b")


# -- run_tool / get_client_tools guards (patched lookups) ---------------------


def test_run_tool_rejects_non_function_tool():
    a = _fresh()
    a._tool_binding.get_tool = AsyncMock(return_value=MagicMock())  # not a FunctionTool
    with pytest.raises(RuntimeError, match="no callable body"):
        asyncio.run(a.tools.run_tool("k", {}))


def test_get_tool_missing_raises():
    a = _fresh()
    a._fast_mcp = MagicMock()
    a._fast_mcp.get_tool = AsyncMock(return_value=None)
    with pytest.raises(RuntimeError, match="No such tool"):
        asyncio.run(a.tools.get_tool("ghost"))


def test_get_client_tools_unknown_name_raises():
    a = _fresh()
    a._tool_binding.get_tools = AsyncMock(return_value={})
    with pytest.raises(RuntimeError, match="No such tool"):
        asyncio.run(a.tools.get_client_tools(["missing"]))


def _fake_function_tool(name):
    from tai_skeleton.tools.binding import FunctionTool

    def fn():
        """A tool."""
        return name

    fn.__name__ = "fn"
    fake = MagicMock(spec=FunctionTool)
    fake.fn = fn
    return fake


def test_get_client_tools_post_truncation_collision_raises():
    a = _fresh()
    long = "x" * 64
    a._tool_binding.get_tools = AsyncMock(
        return_value={long + "_a": _fake_function_tool("a"), long + "_b": _fake_function_tool("b")}
    )
    with pytest.raises(ValueError, match="collide after 64-char truncation"):
        asyncio.run(a.tools.get_client_tools())


def test_get_client_tools_non_branchable_tool_raises():
    a = _fresh()
    # A tool that is neither a FunctionTool nor a TransformedTool has no callable
    # body to branch/expose, so resolving its client runnable raises loudly.
    a._tool_binding.get_tools = AsyncMock(return_value={"t": MagicMock()})
    with pytest.raises(TypeError, match="cannot branch-bind"):
        asyncio.run(a.tools.get_client_tools())


# -- started-app integration --------------------------------------------------


def test_started_app_tool_extension_agent_and_client_tools():
    manifest = Manifest.model_validate(
        {
            "extensions_modules": ["tests.app._fixtures.ext_mod"],
            "tools": [
                {
                    "title": "fxt",
                    "module": "tests.app._fixtures.tools_b",
                    "include": ["shout"],
                    "extensions": {"shout": [["loud"]]},
                },
            ],
            "agents": [
                {"title": "agents", "module": "tests.fixtures.dummy_agent", "include": ["dummy_agent"]},
            ],
        }
    )

    async def run():
        async with app.app_context(manifest):
            tools = await app.tools.get_tools()
            # base + extension-branched tool both bound
            assert {"shout", "shout_loud"} <= set(tools)
            # extension listing
            assert {"name": "loud", "kind": "wrapper"} in app.extensions.available_extensions()
            # extension-branched run
            assert await app.tools.run_tool("shout_loud", {"text": "hi"}) == "HI"
            # agent + its synthesized run tool, driven to its final value
            assert "dummy_agent" in tools
            assert app.agents.get_agent("dummy_agent").tool_name == "dummy_agent"
            assert await app.tools.run_tool("dummy_agent", {"text": "ab", "times": 2}) == "abab|tags=0|item=NoneType"
            # tool_title resolves from the manifest
            tool = await app.tools.get_tool("shout")
            assert isinstance(tool, FunctionTool)  # callable body lives on FunctionTool
            assert isinstance(app.tools.tool_title(tool.fn), str)
            # client tools (LangChain) build for a selected name
            client_tools = await app.tools.get_client_tools(["shout"])
            assert [t.name for t in client_tools] == ["shout"]
            # live_manifest is available once started
            assert isinstance(app.admin.live_manifest, dict)

    asyncio.run(run())


def test_stacked_extensions_compose_descriptions():
    # A single stacked combo ``[first, second]`` chains two WRAPPER extensions.
    # The binder must carry the running description forward, so the second
    # extension composes on the first's output — not the original tool description.
    manifest = Manifest.model_validate(
        {
            "extensions_modules": ["tests.app._fixtures.ext_chain"],
            "tools": [
                {
                    "title": "fxt",
                    "module": "tests.app._fixtures.tools_b",
                    "include": ["shout"],
                    "extensions": {"shout": [["first", "second"]]},
                },
            ],
        }
    )

    async def run():
        async with app.app_context(manifest):
            tools = await app.tools.get_tools()
            assert "shout_first_second" in tools
            tool = await app.tools.get_tool("shout_first_second")
            assert tool.description == "Return the text. | first | second"

    asyncio.run(run())


def test_force_binds_tool_outside_manifest():
    async def run():
        async with app.app_context(Manifest.model_validate({})):

            @app.tools.tool(force=True)
            def forced_tool(x: int) -> int:
                """A forcibly-bound tool."""
                return x * 2

            tools = await app.tools.get_tools()
            assert "forced_tool" in tools
            assert await app.tools.run_tool("forced_tool", {"x": 3}) == 6

    asyncio.run(run())


def test_tool_with_explicit_description_overrides_docstring():
    # An explicit ``description=`` must bind cleanly (no TypeError from passing
    # it twice) and win over the function docstring on the bound tool.
    async def run():
        async with app.app_context(Manifest.model_validate({})):

            @app.tools.tool(force=True, description="Explicit description.")
            def described_tool(x: int) -> int:
                """Docstring that must not become the description."""
                return x

            tool = await app.tools.get_tool("described_tool")
            assert tool.description == "Explicit description."

    asyncio.run(run())


def test_toolkit_binds_adapted_tools():
    manifest = Manifest.model_validate(
        {"tools": [{"title": "widgets", "module": "tests.app._fixtures.toolkit_mod", "include": ["widgets_echo"]}]}
    )

    async def run():
        async with app.app_context(manifest):
            tools = await app.tools.get_tools()
            assert "widgets_echo" in tools
            assert await app.tools.run_tool("widgets_echo", {"value": 5}) == 5

    asyncio.run(run())


def test_extension_returning_same_name_is_rejected():
    # An extension must rename the tool to create a branch; returning the same
    # name raises during binding (which happens at module-import time in start()).
    manifest = Manifest.model_validate(
        {
            "extensions_modules": ["tests.app._fixtures.ext_samename"],
            "tools": [
                {
                    "title": "fxt",
                    "module": "tests.app._fixtures.tools_b",
                    "include": ["shout"],
                    "extensions": {"shout": [["samename"]]},
                }
            ],
        }
    )

    async def run():
        async with app.app_context(manifest):
            pass

    with pytest.raises(ValueError, match="same name"):
        asyncio.run(run())


def test_agent_not_in_include_is_not_registered():
    # Two agents share a module; only ``kept_agent`` is included, so the gate's
    # should_include_agent false branch leaves ``dropped_agent`` unregistered.
    manifest = Manifest.model_validate(
        {"agents": [{"title": "agents", "module": "tests.app._fixtures.agents_mod", "include": ["kept_agent"]}]}
    )

    async def run():
        async with app.app_context(manifest):
            assert app.agents.get_agent("kept_agent").tool_name == "kept_agent"
            with pytest.raises(RuntimeError, match="No such agent"):
                app.agents.get_agent("dropped_agent")

    asyncio.run(run())


def _fake_mcp_tool(name):
    return type(
        "_FakeMcpTool",
        (),
        {
            "name": name,
            "description": name,
            "inputSchema": {"type": "object", "properties": {}},
            "outputSchema": {},
        },
    )()


def test_mcp_tools_binds_probed_tool(monkeypatch):
    cfg = TaiMCPConfig(title="probed", include=[], config=MCPConfig(type="http", url="http://x/mcp"))
    monkeypatch.setattr(app, "_probe_mcp", AsyncMock(return_value=[_fake_mcp_tool("ping")]))

    async def run():
        async with app.app_context(Manifest.model_validate({"mcp": [cfg.model_dump()]})):
            assert "probed_ping" in await app.tools.get_tools()

    asyncio.run(run())


def test_mcp_tools_skips_excluded_tool(monkeypatch):
    # Two probed tools; the manifest excludes one by its normalized name, so
    # ``mcp_tools`` skips binding it (the should_include_mcp_tool false branch).
    cfg = TaiMCPConfig(title="ex", include=[], exclude=["ex_ping2"], config=MCPConfig(type="http", url="http://x/mcp"))
    monkeypatch.setattr(app, "_probe_mcp", AsyncMock(return_value=[_fake_mcp_tool("ping"), _fake_mcp_tool("ping2")]))

    async def run():
        async with app.app_context(Manifest.model_validate({"mcp": [cfg.model_dump()]})):
            tools = await app.tools.get_tools()
            assert "ex_ping" in tools
            assert "ex_ping2" not in tools

    asyncio.run(run())
