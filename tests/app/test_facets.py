"""Facet forwarding: each of the 17 ``tai42_contract.app`` facets is a thin view
that forwards to its feature's impl collaborator (``ToolBinding``,
``AgentBinding``, ``BackendHolder``, the extension registry, ``HttpSurface``) or
to the app's remaining private members. These assert every facet method and
property delegates to the right target with the right arguments.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from tai42_contract.extensions import ExtensionKind
from tai42_contract.presets import PresetBody
from tai42_contract.storage import Storage

import tai42_skeleton.monitoring
from tai42_skeleton.app.facets import (
    AdminFacet,
    AgentsFacet,
    BackendsFacet,
    BackupFacet,
    ConfigFacet,
    ConnectorsFacet,
    ExtensionsFacet,
    HttpFacet,
    LifecycleFacet,
    MonitoringFacet,
    PresetsFacet,
    StorageFacet,
    SubAppFacet,
    ToolsFacet,
    VersioningFacet,
    WebhookVerifiersFacet,
)
from tai42_skeleton.exceptions.exceptions import TaiValidationError


def _app() -> MagicMock:
    app = MagicMock()
    # Async collaborator methods must be awaitable.
    for name in ("get_tool", "get_tools", "get_client_tools", "run_tool"):
        setattr(app._tool_binding, name, AsyncMock(return_value=f"{name}-result"))
    app._run_tool_reload = AsyncMock(return_value="_run_tool_reload-result")
    return app


class _Storage(Storage):
    """Minimal concrete ``Storage`` used as a real ``type[Storage]`` sentinel for
    the storage-facet forwarding assertion."""

    async def load(self, path: str) -> str: ...
    async def list(self) -> list[str]: ...
    async def upload(self, path: str, content: str) -> None: ...
    async def delete(self, path: str) -> None: ...
    async def delete_dir(self, path: str) -> None: ...


def _noop() -> None: ...


# -- ToolsFacet ---------------------------------------------------------------


def test_tools_facet_sync_forwarding():
    app = _app()
    f = ToolsFacet(app)
    # Each facet method both forwards the call AND returns the collaborator's result.
    assert f.tool("a", force=True, name="n") is app._tool_binding.tool.return_value
    app._tool_binding.tool.assert_called_once_with("a", force=True, name="n")
    assert f.toolkit("tk", name="n") is app._tool_binding.toolkit.return_value
    app._tool_binding.toolkit.assert_called_once_with("tk", name="n")
    assert f.tool_title("fn") is app._tool_binding.tool_title.return_value
    app._tool_binding.tool_title.assert_called_once_with("fn")
    assert f.remove_tool("x") is app._tool_binding.remove_tool.return_value
    app._tool_binding.remove_tool.assert_called_once_with("x")
    assert f.register_tool_info("i", [["ext"]]) is app._tool_binding.register_tool_info.return_value
    app._tool_binding.register_tool_info.assert_called_once_with("i", [["ext"]])
    assert f.unregister_tool_info("i") is app._tool_binding.unregister_tool_info.return_value
    app._tool_binding.unregister_tool_info.assert_called_once_with("i")
    assert f.unregister_tool_base("b") is app._tool_binding.unregister_tool_base.return_value
    app._tool_binding.unregister_tool_base.assert_called_once_with("b")


def test_tools_facet_base_of_is_branch_mcp_bound_names_forwarding():
    app = _app()
    f = ToolsFacet(app)
    assert f.base_of("wv_exta") is app._tool_binding.base_of.return_value
    app._tool_binding.base_of.assert_called_once_with("wv_exta")
    assert f.is_branch("wv_exta") is app._tool_binding.is_branch.return_value
    app._tool_binding.is_branch.assert_called_once_with("wv_exta")
    assert f.mcp_bound_names("svc") is app._tool_binding.mcp_bound_names.return_value
    app._tool_binding.mcp_bound_names.assert_called_once_with("svc")


async def test_tools_facet_async_forwarding():
    app = _app()
    f = ToolsFacet(app)
    assert await f.get_tool("k") == "get_tool-result"
    app._tool_binding.get_tool.assert_awaited_once_with("k")
    assert await f.get_tools() == "get_tools-result"
    assert await f.get_client_tools(["a"]) == "get_client_tools-result"
    app._tool_binding.get_client_tools.assert_awaited_once_with(["a"])
    assert await f.run_tool("k", {"x": 1}) == "run_tool-result"
    app._tool_binding.run_tool.assert_awaited_once_with("k", {"x": 1}, offload_sync=False)


# -- AgentsFacet --------------------------------------------------------------


def test_agents_facet_forwarding():
    app = _app()
    f = AgentsFacet(app)
    assert f.agent("n") is app._agent_binding.agent.return_value
    app._agent_binding.agent.assert_called_once_with("n")
    assert f.get_agent("n") is app._agent_binding.get_agent.return_value
    app._agent_binding.get_agent.assert_called_once_with("n")
    assert f.all_agents() is app._agent_binding.all_agents.return_value
    app._agent_binding.all_agents.assert_called_once_with()


# -- WebhookVerifiers / Backup / Versioning / Presets facets ------------------


def test_webhook_verifiers_facet_forwarding():
    app = _app()
    f = WebhookVerifiersFacet(app)
    assert f.register("n", "verifier") is app._webhook_verifier_registry.register.return_value  # pyright: ignore[reportArgumentType]
    app._webhook_verifier_registry.register.assert_called_once_with("n", "verifier")
    assert f.get("n") is app._webhook_verifier_registry.get.return_value
    app._webhook_verifier_registry.get.assert_called_once_with("n")


def test_backup_facet_forwarding():
    app = _app()
    f = BackupFacet(app)

    def _importer(payload: object) -> None: ...

    result = f.register_section("n", _noop, _importer, secret=True)
    assert result is app._backup_registry.register_section.return_value
    app._backup_registry.register_section.assert_called_once_with("n", _noop, _importer, secret=True)
    assert f.sections() is app._backup_registry.sections.return_value
    app._backup_registry.sections.assert_called_once_with()
    assert f.export_section("n") is app._backup_registry.export_section.return_value
    app._backup_registry.export_section.assert_called_once_with("n")
    assert f.import_section("n", {"p": 1}) is app._backup_registry.import_section.return_value
    app._backup_registry.import_section.assert_called_once_with("n", {"p": 1})


def test_versioning_facet_forwarding():
    app = _app()
    app._versioned_store = "vs"
    assert VersioningFacet(app).store == "vs"


async def test_presets_facet_forwarding():
    app = _app()
    app._preset_store = "ps"
    app._preset_bind = AsyncMock(return_value="bound")
    f = PresetsFacet(app)
    assert f.store == "ps"
    result = await f.bind("base", {"k": 1}, name="p", description="d", tags=["t"])
    assert result == "bound"
    app._preset_bind.assert_awaited_once_with(
        "base", {"k": 1}, name="p", description="d", tags=["t"], output_schema=None
    )


async def test_presets_facet_list_active_bodies_validates_each_raw_body():
    app = _app()
    raw = {
        "a": PresetBody(
            base_tool="echo", description="d", fixed_kwargs={"x": 1}, extensions=[["exta"]], tags=["t"]
        ).model_dump(),
    }
    app._versioned_store = MagicMock()
    app._versioned_store.list_active_bodies = AsyncMock(return_value=raw)
    f = PresetsFacet(app)
    bodies = await f.list_active_bodies()
    # Reaches the concrete store's batched read for the "preset" kind and validates
    # each raw body into a typed PresetBody.
    app._versioned_store.list_active_bodies.assert_awaited_once_with("preset")
    assert set(bodies) == {"a"}
    assert isinstance(bodies["a"], PresetBody)
    assert bodies["a"].base_tool == "echo"
    assert bodies["a"].fixed_kwargs == {"x": 1}
    assert bodies["a"].extensions == [["exta"]]


# -- BackendsFacet ------------------------------------------------------------


def test_backends_facet_forwarding():
    app = _app()
    app._backend_holder.backend = "the-backend"
    f = BackendsFacet(app)
    assert f.register_backend(cls=None) is app._backend_holder.register_backend.return_value
    app._backend_holder.register_backend.assert_called_once_with(None)
    assert f.backend == "the-backend"


# -- StorageFacet -------------------------------------------------------------


def test_storage_facet_forwarding():
    app = _app()
    app._resource_manager = "rm"
    f = StorageFacet(app)
    assert f.register_storage(_Storage) is app._register_storage.return_value
    app._register_storage.assert_called_once_with(_Storage)
    assert f.resource_manager == "rm"


# -- MonitoringFacet ----------------------------------------------------------


def test_monitoring_facet_forwarding(monkeypatch):
    recorded = MagicMock()
    monkeypatch.setattr(tai42_skeleton.monitoring, "register_monitoring", recorded)
    f = MonitoringFacet(_app())
    f.register_monitoring(_noop)
    recorded.assert_called_once_with(_noop)


def test_monitoring_facet_active_returns_the_registered_backend(monkeypatch):
    backend = MagicMock()
    monkeypatch.setattr(tai42_skeleton.monitoring, "get_monitoring", lambda: backend)
    f = MonitoringFacet(_app())
    assert f.active is backend


# -- ExtensionsFacet ----------------------------------------------------------


def test_extensions_facet_forwarding():
    app = _app()
    f = ExtensionsFacet(app)
    assert f.extension(None, kind=ExtensionKind.WRAPPER, name="x") is app._extension_registry.extension.return_value
    app._extension_registry.extension.assert_called_once_with(
        None, kind=ExtensionKind.WRAPPER, name="x", requires_body_locality=False
    )
    assert f.available_extensions() is app._extension_registry.available_extensions.return_value
    app._extension_registry.available_extensions.assert_called_once_with()


def test_extensions_facet_validate_combo_valid_delegates_to_registry():
    app = _app()
    registry = app._extension_registry
    registry.available_extensions.return_value = [{"name": "exta"}, {"name": "extb"}]
    registry.validate = MagicMock()
    f = ExtensionsFacet(app)
    # A known combo passes the unknown-name check and delegates to the registry's
    # non-stackable-kind validation.
    f.validate_combo(["exta"])
    registry.validate.assert_called_once_with(["exta"])


def test_extensions_facet_validate_combo_unknown_name_raises_validation_error():
    app = _app()
    registry = app._extension_registry
    registry.available_extensions.return_value = [{"name": "exta"}]
    registry.validate = MagicMock()
    f = ExtensionsFacet(app)
    with pytest.raises(TaiValidationError) as exc:
        f.validate_combo(["ghost", "also_missing"])
    # Every unknown name is reported, and the registry's own validation is never
    # reached once an unknown name is found.
    assert "ghost" in str(exc.value)
    assert "also_missing" in str(exc.value)
    registry.validate.assert_not_called()


def test_extensions_facet_validate_combo_non_stackable_clash_raises_validation_error():
    app = _app()
    registry = app._extension_registry
    registry.available_extensions.return_value = [{"name": "exta"}, {"name": "extb"}]
    registry.validate = MagicMock(side_effect=TaiValidationError("non-stackable kinds clash"))
    f = ExtensionsFacet(app)
    with pytest.raises(TaiValidationError):
        f.validate_combo(["exta", "extb"])


# -- ConnectorsFacet ----------------------------------------------------------


def test_connectors_facet_forwarding():
    app = _app()
    app._token_store = "ts"
    f = ConnectorsFacet(app)
    assert f.register_connector("descriptor") is app._register_connector.return_value  # pyright: ignore[reportArgumentType]
    app._register_connector.assert_called_once_with("descriptor")
    assert f.token_store == "ts"


# -- HttpFacet ----------------------------------------------------------------


def test_http_facet_forwarding():
    app = _app()
    f = HttpFacet(app)
    assert f.middleware(cls=None, opt=1) is app._http_surface.middleware.return_value
    app._http_surface.middleware.assert_called_once_with(None, opt=1)
    route_result = f.custom_route(
        "/p", ["GET"], name="n", include_in_schema=False, summary="P", tags=["t"], response_model=None
    )
    assert route_result is app._http_surface.custom_route.return_value
    app._http_surface.custom_route.assert_called_once_with(
        "/p",
        ["GET"],
        "n",
        False,
        summary="P",
        tags=["t"],
        response_model=None,
        request_model=None,
        authed=True,
        destructive=False,
        action=None,
        declared=None,
    )


# -- LifecycleFacet -----------------------------------------------------------


def test_lifecycle_facet_forwarding():
    app = _app()
    f = LifecycleFacet(app)
    assert f.on_startup(_noop) is app._on_startup.return_value
    app._on_startup.assert_called_once_with(_noop)
    assert f.on_shutdown(_noop) is app._on_shutdown.return_value
    app._on_shutdown.assert_called_once_with(_noop)
    assert f.on_reload(_noop) is app._on_reload.return_value
    app._on_reload.assert_called_once_with(_noop)


# -- AdminFacet ---------------------------------------------------------------


def test_admin_facet_sync_forwarding():
    app = _app()
    app._live_manifest = {"live": True}
    f = AdminFacet(app)
    assert f.reload_mcp("t") is app._reload_mcp.return_value
    app._reload_mcp.assert_called_once_with("t")
    assert f.deregister_mcp("t") is app._deregister_mcp.return_value
    app._deregister_mcp.assert_called_once_with("t")
    assert f.reload_config() is app._reload_config.return_value
    app._reload_config.assert_called_once_with()
    assert f.tool_reloader("kind") is app._tool_reloader.return_value
    app._tool_reloader.assert_called_once_with("kind")
    assert f.reload_failed_mcps() is app._reload_failed_mcps.return_value
    app._reload_failed_mcps.assert_called_once_with()
    assert f.list_failed_mcps() is app._list_failed_mcps.return_value
    app._list_failed_mcps.assert_called_once_with()
    assert f.live_mcp_status() is app._live_mcp_status.return_value
    app._live_mcp_status.assert_called_once_with()
    assert f.live_manifest == {"live": True}


async def test_admin_facet_run_tool_reload_forwarding():
    app = _app()
    f = AdminFacet(app)
    assert await f.run_tool_reload("k", "reload", "n") == "_run_tool_reload-result"
    app._run_tool_reload.assert_awaited_once_with("k", "reload", "n")


def test_admin_facet_live_manifest_typed_forwards_to_require_live_manifest():
    app = _app()
    f = AdminFacet(app)
    assert f.live_manifest_typed is app._require_live_manifest.return_value
    app._require_live_manifest.assert_called_once_with()


# -- ConfigFacet / SubAppFacet ------------------------------------------------


def test_config_facet_forwarding():
    app = _app()
    app._config_manager = "cm"
    assert ConfigFacet(app).config_manager == "cm"


def test_sub_app_facet_forwarding():
    app = _app()
    app._mcp_sub_app_router = "router"
    assert SubAppFacet(app).mcp_sub_app_router == "router"
