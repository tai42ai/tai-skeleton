import logging
from typing import TYPE_CHECKING, Any, Literal, cast

from fastmcp import FastMCP
from fastmcp.server.http import StarletteWithLifespan, create_sse_app
from fastmcp.server.server import Transport
from starlette.middleware import Middleware
from tai_contract.connectors.providers import ProviderDescriptor
from tai_contract.connectors.store import ConnectorTokenStore
from tai_contract.manifest import TaiMCPConfig
from tai_contract.storage import Storage

from tai_skeleton.agent.binding import AgentBinding
from tai_skeleton.app.channels_facet import ChannelsFacet
from tai_skeleton.app.clients import ClientsFacet
from tai_skeleton.app.facets import (
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
from tai_skeleton.app.http import HttpSurface
from tai_skeleton.app.lifecycle import TaiMCPLifecycleMixin
from tai_skeleton.app.reload_gate import reload_gate
from tai_skeleton.app.sessions import ReloadRejectionMiddleware, SessionRegistry, SessionTrackingMiddleware
from tai_skeleton.app.sub_mcp_app import SubMcpAppRouter
from tai_skeleton.backend.registry import BackendHolder
from tai_skeleton.backup import BackupRegistry, register_core_sections
from tai_skeleton.channels.registry import ChannelRegistry
from tai_skeleton.config import ConfigManagerFactory
from tai_skeleton.middleware.body_limit import BodyLimitMiddleware
from tai_skeleton.middleware.rate_limit import RateLimitMiddleware
from tai_skeleton.presets.manager import PresetManager
from tai_skeleton.storage import StorageRegistry
from tai_skeleton.template import ResourceManager
from tai_skeleton.tools.binding import ToolBinding
from tai_skeleton.webhooks.registry import WebhookVerifierRegistry

if TYPE_CHECKING:
    from fastmcp.tools import Tool
    from tai_contract.app import TaiApp
    from tai_contract.presets import PresetStore

    from tai_skeleton.versioning.store import PostgresVersionedStore

logger = logging.getLogger(__name__)


class TaiMCP(TaiMCPLifecycleMixin):
    """The concrete ``tai_contract.app.TaiApp`` impl — owns the FastMCP server and
    exposes the 18 contract facet namespaces as its SOLE feature/contract surface;
    the concrete server additionally exposes a launch surface (``sse_app`` /
    ``http_app`` / ``run`` and friends) that is not part of the facade.

    This class is the composition root only: each feature's impl body lives in
    its feature package (``tools.binding.ToolBinding``, ``agent.binding
    .AgentBinding``, ``backend.registry.BackendHolder``, the extension/monitoring
    registries, ``app.http.HttpSurface``) and each facet forwards straight to its
    collaborator. Callers reach the app's features only through the facets
    (``app.tools.run_tool``, ``app.backends.backend``, ...) or the ``tai_app``
    handle, never a flat member."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._storage_registry: StorageRegistry = StorageRegistry()
        self._resource_manager_cache: ResourceManager | None = None
        self._config_manager = ConfigManagerFactory.create()
        self._mcp_sub_app_router: SubMcpAppRouter = SubMcpAppRouter(app=self)
        self._clients: ClientsFacet = ClientsFacet()

        # Active-MCP-session registry + the middleware that captures live
        # sessions on every incoming message. Built once (the FastMCP server
        # outlives every reload) so the list_changed broadcast primitive sees
        # every connected client. Registered on the raw server here rather than
        # through a facet — it is protocol-level infra, not a feature.
        self._session_registry = SessionRegistry()
        self._fast_mcp.add_middleware(SessionTrackingMiddleware(self._session_registry))
        # A session ``tools/call`` is a run surface too: reject it with the same
        # retriable "reloading" error while the reload gate is held, so the MCP
        # half of the run surface is protected alongside ``POST /api/run-tool``.
        self._fast_mcp.add_middleware(ReloadRejectionMiddleware(reload_gate))
        # Tool-edge authorization for projected operations. Installed on the main
        # server AND (in ``_build_sub_app``) on every sub-MCP mount, so no
        # projected op is dispatchable externally without the same permission
        # decision the HTTP edge makes. Imported locally to avoid an import cycle
        # (authz -> operations -> app -> server).
        from tai_skeleton.authz.middleware import AuthzMiddleware

        self._fast_mcp.add_middleware(AuthzMiddleware(self))

        # Per-feature impl collaborators — the bodies behind the facets.
        self._tool_binding = ToolBinding(self)
        self._agent_binding = AgentBinding(self)
        self._backend_holder = BackendHolder()
        self._http_surface = HttpSurface(self)
        # The public webhook doors (universal_webhook + interactions callback) are
        # exposed by design; their flood limiter is registered here at construction
        # so it is always on (tunable/disable via TAI_RATE_LIMIT_*), never left to a
        # manifest opt-in an operator could forget. It no-ops for every other path.
        self._http_surface.middleware(RateLimitMiddleware)

        # Webhook-verifier registry: reset each start() so a reload re-imports the
        # manifest's verifier modules and re-registers cleanly (mirrors the agent
        # binding reset).
        self._webhook_verifier_registry = WebhookVerifierRegistry()

        # Channel registry: reset each start() so a reload re-imports the
        # manifest's channel modules and re-registers cleanly (mirrors the
        # webhook-verifier registry above).
        self._channel_registry = ChannelRegistry()

        # The backup registry is the host's first consumer of its own AppBackup
        # facet: build it once per app object and register the core host sections
        # here (never on reload, which re-imports modules but keeps this object),
        # so the duplicate-name guard is never tripped.
        self._backup_registry = BackupRegistry()
        register_core_sections(self._backup_registry)

        # The 18 contract facet namespaces, partitioning the feature surface.
        self._tools_facet = ToolsFacet(self)
        self._agents_facet = AgentsFacet(self)
        self._backends_facet = BackendsFacet(self)
        self._storage_facet = StorageFacet(self)
        self._connectors_facet = ConnectorsFacet(self)
        self._webhook_verifiers_facet = WebhookVerifiersFacet(self)
        self._channels_facet = ChannelsFacet(self)
        self._monitoring_facet = MonitoringFacet(self)
        self._extensions_facet = ExtensionsFacet(self)
        self._http_facet = HttpFacet(self)
        self._lifecycle_facet = LifecycleFacet(self)
        self._admin_facet = AdminFacet(self)
        self._config_facet = ConfigFacet(self)
        self._sub_app_facet = SubAppFacet(self)
        self._backup_facet = BackupFacet(self)
        self._versioning_facet = VersioningFacet(self)
        self._presets_facet = PresetsFacet(self)

        # The preset register/reload engine — a process-lifetime singleton (like
        # the tool binding) owning the authoritative spec map + quarantine set, so
        # its state survives a ``reload_config`` that swaps the manifest registries
        # beneath it. Reached by the preset routes and the startup/reload
        # rehydration hook via this concrete instance.
        self._preset_manager = PresetManager(self)

    # -- Facet namespaces (tai_contract.app.TaiApp) --------------------------

    @property
    def tools(self) -> ToolsFacet:
        return self._tools_facet

    @property
    def agents(self) -> AgentsFacet:
        return self._agents_facet

    @property
    def backends(self) -> BackendsFacet:
        return self._backends_facet

    @property
    def storage(self) -> StorageFacet:
        return self._storage_facet

    @property
    def connectors(self) -> ConnectorsFacet:
        return self._connectors_facet

    @property
    def webhook_verifiers(self) -> WebhookVerifiersFacet:
        return self._webhook_verifiers_facet

    @property
    def channels(self) -> ChannelsFacet:
        return self._channels_facet

    @property
    def monitoring(self) -> MonitoringFacet:
        return self._monitoring_facet

    @property
    def extensions(self) -> ExtensionsFacet:
        return self._extensions_facet

    @property
    def http(self) -> HttpFacet:
        return self._http_facet

    @property
    def clients(self) -> ClientsFacet:
        return self._clients

    @property
    def lifecycle(self) -> LifecycleFacet:
        return self._lifecycle_facet

    @property
    def admin(self) -> AdminFacet:
        return self._admin_facet

    @property
    def config(self) -> ConfigFacet:
        return self._config_facet

    @property
    def sub_app(self) -> SubAppFacet:
        return self._sub_app_facet

    @property
    def backup(self) -> BackupFacet:
        return self._backup_facet

    @property
    def versioning(self) -> VersioningFacet:
        return self._versioning_facet

    @property
    def presets(self) -> PresetsFacet:
        return self._presets_facet

    # -- Raw FastMCP escape hatch (skeleton-only, ungoverned) ----------------

    @property
    def fastmcp(self) -> FastMCP:
        """The raw, ungoverned FastMCP server — the escape hatch beneath the
        facets.

        Prefer the facets; reach here only for what the facets don't wrap
        (prompts, resources, ``add_middleware``, sampling, elicit-handlers,
        completions, server metadata such as ``name``/``version``/``auth``, the
        process-global ``fastmcp.settings``). Anything registered THROUGH this
        server skips the platform's governance: manifest gating, the extension
        registry's ``validate()``, and the access-control gate. It is
        deliberately NOT on the ``tai_contract.app.TaiApp`` protocol — the
        contract stays FastMCP-free so an alternative impl remains possible, so
        this accessor is skeleton-specific.

        Named ``fastmcp`` (not ``mcp``) because ``app.sub_app`` already owns the
        sub-MCP namespace; ``mcp`` here would read as the sub-MCP."""
        return self._fast_mcp

    async def emit_list_changed(self, kind: str) -> None:
        """Broadcast a ``list_changed`` notification to every active MCP session
        for the given SINGULAR registry ``kind`` (``tool`` / ``prompt`` /
        ``resource``). The generic in-process registration-mutation path (e.g. a
        dev's runtime ``add_prompt`` via ``app.fastmcp``) awaits this after its
        own registry mutation; the reload path drives the same registry from its
        sync scheduler."""
        await self._session_registry.emit_list_changed(kind)

    # -- Live server-surface members (concrete launch surface, not facets) ----

    @property
    def _live_manifest(self) -> dict[str, Any]:
        if self._manifest is None:
            raise RuntimeError("TaiMCP is not started — call start()/app_context first.")
        return self._manifest.live_manifest.model_dump(mode="json", exclude_none=True)

    def _with_body_limit(self, middleware: list[Middleware] | None) -> list[Middleware]:
        # The app-level body-size cap is the backstop on EVERY route (authed routes
        # read their bodies unbounded otherwise); always on, tune via
        # TAI_BODY_LIMIT_MAX_BODY_BYTES. It MUST sit inside the base app's own
        # Starlette stack (its own ``ServerErrorMiddleware``), not as an outer
        # finalize wrapper: an over-cap escape (``_BodyTooLarge``) has to reach
        # BodyLimitMiddleware and become a 413 before any error handler commits a 500.
        # RateLimitMiddleware, by contrast, rejects before the app is entered, so it
        # stays an outer finalize wrapper.
        return [Middleware(BodyLimitMiddleware), *(middleware or [])]

    def sse_app(
        self,
        path: str | None = None,
        message_path: str | None = None,
        middleware: list[Middleware] | None = None,
    ) -> StarletteWithLifespan:
        actual_path = path if path is not None else "/sse"
        actual_message_path = message_path if message_path is not None else "/messages"

        base_app = create_sse_app(
            server=self._fast_mcp,
            sse_path=actual_path,
            message_path=actual_message_path,
            auth=self._fast_mcp.auth,
            middleware=self._with_body_limit(middleware),
        )

        return self._http_surface.finalize(base_app)

    def http_app(
        self,
        path: str | None = None,
        middleware: list[Middleware] | None = None,
        json_response: bool | None = None,
        stateless_http: bool | None = None,
        transport: Literal["http", "streamable-http", "sse"] = "http",
    ) -> StarletteWithLifespan:

        base_app = self._fast_mcp.http_app(
            path=path,
            middleware=self._with_body_limit(middleware),
            json_response=json_response,
            stateless_http=stateless_http,
            transport=transport,
        )

        return self._http_surface.finalize(base_app)

    async def run_async(
        self, transport: Transport | None = None, show_banner: bool = True, **transport_kwargs: Any
    ) -> None:
        await self._fast_mcp.run_async(transport, show_banner, **transport_kwargs)

    def run(self, transport: Transport | None = None, show_banner: bool = True, **transport_kwargs: Any) -> None:
        self._fast_mcp.run(transport, show_banner, **transport_kwargs)

    async def run_backend(self, args) -> None:
        await self._backend_holder.launch(args)

    # -- Storage / resources -------------------------------------------------

    @property
    def _resource_manager(self) -> ResourceManager:
        if not self._resource_manager_cache:
            self._resource_manager_cache = ResourceManager(self._storage_registry.provider)
        return self._resource_manager_cache

    def _register_storage(self, cls: type[Storage] | None = None):
        return self._storage_registry.register_storage(cls)

    # -- Connectors (AppConnectors facet body) -------------------------------

    def _register_connector(self, descriptor: ProviderDescriptor) -> None:
        """Register an OAuth connector provider from its pure descriptor data.

        Forwarded by the ``tai_app.connectors`` handle when the manifest loads a
        provider plugin. A connector is pure data, so this is a plain call, not a
        decorator — it stores the descriptor in the engine registry."""
        from tai_skeleton.connectors.providers.registry import register_connector

        register_connector(descriptor)

    @property
    def _token_store(self) -> ConnectorTokenStore:
        return self._connector_token_store()

    @staticmethod
    def _connector_token_store() -> ConnectorTokenStore:
        from tai_skeleton.connectors.store import token_store

        return token_store()

    # -- Versioning + presets seams --------------------------------------------
    # ``app.versioning.store`` and ``app.presets.store`` forward here; ``bind`` is
    # the kernel every preset builds its live tool through.

    @property
    def _versioned_store(self) -> "PostgresVersionedStore":
        # Concretely typed (not the ``VersionedStore`` protocol) so the batched
        # ``list_active_bodies`` accessor — a concrete-only method — resolves
        # through this reference. The ``app.versioning.store`` facet re-narrows to
        # the protocol for the contract surface.
        from tai_skeleton.versioning import versioned_store

        return versioned_store()

    @property
    def preset_manager(self) -> PresetManager:
        """The preset register/reload engine (spec map + quarantine set + register/
        reload/remove/rehydrate). Skeleton-only surface — like ``emit_list_changed``
        and ``fastmcp``, it is deliberately not on the ``tai_contract.app.TaiApp``
        protocol; the preset routes and the startup/reload rehydration hook reach
        it through this concrete instance."""
        return self._preset_manager

    @property
    def _preset_store(self) -> "PresetStore":
        from tai_skeleton.presets import preset_store

        # Wire the engine's collision predicate so ``create_preset`` raises
        # ``PresetNameConflictError`` BEFORE any store write when a name collides
        # with a live non-preset base tool.
        return preset_store(name_conflicts=self._preset_manager.name_conflicts)

    async def _preset_bind(
        self,
        base_tool: str,
        fixed_kwargs: dict[str, Any],
        *,
        name: str,
        description: str = "",
        tags: list[str] | None = None,
        output_schema: dict[str, Any] | None = None,
    ) -> "Tool":
        from tai_skeleton.presets import preset_bind

        # The concrete app IS a structural ``TaiApp`` (asserted by the conformance
        # test); the cast bridges pyright's nominal facet-return-type variance.
        return await preset_bind(
            cast("TaiApp", self),
            base_tool,
            fixed_kwargs,
            name=name,
            description=description,
            tags=tags,
            output_schema=output_schema,
        )

    # -- Lifecycle seam --------------------------------------------------------

    def _mcp_tools(self, config: TaiMCPConfig, tools):
        # The mixin's re-init path binds remote-MCP tools through this seam.
        self._tool_binding.mcp_tools(config, tools)
