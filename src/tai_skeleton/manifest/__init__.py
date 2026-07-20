"""Manifest impl: the include/exclude filtering + derived-map building +
surgical reload logic behind the ``tai_contract.manifest.Manifest`` interface.

The model shape, the ``*Config`` models, the model validators, and the public
predicate/query signatures are the contract (``tai_contract.manifest``). This
module owns the logic: ``model_post_init`` plus the private ``_build_*`` /
``_should_include`` helpers that build the derived lookup maps, and the concrete
bodies of the filter predicates / ``replace_mcp`` / ``live_manifest`` /
``find_title``.
"""

from collections import defaultdict
from copy import deepcopy
from typing import Literal

from pydantic import Field
from tai_contract.manifest import (
    AgentsConfig,
    BaseConfig,
    ExtensionElement,
    MCPConfig,
    TaiMCPConfig,
    ToolsConfig,
)
from tai_contract.manifest import (
    Manifest as ManifestContract,
)


class Manifest(ManifestContract):
    """Concrete manifest: builds the derived lookup maps and runs the
    include/exclude filtering declared by ``tai_contract.manifest.Manifest``."""

    # The contract declares the derived lookup fields as optional; this impl
    # always builds them in ``model_post_init``, so they are non-None here.
    tools_list: set[str] = Field(default_factory=set, exclude=True)
    tool_extensions: dict[str, list[list[ExtensionElement]]] = Field(default_factory=dict, exclude=True)
    tools_map: dict[str, ToolsConfig] = Field(default_factory=dict, exclude=True)
    agents_map: dict[str, AgentsConfig] = Field(default_factory=dict, exclude=True)
    mcp_map: dict[str, TaiMCPConfig] = Field(default_factory=dict, exclude=True)
    tools_module_title_map: dict[str, str] = Field(default_factory=dict, exclude=True)

    # Derived include/exclude sets per config key, frozen to match the contract's
    # ``dict[str, frozenset[str]]`` shape. Always built in ``model_post_init``,
    # so non-None here.
    include_module_tools_map: dict[str, frozenset[str]] = Field(default_factory=dict, exclude=True)
    exclude_module_tools_map: dict[str, frozenset[str]] = Field(default_factory=dict, exclude=True)
    include_module_agents_map: dict[str, frozenset[str]] = Field(default_factory=dict, exclude=True)
    exclude_module_agents_map: dict[str, frozenset[str]] = Field(default_factory=dict, exclude=True)
    include_title_mcp_tools_map: dict[str, frozenset[str]] = Field(default_factory=dict, exclude=True)
    exclude_title_mcp_tools_map: dict[str, frozenset[str]] = Field(default_factory=dict, exclude=True)

    # Resolved include/exclude decisions recorded per config key as tools/agents/
    # mcp-tools are gated. Kept OUT of the emitted config (``exclude=True``) so
    # ``live_manifest`` reports the operator's original include/exclude lists
    # rather than being polluted with every auto-resolved base name.
    resolved_includes: dict[str, set[str]] = Field(default_factory=dict, exclude=True)
    resolved_excludes: dict[str, set[str]] = Field(default_factory=dict, exclude=True)

    def model_post_init(self, __context) -> None:
        (
            self.include_module_tools_map,
            self.exclude_module_tools_map,
            self.tools_map,
        ) = self._build_map(self.tools, key="module")
        (
            self.include_module_agents_map,
            self.exclude_module_agents_map,
            self.agents_map,
        ) = self._build_map(self.agents, key="module")
        (
            self.include_title_mcp_tools_map,
            self.exclude_title_mcp_tools_map,
            self.mcp_map,
        ) = self._build_map(self.mcp, key="title")
        self._rebuild_tools_and_extensions()
        self.tools_module_title_map = {cfg.module: cfg.title for cfg in self.tools}

    def _rebuild_tools_and_extensions(self) -> None:
        # Rebuild ``tools_list`` (union of the include lists across tools + mcp)
        # and ``tool_extensions`` from scratch. Shared by ``model_post_init`` and
        # ``replace_mcp`` so the two paths cannot drift.
        self.tools_list = set()
        for cfg in self.tools + self.mcp:
            self.tools_list.update(cfg.include)
        self.tool_extensions = self._build_tool_extensions()

    def _build_tool_extensions(self) -> dict[str, list[list[ExtensionElement]]]:
        # Union the ``extensions`` maps of tools + mcp AND ``api_tools`` into a
        # single attachment map (tool base-name -> combos), applied AFTER
        # selection and independent of it. ``api_tools.extensions`` is the home
        # for combos whose base is a PROJECTED operation, so a combo over a
        # projected op attaches exactly like a ``tools[]``-entry combo.
        tool_extensions: dict[str, list[list[ExtensionElement]]] = defaultdict(list)
        for cfg in self.tools + self.mcp:
            for base, combos in cfg.extensions.items():
                tool_extensions[base].extend(combos)
        for base, combos in self.api_tools.extensions.items():
            tool_extensions[base].extend(combos)
        return dict(tool_extensions)

    def _build_map(
        self, configs, key: Literal["title", "module"] = "title"
    ) -> tuple[dict[str, frozenset[str]], dict[str, frozenset[str]], dict]:
        include_map: dict[str, frozenset[str]] = {}
        exclude_map: dict[str, frozenset[str]] = {}
        config_map: dict = {}
        failures: list[tuple[object, Exception]] = []

        for cfg in configs:
            try:
                k = cfg.title if key == "title" else cfg.module
                include = frozenset(cfg.include)
                exclude = frozenset(cfg.exclude)
            except Exception as exc:
                failures.append((cfg, exc))
                continue
            # A repeated module/title is ambiguous: the config map would keep only
            # the last row while its include/exclude lists still gate loading, so
            # the earlier row silently vanishes from every view. Reject it loudly.
            if k in config_map:
                raise ValueError(f"manifest: duplicate {key} {k!r} — each {key} must map to exactly one config row")
            config_map[k] = deepcopy(cfg)
            include_map[k] = include
            exclude_map[k] = exclude

        # A malformed config entry must fail the build LOUDLY, not be silently
        # omitted from the maps while the build reports success: omitting an entry
        # silently changes which tools/agents/mcp the process loads.
        if failures:
            detail = "; ".join(f"{cfg!r}: {exc}" for cfg, exc in failures)
            raise ValueError(f"manifest: failed to build map entries: {detail}") from failures[0][1]

        return include_map, exclude_map, config_map

    def _should_include(
        self,
        name: str,
        key: str,
        prefix: str,
        is_title_key: bool = False,
    ) -> bool:
        if not is_title_key:
            include_map_attr = f"include_{prefix}_map"
            modules_set: set[str] = set(getattr(self, include_map_attr).keys())
            try:
                key = self._find_module(key, modules_set)
            except ImportError:
                # A tool/agent registered by an explicitly-loaded plugin module —
                # the backend/storage/monitoring module or a lifecycle module named
                # directly in the manifest, not a ``tools:`` entry — has no
                # include/exclude list to filter against. The module was loaded on
                # purpose, so its registrations are always included rather than
                # aborting boot as unknown configuration.
                if self._is_plugin_module(key):
                    return True
                raise
        include_map = getattr(self, f"include_{prefix}_map")
        exclude_map = getattr(self, f"exclude_{prefix}_map")
        should_include = name in include_map[key] or (not include_map[key] and name not in exclude_map[key])
        # Record the resolved decision off to the side, never into the emitted
        # ``cfg.include``/``cfg.exclude`` (which back ``live_manifest``).
        if should_include:
            self.resolved_includes.setdefault(key, set()).add(name)
        else:
            self.resolved_excludes.setdefault(key, set()).add(name)
        return should_include

    def should_include_tool(self, name: str, module: str) -> bool:
        return self._should_include(name, module, "module_tools", is_title_key=False)

    def should_include_agent(self, name: str, module: str) -> bool:
        return self._should_include(name, module, "module_agents", is_title_key=False)

    def should_include_mcp_tool(self, name: str, title: str) -> bool:
        return self._should_include(name, title, "title_mcp_tools", is_title_key=True)

    def replace_mcp(self, mcp: list["TaiMCPConfig"]) -> None:
        """Swap the MCP rows and rebuild the derived maps.

        The surgical-reload path grafts the manifest's CURRENT MCP section
        into a running process (an MCP row added/removed after boot must be
        loadable/removable without a full re-init). ``tools_list`` is rebuilt
        from scratch; the live ToolRegistry holds its own copy, so this only
        affects the next registry built from this manifest.
        """
        # Drop stale resolved decisions keyed by any MCP title (the rows leaving
        # AND the rows arriving) before the rebuild: ``_should_include`` accumulates
        # into these dicts forever, so a narrowed include list would otherwise leave
        # old tool names recorded under the title and corrupt extension ownership.
        # Fresh ``should_include_mcp_tool`` calls during rebind repopulate them.
        for title in {cfg.title for cfg in self.mcp} | {cfg.title for cfg in mcp}:
            self.resolved_includes.pop(title, None)
            self.resolved_excludes.pop(title, None)
        self.mcp = list(mcp)
        (
            self.include_title_mcp_tools_map,
            self.exclude_title_mcp_tools_map,
            self.mcp_map,
        ) = self._build_map(self.mcp, key="title")
        self._rebuild_tools_and_extensions()

    @property
    def live_manifest(self) -> "Manifest":
        manifest = deepcopy(self)
        manifest.tools = list(self.tools_map.values())
        manifest.agents = list(self.agents_map.values())
        manifest.mcp = list(self.mcp_map.values())
        return manifest

    def find_title(self, module: str) -> str:
        parts = module.split(".")
        candidates = [".".join(parts[:i]) for i in range(len(parts), 0, -1)]
        found = [candidate for candidate in candidates if candidate in self.tools_module_title_map]
        if found:
            return self.tools_module_title_map[found[0]]

        return module

    def _is_plugin_module(self, module: str) -> bool:
        """Whether ``module`` belongs to an explicitly-loaded plugin the manifest
        names directly (the backend/storage/monitoring module or a lifecycle
        module) rather than a ``tools:`` entry. Such a module's registered
        tools/agents are always included — they carry no include/exclude list."""
        roots = [self.backend_module, self.storage_module, self.monitoring_module, *self.lifecycle_modules]
        return any(root and (module == root or module.startswith(f"{root}.")) for root in roots)

    @staticmethod
    def _find_module(current_module: str, modules: set[str]) -> str:
        parts = current_module.split(".")
        candidates = [".".join(parts[:i]) for i in range(len(parts), 0, -1)]
        found = [module for module in candidates if module in modules]
        if not found:
            raise ImportError(f"module {current_module} not found in manifest.")

        return found[0]


__all__ = [
    "AgentsConfig",
    "BaseConfig",
    "MCPConfig",
    "Manifest",
    "TaiMCPConfig",
    "ToolsConfig",
]
