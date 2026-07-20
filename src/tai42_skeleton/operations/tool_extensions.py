"""Tool-extension operations — read and author a tool's applied extension combos.

The ``extensions`` MAP on a ``tools``/``mcp`` config is the single source of truth
for which clip-on powers (chain/batch/monitor/…) a tool carries, decoupled from
``include``/``exclude`` (selection). These operations read and write that map
through the single :class:`~tai42_skeleton.config.service.ConfigService` pipeline: the
mutator upserts the providing config's ``extensions`` entry on the PRESERVED
document, then ConfigService validates, persists, reloads locally, and broadcasts
the reload — so the map, the bound branch tools, and the whole-fleet surface stay in
sync automatically (every worker re-reads the persisted manifest and rebinds).

The concrete app singleton (``from tai42_skeleton.app import instance``) is reached
for the live manifest's resolved selection maps, the MCP-bound-tool map, and the
extension registry — none of which ride the emitted ``live_manifest`` dict.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from tai42_contract.manifest import ExtensionElement

from tai42_skeleton.app import instance
from tai42_skeleton.app.boot_rules import BackendNeedsBusError
from tai42_skeleton.config.service import ConfigService
from tai42_skeleton.exceptions.exceptions import TaiValidationError
from tai42_skeleton.manifest import Manifest
from tai42_skeleton.operations import BadRequestError, ConflictError, NotFoundError, operation
from tai42_skeleton.operations._broadcast import apply_response


class ToolExtensionsUpdate(BaseModel):
    """Set a tool's applied extension combos — the full list of combos, authored
    losslessly. Each combo element is an extension name or a ``{"name", "config"}``
    mapping binding author config. An empty list clears them."""

    combos: list[list[ExtensionElement]]


def _live_manifest() -> Manifest:
    return instance.app.admin.live_manifest_typed


def _owning_configs(manifest: Manifest, name: str) -> list[tuple[str, str]]:
    """Every config that PROVIDES ``name``, as ``(kind, key)`` pairs (``key`` is
    the module for a ``tools`` config, the title for an ``mcp`` config). A local
    tool is owned when its resolved selection recorded it under the config's
    module; an mcp tool when resolved selection OR the live mcp-bound-tool map
    recorded it under the config's title.

    A branch/composed tool (the tool an extension combo PRODUCES, e.g.
    ``weather_chain``) is never itself an extension TARGET — only its base tool is —
    so it reports no owner. The mcp-bound-tool map records branch names alongside
    bases, so without this guard an mcp branch tool would resolve a bogus owner and
    corrupt the manifest on write; excluding it makes a branch tool report no owner
    exactly as the base-only ``resolved_includes`` path already does for local
    tools."""
    if instance.app.tools.is_branch(name):
        return []
    owners: list[tuple[str, str]] = []
    for cfg in manifest.tools:
        if name in manifest.resolved_includes.get(cfg.module, set()):
            owners.append(("tools", cfg.module))
    for cfg in manifest.mcp:
        if name in manifest.resolved_includes.get(cfg.title, set()) or name in instance.app.tools.mcp_bound_names(
            cfg.title
        ):
            owners.append(("mcp", cfg.title))
    return owners


def _other_mappers(manifest: Manifest, name: str, owner: tuple[str, str]) -> list[str]:
    """The configs OTHER than the providing one whose ``extensions`` map also
    carries ``name`` — a write would leave their combos behind and the union GET
    would disagree with what was authored, so any such config is a 409."""
    kind, key = owner
    others: list[str] = []
    for cfg in manifest.tools:
        if name in cfg.extensions and not (kind == "tools" and cfg.module == key):
            others.append(f"tools:{cfg.module}")
    for cfg in manifest.mcp:
        if name in cfg.extensions and not (kind == "mcp" and cfg.title == key):
            others.append(f"mcp:{cfg.title}")
    return others


def _validate_combos_against_registry(combos: list[list[ExtensionElement]]) -> None:
    """Reject an unknown extension name and a double-non-stackable-kind combo
    against the LIVE registry, before any persist — one bad combo fails the whole
    write. Delegates to the ``app.extensions.validate_combo`` accessor (the shared
    combo-validation seam) so the manifest can never carry an illegal combo."""
    for combo in combos:
        try:
            instance.app.extensions.validate_combo(combo)
        except TaiValidationError as exc:
            raise BadRequestError(str(exc)) from exc


def _apply_combos(
    manifest_dict: dict[str, Any], owner: tuple[str, str], name: str, combos: list[list[ExtensionElement]]
) -> bool:
    """Upsert the providing config's ``extensions[name]`` entry in the persisted
    manifest dict — ``include``/``exclude`` are untouched. An empty ``combos``
    drops the tool's key (and the whole ``extensions`` map if it empties), never
    writing a present-but-empty value. Returns whether the providing config entry
    was found."""
    kind, key = owner
    section, key_field = ("tools", "module") if kind == "tools" else ("mcp", "title")
    for entry in manifest_dict.get(section, []):
        if entry.get(key_field) == key:
            extensions = dict(entry.get("extensions", {}))
            if combos:
                extensions[name] = combos
            else:
                extensions.pop(name, None)
            if extensions:
                entry["extensions"] = extensions
            else:
                entry.pop("extensions", None)
            return True
    return False


@operation(summary="Get a tool's applied extension combos", tags=["extensions"], errors=[NotFoundError])
async def get_tool_extensions(name: str) -> dict:
    tools = await instance.app.tools.get_tools()
    if name not in tools:
        raise NotFoundError(f"tool {name!r} is not a registered tool")
    manifest = _live_manifest()
    combos = [list(combo) for combo in manifest.tool_extensions.get(name, [])]
    return {"combos": combos, "available": instance.app.extensions.available_extensions()}


@operation(
    summary="Set a tool's applied extension combos",
    tags=["extensions"],
    destructive=True,
    reload_gated=True,
    errors=[BadRequestError, ConflictError],
    request_model=ToolExtensionsUpdate,
)
async def set_tool_extensions(name: str, combos: list[list[ExtensionElement]]) -> object:
    """Author all of a tool's extension combos through the pipeline: persist, reload,
    and broadcast.

    Owner resolution, the consolidation-conflict check, and the registry combo
    validation all run BEFORE the pipeline (nothing persisted on a reject). The
    mutator then upserts the providing config's ``extensions`` entry, and
    ConfigService validates the whole document, persists, reloads locally, and
    broadcasts the reload so every worker rebinds; the per-origin fleet report rides
    the response as its ``fanout`` summary. The persist + local reload have already
    landed, so re-running the apply (or a ``reload_config``) is the recovery for a
    sibling that did not converge.
    """
    manifest = _live_manifest()
    owners = _owning_configs(manifest, name)
    if not owners:
        raise BadRequestError(f"tool {name!r} is not currently provided by any config")
    if len(owners) > 1:
        where = ", ".join(f"{kind}:{key}" for kind, key in owners)
        raise BadRequestError(
            f"tool {name!r} is provided by multiple configs ({where}); cannot determine which to edit"
        )
    owner = owners[0]

    conflicting = _other_mappers(manifest, name, owner)
    if conflicting:
        raise ConflictError(
            f"tool {name!r} extensions are also mapped by other config(s) ({', '.join(conflicting)}); "
            "consolidate the mapping into the providing config first"
        )

    _validate_combos_against_registry(combos)

    def mutator(document: dict[str, Any]) -> None:
        # Upsert the providing config's ``extensions[name]`` on the PRESERVED
        # document. A providing config that vanished from the persisted manifest (a
        # concurrent edit) refuses loudly rather than writing a config that is gone.
        if not _apply_combos(document, owner, name, combos):
            raise BadRequestError(f"providing config for tool {name!r} not found in the manifest")

    # ConfigService validates the whole document with the new map BEFORE persisting,
    # so a malformed entry rejects loudly (400) instead of corrupting the store.
    try:
        result = await ConfigService.from_app().apply_change(mutator)
    except BackendNeedsBusError as exc:
        # The invariant is a RuntimeError (a boot-time refusal must still crash loudly),
        # so the mutate-time path maps it explicitly to a loud, actionable 400 naming
        # TAI_BUS_REDIS_URL rather than letting it escape as a 500.
        raise BadRequestError(str(exc)) from exc
    except ValueError as exc:
        raise BadRequestError(f"invalid extensions for tool {name!r}: {exc}") from exc
    return apply_response(result)
