"""Preset operations — author and read versioned presets over the live engine.

A preset is a base tool + baked ``fixed_kwargs`` + extension combos, persisted as a
versioned document and registered as a live named tool. These operations are the
single source of truth for the preset surface: the HTTP routes in
``routers/presets.py`` are thin adapters over them, and the MCP projection binds
each as a tool (so extensions wrap it and other presets can bake over it).

Every mutating door validates a body CAN bind BEFORE any store write (a bad edit is
a loud 400, never a committed version that can never bind), persists THEN registers,
compensates a residual register failure by re-pointing the store so store + live
never diverge, and fans the rebind/removal out on the worker bus. The concrete app
singleton (``from tai_skeleton.app import instance``) is reached for the store view,
the register/reload engine, the generic versioned store (the HARD delete the view
does not expose), and ``emit_list_changed``. Success values are returned bare — the
route adapter wraps them in ``{"data": ...}`` at the HTTP edge.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Annotated, Any

from pydantic import BaseModel, TypeAdapter, ValidationError
from tai_contract.agent.base import PresetSpec
from tai_contract.manifest import ExtensionElement
from tai_contract.presets import CARRY_FORWARD, PresetBody
from tai_contract.presets.errors import (
    PresetExistsError,
    PresetNameConflictError,
    PresetNotFoundError,
    PresetVersionNotFoundError,
)
from tai_contract.versioning.errors import DocumentVersionNotFoundError
from tai_kit.utils.data.json_schema_util import (
    InvalidJsonSchemaError,
    check_json_schema,
)

from tai_skeleton.app import instance
from tai_skeleton.app.bus import LocalApplyResult, OpOutcome
from tai_skeleton.exceptions.exceptions import TaiValidationError
from tai_skeleton.extensions.registry import extension_name
from tai_skeleton.operations import (
    BadRequestError,
    ConflictError,
    NotFoundError,
    UnavailableError,
    operation,
)
from tai_skeleton.operations._broadcast import log_non_convergence
from tai_skeleton.presets.manager import is_valid_preset_name
from tai_skeleton.versioning import versioned_store_configured

logger = logging.getLogger(__name__)


# -- request models (the emitted spec's requestBody schemas) -----------------


class PresetCreate(BaseModel):
    """A preset-creation request. ``extensions`` is the list of extension combos
    (each element an extension name or a ``{"name", "config"}`` mapping binding
    author config). ``output_schema`` is the optional author-set OUTPUT JSON
    Schema (an object schema)."""

    name: str
    base_tool: str
    description: str = ""
    fixed_kwargs: dict[str, Any] = {}
    tags: list[str] = []
    extensions: list[list[ExtensionElement]] = []
    output_schema: dict[str, Any] | None = None


class PresetVersionSave(BaseModel):
    """A new-preset-version request. At least one field must be present; an
    omitted field carries forward, an explicit ``[]`` clears (the store sentinel
    rule). ``output_schema`` carries forward when omitted, clears on an explicit
    ``null``, and wins on an explicit object schema."""

    fixed_kwargs: dict[str, Any] | None = None
    tags: list[str] | None = None
    extensions: list[list[ExtensionElement]] | None = None
    output_schema: dict[str, Any] | None = None


class PresetRollback(BaseModel):
    """A rollback request — the target version to make active."""

    version: int


class PresetRename(BaseModel):
    """A rename request — the new preset (tool) name."""

    new_name: str


class PresetValidate(BaseModel):
    """A preset validation (dry-run) request — the full create field set. When a
    preset named ``name`` already exists the door validates a NEW VERSION: then
    ``base_tool`` / ``description`` carry forward from the active body (a provided
    value that differs is rejected) and any absent field merges from it, exactly as
    the save-version route merges."""

    name: str
    base_tool: str | None = None
    description: str | None = None
    fixed_kwargs: dict[str, Any] | None = None
    tags: list[str] | None = None
    extensions: list[list[ExtensionElement]] | None = None
    output_schema: dict[str, Any] | None = None


class PresetVersionTags(BaseModel):
    """Replace a preset version's ``tags`` annotation (labels on an immutable
    version body — no rebind)."""

    tags: list[str]


# -- body-structure readers (raise the byte-stable 400 the routes surfaced) --

# These validate the SHAPE of the nested combo / output-schema payload. They are
# shared by the router's HTTP-edge extractors (which parse the create/save bodies)
# and the validate operation (whose create-vs-version reading is mode-dependent, so
# it happens inside the op after the store lookup that resolves the mode). A
# malformed structure raises :class:`BadRequestError`, mapped to the same 400 the
# route handler surfaced.


def read_element(element: Any) -> ExtensionElement:
    """One combo element, structurally validated: a non-empty extension NAME
    (bare string), or a ``{"name": <non-empty str>, "config": <dict>}`` mapping
    binding author config (``config`` REQUIRED — a config-less selection is the
    bare-string form, so a config-free dict is malformed) with no other keys.
    Anything else is a loud 400. Registration of the name is checked later against
    the live registry."""
    if isinstance(element, str):
        if not element:
            raise BadRequestError("an extension name must be a non-empty string")
        return element
    if isinstance(element, dict):
        name = element.get("name")
        if not isinstance(name, str) or not name:
            raise BadRequestError("an extension element must have a non-empty string 'name'")
        config = element.get("config")
        if not isinstance(config, dict):
            raise BadRequestError(f"extension element {name!r} must carry a 'config' mapping")
        extra = set(element) - {"name", "config"}
        if extra:
            raise BadRequestError(f"extension element {name!r} has unexpected keys: {sorted(extra)!r}")
        return {"name": name, "config": dict(config)}
    raise BadRequestError("each combo element must be an extension name or a {'name', 'config'} mapping")


def read_combos(extensions: Any) -> list[list[ExtensionElement]]:
    """A list of extension combos, each a non-empty list of combo elements. The
    empty INNER combo (``[[]]`` or any ``[]`` member) is rejected — mirrors the
    view's rule so a create/edit is guarded before any store write."""
    result: list[list[ExtensionElement]] = []
    for combo in extensions:
        if not isinstance(combo, list) or not combo:
            raise BadRequestError("each extension combo must be a non-empty list of extension elements")
        result.append([read_element(element) for element in combo])
    return result


def read_create_extensions(present: bool, value: Any) -> list[list[ExtensionElement]]:
    """Create's extension combos: an absent field means no extensions, an explicit
    ``extensions: []`` is REJECTED (nothing to clear on create), and an empty inner
    combo is rejected."""
    if not present:
        return []
    if not isinstance(value, list):
        raise BadRequestError("'extensions' must be a list of combos")
    if value == []:
        raise BadRequestError("explicit empty 'extensions' is rejected; omit the field for no extensions")
    return read_combos(value)


def read_edit_extensions(present: bool, value: Any) -> list[list[ExtensionElement]] | None:
    """Save-version's extension combos under the carry-forward sentinel: absent or
    ``null`` carries forward (``None``); ``[]`` clears; an empty inner combo is
    rejected."""
    if not present or value is None:
        return None
    if not isinstance(value, list):
        raise BadRequestError("'extensions' must be a list of combos")
    return read_combos(value)


def read_output_schema(value: Any) -> dict[str, Any] | None:
    """The optional author-set output schema from a request value: ``null`` →
    ``None``; a JSON object → itself; anything else → a loud 400."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise BadRequestError("'output_schema' must be a JSON object (a JSON Schema)")
    return value


# -- agent-authoring validation ----------------------------------------------

# An authored agent is a preset whose ``base_tool`` names an agent's run tool. On top
# of the create route's base-tool rules, an agent base adds these authoring checks: every
# baked ``fixed_kwargs`` field must be preset-bakeable for the agent (a ``spec_runnable``
# agent honors every ``ToolInput`` field; otherwise only the fields it declares in
# ``preset_bakeable_fields``), the baked ``fixed_kwargs`` must be valid partial
# ``ToolInput``, and every spec reference (``tool_names`` / inline ``presets`` /
# nested ``subagents``) must resolve at author time — recursively, at every depth.


def _agent_tool_names() -> set[str]:
    """Every registered agent's declared ``tool_name``.

    A ``tool_name`` can differ from the decorator-registration name: the run tool
    binds under the REGISTRATION name (so that name is already a live tool the
    ``name_conflicts`` guard catches), but the ``tool_name`` may not be a bound tool.
    Keeping the authored-agent name off BOTH sets keeps one unambiguous agent-name
    space."""
    return {agent.tool_name for agent in instance.app.agents.all_agents().values()}


def _spec_reference_error(
    node: dict[str, Any], tools: set[str], preset_names: frozenset[str], where: str
) -> str | None:
    """The first unresolved reference in one spec node, or ``None`` if all resolve.

    Checks the node's ``tool_names`` (each must be a registered tool), its inline
    ``presets`` (each a self-contained ``PresetSpec`` whose ``base_tool`` is a
    registered NON-preset tool — the same flat-preset rule as the create route), and
    recurses into every inline ``subagents`` spec, so a bad reference at ANY depth is
    caught loudly rather than silently dropped."""
    tool_names = node.get("tool_names", [])
    if not isinstance(tool_names, list):
        return f"{where}.tool_names must be a list of tool names"
    for tool_name in tool_names:
        if not isinstance(tool_name, str):
            return f"{where}.tool_names entries must be strings"
        if tool_name not in tools:
            return f"{where}.tool_names references unknown tool {tool_name!r}"

    presets = node.get("presets", [])
    if not isinstance(presets, list):
        return f"{where}.presets must be a list of preset specs"
    for i, entry in enumerate(presets):
        try:
            preset = PresetSpec.model_validate(entry)
        except ValidationError as exc:
            return f"{where}.presets[{i}] is not a valid preset spec: {exc}"
        if preset.base_tool in preset_names:
            return f"{where}.presets[{i}] base_tool {preset.base_tool!r} is itself a preset"
        if preset.base_tool not in tools:
            return f"{where}.presets[{i}] base_tool {preset.base_tool!r} is not a registered tool"

    subagents = node.get("subagents", [])
    if not isinstance(subagents, list):
        return f"{where}.subagents must be a list of sub-agent specs"
    for i, entry in enumerate(subagents):
        if not isinstance(entry, dict):
            return f"{where}.subagents[{i}] must be an object"
        nested = _spec_reference_error(entry, tools, preset_names, f"{where}.subagents[{i}]")
        if nested is not None:
            return nested
    return None


def _node_references_tool(node: dict[str, Any], target: str) -> bool:
    """Whether a spec node names ``target`` in its ``tool_names`` at any depth,
    recursing ``subagents`` — the SAME traversal :func:`_spec_reference_error` walks,
    read-only. Only ``tool_names`` can name a preset: inline ``presets`` entries and a
    ``base_tool`` are rejected at authoring if they name a preset, so neither is
    scanned here."""
    tool_names = node.get("tool_names", [])
    if isinstance(tool_names, list) and target in tool_names:
        return True
    subagents = node.get("subagents", [])
    if isinstance(subagents, list):
        for entry in subagents:
            if isinstance(entry, dict) and _node_references_tool(entry, target):
                return True
    return False


def _referencing_presets(old_name: str, bodies: dict[str, PresetBody]) -> list[str]:
    """Every OTHER preset whose ACTIVE body's ``fixed_kwargs`` composes ``old_name``
    as a tool (``tool_names`` at any depth) — the referees a rename would strand,
    sorted for a stable, fully-listed 409. Only active bodies are walked: a
    non-active historical version may still name the old tool, loud at
    authoring / run time if ever rolled back (delete's existing posture)."""
    return sorted(
        name for name, body in bodies.items() if name != old_name and _node_references_tool(body.fixed_kwargs, old_name)
    )


async def _agent_authoring_error(base_tool: str, fixed_kwargs: dict[str, Any]) -> str | None:
    """When ``base_tool`` names a registered agent, the first authoring violation, or
    ``None`` if the spec is valid. Returns ``None`` for a NON-agent base — a plain
    tool preset is governed by the create route's base rules alone.

    Each baked ``fixed_kwargs`` field must be preset-bakeable for the agent: a
    ``spec_runnable`` agent honors every ``ToolInput`` field, so all of them are
    bakeable; otherwise only the fields the agent declares in
    ``preset_bakeable_fields`` are. A baked field the runtime does not honor is
    rejected here rather than persisted as a silent no-op bake. An EMPTY
    ``fixed_kwargs`` bakes nothing, so there is nothing to gate."""
    agent = instance.app.agents.all_agents().get(base_tool)
    if agent is None:
        return None

    # ``fixed_kwargs`` bakes a PARTIAL spec (only the composable fields), so it is
    # validated field-by-field against the agent's ``ToolInput`` — a full-model
    # construction would spuriously fail on the run-time-only required fields the
    # author deliberately leaves unbaked. The bakeable set is every ``ToolInput``
    # field for a ``spec_runnable`` agent, else exactly the declared honored fields.
    model_fields = agent.ToolInput.model_fields
    bakeable = set(model_fields) if agent.spec_runnable else set(agent.preset_bakeable_fields)
    for key, value in fixed_kwargs.items():
        field = model_fields.get(key)
        if field is None:
            return f"fixed_kwargs field {key!r} is not a field of agent {base_tool!r}'s input"
        if key not in bakeable:
            return (
                f"fixed_kwargs field {key!r} is not preset-bakeable for agent {base_tool!r}: "
                "the agent is not spec_runnable and does not declare it in preset_bakeable_fields"
            )
        # Validate against the full annotation INCLUDING the field's pydantic
        # constraints (``Field(gt=..., min_length=..., pattern=...)``), which live in
        # ``field.metadata`` — validating the bare annotation alone would let a baked
        # value that violates a declared constraint pass author-time validation.
        annotation: Any = field.annotation
        for meta in field.metadata:
            annotation = Annotated[annotation, meta]
        try:
            TypeAdapter(annotation).validate_python(value)
        except ValidationError as exc:
            return f"fixed_kwargs field {key!r} is invalid for agent {base_tool!r}: {exc}"

    tools = set(await instance.app.tools.get_tools())
    preset_names = instance.app.preset_manager.registered_names()
    return _spec_reference_error(fixed_kwargs, tools, preset_names, "fixed_kwargs")


# -- validate-before-commit helpers ------------------------------------------

# create/save-version/rollback all validate a body CAN bind before any store write,
# so a bad edit is a loud 400 rather than a committed version that can never bind
# (which would brick the preset into delete-only). Two checks: each extension combo
# against the live registry, and a dry-run of the bake with no registration.


def _combo_registry_error(extensions: Sequence[Sequence[ExtensionElement]]) -> str | None:
    """The first extension combo that fails the LIVE registry (unknown name or a
    non-stackable-kind clash), as a 400 message, or ``None`` if all combos are
    valid — shared by create/save-version/rollback, via the public
    ``app.extensions.validate_combo`` accessor."""
    for combo in extensions:
        try:
            instance.app.extensions.validate_combo(combo)
        except TaiValidationError as exc:
            return str(exc)
    return None


async def _output_schema_error(
    base_tool: str, output_schema: dict[str, Any] | None, extensions: Sequence[Sequence[ExtensionElement]]
) -> str | None:
    """The first author-time violation of an ``output_schema``, as a 400 message, or
    ``None`` if it is unset or valid — shared by create/save-version/rollback so a
    bad schema is a 400 that never persists nor reaches the bind kernel.

    Rejects, in order: a schema that fails the draft-2020-12 meta-schema; a
    non-object schema (both dispatch paths require an object root); a clash with an
    explicit ``output_schema`` extension entry (the shape declared in two places);
    and an agent base whose run tool does not advertise ``response_format``
    (voting_agent) — that base cannot force structured output, so reject at
    authoring rather than let the bake target a missing parameter at bind time."""
    if output_schema is None:
        return None
    try:
        check_json_schema(output_schema)
    except InvalidJsonSchemaError as exc:
        return f"output_schema is not a valid JSON Schema: {exc}"
    if output_schema.get("type") != "object":
        return 'output_schema must be an object schema ("type": "object")'
    for combo in extensions:
        for element in combo:
            if extension_name(element) == "output_schema":
                return (
                    "output_schema field conflicts with an explicit 'output_schema' extension entry; "
                    "declare the output shape in exactly one place"
                )
    agent = instance.app.agents.all_agents().get(base_tool)
    if agent is not None and "response_format" not in agent.ToolInput.model_fields:
        return f"agent base {base_tool!r} does not support forced structured output (its input has no response_format)"
    return None


async def _dry_run_bind_error(
    base_tool: str,
    fixed_kwargs: dict[str, Any],
    *,
    name: str,
    description: str,
    tags: list[str],
    output_schema: dict[str, Any] | None = None,
) -> str | None:
    """Bake the body through the kernel WITHOUT registering, returning a 400 message
    if the bake raises (unknown base tool, a ``fixed_kwargs`` key that is not an
    argument of the base, an ``output_schema`` the base cannot carry) or ``None`` if
    it binds. The dry run never touches the live registry, so a rejected edit leaves
    both the store and the bindings untouched."""
    try:
        await instance.app.presets.bind(
            base_tool, fixed_kwargs, name=name, description=description, tags=tags, output_schema=output_schema
        )
    except Exception as exc:
        return f"preset {name!r} cannot bind: {exc}"
    return None


# -- record views ------------------------------------------------------------


def _store_record_view(name: str, active_version: int, body: PresetBody) -> dict[str, Any]:
    """A store-backed record row: identity + active-body fields + the
    ``conflicted`` flag (name in the quarantine map) with its ``conflicted_reason``
    (the human-readable cause, ``null`` when not conflicted). Takes the
    already-fetched active ``body`` so the caller batches the read (the list route)
    or reuses one read (the get route) rather than round-tripping per row."""
    mgr = instance.app.preset_manager
    return {
        "name": name,
        "base_tool": body.base_tool,
        "description": body.description,
        "active_version": active_version,
        "tags": list(body.tags),
        "extensions": [list(combo) for combo in body.extensions],
        "output_schema": body.output_schema,
        "conflicted": mgr.is_quarantined(name),
        "conflicted_reason": mgr.quarantine_reason(name),
    }


def _new_record_view(
    name: str,
    base_tool: str,
    description: str,
    tags: list[str],
    extensions: list[list[ExtensionElement]],
    output_schema: dict[str, Any] | None,
    *,
    active_version: int,
) -> dict[str, Any]:
    """The record shape a fresh create returns — built from the just-applied spec
    (a fresh preset is never conflicted), so no extra store read is needed."""
    return {
        "name": name,
        "base_tool": base_tool,
        "description": description,
        "active_version": active_version,
        "tags": list(tags),
        "extensions": [list(combo) for combo in extensions],
        "output_schema": output_schema,
        "conflicted": False,
        "conflicted_reason": None,
    }


async def _wire_snapshot(name: str) -> dict[str, Any] | None:
    """The serialized wire tool for ``name`` (``to_mcp_tool().model_dump()``), or
    ``None`` if the name is not currently bound — the client-visible listing state
    the emit guard diffs across a reload."""
    tools = await instance.app.tools.get_tools()
    tool = tools.get(name)
    return None if tool is None else tool.to_mcp_tool().model_dump()


# -- bus fan-out -------------------------------------------------------------


async def _fanout_reload(name: str) -> None:
    """Broadcast a preset rebind on the worker bus so every worker re-reads the active
    body and rebinds ``name``. The op carries only ``kind`` + ``name``; each worker
    re-reads the store itself. The store write and local rebind have already landed by
    the time this runs (its self entry is a truthful ``applied``), so an unconfirmed
    sibling is surfaced as a loud non-convergence ERROR log, and re-running the
    mutation (or a ``reload_config``) is the recovery.

    The local apply is separate and compensation-rolled-back upstream, so this cannot
    ride ``broadcast()``'s apply-inside model — but it shares its non-convergence
    logging so a stranded sibling is never silent."""
    report = await instance.app.bus.publish(
        {"op": "reload_tool", "kind": "preset", "name": name},
        None,
        LocalApplyResult(outcome=OpOutcome.applied),
    )
    log_non_convergence(report)


async def _fanout_remove(name: str) -> None:
    """Broadcast a preset removal on the worker bus so every worker tears ``name``
    down. Same already-applied self entry + non-convergence logging as
    :func:`_fanout_reload`."""
    report = await instance.app.bus.publish(
        {"op": "remove_tool", "kind": "preset", "name": name},
        None,
        LocalApplyResult(outcome=OpOutcome.applied),
    )
    log_non_convergence(report)


# -- list --------------------------------------------------------------------


@operation(summary="List presets", tags=["presets"])
async def list_presets() -> list[dict[str, Any]]:
    """One row per store-backed record (the presets plus the ``conflicted``
    quarantined ones) — the population the presets management table shows."""
    rows: list[dict[str, Any]] = []
    # A store-less deploy (no versioned store configured) has no presets — skip the
    # Postgres read and serve an empty list.
    if versioned_store_configured():
        records = await instance.app.presets.store.list_presets()
        # One batched active-body read instead of a per-record round-trip (N+1).
        bodies = await instance.app.presets.list_active_bodies()
        # ``records`` and ``bodies`` are two separate reads; a preset deleted between
        # them is gone from ``bodies`` — skip it rather than KeyError, it is no longer
        # a live row to list.
        rows = [
            _store_record_view(rec.name, rec.active_version, bodies[rec.name]) for rec in records if rec.name in bodies
        ]
    return rows


# -- create ------------------------------------------------------------------


@operation(
    summary="Create a preset",
    tags=["presets"],
    destructive=True,
    reload_gated=True,
    errors=[BadRequestError, ConflictError, UnavailableError],
    request_model=PresetCreate,
)
async def create_preset(
    name: str,
    base_tool: str,
    description: str,
    fixed_kwargs: dict[str, Any],
    tags: list[str],
    extensions: list[list[ExtensionElement]],
    output_schema: dict[str, Any] | None,
) -> dict[str, Any]:
    """Create a preset, ATOMIC: ordered name pre-checks run BEFORE any store write,
    then the base rule + agent-authoring + combo/schema/bind validation, then the
    store write THEN register (rolling the row fully back on a register failure), one
    ``list_changed``, and the bus rebind fan-out."""
    # A preset name is a live tool name + a ``{name}`` route segment, so it must be
    # tool-name-safe (a slash-bearing name would never match the routes; an
    # over-long one collides after client-tool truncation).
    if not is_valid_preset_name(name):
        raise BadRequestError(f"invalid preset name {name!r}: must match ^[A-Za-z0-9_-]{{1,64}}$")

    mgr = instance.app.preset_manager
    # Three ordered name pre-checks — the quarantine 409 wins for a name that would
    # also collide, then the collision guard, then the spec-map duplicate.
    if mgr.is_quarantined(name):
        raise ConflictError(f"a quarantined preset {name!r} exists — delete the quarantined record first")
    if await mgr.name_conflicts(name):
        raise ConflictError(f"preset name {name!r} collides with an existing tool")
    # The same collision guard, extended to the agent-name space: an agent's
    # registration name is already a live tool (caught above), but its ``tool_name``
    # may not be — keep the authored name off that set too so one agent-name space
    # stays unambiguous.
    if name in _agent_tool_names():
        raise BadRequestError(f"preset name {name!r} collides with an agent tool name")
    if mgr.is_registered(name):
        raise ConflictError(f"preset {name!r} already exists")

    # A preset's base must be a registered NON-preset tool (a preset cannot be
    # another preset's base — chaining would make rehydration order-dependent).
    if base_tool in mgr.registered_names():
        raise BadRequestError(f"base tool {base_tool!r} is itself a preset")
    if base_tool not in await instance.app.tools.get_tools():
        raise BadRequestError(f"base tool {base_tool!r} is not a registered tool")

    # When the base is an agent tool, this is an authored agent: every baked field
    # must be preset-bakeable for the agent and its baked spec must validate +
    # resolve every reference.
    authoring_error = await _agent_authoring_error(base_tool, fixed_kwargs)
    if authoring_error is not None:
        raise BadRequestError(authoring_error)

    # Validate-before-commit: reject an unknown/illegal extension combo and a body
    # that cannot bake, BEFORE any store write, so a bad create is a 400 that never
    # persists a row.
    combo_error = _combo_registry_error(extensions)
    if combo_error is not None:
        raise BadRequestError(combo_error)
    schema_error = await _output_schema_error(base_tool, output_schema, extensions)
    if schema_error is not None:
        raise BadRequestError(schema_error)
    bind_error = await _dry_run_bind_error(
        base_tool, fixed_kwargs, name=name, description=description, tags=tags, output_schema=output_schema
    )
    if bind_error is not None:
        raise BadRequestError(bind_error)

    # A preset needs the durable store; on a store-less deploy (no
    # VERSIONING_STORE_* credential) refuse cleanly here — the same predicate the
    # list / delete / reconcile paths gate on — rather than let create open Postgres
    # and fail with an opaque 500.
    if not versioned_store_configured():
        raise UnavailableError("presets require a configured versioned-document store")

    # The pre-checks already ran, so the store write is safe. Persist THEN register;
    # if register fails, roll the store row fully back through the generic HARD
    # delete so no stored-but-unregistered preset survives.
    spec = PresetSpec(name=name, description=description, base_tool=base_tool, fixed_kwargs=fixed_kwargs)
    try:
        record = await instance.app.presets.store.create_preset(
            spec, extensions=extensions, tags=tags, output_schema=output_schema
        )
    except PresetNameConflictError as exc:
        raise ConflictError(f"preset name {name!r} collides with an existing tool") from exc
    except PresetExistsError as exc:
        raise ConflictError(f"preset {name!r} already exists") from exc
    try:
        await mgr.register(name, base_tool, fixed_kwargs, extensions, tags, description, output_schema)
    except Exception as register_exc:
        try:
            await instance.app.versioning.store.delete("preset", name)
        except Exception as delete_exc:
            logger.exception("failed to roll back store row for preset %r after a register failure", name)
            raise delete_exc from register_exc
        # A typed clobber error (the name was taken by a foreign tool or a sibling
        # preset in the window between the pre-checks and the register) maps to 409;
        # any other register failure re-raises loudly.
        if isinstance(register_exc, PresetExistsError):
            raise ConflictError(f"preset {name!r} already exists") from register_exc
        if isinstance(register_exc, PresetNameConflictError):
            raise ConflictError(f"preset name {name!r} collides with an existing tool") from register_exc
        raise register_exc
    await instance.app.emit_list_changed("tool")
    await _fanout_reload(name)
    return _new_record_view(
        name, base_tool, description, tags, extensions, output_schema, active_version=record.active_version
    )


# -- get one -----------------------------------------------------------------


@operation(summary="Get a preset", tags=["presets"], errors=[NotFoundError])
async def get_preset(name: str) -> dict[str, Any]:
    """The store record + the active ``fixed_kwargs``; 404 for an absent name."""
    try:
        record = await instance.app.presets.store.get_preset(name)
    except PresetNotFoundError as exc:
        raise NotFoundError(f"preset {name!r} not found") from exc
    # Fetch the active body ONCE and pass it into the record view + the response.
    body = await instance.app.presets.store.get_active_body(name)
    view = _store_record_view(name, record.active_version, body)
    view["fixed_kwargs"] = body.fixed_kwargs
    return view


# -- versions ----------------------------------------------------------------


@operation(summary="List a preset's versions", tags=["presets"], errors=[NotFoundError])
async def list_versions(name: str) -> list[dict[str, Any]]:
    """The full version history for a preset; 404 for an absent name."""
    try:
        versions = await instance.app.presets.store.list_versions(name)
    except PresetNotFoundError as exc:
        raise NotFoundError(f"preset {name!r} not found") from exc
    return [v.model_dump() for v in versions]


@operation(summary="Get a specific preset version", tags=["presets"], errors=[BadRequestError, NotFoundError])
async def get_version(name: str, version: str) -> dict[str, Any]:
    """One version of a preset by its integer version number; a non-integer segment
    is a 400 and an unknown version a 404."""
    try:
        version_num = int(version)
    except ValueError as exc:
        raise BadRequestError("version must be an integer") from exc
    try:
        row = await instance.app.presets.store.get_version(name, version_num)
    except PresetVersionNotFoundError as exc:
        raise NotFoundError(f"preset {name!r} has no version {version_num}") from exc
    return row.model_dump()


@operation(
    summary="Save a new preset version",
    tags=["presets"],
    destructive=True,
    reload_gated=True,
    errors=[BadRequestError, ConflictError, NotFoundError],
    request_model=PresetVersionSave,
)
async def save_version(
    name: str,
    fixed_kwargs: dict[str, Any] | None,
    tags: list[str] | None,
    extensions: list[list[ExtensionElement]] | None,
    output_schema: dict[str, Any] | None,
    output_schema_provided: bool,
) -> dict[str, Any]:
    """Save a new version (carry-forward sentinels on omitted fields) then reload and
    fan out; 409 if the record is conflicted, 404 for an absent name. The
    ``list_changed`` emit is GUARDED on a real change to the serialized wire tool OR
    its extension combos."""
    store = instance.app.presets.store
    if instance.app.preset_manager.is_quarantined(name):
        raise ConflictError(f"preset {name!r} is conflicted and is delete-only")

    # Read the active record + body BEFORE any write: the carried-forward base tool
    # + description come from it, the prior active version is captured for the
    # residual-failure re-point, and it gives the 404 for an absent name.
    try:
        prior_record = await store.get_preset(name)
        active = await store.get_active_body(name)
    except PresetNotFoundError as exc:
        raise NotFoundError(f"preset {name!r} not found") from exc

    # The EFFECTIVE new body under the carry-forward sentinels (omitted → carry the
    # active value; an explicit value — including a clearing ``[]`` — wins). The
    # store applies the same rule on write; these mirror it for pre-write validation.
    base_tool = active.base_tool
    new_fixed_kwargs = active.fixed_kwargs if fixed_kwargs is None else fixed_kwargs
    new_extensions = active.extensions if extensions is None else extensions
    new_tags = active.tags if tags is None else tags
    new_output_schema = active.output_schema if not output_schema_provided else output_schema

    # Validate-before-commit — run the SAME checks create runs so a bad edit is a
    # 400 that commits nothing (never a version that can never bind, which would
    # brick the preset into delete-only).
    if fixed_kwargs is not None:
        authoring_error = await _agent_authoring_error(base_tool, new_fixed_kwargs)
        if authoring_error is not None:
            raise BadRequestError(authoring_error)
    combo_error = _combo_registry_error(new_extensions)
    if combo_error is not None:
        raise BadRequestError(combo_error)
    schema_error = await _output_schema_error(base_tool, new_output_schema, new_extensions)
    if schema_error is not None:
        raise BadRequestError(schema_error)
    bind_error = await _dry_run_bind_error(
        base_tool,
        new_fixed_kwargs,
        name=name,
        description=active.description,
        tags=new_tags,
        output_schema=new_output_schema,
    )
    if bind_error is not None:
        raise BadRequestError(bind_error)

    # Snapshot the OLD wire tool + its extension combos BEFORE the store write —
    # reload tears the old tool down, and after the write the active body already
    # holds the new value.
    old_extensions = active.extensions
    old_wire = await _wire_snapshot(name)
    prior_active = prior_record.active_version

    try:
        row = await store.save_version(
            name,
            fixed_kwargs=fixed_kwargs,
            tags=tags,
            extensions=extensions,
            output_schema=output_schema if output_schema_provided else CARRY_FORWARD,
        )
    except ValueError as exc:
        raise BadRequestError(str(exc)) from exc
    except PresetNotFoundError as exc:
        raise NotFoundError(f"preset {name!r} not found") from exc

    # A residual re-register failure (the environment changed after the pre-write
    # validation) re-points the store's active version back to the prior one so the
    # committed row stays as inert history and store + live never diverge, then
    # re-raises loudly; the emit below is never reached, so a failed save fires
    # nothing.
    try:
        await instance.app.preset_manager.reload(name)
    except Exception:
        await store.rollback(name, prior_active)
        raise

    new_actual_extensions = (await store.get_active_body(name)).extensions
    new_wire = await _wire_snapshot(name)
    if old_wire != new_wire or old_extensions != new_actual_extensions:
        await instance.app.emit_list_changed("tool")
    # The rebind fans out regardless of the emit guard: siblings must re-read the
    # active body even when the wire tool is byte-identical (a baked VALUE changed).
    await _fanout_reload(name)
    return row.model_dump()


# -- rollback ----------------------------------------------------------------


@operation(
    summary="Roll a preset back to a version",
    tags=["presets"],
    destructive=True,
    reload_gated=True,
    errors=[BadRequestError, ConflictError, NotFoundError],
    request_model=PresetRollback,
)
async def rollback_preset(name: str, version: int) -> dict[str, Any]:
    """Re-point the active version then reload and fan out; 409 if the record is
    conflicted, 404 for an absent name or version, 400 if the target version cannot
    bind against the current live registry."""
    store = instance.app.presets.store
    if instance.app.preset_manager.is_quarantined(name):
        raise ConflictError(f"preset {name!r} is conflicted and is delete-only")

    try:
        prior_record = await store.get_preset(name)
        old_extensions = (await store.get_active_body(name)).extensions
    except PresetNotFoundError as exc:
        raise NotFoundError(f"preset {name!r} not found") from exc
    old_wire = await _wire_snapshot(name)

    # Rollback has NO carry-forward: read the TARGET version body and validate THAT
    # against the CURRENT live registry (a base tool or extension it named may have
    # been removed since the version was authored), so a rollback to an unbindable
    # version is a 400 that commits nothing rather than a bricking re-point.
    try:
        target = await store.get_version(name, version)
    except PresetVersionNotFoundError as exc:
        raise NotFoundError(f"preset {name!r} has no version {version}") from exc
    target_body = PresetBody.model_validate(target.body)
    combo_error = _combo_registry_error(target_body.extensions)
    if combo_error is not None:
        raise BadRequestError(combo_error)
    schema_error = await _output_schema_error(target_body.base_tool, target_body.output_schema, target_body.extensions)
    if schema_error is not None:
        raise BadRequestError(schema_error)
    bind_error = await _dry_run_bind_error(
        target_body.base_tool,
        target_body.fixed_kwargs,
        name=name,
        description=target_body.description,
        tags=target_body.tags,
        output_schema=target_body.output_schema,
    )
    if bind_error is not None:
        raise BadRequestError(bind_error)

    prior_active = prior_record.active_version
    record = await store.rollback(name, version)

    # Residual re-register failure: re-point the active version back to the prior
    # one so store + live never diverge, then re-raise loudly.
    try:
        await instance.app.preset_manager.reload(name)
    except Exception:
        await store.rollback(name, prior_active)
        raise

    new_extensions = (await store.get_active_body(name)).extensions
    new_wire = await _wire_snapshot(name)
    if old_wire != new_wire or old_extensions != new_extensions:
        await instance.app.emit_list_changed("tool")
    # The rebind fans out regardless of the emit guard: siblings must re-read the
    # active body even when the wire tool is byte-identical (a baked VALUE changed).
    await _fanout_reload(name)
    return {"name": name, "active_version": record.active_version}


# -- rename ------------------------------------------------------------------


@operation(
    summary="Rename a preset",
    tags=["presets"],
    destructive=True,
    reload_gated=True,
    errors=[BadRequestError, ConflictError, NotFoundError],
    request_model=PresetRename,
)
async def rename_preset(name: str, new_name: str) -> dict[str, Any]:
    """Rename a preset, ATOMIC (a preset's name IS its live tool name). Runs create's
    ordered name pre-checks on the NEW name, BLOCKS with a 409 listing every referee
    if another preset composes the current name, binds the new tool BEFORE tearing
    the old one down, fires one ``list_changed``, and fans the rebind out NEW-first
    then the old removal."""
    # The new name is a live tool name + a ``{name}`` route segment, so it must be
    # tool-name-safe — the same rule create enforces.
    if not is_valid_preset_name(new_name):
        raise BadRequestError(f"invalid preset name {new_name!r}: must match ^[A-Za-z0-9_-]{{1,64}}$")
    # A no-op rename is a caller error, surfaced loudly — never a silent 200.
    if new_name == name:
        raise BadRequestError("new name must differ from the current name")

    mgr = instance.app.preset_manager
    # A conflicted record was never registered and its name may be owned by a foreign
    # tool — rename must not touch it, nor launder a quarantined record into a clean
    # name (the delete-only stance save/rollback take).
    if mgr.is_quarantined(name):
        raise ConflictError(f"preset {name!r} is conflicted and is delete-only")

    # A store-less deploy holds no preset, so an unquarantined name is a genuine 404
    # without a Postgres open (delete's reasoning).
    if not versioned_store_configured():
        raise NotFoundError(f"preset {name!r} not found")

    store = instance.app.presets.store
    try:
        await store.get_preset(name)
    except PresetNotFoundError as exc:
        raise NotFoundError(f"preset {name!r} not found") from exc

    # NEW-name pre-checks — create's exact order and codes: quarantine 409 → live-tool
    # collision 409 → agent tool-name collision 400 → duplicate-preset 409.
    if mgr.is_quarantined(new_name):
        raise ConflictError(f"a quarantined preset {new_name!r} exists — delete the quarantined record first")
    if await mgr.name_conflicts(new_name):
        raise ConflictError(f"preset name {new_name!r} collides with an existing tool")
    if new_name in _agent_tool_names():
        raise BadRequestError(f"preset name {new_name!r} collides with an agent tool name")
    if mgr.is_registered(new_name):
        raise ConflictError(f"preset {new_name!r} already exists")

    # Referential integrity: BLOCK (never silently cascade-rewrite) a rename that
    # would strand another preset's authored-agent composition, listing every referee
    # so the author updates them first. One batched active-body read, no N+1.
    referees = _referencing_presets(name, await instance.app.presets.list_active_bodies())
    if referees:
        raise ConflictError(
            f"preset {name!r} cannot be renamed: it is referenced by preset(s) {referees}; update those presets first"
        )

    # Move the store key. The pre-checks make the typed conflicts race-window catches,
    # mapped exactly as create maps its post-write errors.
    try:
        record = await store.rename_preset(name, new_name)
    except PresetNotFoundError as exc:
        raise NotFoundError(f"preset {name!r} not found") from exc
    except PresetExistsError as exc:
        raise ConflictError(f"preset {new_name!r} already exists") from exc
    except PresetNameConflictError as exc:
        raise ConflictError(f"preset name {new_name!r} collides with an existing tool") from exc

    # Local apply, NEW FIRST: bind ``new_name`` from the moved store row's active body.
    # Reload-before-remove keeps every call resolvable — during the window BOTH names
    # are bound and the old binding still works (its baked spec is in-memory, its base
    # tool untouched). On a re-register failure, compensate by re-pointing the store
    # back so store + live never diverge, then surface loudly; the OLD binding was
    # never touched, so the preset stays fully live under its old name.
    try:
        await mgr.reload(new_name)
    except Exception as reload_exc:
        try:
            await store.rename_preset(new_name, name)
        except Exception as compensate_exc:
            logger.exception("failed to re-point store row for preset %r after a rename re-register failure", name)
            raise compensate_exc from reload_exc
        if isinstance(reload_exc, PresetExistsError):
            raise ConflictError(f"preset {new_name!r} already exists") from reload_exc
        if isinstance(reload_exc, PresetNameConflictError):
            raise ConflictError(f"preset name {new_name!r} collides with an existing tool") from reload_exc
        raise reload_exc

    # Then tear the OLD binding down. A failure here leaves BOTH names bound (old is
    # stale-but-functional: its baked spec is in-memory and its base tool is untouched);
    # it re-raises loudly and ``reload_config`` is the documented recovery (rehydration
    # rebuilds from the store, which now knows only ``new_name``). The store move is
    # never unwound here — ``new_name`` is live and correct.
    await mgr.remove(name)

    # A rename changes the tool listing by definition (old gone, new present), so the
    # emit is unconditional — no wire-diff guard.
    await instance.app.emit_list_changed("tool")
    # Fan out NEW FIRST — reload ``new_name`` on every worker BEFORE removing ``old``
    # (both briefly alive beats neither): the two are sequentially awaited confirmed
    # broadcasts, so every worker applies the reload before any is asked to remove.
    await _fanout_reload(new_name)
    await _fanout_remove(name)
    return {"name": new_name, "renamed_from": name, "active_version": record.active_version}


# -- delete ------------------------------------------------------------------


@operation(
    summary="Delete a preset",
    tags=["presets"],
    reload_gated=True,
    errors=[NotFoundError],
)
async def delete_preset(name: str) -> dict[str, Any]:
    """Delete a preset. A non-conflicted record is soft-deleted and its base + branch
    tools torn down (one ``list_changed``); a conflicted record is removed store-side
    ONLY (HARD delete + drop the quarantine entry), touching no registration and
    firing no emit. Both branches fan the removal out on the bus."""
    mgr = instance.app.preset_manager

    if mgr.is_quarantined(name):
        # A conflicted record was never registered — remove ONLY the stored
        # document (HARD delete, so no ghost/version history lingers), drop the
        # quarantine entry immediately, and touch NO registration (the name may be
        # owned by a foreign tool) and fire NO emit. The removal still fans out so a
        # sibling's in-memory quarantine entry is cleared too. A quarantine entry only
        # ever arises from a store-backed preset (rehydrate or reconcile of a created
        # preset — both require a configured store), so the hard-delete here always has
        # a store to talk to; the store-config guard below is only for the
        # non-quarantined path.
        try:
            await instance.app.versioning.store.delete("preset", name)
        except Exception:
            logger.exception("failed to hard-delete conflicted preset record %r", name)
            raise
        mgr.drop_quarantine(name)
        await _fanout_remove(name)
        return {"name": name, "deleted": True}

    # A store-less deploy (no versioned store configured) can hold no preset, so a
    # name that is not quarantined is a genuine 404 without a Postgres read.
    if not versioned_store_configured():
        raise NotFoundError(f"preset {name!r} not found")
    try:
        await instance.app.presets.store.soft_delete(name)
    except PresetNotFoundError as exc:
        raise NotFoundError(f"preset {name!r} not found") from exc
    await mgr.remove(name)
    await instance.app.emit_list_changed("tool")
    await _fanout_remove(name)
    return {"name": name, "deleted": True}


# -- referees ----------------------------------------------------------------


@operation(summary="List presets that reference this preset", tags=["presets"], errors=[NotFoundError])
async def preset_referees(name: str) -> dict[str, Any]:
    """Every OTHER preset whose active authored-agent composition names this one as
    a tool — the referees a rename would strand, exposed so the UI can preflight a
    rename. 404 for an unknown preset, the same existence check the rename door runs
    first."""
    # A store-less deploy holds no preset, so an unknown name is a genuine 404
    # without a Postgres open (the rename/delete doors' reasoning).
    if not versioned_store_configured():
        raise NotFoundError(f"preset {name!r} not found")
    try:
        await instance.app.presets.store.get_preset(name)
    except PresetNotFoundError as exc:
        raise NotFoundError(f"preset {name!r} not found") from exc
    referees = _referencing_presets(name, await instance.app.presets.list_active_bodies())
    return {"name": name, "referees": referees}


# -- validate (dry-run) ------------------------------------------------------


def _verdict(error: str | None) -> dict[str, Any]:
    """A validation verdict — ``valid`` is the absence of an ``error``. The op
    returns 200 for BOTH outcomes: an invalid draft is a SUCCESSFUL validation, not
    a request failure."""
    return {"valid": error is None, "error": error}


async def _validate_create(
    name: str,
    base_tool: str,
    description: str,
    fixed_kwargs: dict[str, Any],
    tags: list[str],
    extensions: list[list[ExtensionElement]],
    output_schema: dict[str, Any] | None,
) -> dict[str, Any]:
    """The create route's full pre-store verdict for a brand-new preset — the exact
    ordered checks create runs before its store write (name safety → quarantine →
    tool collision → agent-name collision → duplicate → base rules → agent
    authoring), then combo → schema → dry-run bake — as a ``valid``/``error``
    verdict rather than a write."""
    if not is_valid_preset_name(name):
        return _verdict(f"invalid preset name {name!r}: must match ^[A-Za-z0-9_-]{{1,64}}$")
    mgr = instance.app.preset_manager
    if mgr.is_quarantined(name):
        return _verdict(f"a quarantined preset {name!r} exists — delete the quarantined record first")
    if await mgr.name_conflicts(name):
        return _verdict(f"preset name {name!r} collides with an existing tool")
    if name in _agent_tool_names():
        return _verdict(f"preset name {name!r} collides with an agent tool name")
    if mgr.is_registered(name):
        return _verdict(f"preset {name!r} already exists")
    if base_tool in mgr.registered_names():
        return _verdict(f"base tool {base_tool!r} is itself a preset")
    if base_tool not in await instance.app.tools.get_tools():
        return _verdict(f"base tool {base_tool!r} is not a registered tool")
    authoring_error = await _agent_authoring_error(base_tool, fixed_kwargs)
    if authoring_error is not None:
        return _verdict(authoring_error)
    return await _verdict_bind_chain(
        base_tool,
        fixed_kwargs,
        name=name,
        description=description,
        tags=tags,
        output_schema=output_schema,
        extensions=extensions,
    )


async def _verdict_bind_chain(
    base_tool: str,
    fixed_kwargs: dict[str, Any],
    *,
    name: str,
    description: str,
    tags: list[str],
    output_schema: dict[str, Any] | None,
    extensions: list[list[ExtensionElement]] | None = None,
) -> dict[str, Any]:
    """The shared tail both modes run: combo registry → output schema → dry-run
    bake, as a verdict. ``extensions`` defaults to no combos for the bind chain's
    combo/schema checks."""
    combos: list[list[ExtensionElement]] = extensions or []
    combo_error = _combo_registry_error(combos)
    if combo_error is not None:
        return _verdict(combo_error)
    schema_error = await _output_schema_error(base_tool, output_schema, combos)
    if schema_error is not None:
        return _verdict(schema_error)
    bind_error = await _dry_run_bind_error(
        base_tool, fixed_kwargs, name=name, description=description, tags=tags, output_schema=output_schema
    )
    if bind_error is not None:
        return _verdict(bind_error)
    return _verdict(None)


@operation(
    summary="Validate a preset draft (dry-run)",
    tags=["presets"],
    errors=[BadRequestError, UnavailableError],
    request_model=PresetValidate,
)
async def validate_preset(
    name: str,
    base_tool: str | None = None,
    description: str | None = None,
    fixed_kwargs: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    extensions_present: bool = False,
    extensions_value: Any = None,
    output_schema_present: bool = False,
    output_schema_value: Any = None,
) -> dict[str, Any]:
    """Report whether a preset draft would be accepted, running the SAME pre-store
    verdict the corresponding write route would — CREATE mode when no preset named
    ``name`` exists, VERSION mode when one does (mode-resolved by a store lookup).
    Both verdicts return 200; only a malformed body is a 400."""
    # Mode resolution needs the store; refuse cleanly on a store-less deploy exactly
    # as the create route does before anything else.
    if not versioned_store_configured():
        raise UnavailableError("presets require a configured versioned-document store")

    store = instance.app.presets.store
    try:
        await store.get_preset(name)
        active = await store.get_active_body(name)
    except PresetNotFoundError:
        active = None

    if active is None:
        # CREATE mode — base_tool is required (mirrors create's own 400), and the
        # extension combos read under create semantics (explicit ``[]`` is rejected).
        if base_tool is None:
            raise BadRequestError("body must contain a non-empty string 'base_tool'")
        extensions = read_create_extensions(extensions_present, extensions_value)
        output_schema = read_output_schema(output_schema_value) if output_schema_present else None
        return await _validate_create(
            name,
            base_tool,
            description or "",
            fixed_kwargs or {},
            tags or [],
            extensions,
            output_schema,
        )

    # VERSION mode. The corresponding write route is save_version, whose FIRST
    # pre-store gate rejects a quarantined record — mirror that verdict (never a
    # partial check) so a quarantined-but-still-bindable preset validates as invalid,
    # not as valid.
    if instance.app.preset_manager.is_quarantined(name):
        return _verdict(f"preset {name!r} is conflicted and is delete-only")

    # base_tool + description carry forward and are not version fields; a provided
    # value that differs is a loud verdict, never ignored.
    if base_tool is not None and base_tool != active.base_tool:
        return _verdict("base_tool differs from the preset's active base tool; a version cannot change the base tool")
    if description is not None and description != active.description:
        return _verdict("description is not a version field; it carries forward from the preset")
    edit_extensions = read_edit_extensions(extensions_present, extensions_value)
    new_extensions = active.extensions if edit_extensions is None else edit_extensions
    new_output_schema = read_output_schema(output_schema_value) if output_schema_present else active.output_schema

    new_fixed_kwargs = active.fixed_kwargs if fixed_kwargs is None else fixed_kwargs
    new_tags = active.tags if tags is None else tags
    # An authored-agent (``fixed_kwargs``) edit runs the full authoring validation
    # over the carried-forward base tool, exactly as save-version does — only when
    # fixed_kwargs was provided.
    if fixed_kwargs is not None:
        authoring_error = await _agent_authoring_error(active.base_tool, new_fixed_kwargs)
        if authoring_error is not None:
            return _verdict(authoring_error)
    return await _verdict_bind_chain(
        active.base_tool,
        new_fixed_kwargs,
        name=name,
        description=active.description,
        tags=new_tags,
        output_schema=new_output_schema,
        extensions=new_extensions,
    )


# -- version tags ------------------------------------------------------------


@operation(
    summary="Set a preset version's tags",
    tags=["presets"],
    destructive=True,
    errors=[BadRequestError, NotFoundError, UnavailableError],
    request_model=PresetVersionTags,
)
async def set_preset_version_tags(name: str, version: str, tags: list[str]) -> dict[str, Any]:
    """Replace one version's ``tags`` annotation. Tags are labels on an immutable
    version body, so this edits only the annotation and never rebinds the live tool
    (no reload / no fan-out). 404 for an unknown preset or version; the create
    route's 503 on a store-less deploy."""
    try:
        version_num = int(version)
    except ValueError as exc:
        raise BadRequestError("version must be an integer") from exc

    if not versioned_store_configured():
        raise UnavailableError("presets require a configured versioned-document store")

    try:
        await instance.app.presets.set_version_tags(name, version_num, tags)
    except DocumentVersionNotFoundError as exc:
        raise NotFoundError(f"preset {name!r} has no version {version_num}") from exc
    return {"name": name, "version": version_num, "tags": tags}
