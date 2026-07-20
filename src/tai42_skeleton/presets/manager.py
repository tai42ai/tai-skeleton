"""The preset register/reload engine — the skeleton ``PresetManager``.

Turns a preset (a base tool + baked ``fixed_kwargs`` + extension combos) into a
live named tool, and re-registers it whenever the runtime tool registry is
rebuilt. It owns three pieces of process-lifetime state, held in lockstep with
what is actually bound:

* an **authoritative in-memory spec map** ``name -> PresetBody`` (``base_tool``,
  ``description``, ``fixed_kwargs``, ``extensions``, ``tags``) — the source of
  truth for a registered preset's baked kwargs. The kernel bakes the values as
  hidden ``ArgTransform`` defaults with no readable closure, so both the tool
  face and the ephemeral-agent run path read baked kwargs FROM THIS MAP
  (:meth:`get_spec` / :meth:`baked_kwargs`), never from the tool object;
* a **quarantine map** of names whose stored preset could not register at load
  (name taken by a foreign tool, or a missing / preset-owned base tool) to the
  human-readable reason — surfaced as ``conflicted`` records carrying that reason
  rather than bricking boot;
* the register/reload/remove/rehydrate operations over both, plus the
  :meth:`name_conflicts` predicate the :class:`PresetStoreView` consults so a
  colliding name raises before any store write.

The engine NEVER emits ``list_changed`` — routes and startup handlers own that.
It binds/rebinds the live tool from a body it is handed (or reads the ACTIVE
store body on :meth:`reload`); persistence is the route's job.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
from collections import defaultdict
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from tai42_contract.manifest import ExtensionElement
from tai42_contract.presets import PresetBody
from tai42_contract.presets.errors import PresetExistsError, PresetNameConflictError, PresetNotFoundError

if TYPE_CHECKING:
    from tai42_skeleton.app.server import TaiMCP

logger = logging.getLogger(__name__)

# A preset name is a live MCP tool name AND a ``{name}`` path segment, so it is
# constrained to the tool-name-safe alphabet and the client-tool length cap
# (``CLIENT_TOOL_NAME_MAX_LEN``): a slash-bearing name would never match the
# ``/api/presets/{name}`` routes, and an over-long one collides after truncation.
_PRESET_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def is_valid_preset_name(name: str) -> bool:
    """Whether ``name`` is a valid preset (tool) name — the create route's 400
    guard and :meth:`PresetManager.register`'s backstop share this rule."""
    return _PRESET_NAME_RE.fullmatch(name) is not None


class PresetManager:
    """Register/reload engine + authoritative spec map + quarantine map."""

    def __init__(self, app: TaiMCP) -> None:
        self._app = app
        # name -> the spec used to BUILD the live tool; written on register,
        # dropped on teardown, rebuilt wholesale on rehydration — so it always
        # mirrors what is bound.
        self._specs: dict[str, PresetBody] = {}
        # names whose stored preset could not register at load (conflicted),
        # mapped to the human-readable reason surfaced as ``conflicted_reason``.
        self._quarantine: dict[str, str] = {}
        # Per-name serialization: register/reload/remove/reconcile of ONE name
        # hold this lock across their whole teardown+re-register window so two
        # concurrent create/edit ops on the same name never interleave and clobber
        # each other. The lock is NOT reentrant, so the public methods delegate to
        # the internal UNLOCKED ``_register``/``_remove_registration``: a lock-holding
        # path (``reload``, ``reconcile_bases``) re-registers through the internal
        # form, so a path already holding a name's lock never re-acquires it. An entry
        # is created only for a VALID preset name that reaches a register/reload/
        # remove/reconcile op (name validation precedes the lock), never for a rejected
        # name; entries are not evicted on remove, so create/delete churn over distinct
        # names accretes one tiny lock each — bounded in practice by the operational
        # preset namespace, not by untrusted input.
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    # -- authoritative spec map (feeds the run path) --------------------------

    def get_spec(self, name: str) -> PresetBody:
        """The active spec of a REGISTERED preset. Raise
        :class:`PresetNotFoundError` if no preset by that name is registered."""
        try:
            return self._specs[name]
        except KeyError:
            raise PresetNotFoundError(name) from None

    def baked_kwargs(self, name: str) -> dict[str, Any]:
        """The active baked ``fixed_kwargs`` of a REGISTERED preset — the value
        both the tool face and the ephemeral-agent run path serve, read from the
        spec map (never the tool closure)."""
        return self.get_spec(name).fixed_kwargs

    def is_registered(self, name: str) -> bool:
        """Whether ``name`` is a live registered preset."""
        return name in self._specs

    def registered_names(self) -> frozenset[str]:
        """Every live registered preset name."""
        return frozenset(self._specs)

    # -- quarantine map (the ``conflicted`` mechanism) ------------------------

    def is_quarantined(self, name: str) -> bool:
        return name in self._quarantine

    def quarantined_names(self) -> frozenset[str]:
        return frozenset(self._quarantine)

    def quarantine_reason(self, name: str) -> str | None:
        """The human-readable reason ``name`` is quarantined, or ``None`` when it is
        not — the ``conflicted_reason`` a preset row carries alongside its
        ``conflicted`` flag."""
        return self._quarantine.get(name)

    def drop_quarantine(self, name: str) -> None:
        """Remove ``name`` from the quarantine map with immediate effect — the
        single between-rehydration mutation, driven by the DELETE route's
        conflicted branch so a later create of that name starts clean."""
        self._quarantine.pop(name, None)

    # -- name-collision predicate (wired into the PresetStore view) -----------

    async def name_conflicts(self, name: str) -> bool:
        """Whether ``name`` collides with a LIVE non-preset base tool.

        A preset must never silently shadow a real tool. A name already owned by
        one of OUR presets is not a collision here (a duplicate is caught by the
        create route's spec-map / store-write checks); a name owned by a foreign
        registered tool is. This checks LIVE (bound) tools; a name merely REQUESTED
        by a manifest tool that is not yet bound is handled at :meth:`register`,
        which clears the stale requested entry before seeding so the preset's own
        combos still bind."""
        if name in self._specs:
            return False
        return name in await self._app.tools.get_tools()

    # -- register one ---------------------------------------------------------

    async def register(
        self,
        name: str,
        base_tool: str,
        fixed_kwargs: dict[str, Any],
        extensions: Sequence[Sequence[ExtensionElement]],
        tags: list[str],
        description: str = "",
        output_schema: dict[str, Any] | None = None,
    ) -> None:
        """Bind ``name`` as a runnable tool from its full spec (public entry).

        Validates the name, takes the per-name lock, and delegates to the internal
        :meth:`_register`. A name already owned by one of OUR presets raises
        :class:`PresetExistsError`; a name held by a foreign live tool raises
        :class:`PresetNameConflictError` — the manager never silently clobbers a
        live tool or sibling preset."""
        if not is_valid_preset_name(name):
            raise ValueError(f"invalid preset name {name!r}: must match {_PRESET_NAME_RE.pattern}")
        async with self._locks[name]:
            await self._register(name, base_tool, fixed_kwargs, extensions, tags, description, output_schema)

    async def _register(
        self,
        name: str,
        base_tool: str,
        fixed_kwargs: dict[str, Any],
        extensions: Sequence[Sequence[ExtensionElement]],
        tags: list[str],
        description: str = "",
        output_schema: dict[str, Any] | None = None,
    ) -> None:
        """Bind ``name`` as a runnable tool from its full spec — the UNLOCKED core.

        Builds the baked tool through the kernel (``app.presets.bind`` —
        programmatic ``Tool.from_tool`` with each ``fixed_kwargs`` key hidden and
        fixed), seeds the preset's extension COMBOS straight into the structured
        registry in ONE call, then force-registers the prebuilt tool past manifest
        gating. All-or-nothing: a mid-bind failure tears down any partial
        registration and re-raises loudly, so the registry never holds a
        half-bound preset. It only binds the live tool — it never writes the store.

        Self-guards against a clobber: a name already in the spec map raises
        :class:`PresetExistsError`, and a name held by a foreign LIVE tool raises
        :class:`PresetNameConflictError`, before any bind. Callers that
        deliberately re-bind an existing name (reload, reconcile) tear the old
        registration down first, so the guard sees a free name."""
        if name in self._specs:
            raise PresetExistsError(name)
        if name in await self._app.tools.get_tools():
            raise PresetNameConflictError(name)
        tool_obj = await self._app.presets.bind(
            base_tool, fixed_kwargs, name=name, description=description, tags=tags, output_schema=output_schema
        )
        try:
            # Clear any stale registry entry for ``name`` first — the seed below is
            # a no-op when the name is already a requested tool, so honour that
            # "unregister-first" contract here too. Normally a no-op (a fresh preset
            # name is unknown); the one case it clears is a name requested by a
            # manifest tool whose server was down at bind time (present in the
            # registry's requested set but unbound, so it slipped past the
            # bound-only collision guard) — without this its combos would be
            # silently dropped and no branch tool would bind.
            self._app.tools.unregister_tool_info(name)
            # The combos list is the SAME shape the structured registry takes —
            # passed straight through in ONE call (no per-combo loop, no colon
            # form) so ``_tools[name]`` becomes ``[[]] + combos`` and the branch
            # tools bind off the bare runnable.
            self._app.tools.register_tool_info(name, extensions)
            self._app.tools.tool(tool_obj, force=True)
        except Exception:
            self._remove_registration(name)
            raise
        self._specs[name] = PresetBody(
            base_tool=base_tool,
            description=description,
            fixed_kwargs=fixed_kwargs,
            extensions=[list(combo) for combo in extensions],
            tags=tags,
            output_schema=output_schema,
        )

    # -- reload one (edit path) -----------------------------------------------

    async def reload(self, name: str) -> None:
        """Re-register ``name`` from its store ACTIVE version body after a
        ``save_version`` / ``rollback`` has committed.

        Tears the base AND every branch tool down BEFORE re-registering (so a
        reload never re-binds over a live branch), then registers from the whole
        active body (``base_tool``, ``description``, ``fixed_kwargs``,
        ``extensions``, ``tags``). Never drops the live primitive: the
        currently-live spec is captured before teardown and, if the re-register
        from the new body raises, RESTORED so base + branches survive, then the
        register error is re-raised loudly. The already-committed store bump is
        deliberately NOT unwound — the next reload re-attempts it, and the
        divergence stays loud, never silent.

        Holds the per-name lock across the WHOLE teardown + re-register window (the
        ``get_active_body`` await is the race) and re-registers through the
        internal :meth:`_register`, never the public :meth:`register`, so the held
        per-name lock is never re-acquired (the lock is not reentrant)."""
        async with self._locks[name]:
            captured = self._specs.get(name)
            self._remove_registration(name)
            body = await self._app.presets.store.get_active_body(name)
            try:
                await self._register(
                    name,
                    body.base_tool,
                    body.fixed_kwargs,
                    body.extensions,
                    body.tags,
                    body.description,
                    body.output_schema,
                )
            except Exception:
                if captured is not None:
                    await self._register(
                        name,
                        captured.base_tool,
                        captured.fixed_kwargs,
                        captured.extensions,
                        captured.tags,
                        captured.description,
                        captured.output_schema,
                    )
                raise

    # -- remove one (delete path) ---------------------------------------------

    async def remove(self, name: str) -> None:
        """Tear down ``name``'s base AND every branch tool, dropping its spec.

        The teardown counterpart of :meth:`register`, used by the DELETE route's
        non-conflicted branch (a conflicted record was never registered, so its
        delete is store-side only and must NOT reach here). Takes the per-name lock
        so a delete never races a concurrent register/reload of the same name."""
        async with self._locks[name]:
            self._remove_registration(name)

    # -- reconcile after a scoped MCP reload/deregister -----------------------

    async def reconcile_bases(self, affected_bases: set[str]) -> None:
        """Reconcile presets whose base tool is in ``affected_bases`` after a scoped
        MCP reload or deregister changed the live tool bindings.

        Reconciles from the in-memory spec map, never the store: a preset whose base
        tool is still bound is re-registered from its spec so its ``TransformedTool``
        closure tracks the freshly-bound base instead of the stale pre-reload one; a
        preset whose base tool vanished — or that fails to re-register — is
        QUARANTINED, so its store row surfaces as ``conflicted`` rather than staying
        bound to a base that no longer exists. Each name is reconciled under its own
        lock through the internal register/teardown, so a concurrent edit of the same
        preset never interleaves."""
        affected = [name for name, body in self._specs.items() if body.base_tool in affected_bases]
        if not affected:
            return
        live = set(await self._app.tools.get_tools())
        for name in affected:
            async with self._locks[name]:
                body = self._specs.get(name)
                if body is None:
                    continue
                self._remove_registration(name)
                if body.base_tool in live:
                    try:
                        await self._register(
                            name,
                            body.base_tool,
                            body.fixed_kwargs,
                            body.extensions,
                            body.tags,
                            body.description,
                            body.output_schema,
                        )
                        continue
                    except Exception:
                        reason = "re-register after MCP reload failed"
                        self._quarantine[name] = reason
                        logger.exception("preset %r quarantined: %s", name, reason)
                        continue
                reason = f"its base tool {body.base_tool!r} vanished on an MCP change"
                self._quarantine[name] = reason
                logger.error("preset %r quarantined: %s", name, reason)

    # -- rehydrate all (startup/reload hook body) -----------------------------

    async def rehydrate(self) -> None:
        """Re-register every VERSIONED preset from the store — the body a startup
        and reload hook run.

        ``reload_config`` wipes the whole runtime tool registry, so the spec map +
        quarantine map are cleared WHOLESALE first, then every preset is rebuilt from
        its active store body. A stale preset is QUARANTINED
        (logged loudly, surfaced as ``conflicted``) rather than registered when its
        name is taken by a foreign tool, or its base tool is missing or itself a
        preset. A preset already bound as THIS SAME spec is re-adopted without a
        rebind (idempotent self-registration). One bad name never aborts the
        boot."""
        prior = self._specs
        self._specs = {}
        self._quarantine = {}
        records = await self._app.presets.store.list_presets()
        # One batched active-body read instead of a per-record round-trip.
        bodies = await self._app.presets.list_active_bodies()
        preset_names = {rec.name for rec in records}
        # Snapshot the live tools ONCE, before registering any preset, so a name
        # found here that is NOT one of ours is genuinely a foreign tool.
        live = set(await self._app.tools.get_tools())
        for rec in records:
            body = bodies.get(rec.name)
            if body is None:
                # The record list and the active-body map are two separate store
                # round-trips; a delete landing between them leaves a record whose
                # active body is already gone. Skip it — the list route makes the
                # same call for this divergence, and the next reload reconciles it —
                # rather than let a bare ``bodies[rec.name]`` KeyError abort the whole
                # boot/reload under the hook's ``raise_on_error``.
                logger.warning(
                    "preset %r skipped during rehydration: its active body is absent "
                    "(store read-skew or a concurrent delete); the next reload reconciles it",
                    rec.name,
                )
                continue
            await self._rehydrate_one(rec.name, body, prior, preset_names, live)

    async def _rehydrate_one(
        self, name: str, body: PresetBody, prior: dict[str, PresetBody], preset_names: set[str], live: set[str]
    ) -> None:
        bound = name in live
        ours = name in prior
        if bound and not ours:
            reason = "its name is occupied by an existing tool"
            self._quarantine[name] = reason
            logger.error("preset %r quarantined: %s", name, reason)
            return
        if bound and ours and prior[name] == body:
            # Already registered as this exact preset — re-adopt the spec without a
            # rebind (the tool + its branches are untouched).
            self._specs[name] = body
            return
        if bound:
            # Ours but its active body changed — tear the stale registration down
            # before re-binding the new body.
            self._remove_registration(name)
        base = body.base_tool
        if base in preset_names:
            reason = f"its base tool {base!r} is itself a preset"
            self._quarantine[name] = reason
            logger.error("preset %r quarantined: %s", name, reason)
            return
        if base not in live:
            reason = f"its base tool {base!r} is not a registered tool"
            self._quarantine[name] = reason
            logger.error("preset %r quarantined: %s", name, reason)
            return
        try:
            await self._register(
                name, base, body.fixed_kwargs, body.extensions, body.tags, body.description, body.output_schema
            )
        except Exception:
            reason = "registration failed"
            self._quarantine[name] = reason
            logger.exception("preset %r quarantined: %s", name, reason)

    # -- internal teardown ----------------------------------------------------

    def _remove_registration(self, name: str) -> None:
        """Drop ``name``'s base + branch tools, its structured-registry entry, and
        its spec — the shared teardown for reload, remove, and register rollback."""
        branches = self._app.tools.unregister_tool_base(name)
        for branch in branches:
            self._safe_remove(branch)
        self._safe_remove(name)
        self._specs.pop(name, None)

    def _safe_remove(self, name: str) -> None:
        """Remove a bound tool, tolerating a name that was never bound.

        Only ``KeyError`` (the FastMCP provider's "no such tool" for an
        idempotent teardown / a partially-bound register rollback) is tolerated;
        every other failure propagates."""
        with contextlib.suppress(KeyError):
            self._app.tools.remove_tool(name)
