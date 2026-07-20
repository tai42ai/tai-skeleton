"""The concrete :class:`~tai_contract.presets.PresetStore` view.

A thin typed wrapper over the generic
:class:`~tai_contract.versioning.VersionedStore` with ``kind="preset"`` and body
:class:`~tai_contract.presets.PresetBody` (``{base_tool, description,
fixed_kwargs, extensions, tags}``). It holds NO SQL of its own — all persistence
is the generic store. Its jobs are:

* **body validation/reshaping** — an empty INNER extension combo (``[[]]`` or any
  ``[]`` member) is REJECTED loudly; the empty OUTER list ``[]`` is a valid value
  (no extensions on create, an explicit clear on a version save);
* **carry-forward on a version save** — a version body is always the FULL
  :class:`PresetBody`, so ``base_tool`` and ``description`` are ALWAYS carried
  forward from the active version, and each of ``fixed_kwargs``, ``tags``,
  ``extensions`` is carried forward wherever its argument was not provided
  (``None``); an explicitly provided value — including a clearing ``[]`` — wins.
  The new body never DROPS a field;
* **error mapping** — the generic store's errors become the preset error types
  (:class:`PresetNotFoundError` / :class:`PresetExistsError` /
  :class:`PresetVersionNotFoundError`), plus :class:`PresetNameConflictError` when
  the create name collides with an existing base tool.

The name-collision check is an injected async predicate (``name_conflicts``):
the store has no view of the tool registry, so the register/reload engine wires
the real predicate (existing non-preset base tool) when it builds the view. Left
``None``, the view performs no collision check and the create path's own guard is
the authority.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from tai_contract.agent.base import PresetSpec
from tai_contract.manifest import ExtensionElement
from tai_contract.presets import CARRY_FORWARD, CarryForward, PresetBody, PresetStore
from tai_contract.presets.errors import (
    PresetExistsError,
    PresetNameConflictError,
    PresetNotFoundError,
    PresetVersionNotFoundError,
)
from tai_contract.versioning import VersionedStore
from tai_contract.versioning.errors import DocumentExistsError, DocumentNotFoundError, DocumentVersionNotFoundError
from tai_contract.versioning.models import DocumentRecord, DocumentVersion

_KIND = "preset"

# The carry-forward sentinel is ``None`` (the argument default): omitted/``None``
# carries the active value forward; an explicit value the caller passed — including
# a clearing ``[]`` / ``{}`` — is a deliberate value, never "not provided".


def _validate_extensions(extensions: Sequence[Sequence[ExtensionElement]]) -> None:
    """Reject an empty INNER combo. The empty OUTER list is valid (no extensions);
    an empty member combo (``[[]]`` or any ``[]``) is never a legal value."""
    for combo in extensions:
        if not combo:
            raise ValueError("preset extensions must not contain an empty combo")


class PresetStoreView(PresetStore):
    """Typed preset view delegating to a generic :class:`VersionedStore`."""

    def __init__(
        self,
        store: VersionedStore,
        *,
        name_conflicts: Callable[[str], Awaitable[bool]] | None = None,
    ) -> None:
        self._store = store
        self._name_conflicts = name_conflicts

    async def create_preset(
        self,
        spec: PresetSpec,
        extensions: Sequence[Sequence[ExtensionElement]],
        tags: list[str],
        output_schema: dict[str, Any] | None = None,
    ) -> DocumentRecord:
        _validate_extensions(extensions)
        if self._name_conflicts is not None and await self._name_conflicts(spec.name):
            raise PresetNameConflictError(spec.name)
        body = PresetBody(
            base_tool=spec.base_tool,
            description=spec.description,
            fixed_kwargs=spec.fixed_kwargs,
            extensions=[list(combo) for combo in extensions],
            tags=tags,
            output_schema=output_schema,
        )
        try:
            return await self._store.create(_KIND, spec.name, body.model_dump())
        except DocumentExistsError as exc:
            raise PresetExistsError(spec.name) from exc

    async def save_version(
        self,
        name: str,
        fixed_kwargs: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        extensions: Sequence[Sequence[ExtensionElement]] | None = None,
        output_schema: dict[str, Any] | None | CarryForward = CARRY_FORWARD,
    ) -> DocumentVersion:
        active = await self._active_body(name)
        new_extensions = active.extensions if extensions is None else extensions
        _validate_extensions(new_extensions)
        new_output_schema = active.output_schema if isinstance(output_schema, CarryForward) else output_schema
        new_body = PresetBody(
            base_tool=active.base_tool,
            description=active.description,
            fixed_kwargs=active.fixed_kwargs if fixed_kwargs is None else fixed_kwargs,
            extensions=[list(combo) for combo in new_extensions],
            tags=active.tags if tags is None else tags,
            output_schema=new_output_schema,
        )
        try:
            return await self._store.save_version(_KIND, name, new_body.model_dump())
        except DocumentNotFoundError as exc:
            raise PresetNotFoundError(name) from exc

    async def list_presets(self) -> list[DocumentRecord]:
        return await self._store.list(_KIND)

    async def get_preset(self, name: str) -> DocumentRecord:
        try:
            return await self._store.get(_KIND, name)
        except DocumentNotFoundError as exc:
            raise PresetNotFoundError(name) from exc

    async def get_active_kwargs(self, name: str) -> dict[str, Any]:
        return (await self._active_body(name)).fixed_kwargs

    async def list_versions(self, name: str) -> list[DocumentVersion]:
        try:
            return await self._store.list_versions(_KIND, name)
        except DocumentNotFoundError as exc:
            raise PresetNotFoundError(name) from exc

    async def get_version(self, name: str, version: int) -> DocumentVersion:
        try:
            return await self._store.get_version(_KIND, name, version)
        except DocumentVersionNotFoundError as exc:
            raise PresetVersionNotFoundError(name, version) from exc

    async def get_active_body(self, name: str) -> PresetBody:
        return await self._active_body(name)

    async def rollback(self, name: str, version: int) -> DocumentRecord:
        try:
            return await self._store.rollback(_KIND, name, version)
        except DocumentVersionNotFoundError as exc:
            raise PresetVersionNotFoundError(name, version) from exc

    async def soft_delete(self, name: str) -> None:
        try:
            await self._store.soft_delete(_KIND, name)
        except DocumentNotFoundError as exc:
            raise PresetNotFoundError(name) from exc

    async def rename_preset(self, name: str, new_name: str) -> DocumentRecord:
        # A rename must never silently shadow a live tool — the same rule create
        # enforces before any store write — so the injected predicate gates the NEW
        # name first. The body is untouched (rename moves a key), so there is no body
        # validation here.
        if self._name_conflicts is not None and await self._name_conflicts(new_name):
            raise PresetNameConflictError(new_name)
        try:
            return await self._store.rename(_KIND, name, new_name)
        except DocumentNotFoundError as exc:
            raise PresetNotFoundError(name) from exc
        except DocumentExistsError as exc:
            raise PresetExistsError(new_name) from exc

    async def _active_body(self, name: str) -> PresetBody:
        try:
            raw = await self._store.get_active_body(_KIND, name)
        except DocumentNotFoundError as exc:
            raise PresetNotFoundError(name) from exc
        return PresetBody.model_validate(raw)


def preset_store(*, name_conflicts: Callable[[str], Awaitable[bool]] | None = None) -> PresetStoreView:
    """Build the active preset view over the generic versioned-document store."""
    from tai_skeleton.versioning import versioned_store

    return PresetStoreView(versioned_store(), name_conflicts=name_conflicts)
