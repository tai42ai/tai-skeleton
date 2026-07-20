"""Presets: the concrete typed store view over the generic versioned-document
store, plus the bind kernel every preset builds its live tool through.

A *preset* is a base tool with a partial set of keyword arguments baked in,
exposed as a new named tool. :func:`preset_bind` is the kernel that builds the
live tool (a hidden/fixed transform of the base tool); :class:`PresetStoreView`
is the typed view over ``app.versioning.store`` with ``kind="preset"`` that
persists and versions it. tai-contract owns the ``PresetStore`` Protocol +
:class:`~tai_contract.presets.PresetBody` model + the preset errors; this package
holds the concrete view + the bind kernel.

Baked values are a *may-run* boundary, not a *may-read* one. A preset's (and an
authored agent's) baked ``fixed_kwargs`` are hidden from framework-manufactured
surfaces — the tool/agent input schema and the validation-error messages — so a
caller authorized only to RUN the preset does not see them through those channels.
The baked values are, however, present in the agent's runtime context: anything
the agent emits — assistant messages, tool-call arguments and traces, and the
``stream.error`` frame's message text — MAY contain them, and the framework does
not (and cannot generically) scrub agent-authored output or exception text. Bake
provider/secret *references* — names resolved server-side (e.g. an ``llm_provider``
name; connector tokens injected at runtime from the token store) — into
``fixed_kwargs``, never raw credentials. If the baked config is sensitive, treat
run authorization as "may run" rather than "may read the baked config" and
restrict which keys or routes a scoped caller may reach.
"""

from __future__ import annotations

from tai_contract.presets import PresetSpec

from tai_skeleton.presets.bind import preset_bind
from tai_skeleton.presets.manager import PresetManager
from tai_skeleton.presets.store import PresetStoreView, preset_store

__all__ = ["PresetManager", "PresetSpec", "PresetStoreView", "preset_bind", "preset_store"]
