"""The boot-time plugin-quarantine registry — name → human-readable reason.

A quarantined plugin is one the boot pass SKIPPED instead of loading: its
compat verdict against the running ``tai42-contract`` came back incompatible,
or its import/load raised. Quarantining is what lets the server boot (and keep
serving everything else) instead of crash-looping on one broken plugin — for
ADDITIVE plugins only; the scalar backend/storage/monitoring slots still abort
boot, because the server cannot run without them.

Invariants:

* The map is reset at the top of every boot/reload pass and repopulated by that
  pass alone (the lifecycle's additive-module imports, then the Studio-plugin
  registry rebuild handler), so it always describes the CURRENT generation —
  never an accumulation across reloads.
* Every quarantine is LOUD: :func:`quarantine_plugin` emits one error log line
  per plugin, and the map is surfaced through the marketplace installed listing
  and counted in the readiness payload. The count never flips readiness red —
  the server is serving; the quarantine is operator-visible degradation, not
  unreadiness.
* This module imports nothing beyond the stdlib so the readiness route and the
  operations layer can read it without pulling the marketplace package in.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_quarantined: dict[str, str] = {}


def reset_quarantine() -> None:
    """Clear the map at the top of a boot/reload pass — the pass that follows
    owns repopulating it."""
    _quarantined.clear()


def quarantine_plugin(name: str, reason: str) -> None:
    """Record ``name`` as quarantined with its human-readable ``reason``,
    emitting the one loud startup log line the quarantine contract requires."""
    _quarantined[name] = reason
    logger.error("plugin quarantined: %s — %s", name, reason)


def quarantined_plugins() -> dict[str, str]:
    """A snapshot copy of the map (name → reason), safe to hand out."""
    return dict(_quarantined)


def quarantine_count() -> int:
    """The number of quarantined plugins — the readiness payload's count."""
    return len(_quarantined)
