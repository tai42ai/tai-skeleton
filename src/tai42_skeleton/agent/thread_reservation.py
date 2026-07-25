"""The reserved ``bridge:`` thread namespace.

A thread under this prefix carries the messaging bridge's per-conversation memory, so only
the bridge may address one. Every door that maps caller-supplied tool input to agent run
kwargs maps through :func:`run_kwargs_from_tool_input`, which is where the reservation is
enforced — an agent run reached by any other spelling would bypass it.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from tai42_contract.agent import Agent

# The reserved thread namespace the messaging bridge alone writes.
BRIDGE_THREAD_PREFIX = "bridge:"

# LangGraph thread-scoping keys a caller could steer into the reserved namespace.
_RESERVED_CONFIGURABLE_KEYS = ("thread_id", "checkpoint_id")


class ReservedThreadNamespaceError(ValueError):
    """A caller steered an agent run at the reserved ``bridge:`` thread namespace."""


def reserved_thread_namespace_error(run_kwargs: dict[str, Any]) -> str | None:
    """The message for a caller-supplied ``bridge:``-prefixed ``thread_id``/``checkpoint_id``
    anywhere in ``run_kwargs``, or ``None`` when none is present.

    These ids ride inside a ``configurable`` mapping on a config-shaped run kwarg, and a run
    can carry several (``langgraph_config``, a voting agent's ``judge_``/``voter_``
    variants), each an equal steering vector — so EVERY config-bearing value is scanned."""
    for value in run_kwargs.values():
        if not isinstance(value, dict):
            continue
        configurable = value.get("configurable")
        if not isinstance(configurable, dict):
            continue
        for key in _RESERVED_CONFIGURABLE_KEYS:
            candidate = configurable.get(key)
            if isinstance(candidate, str) and candidate.startswith(BRIDGE_THREAD_PREFIX):
                return f"{key} may not use the reserved {BRIDGE_THREAD_PREFIX!r} namespace"
    return None


def run_kwargs_from_tool_input(agent: Agent, validated: BaseModel) -> dict[str, Any]:
    """Map ``validated`` to ``agent``'s run kwargs and refuse a reserved thread id — the one
    seam every caller-driven agent run passes, so no door dispatches around the reservation.
    Raises :class:`ReservedThreadNamespaceError` on a reserved id, ``ValueError`` on an input
    the agent's own mapping rejects."""
    run_kwargs = agent.from_tool_input(validated)
    message = reserved_thread_namespace_error(run_kwargs)
    if message is not None:
        raise ReservedThreadNamespaceError(message)
    return run_kwargs


__all__ = [
    "BRIDGE_THREAD_PREFIX",
    "ReservedThreadNamespaceError",
    "reserved_thread_namespace_error",
    "run_kwargs_from_tool_input",
]
