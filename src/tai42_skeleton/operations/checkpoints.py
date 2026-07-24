"""Checkpoint retention — the sweep that expires idle conversation threads.

``sweep_checkpoints`` deletes every thread whose newest checkpoint is older than
``checkpoint_ttl_minutes`` (the kit ``LLMProviderSettings``). It is the retention
mechanism for the DB-backed providers (``postgres``/``sqlite``); ``redis`` carries
its own native key TTL and ``memory`` is process-lifetime, so both are a no-op
here, as is an unset TTL. Deletion uses the saver's own ``adelete_thread`` surface.

As an operation it projects as a tool, so it is schedulable through the existing
``/api/schedules`` surface with no extra runner.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from tai42_kit.llm.checkpoint.checkpoint_registry import checkpoint_registry
from tai42_kit.llm.settings import llm_provider_settings

from tai42_skeleton.operations import operation

# Providers with a persisted store the sweep walks; redis uses a native key TTL, memory is process-lifetime.
_SWEEPABLE_PROVIDERS = frozenset({"postgres", "sqlite"})


@operation(
    summary="Sweep expired conversation checkpoints",
    tags=["checkpoints"],
    destructive=True,
    reload_gated=True,
)
async def sweep_checkpoints() -> dict[str, Any]:
    """Delete conversation threads whose newest checkpoint is older than the
    configured idle lifetime; return the provider, the TTL, and the swept threads.

    A no-op (nothing deleted) when the TTL is unset, or the provider is ``redis``
    (native key TTL) or ``memory`` (process-lifetime) — each reported in ``skipped``.
    """
    settings = llm_provider_settings()
    provider = settings.checkpoint
    ttl_minutes = settings.checkpoint_ttl_minutes

    if provider not in _SWEEPABLE_PROVIDERS:
        return {
            "provider": provider,
            "ttl_minutes": ttl_minutes,
            "swept_count": 0,
            "swept_threads": [],
            "skipped": f"provider {provider!r} has no swept store (redis uses a key TTL; memory is process-lifetime)",
        }

    if ttl_minutes is None:
        return {
            "provider": provider,
            "ttl_minutes": None,
            "swept_count": 0,
            "swept_threads": [],
            "skipped": "retention disabled (checkpoint_ttl_minutes unset); checkpoints are kept forever",
        }

    saver = await checkpoint_registry().get_checkpointer(provider=provider, conn_string=settings.checkpoint_conn_string)
    cutoff = datetime.now(UTC) - timedelta(minutes=ttl_minutes)

    # Staleness = each thread's newest checkpoint timestamp vs the cutoff.
    newest_by_thread: dict[str, datetime] = {}
    async for tup in saver.alist(None):
        configurable = tup.config.get("configurable") or {}
        thread_id = configurable["thread_id"]
        ts = datetime.fromisoformat(tup.checkpoint["ts"])
        current = newest_by_thread.get(thread_id)
        if current is None or ts > current:
            newest_by_thread[thread_id] = ts

    stale = sorted(thread_id for thread_id, ts in newest_by_thread.items() if ts < cutoff)
    for thread_id in stale:
        await saver.adelete_thread(thread_id)

    return {
        "provider": provider,
        "ttl_minutes": ttl_minutes,
        "swept_count": len(stale),
        "swept_threads": stale,
    }
