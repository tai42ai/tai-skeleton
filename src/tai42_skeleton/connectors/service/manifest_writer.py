"""Connector-managed manifest entries — in-place add / remove mutators.

Each enabled sub-service produces one ``TaiMCPConfig`` entry titled
``{provider}_{sub_service}_{alias}`` with a ``managed`` back-reference.
``config.headers`` stays empty — the Authorization header is injected at call
time by the runtime resolver. Hand-authored entries (``managed is None``) are
never touched; this module only appends or removes entries whose
``managed.connection_id`` it owns.

:func:`add_managed_entries` / :func:`remove_managed_entries` edit the PRESERVED
manifest document IN PLACE and return the titles they touched; the connection
service feeds them to :meth:`~tai42_skeleton.config.service.ConfigService.apply_change`,
which owns the read-modify-write transaction, validation, local reload, and the
fleet broadcast. They carry no config-backend coupling — the pipeline hands the
document in and persists it back.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from tai42_contract.connectors.models import ConnectorRef
from tai42_contract.connectors.providers import ProviderDescriptor
from tai42_contract.manifest import MCPConfig, TaiMCPConfig

from tai42_skeleton.connectors.runtime.launch import resolve_mcp_server

logger = logging.getLogger(__name__)


def managed_title(provider_id: str, sub_service: str, alias: str) -> str:
    return f"{provider_id}_{sub_service}_{alias}"


def _build_entry(
    *,
    descriptor: ProviderDescriptor,
    sub_service_id: str,
    alias: str,
    connection_id: str,
) -> TaiMCPConfig:
    server = resolve_mcp_server(descriptor, sub_service_id)
    if server.type == "stdio":
        # env carries static transport config only; the access token is injected
        # per-call via JSON-RPC _meta, never via env.
        inner = MCPConfig(
            type="stdio",
            command=server.command,
            args=list(server.args),
            env=dict(server.env),
        )
    else:
        inner = MCPConfig(
            type=server.type,
            url=server.url,
            headers=dict(server.extra_headers),
            env={},
        )
    return TaiMCPConfig(
        title=managed_title(descriptor.id, sub_service_id, alias),
        include=[],
        exclude=[],
        config=inner,
        managed=ConnectorRef(
            connection_id=connection_id,
            provider_id=descriptor.id,
            sub_service=sub_service_id,
        ),
    )


def add_managed_entries(
    document: dict[str, Any],
    *,
    descriptor: ProviderDescriptor,
    enabled_sub_services: Iterable[str],
    alias: str,
    connection_id: str,
) -> list[str]:
    """Append one managed entry per sub-service to the preserved manifest
    ``document`` in place; return the titles added.

    Idempotent: an entry already owned by this connection is left in place. A
    title owned by a different connection raises (collision is operator-visible).
    """
    entries: list[Any] = list(document.get("mcp") or [])
    existing_by_title = {entry["title"]: entry for entry in entries}
    added: list[str] = []

    for sub_id in enabled_sub_services:
        entry = _build_entry(
            descriptor=descriptor,
            sub_service_id=sub_id,
            alias=alias,
            connection_id=connection_id,
        )
        current = existing_by_title.get(entry.title)
        if current is not None:
            managed = current.get("managed")
            if managed is not None and managed.get("connection_id") == connection_id:
                continue
            raise ValueError(
                f"manifest title collision: {entry.title!r} already exists "
                f"and is not owned by connection {connection_id}"
            )
        entries.append(entry.model_dump(mode="json", exclude_none=True))
        added.append(entry.title)

    document["mcp"] = entries
    logger.info(
        "connectors: added %d managed manifest entries (connection=%s)",
        len(added),
        connection_id,
    )
    return added


def remove_managed_entries(
    document: dict[str, Any],
    *,
    connection_id: str,
    sub_services: Iterable[str] | None = None,
) -> list[str]:
    """Remove managed entries owned by ``connection_id`` from the preserved
    manifest ``document`` in place; return the titles removed.

    ``sub_services is None`` removes all of the connection's entries (Disconnect);
    otherwise only those whose ``sub_service`` is in the set (toggle-off).
    """
    sub_set = set(sub_services) if sub_services is not None else None
    keep: list[Any] = []
    removed: list[str] = []

    for entry in document.get("mcp") or []:
        managed = entry.get("managed")
        if (
            managed is not None
            and managed.get("connection_id") == connection_id
            and (sub_set is None or managed.get("sub_service") in sub_set)
        ):
            removed.append(entry["title"])
            continue
        keep.append(entry)

    document["mcp"] = keep
    logger.info(
        "connectors: removed %d managed manifest entries (connection=%s)",
        len(removed),
        connection_id,
    )
    return removed
