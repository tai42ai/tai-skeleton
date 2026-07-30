"""The tool-metadata overlay: the concrete Postgres store behind the
:class:`~tai42_contract.tool_meta.ToolMetaStore` contract.

An UNVERSIONED organizational layer over ANY tool in the namespace — folders (real
nesting entities) plus a per-tool row of ``display_name`` / ``folder_id`` / ``tags``
/ ``hidden``, keyed by tool name. The store owns the invariants the tables cannot
express (cycle-freedom, empty-folder deletes, clean-slate name reclaim).
"""

from __future__ import annotations

from tai42_skeleton.tool_meta.store import PostgresToolMetaStore, tool_meta_store

__all__ = ["PostgresToolMetaStore", "tool_meta_store"]
