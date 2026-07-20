"""Argv-injection safety guard for synthesized stdio launch specs.

The engine resolver synthesizes a stdio ``(command, args)`` from a provider's
``pkg_manager`` + a sub-service ``entry_point`` (see
:mod:`tai42_skeleton.connectors.runtime.launch`). ``uvx`` / ``npx`` parse argv
positionally, so every interpolated value passes through
:func:`reject_leading_dash`: a ``-`` prefix would be eaten as a flag and could
smuggle extra packages into the spawned environment.
"""

from __future__ import annotations


def reject_leading_dash(value: str, *, field: str) -> None:
    if value.startswith("-"):
        raise ValueError(f"{field} must not start with '-' (would be parsed as a launcher flag)")
