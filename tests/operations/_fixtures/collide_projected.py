"""A tools module that binds a tool named ``reload_config`` — the name a
projected operation claims.

Naming this module in a manifest ``tools[]`` entry while ``api_tools`` projects
the same op is the duplicate-bind collision the D.1 guard defends against: there
must never be a running window with both a hand-bound tool and the projected op
of the same name. The tool binding raises on the duplicate name at boot.
"""

from __future__ import annotations

from tai_contract.app import tai_app


@tai_app.tools.tool
def reload_config() -> str:
    """A rival hand-bound tool colliding with the projected ``reload_config`` op."""
    return "collision"
