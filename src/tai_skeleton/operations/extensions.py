"""Extension operations — the authed catalog read for the UI's picker.

``list_extensions`` returns the flat ``{"name", "kind"}`` list of every
registered extension (kind carried as its lowercase enum value so the UI can
group and single-select the non-stackable BACKEND kind). A thin skin over the
``tai_app.extensions`` facet — no extension logic lives here.
"""

from __future__ import annotations

from tai_contract.app import tai_app

from tai_skeleton.operations import operation


@operation(summary="List every registered extension", tags=["extensions"])
async def list_extensions() -> list[dict[str, str]]:
    return tai_app.extensions.available_extensions()
