"""Fixture lifecycle module that registers a static resource and a templated
resource through the ``app.fastmcp`` escape hatch on import.

A manifest listing this module under ``lifecycle_modules`` registers both on
import; every ``start()`` reimport re-fires the decorators, so the reset must
clear both the ``Resource`` and the ``ResourceTemplate`` surface for the re-fire
to stay idempotent.
"""

from tai_skeleton.app.instance import app


@app.fastmcp.resource("fixture://static")
def fixture_static_resource() -> str:
    """A fixture static resource."""
    return "static resource body"


@app.fastmcp.resource("fixture://item/{item_id}")
def fixture_templated_resource(item_id: str) -> str:
    """A fixture templated resource — registers a ResourceTemplate."""
    return f"item {item_id}"
