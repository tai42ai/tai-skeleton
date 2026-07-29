"""Fixture tool module imported by a manifest ``tools:`` entry.

Importing it fires ``@tai42_app.tools.tool`` against the bound app, registering a plain
tool. Used by the lifecycle/server tests that drive the app through a manifest.
"""

from tai42_contract.app import tai42_app


@tai42_app.tools.tool
def greet(name: str) -> str:
    """Greet someone by name."""
    return f"hello {name}"
