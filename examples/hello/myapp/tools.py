"""A minimal tool module: one function registered with ``@tai_app.tools.tool``.

The manifest's ``tools:`` section imports this module by its import path
(``myapp.tools``); importing it runs the decorator, which registers ``greet``
as an MCP tool on the server.
"""

from tai_contract.app import tai_app


@tai_app.tools.tool
def greet(name: str) -> str:
    """Greet a person by name."""
    return f"Hello, {name}!"
