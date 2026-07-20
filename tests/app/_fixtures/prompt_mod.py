"""Fixture lifecycle module that registers a prompt via the ``app.fastmcp``
escape hatch — the supported path for a plugin to add a prompt.

A manifest listing this module under ``lifecycle_modules`` registers
``fixture_prompt`` on import; a reload that DROPS the module must remove the
prompt (the reload prompt/resource symmetry), which the lifecycle tests assert.
"""

from tai_skeleton.app.instance import app


@app.fastmcp.prompt
def fixture_prompt() -> str:
    """A fixture prompt registered through the escape hatch."""
    return "fixture prompt body"
