"""A THIRD tools module that also defines ``shout`` — a different module than
``tests.app._fixtures.tools_b``'s ``shout``.

Two ``tools`` configs each providing a tool of the same name is the genuine
ambiguity the extension-apply route's owning-config guard defends against: the
route cannot know which config to edit, so it raises."""

from __future__ import annotations

from tai_contract.app import tai_app


@tai_app.tools.tool
def shout(text: str) -> str:
    """A second definition of ``shout`` from a rival module."""
    return text
