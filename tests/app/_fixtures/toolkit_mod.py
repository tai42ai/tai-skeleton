"""Fixture toolkit module: ``@tai_app.toolkit`` adapts each LangChain tool the
toolkit yields into a bound MCP tool (named ``<prefix>_<tool>``)."""

from langchain_core.tools import StructuredTool
from tai_contract.app import tai_app


def _echo(value: int) -> int:
    """Echo the integer back."""
    return value


def _double(value: int) -> int:
    """Double the integer."""
    return value * 2


class _Kit:
    def get_tools(self):
        # Two tools so a manifest that includes only one exercises the toolkit's
        # should-include skip branch (the excluded ``double`` is not bound).
        return [
            StructuredTool.from_function(_echo, name="echo", description="echo an int"),
            StructuredTool.from_function(_double, name="double", description="double an int"),
        ]


@tai_app.tools.toolkit
def widgets():
    """A toolkit yielding ``echo`` and ``double`` tools."""
    return _Kit()
