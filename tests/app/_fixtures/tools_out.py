"""Fixture tool with a declared OBJECT output schema for branch-tool
output-schema propagation. ``report`` returns a pydantic model, so
FastMCP derives a real object output schema that shape-preserving branches must
inherit."""

from pydantic import BaseModel
from tai42_contract.app import tai42_app


class Report(BaseModel):
    title: str
    score: int


@tai42_app.tools.tool
def report(text: str) -> Report:
    """Build a report from the text."""
    return Report(title=text, score=len(text))
