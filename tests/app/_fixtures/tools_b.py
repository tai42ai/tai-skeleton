"""Fixture tool module whose tool ``shout`` is extended by the ``loud`` wrapper
when the manifest attaches it via ``extensions: {shout: [loud]}``."""

from tai_contract.app import tai_app


@tai_app.tools.tool
def shout(text: str) -> str:
    """Return the text."""
    return text


@tai_app.tools.tool
def ping() -> str:
    """Return pong."""
    return "pong"
