"""Fixture tool module whose tool ``shout`` is extended by the ``loud`` wrapper
when the manifest attaches it via ``extensions: {shout: [loud]}``."""

from tai42_contract.app import tai42_app


@tai42_app.tools.tool
def shout(text: str) -> str:
    """Return the text."""
    return text


@tai42_app.tools.tool
def ping() -> str:
    """Return pong."""
    return "pong"
