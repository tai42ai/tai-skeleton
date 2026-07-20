"""Fixture tool for the ``ask_external`` transformer: it accepts ``callback_url``
and returns the external URL the human visits (here a fake signing link)."""

from tai_contract.app import tai_app


@tai_app.tools.tool
async def make_signature(document: str, callback_url: str) -> str:
    """Create an external signature request and return its URL."""
    return f"https://sign.example/{document}?cb={callback_url}"
