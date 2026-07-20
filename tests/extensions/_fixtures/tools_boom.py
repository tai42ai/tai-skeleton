"""Fixture tool that raises, for exercising the ``monitor`` extension's error
path: the span is marked ``ERROR`` before the exception propagates."""

from tai_contract.app import tai_app


@tai_app.tools.tool
def boom() -> str:
    """Always raise."""
    raise RuntimeError("kaboom")
