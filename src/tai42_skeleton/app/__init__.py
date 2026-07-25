"""The app facade package: ``TaiMCP`` (the concrete ``tai42_contract.app.TaiApp``
impl) and its lifecycle/facets.

The process singleton lives in :mod:`tai42_skeleton.app.instance` (imported
explicitly so the heavy app object is built only when a launcher wants it).
"""

from tai42_skeleton.app.server import TaiMCP

__all__ = ["TaiMCP"]
