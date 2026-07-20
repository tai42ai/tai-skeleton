"""tai42-skeleton — the concrete TaiMCP application skeleton for the TAI ecosystem.

Implements the ``tai42-contract`` protocols (manifest loading, the ``tai42_app``
handle, provider/connector wiring) as a runnable MCP application, composed from
the generic leaf helpers, factories, and clients provided by ``tai42-kit``. This
is the engine that plugins plug into; it depends on ``tai42-contract`` and
``tai42-kit`` only.

Deliberately light: nothing is re-exported here, so ``import tai42_skeleton``
constructs no app singleton and reads no settings. Import the subpackage you
need — construction happens only when the app entry point runs.
"""

from importlib.metadata import version

__version__ = version("tai42-skeleton")

__all__ = ["__version__"]
