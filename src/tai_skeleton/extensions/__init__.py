"""Extension subsystem — taxonomy and registry.

Extensions are plain callables registered against an :class:`ExtensionKind`
(``WRAPPER`` / ``TRANSFORMER`` / ``BACKEND``) and applied to a tool. The
:class:`ExtensionRegistry` holds the registered extensions and enforces the
``multiple`` cardinality rule per kind.
"""

from tai_contract.extensions import ExtensionKind

from tai_skeleton.extensions.registry import ExtensionRegistry

__all__ = ["ExtensionKind", "ExtensionRegistry"]
