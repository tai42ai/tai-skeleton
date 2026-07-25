"""Fixture extension module: registers a WRAPPER extension that branches a tool
into a renamed variant. Imported via a manifest ``extensions:`` entry.

The wrapper preserves the wrapped tool's call signature (via ``functools.wraps``)
so the renamed branch is a valid FastMCP tool, then renames it so the extension
creates a distinct branch rather than colliding with the base tool name.
"""

import functools

from tai42_contract.app import tai42_app
from tai42_contract.extensions import ExtensionKind


@tai42_app.extensions.extension(kind=ExtensionKind.WRAPPER, name="loud")
def loud(func, name, desc, config=None):
    """Branch ``func`` into a renamed variant whose string result is upper-cased."""

    @functools.wraps(func)
    def loud_variant(*args, **kwargs):
        result = func(*args, **kwargs)
        return result.upper() if isinstance(result, str) else result

    loud_variant.__name__ = f"{name}_loud"
    loud_variant.__qualname__ = f"{name}_loud"
    loud_variant.__doc__ = (desc or "") + " (loud)"
    return loud_variant
