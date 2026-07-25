"""Schema-preserving wrapper extensions for the preset-engine tests.

Listed as the engine tests' ``extensions_modules`` entry (kept separate from the
``tools`` module so no single module is imported by two manifest sections — a
double import would re-bind the tools module's tools, a hard collision under the
server's ``on_duplicate="error"``). A preset's extension COMBOS attach these two
independent wrappers as branches.
"""

from __future__ import annotations

import functools

from tai42_contract.app import tai42_app
from tai42_contract.extensions import ExtensionKind


@tai42_app.extensions.extension(kind=ExtensionKind.WRAPPER, name="exta")
def exta(func, name, desc, config=None):
    """Independent order-marker wrapper: append ``|a`` to a string result;
    schema-preserving so it branches any base cleanly."""

    @functools.wraps(func)
    def variant(*args, **kwargs):
        result = func(*args, **kwargs)
        return f"{result}|a" if isinstance(result, str) else result

    variant.__name__ = f"{name}_exta"
    variant.__qualname__ = variant.__name__
    return variant


@tai42_app.extensions.extension(kind=ExtensionKind.WRAPPER, name="extb")
def extb(func, name, desc, config=None):
    """Independent order-marker wrapper: append ``|b`` to a string result."""

    @functools.wraps(func)
    def variant(*args, **kwargs):
        result = func(*args, **kwargs)
        return f"{result}|b" if isinstance(result, str) else result

    variant.__name__ = f"{name}_extb"
    variant.__qualname__ = variant.__name__
    return variant
