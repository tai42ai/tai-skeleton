"""Fixture extensions for a two-link stacked combo (``{shout: [[first, second]]}``).

Each WRAPPER branches the tool into a renamed variant and appends its own marker
to the description it receives. Stacked, the second must compose on the first's
running description — so a tool built through the ``[first, second]`` combo ends
with ``"... | first | second"``, proving the binder carries the running
description forward across the chain rather than resetting to the original.
"""

import functools

from tai42_contract.app import tai42_app
from tai42_contract.extensions import ExtensionKind


@tai42_app.extensions.extension(kind=ExtensionKind.WRAPPER, name="first")
def first(func, name, desc, config=None):
    @functools.wraps(func)
    def first_variant(*args, **kwargs):
        return func(*args, **kwargs)

    first_variant.__name__ = f"{name}_first"
    first_variant.__qualname__ = f"{name}_first"
    first_variant.__doc__ = (desc or "") + " | first"
    return first_variant


@tai42_app.extensions.extension(kind=ExtensionKind.WRAPPER, name="second")
def second(func, name, desc, config=None):
    @functools.wraps(func)
    def second_variant(*args, **kwargs):
        return func(*args, **kwargs)

    second_variant.__name__ = f"{name}_second"
    second_variant.__qualname__ = f"{name}_second"
    second_variant.__doc__ = (desc or "") + " | second"
    return second_variant
