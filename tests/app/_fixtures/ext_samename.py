"""Fixture extension that returns the tool unchanged (same ``__name__``). The
binder must reject this: an extension has to rename the tool to create a branch,
so a same-name return raises ValueError."""

from tai42_contract.app import tai42_app
from tai42_contract.extensions import ExtensionKind


@tai42_app.extensions.extension(kind=ExtensionKind.WRAPPER, name="samename")
def samename(func, name, desc, config=None):
    return func
