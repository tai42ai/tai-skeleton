"""Fixture extensions for output-schema propagation across a branch, keyed on
``ExtensionKind.preserves_output_shape``.

Against the base tool ``report(text: str) -> Report`` (object output schema):

* ``passw`` (WRAPPER) and ``passb`` (BACKEND) return the base result unchanged
  behind a signature that carries NO return annotation, so each branch declares
  no output schema of its own — they must INHERIT the base's object schema.
* ``listtf`` (TRANSFORMER) reshapes the result into a list, so its branch must
  NOT inherit the base's single-object output schema.
* ``ownw`` (WRAPPER) declares its OWN object output schema — the gate's
  first conjunct ("branch declares none") must leave it untouched.
"""

from makefun import create_function
from pydantic import BaseModel
from tai42_contract.app import tai42_app
from tai42_contract.extensions import ExtensionKind


@tai42_app.extensions.extension(kind=ExtensionKind.WRAPPER, name="passw")
def passw(func, name, desc, config=None):
    """Shape-preserving wrapper with no return annotation — branch inherits."""

    def impl(*args, **kwargs):
        return func(*args, **kwargs)

    return create_function(f"{name}_passw(text: str)", impl, func_name=f"{name}_passw")


@tai42_app.extensions.extension(kind=ExtensionKind.BACKEND, name="passb")
def passb(func, name, desc, config=None):
    """Shape-preserving backend swap with no return annotation — branch inherits."""

    def impl(*args, **kwargs):
        return func(*args, **kwargs)

    return create_function(f"{name}_passb(text: str)", impl, func_name=f"{name}_passb")


@tai42_app.extensions.extension(kind=ExtensionKind.TRANSFORMER, name="listtf")
def listtf(func, name, desc, config=None):
    """Reshape the single result into a list — its branch must NOT inherit the
    base's single-object output schema. Presents a concrete signature (the
    transformer input-schema rule)."""

    def impl(*args, **kwargs):
        return [func(*args, **kwargs)]

    return create_function(f"{name}_listtf(text: str)", impl, func_name=f"{name}_listtf")


class OwnOut(BaseModel):
    kept: bool


@tai42_app.extensions.extension(kind=ExtensionKind.WRAPPER, name="ownw")
def ownw(func, name, desc, config=None):
    """Shape-preserving wrapper that declares its OWN object output schema — the
    gate must leave it, never force the base schema over it."""

    def variant(text: str) -> OwnOut:
        return OwnOut(kept=bool(func(text=text)))

    variant.__name__ = f"{name}_ownw"
    variant.__qualname__ = variant.__name__
    return variant
