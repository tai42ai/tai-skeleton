"""BaseRegistry is the shared abstract base for the manifest-driven registries:
it declares ``validation`` as the one method every concrete registry must
provide, so an impl that omits it cannot instantiate."""

from __future__ import annotations

import pytest

from tai42_skeleton.core.registry.base_registry import BaseRegistry


def test_cannot_instantiate_without_validation() -> None:
    """The abstract ``validation`` method blocks a registry that does not
    implement it from being instantiated."""
    with pytest.raises(TypeError):
        BaseRegistry()  # type: ignore[abstract]


def test_incomplete_subclass_still_abstract() -> None:
    class Incomplete(BaseRegistry):
        pass

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]


def test_concrete_subclass_instantiates_and_runs_validation() -> None:
    class Concrete(BaseRegistry):
        def __init__(self) -> None:
            self.validated = False

        def validation(self) -> None:
            self.validated = True

    reg = Concrete()
    assert isinstance(reg, BaseRegistry)
    reg.validation()
    assert reg.validated is True
