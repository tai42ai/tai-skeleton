from abc import ABC, abstractmethod


class BaseRegistry(ABC):
    """Common base for the manifest-driven registries.

    Every registry collects requested entries and then verifies them against
    what was actually registered, so ``validation`` is the one method each
    concrete registry must provide.
    """

    @abstractmethod
    def validation(self) -> None: ...
