"""Storage seam tests: a provider registered through the registry conforms to
the contract ``Storage`` ABC, and the registry is dead by default."""

from __future__ import annotations

import pytest
from tai42_contract.storage import Storage

from tai42_skeleton.storage import StorageRegistry


class _InMemoryStorage(Storage):
    def __init__(self) -> None:
        self.items: dict[str, str] = {}

    async def load(self, path: str) -> str:
        try:
            return self.items[path]
        except KeyError as exc:
            raise FileNotFoundError(path) from exc

    async def list(self) -> list[str]:
        return sorted(self.items)

    async def upload(self, path: str, content: str) -> None:
        self.items[path] = content

    async def delete(self, path: str) -> None:
        self.items.pop(path, None)

    async def delete_dir(self, path: str) -> None:
        prefix = path.rstrip("/") + "/"
        for key in [k for k in self.items if k.startswith(prefix)]:
            del self.items[key]


def test_dead_by_default() -> None:
    """A fresh registry holds no provider until one is registered."""
    assert StorageRegistry().provider is None


def test_register_storage_conforms_to_contract() -> None:
    """A class registered via ``register_storage`` is instantiated and is a
    genuine ``Storage`` (satisfies the contract ABC)."""
    registry = StorageRegistry()

    @registry.register_storage
    class Provider(_InMemoryStorage):
        pass

    provider = registry.provider
    assert provider is not None
    assert isinstance(provider, Storage)
    # The registered class itself was instantiated (not a default): the provider
    # is an instance of the concrete class registered here.
    assert isinstance(provider, _InMemoryStorage)


def test_register_storage_called_form() -> None:
    """The called form ``@register_storage()`` works identically to the bare form."""
    registry = StorageRegistry()

    @registry.register_storage()
    class Provider(_InMemoryStorage):
        pass

    assert isinstance(registry.provider, Storage)


def test_incomplete_storage_cannot_instantiate() -> None:
    """An impl missing an abstract method is not a valid ``Storage`` — the ABC
    refuses to instantiate it, so registration raises loudly."""
    registry = StorageRegistry()

    with pytest.raises(TypeError):

        @registry.register_storage
        class Partial(Storage):  # missing load/list/upload/delete/delete_dir
            pass
