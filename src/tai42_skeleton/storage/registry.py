"""The Storage provider registry — the skeleton's registration seam.

The skeleton ships **no** concrete :class:`~tai42_contract.storage.Storage` backend:
storage is *dead by default*. Until a provider is registered through
:meth:`StorageRegistry.register_storage`, the registry holds none, and a
:class:`~tai42_skeleton.template.ResourceManager` built on it raises loudly the
moment it is used (never a silent no-op). A backend ships as a separate
``tai42-contract``-only plugin, loaded by the manifest, that registers through the
app's ``register_storage`` handle.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar, overload

from tai42_contract.storage import Storage

_StorageT = TypeVar("_StorageT", bound=Storage)


class StorageRegistry:
    """Holds the active :class:`Storage` provider for one app instance."""

    def __init__(self) -> None:
        self._provider: Storage | None = None

    @overload
    def register_storage(self, cls: type[_StorageT]) -> type[_StorageT]: ...

    @overload
    def register_storage(self, cls: None = None) -> Callable[[type[Storage]], type[Storage]]: ...

    def register_storage(
        self, cls: type[Storage] | None = None
    ) -> Callable[[type[Storage]], type[Storage]] | type[Storage]:
        """Register a :class:`Storage` subclass as the active provider.

        Usable bare (``@register_storage``) or called (``@register_storage()``).
        The class is instantiated immediately and replaces any prior provider.
        """
        if cls is not None:
            return self.register_storage()(cls)

        def decorator(cls: type[Storage]) -> type[Storage]:
            self._provider = cls()
            return cls

        return decorator

    @property
    def provider(self) -> Storage | None:
        """The registered provider, or ``None`` while dead by default.

        Returning ``None`` is the unconfigured state, surfaced to the caller as-is;
        the loud failure happens in :class:`ResourceManager` when an unconfigured
        manager is actually used.
        """
        return self._provider


__all__ = ["StorageRegistry"]
