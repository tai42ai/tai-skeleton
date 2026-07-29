"""Storage seam: the registration registry, over the contract ``Storage`` ABC.

The ABC and its root-delete guard live in :mod:`tai42_contract.storage`; the
skeleton adds the :class:`StorageRegistry` that collects the active provider.
Re-exported here so the contract symbols share the skeleton's storage namespace.
"""

from tai42_contract.storage import Storage, assert_not_root

from tai42_skeleton.storage.registry import StorageRegistry

__all__ = ["Storage", "StorageRegistry", "assert_not_root"]
