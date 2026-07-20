"""The operations layer — typed async functions + declared metadata.

An operation is a plain async function decorated with :func:`operation`; the
decorator records its metadata into :data:`operation_registry`. Routes are thin
adapters over operations (:func:`register_operation_route`), the OpenAPI spec and
CLI derive from those routes, and the MCP tool surface projects from the registry
(:func:`project_operations`). One source; every management surface derives from it.
"""

import pkgutil

from tai_skeleton.operations.adapter import register_operation_route
from tai_skeleton.operations.decorator import operation, operation_metadata_of
from tai_skeleton.operations.errors import (
    BadRequestError,
    ConflictError,
    ForbiddenError,
    NotFoundError,
    NotSupportedError,
    OperationError,
    OperationFailed,
    PayloadTooLargeError,
    PermissionDenied,
    UnavailableError,
    UpstreamError,
    ValidationRejected,
)
from tai_skeleton.operations.projection import project_operations
from tai_skeleton.operations.registry import (
    OperationMetadata,
    OperationRegistry,
    operation_registry,
)

# The infrastructure modules of this package — the decorator, the registry, the
# error types, the route adapter, and the projection. They declare NO operations
# and must NEVER be reloaded: reloading ``registry`` or ``decorator`` would mint a
# NEW ``operation_registry`` object and orphan the singleton the lifecycle, the
# projection, and the tool-edge authorization all hold. Every OTHER module in this
# package is an operation-declaration LEAF whose ``@operation`` decorators register
# into that singleton at import.
_INFRA_MODULES = frozenset({"__init__", "adapter", "decorator", "errors", "projection", "registry", "_broadcast"})


def operation_leaf_modules() -> list[str]:
    """The fully-qualified name of every operation-declaration leaf module.

    Discovered dynamically from the package directory, so a new operations leaf
    module is discovered automatically — a new declaration file can never be
    silently dropped from re-registration for want of a hand-maintained list.
    """
    return sorted(
        f"{__name__}.{info.name}" for info in pkgutil.iter_modules(__path__) if info.name not in _INFRA_MODULES
    )


# The leaf operations captured from the FIRST re-import — the STABLE in-memory
# source every later reload repopulates the cleared registry from. The leaf modules
# in this package are static (a deployed process never gains or edits one across an
# in-process reload), and the metadata objects are exactly the records the still-cached
# leaf functions carry as ``__operation__``, so re-adding them makes the registry hold
# the record a router's ``register_operation_route`` decorates and the projection reads.
_leaf_snapshot: list[OperationMetadata] | None = None


def reregister_operations() -> list[str]:
    """Repopulate the cleared ``operation_registry`` with this package's leaf
    operations; return the leaf module names re-imported (empty after the first call).

    The registry is process-global and a decorator fires exactly once per interpreter,
    so a plain re-import of a router that merely ``from operations.<domain> import <op>``
    never re-registers a leaf that stayed cached in ``sys.modules`` — and the reload
    path clears the registry, so without this the surface would project nothing.

    The FIRST call pops each leaf from ``sys.modules`` and re-imports it (never this
    package or its infra, so the singleton is preserved), re-firing every ``@operation``
    into the cleared registry, then snapshots the registered records. Every LATER call
    re-adds that snapshot without touching ``sys.modules``: the leaf modules are static
    across an in-process reload and the snapshot records are the SAME objects the cached
    leaf functions carry as ``__operation__``, so the routers re-attach their route
    templates to the very records now back in the registry. Avoiding the per-reload
    sys.modules churn keeps the reload off the import machinery's blocking I/O, whose
    yields would otherwise let a reload's synchronous re-import interleave with the
    concurrently-running worker-bus subscription task. A leaf import that fails on the first
    call propagates loudly.
    Runs after ``operation_registry.clear()`` and before the routers re-attach their
    route templates, so the registered metadata is the record the routes decorate and the
    projection reads.
    """
    global _leaf_snapshot

    if _leaf_snapshot is not None:
        for metadata in _leaf_snapshot:
            operation_registry.register(metadata)
        return []

    # Imported lazily: this package is imported very early and by the app package
    # itself, so a module-level import of the app importer would be circular.
    from tai_skeleton.app.importer import import_or_reload_package

    reloaded: list[str] = []
    for module in operation_leaf_modules():
        reloaded.extend(import_or_reload_package(module))
    _leaf_snapshot = operation_registry.all()
    return reloaded


__all__ = [
    "BadRequestError",
    "ConflictError",
    "ForbiddenError",
    "NotFoundError",
    "NotSupportedError",
    "OperationError",
    "OperationFailed",
    "OperationMetadata",
    "OperationRegistry",
    "PayloadTooLargeError",
    "PermissionDenied",
    "UnavailableError",
    "UpstreamError",
    "ValidationRejected",
    "operation",
    "operation_leaf_modules",
    "operation_metadata_of",
    "operation_registry",
    "project_operations",
    "register_operation_route",
    "reregister_operations",
]
