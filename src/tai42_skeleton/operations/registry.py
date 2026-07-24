"""The operation metadata record and the process-wide operation registry.

An :class:`OperationMetadata` carries everything the four downstream consumers
need from one declaration: the route adapter (route template + method + request
model + declared errors), the OpenAPI emitter (``destructive`` → ``x-destructive``),
the tool projection (name, destructive → ``destructiveHint``, reload gate), and
the tool-edge authorization (route template + method for the concrete-path
synthesis the scope verifier and fences key on).

The decorator fills the operation-level fields; the adapter registration attaches
the route template and HTTP method to the SAME record, because authz's path
synthesis and the fences' ``{"method", "path"}`` context need both.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from tai42_skeleton.operations.errors import OperationError


@dataclass
class OperationMetadata:
    """One operation's declared metadata — the single source the route, the CLI,
    the OpenAPI spec, the MCP tool, and the authorization check all derive from."""

    name: str
    func: Callable[..., Awaitable[object]]
    summary: str
    tags: tuple[str, ...] = ()
    destructive: bool = False
    reload_gated: bool = False
    meta_executor: bool = False
    caller_context: bool = False
    authority_changing: bool = False
    error_classes: tuple[type[OperationError], ...] = ()
    request_model: type[BaseModel] | None = None
    response_model: type[BaseModel] | None = None

    # Attached by the route-adapter registration (not the decorator): the route
    # template (``/api/tools/{name}``) and the HTTP method the operation serves.
    # Authz synthesizes the concrete path from these + the call's path args.
    route_template: str | None = None
    http_method: str | None = None

    # Filled from the adapter's ``path_params`` so authz knows which call
    # arguments name path segments in ``route_template``.
    path_params: tuple[str, ...] = field(default_factory=tuple)


class OperationRegistry:
    """In-memory map of every registered operation, keyed by operation name.

    Registering the SAME name twice raises loudly — mirroring the tool binding's
    duplicate-name guard — so two operations cannot silently claim one name.

    The registry also states whether it is SETTLED, i.e. whether its silence about a
    name is an answer. A boot/reload tears it down and replays it while the serving
    loop keeps dispatching, so a reader that must distinguish "not an operation" from
    "not replayed yet" consults :attr:`settled` rather than guessing from the contents.
    """

    def __init__(self) -> None:
        self._operations: dict[str, OperationMetadata] = {}
        self._rebuild_depth = 0

    def register(self, metadata: OperationMetadata) -> None:
        existing = self._operations.get(metadata.name)
        if existing is not None and existing is not metadata:
            raise ValueError(
                f"Operation {metadata.name!r} is already registered; "
                "an operation name must be unique across the registry."
            )
        self._operations[metadata.name] = metadata

    def get(self, name: str) -> OperationMetadata:
        try:
            return self._operations[name]
        except KeyError:
            raise KeyError(f"Operation {name!r} is not registered.") from None

    def has(self, name: str) -> bool:
        return name in self._operations

    def all(self) -> list[OperationMetadata]:
        """Every registered operation, ordered by name for a stable surface."""
        return [self._operations[name] for name in sorted(self._operations)]

    def names(self) -> frozenset[str]:
        return frozenset(self._operations)

    def clear(self) -> None:
        """Drop every registration — the reload path rebuilds from module import."""
        self._operations.clear()

    @contextmanager
    def rebuilding(self) -> Iterator[None]:
        """Mark the surface UNSETTLED for the duration of a teardown + replay.

        The boot/reload body wraps the whole rebuild in this — the ``clear()``, the
        snapshot replay, and the router re-attachment that puts each record's route
        template back — because a record is only usable once all three have run. It
        nests (a reload reached from inside another rebuild keeps the mark held until
        the outermost one finishes) and releases in a ``finally``, so a rebuild that
        raises cannot leave the surface permanently marked torn.
        """
        self._rebuild_depth += 1
        try:
            yield
        finally:
            self._rebuild_depth -= 1

    @property
    def settled(self) -> bool:
        """Whether this registry's contents are an ANSWER: no rebuild is in flight AND
        it is non-empty.

        Emptiness alone is decisive: a started app always holds this package's leaf
        operations, so an empty registry is only ever the cleared half of a rebuild (or
        an app that never started). Both halves must hold — a rebuild that has already
        replayed some records is populated but not yet an answer about the rest.
        """
        return self._rebuild_depth == 0 and bool(self._operations)


# The one process-wide operation registry. The ``@operation`` decorator records
# into it; the adapter, the emitter, the projection, and authz read it.
operation_registry = OperationRegistry()
