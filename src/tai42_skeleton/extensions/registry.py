import inspect
from collections.abc import Callable, Iterator, Sequence
from typing import Any

from tai42_contract.extensions import ExtensionKind
from tai42_contract.manifest import ExtensionElement

from tai42_skeleton.core.registry.base_registry import BaseRegistry
from tai42_skeleton.exceptions.exceptions import TaiValidationError


def extension_name(element: ExtensionElement) -> str:
    """The extension NAME of a combo element — the bare string itself, or the
    ``name`` of a ``{"name", "config"}`` mapping."""
    if isinstance(element, str):
        return element
    return element["name"]


def extension_config(element: ExtensionElement) -> dict[str, Any]:
    """The author-bound config of a combo element — empty for a bare name, the
    ``config`` mapping of a ``{"name", "config"}`` element."""
    if isinstance(element, str):
        return {}
    return element["config"]


def factory_accepts_config(func: Callable) -> bool:
    """Whether an extension factory accepts the author-config argument.

    A factory that closes over author config declares a ``config`` parameter (after
    ``func``/``name``/``description``); the apply site passes it by keyword, so the
    parameter is found whether it is positional-or-keyword or keyword-only. A
    config-agnostic factory omits ``config`` and is called with three arguments, so
    an extension that takes no config needs no change to opt out."""
    try:
        params = inspect.signature(func).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == "config" and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY) for p in params)


class ExtensionRegistry(BaseRegistry):
    def __init__(self, requested_extensions: frozenset[str]):
        self._requested_extensions = requested_extensions

        self._kinds: dict[str, ExtensionKind] = {}
        self._extensions: dict[str, Callable] = {}
        self._available_extensions: set[str] = set()
        self._requires_body_locality: dict[str, bool] = {}

    def register_extension(
        self, func: Callable, kind: ExtensionKind, name: str, *, requires_body_locality: bool = False
    ):
        # The registry is rebuilt fresh on every start/reload, so a name already
        # present within one build is a genuine collision (two extensions claiming
        # one name), never a legitimate re-register — fail loud rather than let the
        # last one silently win.
        if name in self._extensions:
            raise TaiValidationError(f"Extension '{name}' is already registered.")
        self._extensions[name] = func
        self._kinds[name] = kind
        self._requires_body_locality[name] = requires_body_locality
        self._available_extensions.add(name)

    def extension(
        self,
        f: Callable | None = None,
        *,
        kind: ExtensionKind,
        name: str | None = None,
        requires_body_locality: bool = False,
    ):
        """Decorator form of :meth:`register_extension` (the ``app.extensions``
        facet body): usable bare or with arguments.

        ``requires_body_locality`` marks an extension whose wrapper only works in
        the process running the tool body it wraps; the bind engine reads it to
        require the extension to sit INSIDE any execution-relocating
        (``ExtensionKind.relocates_execution``) extension in a stacked combo."""
        if f and callable(f):
            return self.extension(kind=kind, name=name, requires_body_locality=requires_body_locality)(f)

        def decorator(func):
            self.register_extension(func, kind, name or func.__name__, requires_body_locality=requires_body_locality)
            return func

        return decorator

    def get_extension(self, element: ExtensionElement) -> Callable:
        name = extension_name(element)
        try:
            return self._extensions[name]
        except KeyError:
            raise TaiValidationError(f"Extension '{name}' is not registered.") from None

    def get_kind(self, element: ExtensionElement) -> ExtensionKind:
        name = extension_name(element)
        try:
            return self._kinds[name]
        except KeyError:
            raise TaiValidationError(f"Extension '{name}' is not registered.") from None

    def requires_body_locality(self, element: ExtensionElement) -> bool:
        """Whether the extension registered with ``requires_body_locality`` — its
        wrapper only works in the process running the tool body it wraps, so the
        bind engine must place it INSIDE any execution-relocating extension."""
        name = extension_name(element)
        try:
            return self._requires_body_locality[name]
        except KeyError:
            raise TaiValidationError(f"Extension '{name}' is not registered.") from None

    def available_extensions(self) -> list[dict[str, str]]:
        # Serialize the enum VALUE (lowercase, e.g. "backend"), not its member
        # name ("BACKEND"): only the value round-trips back through
        # ``ExtensionKind("backend")``.
        return [{"name": name, "kind": self._kinds[name].value} for name in sorted(self._available_extensions)]

    def validate(self, elements: Sequence[ExtensionElement]):
        # Resolve each combo element to its extension NAME (a bare name or a
        # ``{"name", "config"}`` mapping) before the registry lookups.
        names = [extension_name(element) for element in elements]
        # Membership-check first so an unknown name raises a descriptive error
        # rather than a bare KeyError when the cardinality loop indexes ``_kinds``.
        unknown = [name for name in names if name not in self._kinds]
        if unknown:
            raise TaiValidationError(f"Extension(s) '{', '.join(sorted(unknown))}' not found in server.")
        for kind in self.kind_iterator():
            kind_names = [name for name in names if self._kinds[name] == kind]
            if not kind.multiple and len(kind_names) > 1:
                raise TaiValidationError(f"Only one '{kind}' extension is allowed. Found multiple: {kind_names}")

    def missing_extensions(self) -> set[str]:
        return set(self._requested_extensions - self._available_extensions)

    @staticmethod
    def kind_iterator() -> Iterator[ExtensionKind]:
        return iter(ExtensionKind)

    def validation(self):
        missing_extensions = self.missing_extensions()
        if missing_extensions:
            raise TaiValidationError(f"Extension(s) '{', '.join(sorted(missing_extensions))}' not found in server.")
