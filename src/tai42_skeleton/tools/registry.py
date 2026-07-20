from collections import defaultdict
from collections.abc import Iterator, Sequence

from tai42_contract.manifest import ExtensionElement

from tai42_skeleton.core.registry.base_registry import BaseRegistry
from tai42_skeleton.exceptions.exceptions import TaiValidationError
from tai42_skeleton.extensions.registry import extension_name


class ToolRegistry(BaseRegistry):
    def __init__(self, requested_tools: set[str], tool_extensions: dict[str, list[list[ExtensionElement]]]):
        # Copy, never alias: the caller passes ``manifest.tools_list`` (the same
        # set object), and register_tool/unregister_tool mutate this — aliasing
        # would silently rewrite the manifest's tool list.
        self._requested_tools: set[str] = set(requested_tools)
        self._tools: dict[str, list[list[ExtensionElement]]] = defaultdict(list)
        self._extend_tools: dict[str, str] = {}
        self._register_tools(tool_extensions)

    def register_extend_tool(self, tool_name, extend_tool_name):
        self._extend_tools[extend_tool_name] = tool_name

    def register_tool(self, name: str, combos: Sequence[Sequence[ExtensionElement]] | None = None) -> None:
        # Bare ``[]`` present-check PLUS the combos — symmetric with the manifest
        # seeding in ``_register_tools``. Each combo is copied to a list so the
        # stored shape is always ``list[list[ExtensionElement]]`` regardless of the
        # sequence kind the caller passed.
        tracked: list[list[ExtensionElement]] = [[], *([list(combo) for combo in combos] if combos else [])]

        if name in self._requested_tools:
            # An already-requested name re-registered with the SAME combos is a
            # true idempotent no-op; one carrying DIFFERENT combos is a caller bug
            # (the reload path must ``unregister_tool_base`` first), so fail loud
            # rather than silently discard the new combos.
            if self._tools.get(name, []) != tracked:
                raise TaiValidationError(
                    f"Tool '{name}' is already registered with different extension combos; "
                    "unregister it before re-registering with a different combo set."
                )
            return

        self._requested_tools.add(name)
        self._tools[name] = tracked

    def unregister_tool(self, name: str) -> None:
        if name not in self._requested_tools:
            return

        self._requested_tools.discard(name)
        self._tools.pop(name, None)

    def unregister_tool_base(self, tool_name: str) -> list[str]:
        """Tear a base tool down: drop its combos, its extension BRANCH tools,
        and its selection entry, returning the removed branch names.

        ``_extend_tools`` always holds a base SELF-ENTRY ``tool_name ->
        tool_name`` (every bound tool records itself, including the base where
        ``curr_name == orig_name``) which is load-bearing for ``missing_tools``.
        The returned branch list EXCLUDES that self-entry (``b != tool_name``)
        so the caller removes each branch and the base separately without
        double-removing the base; the self-entry is still cleared here.
        """
        branches = [b for b, base in self._extend_tools.items() if base == tool_name and b != tool_name]
        for b in branches:
            del self._extend_tools[b]
        self._extend_tools.pop(tool_name, None)
        self._tools.pop(tool_name, None)
        self._requested_tools.discard(tool_name)
        return branches

    def base_of(self, name: str) -> str:
        """The base tool ``name`` was produced from — ``name`` itself for a base
        (every bound tool records a self-entry), the origin base for a branch
        (composed) tool. An unbound name reports itself."""
        return self._extend_tools.get(name, name)

    def is_branch(self, name: str) -> bool:
        """Whether ``name`` is an extension BRANCH tool (produced by a combo) rather
        than a base tool — ``base_of(name) != name``."""
        return self.base_of(name) != name

    def tool_extensions_iterator(self, tool_name: str) -> Iterator[list[ExtensionElement]]:
        # ``.get`` — reading an unknown name must not plant an empty defaultdict
        # entry that later shows up in ``missing_tools``.
        yield from self._tools.get(tool_name, [])

    def _register_tools(self, tool_extensions: dict[str, list[list[ExtensionElement]]]) -> None:
        # Selection request -> a bare ``[]`` present-check combo per name.
        for name in self._requested_tools:
            self._tools[name].append([])
        # Attachment -> the mapped combos, appended after selection. A base in
        # the map but NOT selected has no ``[]`` seed and never binds, so it
        # stays in ``missing_tools`` and raises at ``validation``.
        for base, combos in tool_extensions.items():
            self._tools[base].extend(combos)

    @property
    def used_extensions(self) -> frozenset[str]:
        # Extract the NAME of each combo element (a bare name or a
        # ``{"name", "config"}`` mapping) — the used-extension set keys on names,
        # and a mapping element is unhashable.
        used_extensions: set[str] = set()
        for combos in self._tools.values():
            for combo in combos:
                used_extensions.update(extension_name(element) for element in combo)
        return frozenset(used_extensions)

    @property
    def missing_tools(self) -> frozenset[str]:
        return frozenset(self._tools.keys()) - frozenset(self._extend_tools.values())

    def validation(self, ignore: frozenset[str] = frozenset()):
        # ``ignore`` holds tool names owned by MCP servers that failed their
        # viability check. They are legitimately absent (the server is down),
        # so they must not abort startup — they are tracked in
        # ``app.list_failed_mcps()`` for later reload instead.
        missing_tools = self.missing_tools - frozenset(ignore)
        if missing_tools:
            raise TaiValidationError(f"Tools '{', '.join(sorted(missing_tools))}' not found in server.")
