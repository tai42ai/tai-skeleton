import importlib
import importlib.util
import os
import pkgutil
import sys


def _discover_all_modules(root_pkg_name: str) -> set[str]:
    spec = importlib.util.find_spec(root_pkg_name)
    if not spec:
        raise ImportError(f"Cannot find module {root_pkg_name}")

    names = {root_pkg_name}
    search_locations = spec.submodule_search_locations
    if not search_locations:
        return names

    # Enumerate submodule names WITHOUT importing them: ``pkgutil.iter_modules``
    # only lists a directory's modules, whereas ``walk_packages`` imports each
    # subpackage to obtain its ``__path__`` and recurse. Subpackage search paths
    # are instead built from the filesystem, so discovery imports nothing and the
    # caller's pop+reimport step is the sole import of each module (an ``__init__``
    # side effect runs exactly once per start()).
    stack: list[tuple[list[str], str]] = [(list(search_locations), root_pkg_name)]
    while stack:
        paths, prefix = stack.pop()
        for module_info in pkgutil.iter_modules(paths):
            full_name = f"{prefix}.{module_info.name}"
            names.add(full_name)
            if module_info.ispkg:
                stack.append(([os.path.join(p, module_info.name) for p in paths], full_name))
    return names


def _stable_cycle_fallback(nodes: set[str]) -> list[str]:
    return sorted(nodes, key=lambda n: (n.count("."), n))


def import_or_reload_package(root_pkg_name: str | None) -> list[str]:
    if not root_pkg_name:
        return []

    importlib.invalidate_caches()

    # A manifest-named package that cannot be found is corrupt configuration:
    # the discovery failure propagates and aborts startup loudly rather than
    # booting a server silently missing its modules.
    managed = _discover_all_modules(root_pkg_name)

    # Remove all managed modules from sys.modules
    for name in managed:
        sys.modules.pop(name, None)

    # Re-import all in a stable order; imports will handle dependencies automatically
    order = _stable_cycle_fallback(managed)
    reloaded = []
    for name in order:
        try:
            importlib.import_module(name)
            reloaded.append(name)
        except ImportError as e:
            # A manifest-named module that fails to import is corrupt
            # configuration — abort startup loudly, naming the module.
            raise ImportError(f"Failed to import module {name}: {e}") from e
    return reloaded
