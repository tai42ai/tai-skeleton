"""The config-manager raw-write seal.

Every manifest / env mutation must cross the one pipeline
(:class:`~tai_skeleton.config.service.ConfigService`), so feature code must never call
the config manager's raw write seams directly — ``mutate_manifest``,
``replace_manifest``, ``write_manifest``, and ``write_env`` — only the config layer
(``config/``, including ``ConfigService``) and the config managers themselves may.
This test statically scans the shipped ``tai_skeleton`` feature modules and asserts
ZERO direct calls, so a converged writer cannot quietly regress to a raw seam.

The config manager's raw write seams are instance methods, reachable in source only as
an attribute call ``<manager>.mutate_manifest(...)`` and so on (a bare re-export would
still call the method as an attribute at its own body). Detecting attribute calls by
method name therefore seals every seam completely; scanning the AST (not the text)
ignores the method names where they appear only in docstrings, comments, or the
``ConfigService`` protocol declaration.
"""

from __future__ import annotations

import ast
from pathlib import Path

import tai_skeleton

# The config layer owns these seams (``ConfigService`` and the file config manager
# live here), so it is the only package allowed to drive them directly.
_ALLOWED_PACKAGE = "config"

# The four raw config-manager write seams ConfigService persists through.
_SEALED_METHODS = frozenset({"mutate_manifest", "replace_manifest", "write_manifest", "write_env"})

_PACKAGE_ROOT = Path(tai_skeleton.__file__).parent


def _feature_modules() -> list[Path]:
    """Every shipped ``tai_skeleton`` module OUTSIDE the config layer."""
    config_dir = _PACKAGE_ROOT / _ALLOWED_PACKAGE
    return [path for path in _PACKAGE_ROOT.rglob("*.py") if config_dir not in path.parents]


def _sealed_call_sites(path: Path) -> list[tuple[int, str]]:
    """The ``(lineno, method)`` of every direct call to a sealed write seam in
    ``path`` — an attribute call whose method name is one of the sealed methods."""
    tree = ast.parse(path.read_text(), filename=str(path))
    sites: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _SEALED_METHODS:
            sites.append((node.lineno, node.func.attr))
    return sites


def test_feature_modules_never_call_write_manifest_or_write_env() -> None:
    modules = _feature_modules()
    # The scan must have something to scan — a mis-resolved root would vacuously pass.
    assert modules, "no feature modules discovered under the tai_skeleton package"

    violations: list[str] = []
    for path in modules:
        for lineno, method in _sealed_call_sites(path):
            violations.append(f"{path.relative_to(_PACKAGE_ROOT)}:{lineno} calls .{method}(...)")

    assert not violations, (
        "feature code must mutate the manifest / env only through ConfigService, never the "
        "raw config-manager seam:\n  " + "\n  ".join(sorted(violations))
    )
