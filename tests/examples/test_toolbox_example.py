"""Guards for the ``examples/toolbox`` starter and the toolbox homing rule.

Two things kept honest here:

* The starter manifest still BOOTS with tai-toolbox installed and its wired
  extensions/tools actually register — so the example can't rot silently. The
  manifest is loaded exactly as the server loads it (``pyaml_env.parse_config``
  → ``Manifest.model_validate``), and the toolbox modules reach the running app
  only through that manifest — never a direct import in skeleton code.
* No skeleton source module imports ``tai_toolbox``. Toolbox is a *dependency*
  (an optional extra + a manifest-loaded package), never a code import: the
  homing rule that keeps the contrib package out of the framework body.
"""

from __future__ import annotations

from pathlib import Path

from pyaml_env import parse_config

from tai_skeleton.app.instance import app
from tai_skeleton.manifest import Manifest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src" / "tai_skeleton"
_MANIFEST_PATH = _REPO_ROOT / "examples" / "toolbox" / "manifest.yml"


def _load_example_manifest() -> Manifest:
    """Load the starter manifest the way the running server does."""
    data = parse_config(data=_MANIFEST_PATH.read_text()) or {}
    return Manifest.model_validate(data)


async def test_toolbox_starter_boots_and_registers_extensions_and_tools():
    manifest = _load_example_manifest()

    async with app.app_context(manifest):
        # The two extensions_modules imported and registered.
        extensions = {e["name"]: e["kind"] for e in app.extensions.available_extensions()}
        assert extensions.get("batch") == "transformer"
        assert extensions.get("chain") == "transformer"

        tools = await app.tools.get_tools()

        # Both plain toolbox tools bound.
        assert "generate_uuid" in tools
        assert "current_time_info" in tools

        # The composition extensions branched generate_uuid into its variants.
        assert "generate_uuid_batch" in tools
        assert "generate_uuid_chain" in tools


def test_skeleton_source_never_imports_tai_toolbox():
    # Homing rule: tai-toolbox is a dependency (optional extra + manifest-loaded
    # package), never a code import. Any ``tai_toolbox`` reference in the
    # framework body would break it.
    offenders = [py.relative_to(_REPO_ROOT) for py in _SRC.rglob("*.py") if "tai_toolbox" in py.read_text()]
    assert not offenders, f"tai_toolbox must not be imported from skeleton source; found in: {offenders}"
