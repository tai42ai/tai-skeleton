"""The builtin-loading mechanism: a manifest whose ``tools[].module`` and
``extensions_modules`` entries name the builtin modules must register every listed
tool and the ``monitor`` tool extension through the manifest-driven loader alone —
no manual import. Importing a module by hand would prove nothing; this drives the
real ``app.app_context`` boot the runtime uses.
"""

from __future__ import annotations

import asyncio

from tai42_skeleton.app.instance import app
from tai42_skeleton.manifest import Manifest

BUILTIN_TOOLS: dict[str, list[str]] = {
    "tai42_skeleton.tools.builtin.file_loader": ["file_loader"],
    "tai42_skeleton.tools.builtin.interactions": ["ask_user"],
}

BUILTIN_EXTENSION_MODULE = "tai42_skeleton.extensions.builtin.monitor"


def test_manifest_registers_every_builtin_tool_and_extension() -> None:
    manifest = Manifest.model_validate(
        {
            "tools": [
                {"title": module.rsplit(".", 1)[-1], "module": module, "include": names}
                for module, names in BUILTIN_TOOLS.items()
            ],
            "extensions_modules": [BUILTIN_EXTENSION_MODULE],
        }
    )
    expected_tools = {name for names in BUILTIN_TOOLS.values() for name in names}

    async def run() -> None:
        async with app.app_context(manifest):
            bound = await app.tools.get_tools()
            missing = expected_tools - set(bound)
            assert not missing, f"builtin tools not registered by the manifest loader: {sorted(missing)}"

            registered = {ext["name"] for ext in app.extensions.available_extensions()}
            assert "monitor" in registered, (
                f"monitor tool extension not registered by the manifest loader: {sorted(registered)}"
            )

    asyncio.run(run())
