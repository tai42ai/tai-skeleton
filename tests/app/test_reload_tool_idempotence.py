"""``start()`` is idempotent for the live component surface.

Every module-decorator-registered component — agents, webhook verifiers,
channels, connectors, identity providers, and the tool/prompt/resource/template
surface — is reset inside ``start()`` right before ``_initialize_components``
re-imports the manifest modules and re-fires its registration decorator, so the
re-registration always lands in a clean surface. A module-level
``@tai42_app.tools.tool`` (or a prompt / resource / templated resource registered
through ``app.fastmcp`` in a manifest module) re-fires on every reload; the reset
makes that re-fire's ``add_*`` idempotent under the server-wide
``on_duplicate="error"`` rather than a duplicate crash.

The reload path (``_update``) relies on this: it does not clear the surface in
the caller. A reload whose caller-side removal did not fully clear the surface
would otherwise re-fire the decorator against a still-present component and raise
``Component already exists``.
"""

from __future__ import annotations

import asyncio

from tai42_skeleton.app.instance import app
from tai42_skeleton.manifest import Manifest

# A tool module (``@tai42_app.tools.tool greet``) plus lifecycle modules that
# register a prompt, a static resource, and a templated resource on import — each
# re-fires its decorator on every ``start()`` reimport, covering all four
# component kinds ``_reset_component_surface`` clears.
_MANIFEST = Manifest.model_validate(
    {
        "tools": [{"title": "fxt", "module": "tests.app._fixtures.tools_a"}],
        "lifecycle_modules": [
            "tests.app._fixtures.prompt_mod",
            "tests.app._fixtures.resource_mod",
        ],
    }
)


def test_start_is_idempotent_for_the_component_surface():
    """A second ``start()`` while the tool, prompt, resource, and resource template
    are already registered — the re-fire a reload performs, with no caller-side
    removal — must not raise a duplicate and must leave each present exactly once,
    mirroring the agent/webhook/channel/connector/identity reset."""

    async def run() -> None:
        async with app.app_context(_MANIFEST):
            tools = {t.name for t in await app._fast_mcp.list_tools()}
            prompts = {p.name for p in await app._fast_mcp.list_prompts()}
            resources = {str(r.uri) for r in await app._fast_mcp.list_resources()}
            templates = {t.uri_template for t in await app._fast_mcp.list_resource_templates()}
            assert "greet" in tools
            assert "fixture_prompt" in prompts
            assert "fixture://static" in resources
            assert "fixture://item/{item_id}" in templates

            # Re-fire every module-level decorator with all four still present,
            # WITHOUT any caller-side removal — start() itself clears first.
            app.start(_MANIFEST)

            after_tools = [t.name for t in await app._fast_mcp.list_tools()]
            after_prompts = [p.name for p in await app._fast_mcp.list_prompts()]
            after_resources = [str(r.uri) for r in await app._fast_mcp.list_resources()]
            after_templates = [t.uri_template for t in await app._fast_mcp.list_resource_templates()]
            assert after_tools.count("greet") == 1, after_tools
            assert after_prompts.count("fixture_prompt") == 1, after_prompts
            assert after_resources.count("fixture://static") == 1, after_resources
            assert after_templates.count("fixture://item/{item_id}") == 1, after_templates

    asyncio.run(run())
