"""Base tools for the preset-engine tests.

Listed as the engine tests' ``tools`` module (``weather`` / ``echo``): a preset
bakes kwargs over one of these, and its extension COMBOS attach the wrappers from
:mod:`tests.presets._ext_fixtures` as branches. The wrappers live in a separate
module so no single module is imported by two manifest sections.
"""

from __future__ import annotations

from tai42_contract.app import tai42_app


@tai42_app.tools.tool
def weather(city: str, units: str = "metric") -> dict:
    """Report the weather for a city."""
    return {"city": city, "units": units}


@tai42_app.tools.tool
def echo(text: str) -> str:
    """Echo the text back."""
    return text
