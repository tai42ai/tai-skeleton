"""A config-agnostic extension fixture: it keeps the three-argument
``(func, name, description)`` factory signature and takes no author config, so the
apply site calls it without a config argument and rejects any config bound to it."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tai42_contract.app import tai42_app
from tai42_contract.extensions import ExtensionKind


@tai42_app.extensions.extension(kind=ExtensionKind.TRANSFORMER, name="noconfig")
def noconfig(func: Callable[..., Any], name: str, description: str) -> Callable[..., Any]:
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        return await func(*args, **kwargs)

    wrapper.__name__ = f"{name}_noconfig"
    return wrapper
