"""Typed seams over redis-py 7.x async client stubs.

redis-py 7.x annotates command methods with the shared sync/async signature
(``Awaitable[T] | T``), so ``await client.eval(...)`` fails type checking even
though the async client always returns an awaitable at runtime. These helpers
confine the required casts to one place; they add no runtime behavior beyond a
plain function call.
"""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Any, cast

from redis.asyncio import Redis as AsyncRedis


def awaited[T](result: Awaitable[T] | T) -> Awaitable[T]:
    """Pin a redis-py async command result to its awaitable half."""
    return cast("Awaitable[T]", result)


def eval_script(client: AsyncRedis, script: str, numkeys: int, *keys_and_args: object) -> Awaitable[Any]:
    """``client.eval(script, numkeys, *keys_and_args)`` accepting redis's real
    ``EncodableT`` arguments (``bytes``/``int``/``str``), which the async stub
    narrows to ``str``. Returns the async client's true awaitable result."""
    return cast("Awaitable[Any]", client.eval(script, numkeys, *cast("tuple[str, ...]", keys_and_args)))


__all__ = ["awaited", "eval_script"]
