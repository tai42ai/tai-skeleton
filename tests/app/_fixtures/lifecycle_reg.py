"""Fixture registering a startup handler, a shutdown handler, and a pass-through
middleware ON IMPORT.

Loaded via a manifest ``lifecycle_modules`` entry so each ``start()`` re-imports
it and re-fires the decorators — used to prove module-registered handlers and
middlewares are idempotent (qualname-keyed) rather than accumulating across
repeated ``update()``.
"""

from tai42_contract.app import tai42_app


@tai42_app.lifecycle.on_startup
def startup_marker() -> None:
    pass


@tai42_app.lifecycle.on_shutdown
def shutdown_marker() -> None:
    pass


@tai42_app.http.middleware
class MarkerMiddleware:
    def __init__(self, app, **kwargs) -> None:
        self._app = app

    async def __call__(self, scope, receive, send) -> None:
        await self._app(scope, receive, send)
