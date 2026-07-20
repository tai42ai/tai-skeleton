"""Backend registration — the impl body behind the ``app.backends`` facet."""

from tai_contract.backend import Backend

from tai_skeleton.settings.cache import backend_provider


class BackendHolder:
    """Holds the process's single registered :class:`Backend` instance."""

    def __init__(self) -> None:
        self._backend: Backend | None = None

    @property
    def backend(self) -> Backend | None:
        return self._backend

    def register_backend(self, cls: type | None = None):
        if cls:
            return self.register_backend()(cls)

        def decorator(klass):
            self._backend = klass()
            return klass

        return decorator

    async def launch(self, args) -> None:
        if self._backend is None:
            raise RuntimeError(f"Backend provider {backend_provider()} is not configured")
        await self._backend.launch(args)
