"""The app's client surface — a thin delegate over the tai-kit pooled-client framework.

Surfaces the pooling on the app: ``client_ctx(SomeClient, settings=…, fresh=…)``
yields a connected client, pooled per event loop + connection params (or a one-shot
off-pool client when ``fresh=True``); ``shutdown_clients()`` closes every live pool
for the running loop at teardown. The pooling itself lives in ``tai_kit.clients`` —
this only exposes it on the app, so callers reach a client by class and never import
a concrete client here.
"""

from contextlib import AbstractAsyncContextManager
from typing import Any, cast

from tai_contract.clients import BaseClient
from tai_kit.clients import PooledClient, shutdown_all_clients
from tai_kit.clients import client_ctx as _pool_client_ctx


class ClientsFacet:
    """``app.clients`` — concrete ``tai_contract.app.AppClients`` facet, forwarding
    to the tai-kit pool."""

    def client_ctx[ClientT](
        self,
        client_cls: type[BaseClient[ClientT]],
        settings: Any = None,
        *,
        fresh: bool = False,
        **kwargs: Any,
    ) -> AbstractAsyncContextManager[ClientT]:
        # The contract types the client class as the BaseClient interface; the kit
        # pool needs the PooledClient impl it always is at runtime (every app-owned
        # client subclasses PooledClient). This delegate is the seam that knows the
        # two are the same object.
        return _pool_client_ctx(cast("type[PooledClient[ClientT]]", client_cls), settings, fresh=fresh, **kwargs)

    async def shutdown_clients(self) -> None:
        await shutdown_all_clients()
