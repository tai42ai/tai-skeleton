"""``ClientsFacet`` delegate: it forwards the app client surface onto the tai42-kit
pool (per-key pooling, fresh off-pool clients, shutdown, settings/kwargs guard).
"""

from __future__ import annotations

import pytest
from tai42_contract.app.facets import AppClients as AppClientsProtocol
from tai42_kit.clients import PooledClient

from tai42_skeleton.app.clients import ClientsFacet


class _DummyClient(PooledClient[dict]):
    creates = 0
    closes = 0

    async def _create(self, **kwargs) -> dict:
        _DummyClient.creates += 1
        return {"id": _DummyClient.creates, "kwargs": kwargs}

    async def _close(self, client: dict) -> None:
        _DummyClient.closes += 1


def test_satisfies_contract_protocol():
    assert isinstance(ClientsFacet(), AppClientsProtocol)


async def test_pools_same_client_per_key_and_shutdown_closes():
    _DummyClient.creates = _DummyClient.closes = 0
    app = ClientsFacet()
    async with app.client_ctx(_DummyClient, host="a") as c1, app.client_ctx(_DummyClient, host="a") as c2:
        assert c1 is c2  # reused from the pool, not recreated
    assert _DummyClient.creates == 1
    assert _DummyClient.closes == 0  # pooled clients outlive the context
    await app.shutdown_clients()
    assert _DummyClient.closes == 1


async def test_fresh_is_off_pool_and_closed_on_exit():
    _DummyClient.creates = _DummyClient.closes = 0
    app = ClientsFacet()
    async with app.client_ctx(_DummyClient, fresh=True, host="b") as c:
        assert c["id"] == 1
    assert _DummyClient.closes == 1  # one-shot closed on exit
    await app.shutdown_clients()
    assert _DummyClient.closes == 1  # nothing left pooled to close


async def test_settings_and_kwargs_are_mutually_exclusive():
    class _Settings:
        def client_kwargs(self) -> dict:
            return {"host": "x"}

    app = ClientsFacet()
    with pytest.raises(ValueError, match="not both"):
        app.client_ctx(_DummyClient, settings=_Settings(), host="y")
