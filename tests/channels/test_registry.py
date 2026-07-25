"""The channel registry + the ``app.channels`` facet."""

from __future__ import annotations

import pytest

from tai42_skeleton.channels.registry import ChannelRegistry
from tests._helpers import DeliverOnlyChannel


class _Chan(DeliverOnlyChannel):
    async def deliver(self, delivery) -> None:
        return None


def test_register_get_round_trip() -> None:
    reg = ChannelRegistry()
    c = _Chan()
    reg.register("prov", c)
    assert reg.get("prov") is c
    assert reg.names() == ["prov"]


def test_names_sorted() -> None:
    reg = ChannelRegistry()
    reg.register("zeta", _Chan())
    reg.register("alpha", _Chan())
    assert reg.names() == ["alpha", "zeta"]


def test_duplicate_name_raises() -> None:
    reg = ChannelRegistry()
    reg.register("prov", _Chan())
    with pytest.raises(ValueError, match="already registered"):
        reg.register("prov", _Chan())


def test_unknown_name_raises_loudly() -> None:
    reg = ChannelRegistry()
    with pytest.raises(KeyError, match="unknown channel"):
        reg.get("nope")


def test_reset_clears() -> None:
    reg = ChannelRegistry()
    reg.register("prov", _Chan())
    reg.reset()
    assert reg.names() == []


def test_facet_registers_and_resolves_through_app() -> None:
    from tai42_contract.app import tai42_app

    from tai42_skeleton.app.instance import build_app

    app = build_app()
    tai42_app.bind(app)
    app._channel_registry.reset()
    try:
        c = _Chan()
        tai42_app.channels.register("facet-prov", c)
        assert tai42_app.channels.get("facet-prov") is c
        assert tai42_app.channels.names() == ["facet-prov"]
    finally:
        app._channel_registry.reset()
