"""Every request-scoped Redis connection settings subclass carries a bounded
socket connect + read timeout, so a black-holed Redis fails the operation loudly
instead of hanging the request/loop task."""

import pytest

from tai_skeleton.access_control.settings import AccessControlRedisSettings
from tai_skeleton.connectors.settings import ConnectorStoreRedisSettings
from tai_skeleton.hooks.settings import HooksRedisSettings
from tai_skeleton.interactions.settings import (
    InteractionsRedisSettings,
    InteractionsSettings,
    interactions_settings,
)
from tai_skeleton.routers.tool_runs_settings import ToolRunsRedisSettings
from tai_skeleton.settings.rate_limit import RateLimitRedisSettings


@pytest.mark.parametrize(
    "cls",
    [
        ToolRunsRedisSettings,
        RateLimitRedisSettings,
        HooksRedisSettings,
        ConnectorStoreRedisSettings,
        InteractionsRedisSettings,
    ],
)
def test_client_kwargs_carries_both_socket_timeouts(cls):
    kwargs = cls().client_kwargs()
    assert kwargs["socket_connect_timeout"] == 5
    assert kwargs["socket_timeout"] == 5


def test_access_control_carries_connect_and_existing_read_timeout():
    # Access control already shipped ``socket_timeout=5``; this adds only the
    # connect-phase bound, so client_kwargs must now carry BOTH.
    kwargs = AccessControlRedisSettings().client_kwargs()
    assert kwargs["socket_connect_timeout"] == 5
    assert kwargs["socket_timeout"] == 5


def test_socket_timeouts_reject_non_positive(monkeypatch):
    # The global gt=0 rule: a zero/negative override is rejected loudly at load.
    monkeypatch.setenv("TAI_TOOL_RUNS_SOCKET_TIMEOUT", "0")
    with pytest.raises(ValueError, match="socket_timeout"):
        ToolRunsRedisSettings()


def test_blocking_grace_seconds_default_and_positive(monkeypatch):
    assert interactions_settings().blocking_grace_seconds == 5
    monkeypatch.setenv("INTERACTIONS_BLOCKING_GRACE_SECONDS", "-1")
    with pytest.raises(ValueError, match="blocking_grace_seconds"):
        InteractionsSettings()
