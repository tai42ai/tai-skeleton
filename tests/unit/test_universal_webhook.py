"""Webhook fire path through the hooks package: POST
/universal_webhook/{topic} parses the payload, looks up registered hooks via
``tai42_skeleton.hooks`` and runs each hook's tool through ``tai42_app``.
"""

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock

import pytest
from starlette.requests import Request
from tai42_contract.hooks.models import HookParams

# The unit-suite conftest bound this singleton as the global ``tai42_app`` handle
# (the routers module below registers its route through the handle at import);
# entering ``app.app_context`` below re-binds the same singleton, which the
# routers/hooks chain resolves against at request time.
from tai42_skeleton.app.instance import app
from tai42_skeleton.hooks.cache import get_hooks_manager
from tai42_skeleton.manifest import Manifest
from tai42_skeleton.routers.hooks import universal_webhook

_MANIFEST = {}


@pytest.fixture(autouse=True)
def _in_memory_hooks(monkeypatch: pytest.MonkeyPatch):
    for var in ("HOOKS_REDIS_URL", "HOOKS_REDIS_MAX_CONNECTIONS", "HOOKS_MAX_WORKERS", "HOOKS_PREFIX"):
        monkeypatch.delenv(var, raising=False)
    get_hooks_manager.cache_clear()
    yield
    get_hooks_manager.cache_clear()


def _request(topic: str, body: dict) -> Request:
    payload = json.dumps(body).encode()
    sent = {"called": False}

    async def receive() -> dict[str, Any]:
        if sent["called"]:
            return {"type": "http.disconnect"}
        sent["called"] = True
        return {"type": "http.request", "body": payload, "more_body": False}

    scope = {
        "type": "http",
        "method": "POST",
        "path": f"/universal_webhook/{topic}",
        "raw_path": f"/universal_webhook/{topic}".encode(),
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
        "path_params": {"topic": topic},
        "app": None,
    }
    return Request(scope, receive=receive)


def test_webhook_fires_registered_hook_tool(monkeypatch: pytest.MonkeyPatch):
    async def run():
        async with app.app_context(Manifest.model_validate(_MANIFEST)):
            run_tool = AsyncMock()
            monkeypatch.setattr(app._tool_binding, "run_tool", run_tool)

            manager = get_hooks_manager()
            await manager.register(
                HookParams(
                    name="on-order",
                    topic="orders",
                    tool="run_order_flow",
                    expr=".payload",
                    tool_kwargs={"source": "webhook"},
                )
            )

            resp = await universal_webhook(_request("orders", {"payload": {"id": 7}}))
            assert json.loads(bytes(resp.body)) == {"status": "accepted", "topic": "orders"}

            assert resp.background is not None
            await resp.background()
            run_tool.assert_awaited_once_with("run_order_flow", {"id": 7, "source": "webhook"}, offload_sync=False)

    asyncio.run(run())


def test_webhook_unknown_topic_accepts_and_runs_nothing(monkeypatch: pytest.MonkeyPatch):
    async def run():
        async with app.app_context(Manifest.model_validate(_MANIFEST)):
            run_tool = AsyncMock()
            monkeypatch.setattr(app._tool_binding, "run_tool", run_tool)

            resp = await universal_webhook(_request("nothing-registered", {"x": 1}))
            assert json.loads(bytes(resp.body)) == {"status": "accepted", "topic": "nothing-registered"}

            assert resp.background is not None
            await resp.background()
            run_tool.assert_not_awaited()

    asyncio.run(run())


def test_malformed_body_is_rejected_with_400() -> None:
    raw = b"{not json"
    sent = {"called": False}

    async def receive() -> dict[str, Any]:
        if sent["called"]:
            return {"type": "http.disconnect"}
        sent["called"] = True
        return {"type": "http.request", "body": raw, "more_body": False}

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/universal_webhook/orders",
        "raw_path": b"/universal_webhook/orders",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
        "path_params": {"topic": "orders"},
        "app": None,
    }

    async def run():
        return await universal_webhook(Request(scope, receive))

    response = asyncio.run(run())
    assert response.status_code == 400
    body = json.loads(bytes(response.body))
    assert body["status"] == "rejected"
    assert "malformed JSON body" in body["error"]
