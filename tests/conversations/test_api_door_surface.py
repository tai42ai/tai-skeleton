"""The API send door's edge/error surface: the 400 body guards, the wait-seconds clamp,
the typed-error status mappings, and the 200/202 answer shapes.

The turn engine is stubbed so the HANDLER's own translation of a body or a typed error into
an HTTP response is what is under test — the auth stack is exercised in
``test_api_door_authz``.
"""

from __future__ import annotations

import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.routing import Route
from starlette.testclient import TestClient
from tai42_contract.conversations import ConversationAnswer

import tai42_skeleton.conversations as conversations_package
from tai42_skeleton.conversations.caps import AddressRateLimitedError, ThreadQueueOverflowError
from tai42_skeleton.conversations.turn import ApiSubmitResult, ConversationRouteResolutionError
from tai42_skeleton.operations import BadRequestError
from tai42_skeleton.operations.errors import NotSupportedError

_PATH = "/api/conversations/support/messages"


def _router():
    from tai42_contract.app import tai42_app

    from tai42_skeleton.app.instance import app as skeleton_app

    with tai42_app.bound(skeleton_app):
        from tai42_skeleton.routers import conversations as router

    return router


class _Engine:
    """A stub turn engine: it records its calls and either returns a fixed result or raises
    a fixed exception."""

    def __init__(self, *, result: ApiSubmitResult | None = None, raises: Exception | None = None) -> None:
        self._result = result
        self._raises = raises
        self.calls: list[tuple] = []

    async def __call__(self, route_name, external_user_id, text, caller_principal, wait_seconds):
        self.calls.append((route_name, external_user_id, text, caller_principal, wait_seconds))
        if self._raises is not None:
            raise self._raises
        return self._result or ApiSubmitResult(message_id="m-1", thread_id="t-1", answer=None)


def _client(monkeypatch, engine: _Engine) -> TestClient:
    router = _router()
    monkeypatch.setattr(conversations_package, "submit_api_message", engine)
    monkeypatch.setattr(router, "get_current_user_id", lambda: "caller")
    routes = [Route("/api/conversations/{route_name}/messages", router.send_conversation_message, methods=["POST"])]
    return TestClient(Starlette(routes=routes))


def _post(client: TestClient, path: str = _PATH, body=None):
    return client.post(path, json={"external_user_id": "u-7", "text": "hi"} if body is None else body)


# -- the 400 body guards -------------------------------------------------------


def test_a_reload_locked_door_rejects_before_the_turn(monkeypatch):
    engine = _Engine()
    router = _router()
    monkeypatch.setattr(conversations_package, "submit_api_message", engine)

    class _Locked:
        locked = True

        def reject_response(self):
            from starlette.responses import JSONResponse

            return JSONResponse({"error": "reloading"}, status_code=503)

    monkeypatch.setattr(router, "reload_gate", _Locked())
    routes = [Route("/api/conversations/{route_name}/messages", router.send_conversation_message, methods=["POST"])]
    response = TestClient(Starlette(routes=routes)).post(_PATH, json={"external_user_id": "u", "text": "hi"})
    assert response.status_code == 503
    assert engine.calls == []


def test_an_unparseable_body_is_a_400(monkeypatch):
    engine = _Engine()
    client = _client(monkeypatch, engine)
    response = client.post(_PATH, content=b"not json", headers={"Content-Type": "application/json"})
    assert response.status_code == 400
    assert "invalid JSON body" in response.json()["error"]
    assert engine.calls == []


def test_a_non_object_body_is_a_400(monkeypatch):
    engine = _Engine()
    client = _client(monkeypatch, engine)
    response = client.post(_PATH, json=["not", "an", "object"])
    assert response.status_code == 400
    assert "must be a JSON object" in response.json()["error"]
    assert engine.calls == []


def test_a_body_that_is_not_a_conversation_message_is_a_400(monkeypatch):
    engine = _Engine()
    client = _client(monkeypatch, engine)
    response = client.post(_PATH, json={"external_user_id": "u-7"})  # missing ``text``
    assert response.status_code == 400
    assert "invalid conversation message" in response.json()["error"]
    assert engine.calls == []


def test_a_non_integer_wait_seconds_is_a_400(monkeypatch):
    engine = _Engine()
    client = _client(monkeypatch, engine)
    response = client.post(_PATH, json={"external_user_id": "u-7", "text": "hi", "wait_seconds": "abc"})
    assert response.status_code == 400
    assert "invalid conversation message" in response.json()["error"]
    assert engine.calls == []


def test_a_negative_wait_seconds_is_a_400(monkeypatch):
    engine = _Engine()
    client = _client(monkeypatch, engine)
    response = client.post(_PATH, json={"external_user_id": "u-7", "text": "hi", "wait_seconds": -1})
    assert response.status_code == 400
    assert "invalid conversation message" in response.json()["error"]
    assert engine.calls == []


# -- the wait-seconds clamp ----------------------------------------------------


def test_an_absent_wait_seconds_is_the_pure_async_default(monkeypatch):
    engine = _Engine()
    client = _client(monkeypatch, engine)
    assert _post(client).status_code == 202
    assert engine.calls[0][4] == 0


def test_a_below_cap_wait_seconds_passes_through_unclamped(monkeypatch):
    engine = _Engine()
    client = _client(monkeypatch, engine)
    client.post(_PATH, json={"external_user_id": "u-7", "text": "hi", "wait_seconds": 5})
    assert engine.calls[0][4] == 5


def test_an_oversized_wait_seconds_clamps_to_the_settings_max(monkeypatch):
    engine = _Engine()
    client = _client(monkeypatch, engine)
    client.post(_PATH, json={"external_user_id": "u-7", "text": "hi", "wait_seconds": 99999})
    assert engine.calls[0][4] == 120  # ConversationsSettings().sync_wait_max_seconds default


# -- the typed-error status mappings -------------------------------------------


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (ConversationRouteResolutionError("no such route"), 404),
        (AddressRateLimitedError("over cap"), 429),
        (ThreadQueueOverflowError("full"), 503),
        (NotSupportedError("no backend"), 501),
    ],
)
def test_a_typed_engine_error_maps_to_its_status(monkeypatch, error, status):
    engine = _Engine(raises=error)
    client = _client(monkeypatch, engine)
    response = _post(client)
    assert response.status_code == status
    assert response.json()["error"] == str(error)


# -- the answer shapes ---------------------------------------------------------


def test_a_finished_turn_answers_200_with_the_answer_inline(monkeypatch):
    answer = ConversationAnswer(message_id="m-1", thread_id="t-1", status="answered", answer="hello back")
    engine = _Engine(result=ApiSubmitResult(message_id="m-1", thread_id="t-1", answer=answer))
    client = _client(monkeypatch, engine)
    response = _post(client)
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["message_id"] == "m-1"
    assert data["answer"]["answer"] == "hello back"


def test_an_unfinished_turn_answers_202_without_an_answer(monkeypatch):
    engine = _Engine(result=ApiSubmitResult(message_id="m-1", thread_id="t-1", answer=None))
    client = _client(monkeypatch, engine)
    response = _post(client)
    assert response.status_code == 202
    assert "answer" not in response.json()["data"]


# -- the route-create body extractor ------------------------------------------


def _create_request(body: bytes, route_name: str = "support") -> Request:
    async def _receive():
        return {"type": "http.request", "body": body, "more_body": False}

    scope = {
        "type": "http",
        "method": "POST",
        "headers": [],
        "path_params": {"route_name": route_name},
        "query_string": b"",
    }
    return Request(scope, _receive)


async def test_extract_route_create_injects_the_path_route_name():
    router = _router()
    fields = await router._extract_route_create(
        _create_request(
            b'{"door": "api", "agent_name": "triage", "execution_key": "svc", "callback_url": "https://cb.example/x"}'
        )
    )
    assert fields["route_name"] == "support"
    assert fields["door"] == "api"


async def test_extract_route_create_rejects_invalid_json():
    router = _router()
    with pytest.raises(BadRequestError, match="invalid JSON body"):
        await router._extract_route_create(_create_request(b"not json"))


async def test_extract_route_create_rejects_a_non_object_body():
    router = _router()
    with pytest.raises(BadRequestError, match="JSON object"):
        await router._extract_route_create(_create_request(b"[1, 2, 3]"))


async def test_extract_route_create_rejects_a_body_route_name_disagreeing_with_the_path():
    router = _router()
    with pytest.raises(BadRequestError, match="must match"):
        await router._extract_route_create(
            _create_request(b'{"route_name": "other", "door": "api", "agent_name": "triage", "execution_key": "svc"}')
        )


async def test_extract_route_create_rejects_an_invalid_route_body():
    router = _router()
    with pytest.raises(BadRequestError, match="invalid conversation route"):
        await router._extract_route_create(_create_request(b'{"door": "channel"}'))
