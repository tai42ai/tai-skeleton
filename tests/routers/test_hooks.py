"""The /universal_webhook route: a parsed payload is accepted and handed to the
hooks manager on a background task; a payload that fails to parse is rejected
with a 400. Parsing and the manager are faked at their seams."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import cast

import pytest
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from tai42_contract.hooks import HookParams

from tai42_skeleton.operations import hooks as hooks_ops
from tai42_skeleton.routers import hooks


class _FakeRequest:
    """A structural stand-in for the ingress route: it reads ``path_params``,
    ``url.query``, ``method``, ``headers``, streams a bounded body, and tolerates
    the route caching ``_body`` on it."""

    def __init__(
        self, topic: str, *, query: str = "", method: str = "POST", body: bytes = b"", headers: dict | None = None
    ) -> None:
        self.path_params = {"topic": topic}
        self.url = SimpleNamespace(query=query)
        self.method = method
        self.headers = headers or {}
        self._chunks = [body] if body else []

    async def stream(self):
        for chunk in self._chunks:
            yield chunk
        yield b""


class _FakeManager:
    def __init__(self, verifier_binding: dict | None = None) -> None:
        self.events: list[tuple[str, dict]] = []
        self._binding = verifier_binding

    async def on_event(self, topic: str, payload: dict) -> None:
        self.events.append((topic, payload))

    async def get_topic_verifier(self, topic: str) -> dict | None:
        return self._binding


def _body(response: Response) -> dict:
    return json.loads(bytes(response.body))


async def test_accepts_parsed_payload_and_schedules_event(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _FakeManager()

    async def fake_parse(request, include_query=True):
        return {"hello": "world"}

    monkeypatch.setattr(hooks, "parse_any_payload", fake_parse)
    monkeypatch.setattr(hooks, "get_hooks_manager", lambda: manager)

    # starlette's Request is a concrete class the handler only reads
    # ``path_params`` off; it can't be matched structurally, so cast the stand-in.
    response = await hooks.universal_webhook(cast(Request, _FakeRequest("orders")))

    assert isinstance(response, JSONResponse)
    assert response.status_code == 200
    assert _body(response) == {"status": "accepted", "topic": "orders"}

    # The manager is invoked via the response's background task.
    assert response.background is not None
    await response.background()
    assert manager.events == [("orders", {"hello": "world"})]


async def test_rejects_unparseable_payload_with_400(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_parse(request, include_query=True):
        raise ValueError("bad body")

    manager = _FakeManager()
    monkeypatch.setattr(hooks, "parse_any_payload", fake_parse)
    monkeypatch.setattr(hooks, "get_hooks_manager", lambda: manager)

    # See the note above: cast the structural stand-in to the concrete Request.
    response = await hooks.universal_webhook(cast(Request, _FakeRequest("orders", body=b"{bad")))

    assert isinstance(response, JSONResponse)
    assert response.status_code == 400
    body = _body(response)
    assert body["status"] == "rejected"
    assert body["topic"] == "orders"
    assert "bad body" in body["error"]
    # A parse rejection dispatches nothing — no background task, no event fired.
    assert response.background is None
    assert manager.events == []


# -- Hook management routes (AUTHED) -----------------------------------------


class _MgmtManager:
    """A hooks manager stubbed at the methods the management routes drive."""

    def __init__(self, existing: dict[str, HookParams] | None = None, verifiers: dict[str, dict] | None = None) -> None:
        self._hooks: dict[str, HookParams] = existing or {}
        self._verifiers: dict[str, dict] = verifiers or {}
        self.registered: list[HookParams] = []
        self.unregistered: list[str] = []

    async def list_hooks(self) -> dict[str, HookParams]:
        return self._hooks

    async def list_hooks_by_topic(self, topic: str) -> dict[str, HookParams]:
        return {name: p for name, p in self._hooks.items() if p.topic == topic}

    async def register(self, params: HookParams) -> bool:
        self.registered.append(params)
        return True

    async def unregister(self, name: str) -> bool:
        self.unregistered.append(name)
        return name in self._hooks

    async def all_topic_verifiers(self) -> dict[str, dict]:
        return dict(self._verifiers)

    async def set_topic_verifier(self, topic: str, binding: dict) -> None:
        self._verifiers[topic] = binding

    async def delete_topic_verifier(self, topic: str) -> bool:
        return self._verifiers.pop(topic, None) is not None


class _GetReq:
    def __init__(self, topic: str | None = None) -> None:
        self.query_params: dict[str, str] = {} if topic is None else {"topic": topic}


class _JsonReq:
    def __init__(self, body: object) -> None:
        self._body = body

    async def json(self) -> object:
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class _DelReq:
    def __init__(self, name: str) -> None:
        self.path_params = {"name": name}


def _hooks_fixture() -> dict[str, HookParams]:
    return {
        "a": HookParams(name="a", topic="orders", tool="notify"),
        "b": HookParams(name="b", topic="alerts", tool="page"),
    }


async def test_list_hooks_returns_all(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _MgmtManager(_hooks_fixture(), verifiers={"orders": {"verifier": "shared_secret", "config": {}}})
    monkeypatch.setattr(hooks_ops, "get_hooks_manager", lambda: manager)
    response = await hooks.list_hooks(cast(Request, _GetReq()))
    body = _body(response)
    assert body["data"]["total"] == 2
    assert {item["name"] for item in body["data"]["items"]} == {"a", "b"}
    # The GET response carries the per-topic verifier bindings.
    assert body["data"]["topic_verifiers"] == {"orders": {"verifier": "shared_secret", "config": {}}}


async def test_list_hooks_filters_by_topic(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _MgmtManager(_hooks_fixture())
    monkeypatch.setattr(hooks_ops, "get_hooks_manager", lambda: manager)
    response = await hooks.list_hooks(cast(Request, _GetReq(topic="alerts")))
    body = _body(response)
    assert body["data"]["total"] == 1
    assert body["data"]["items"][0]["name"] == "b"


async def test_register_hook_validates_and_registers(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _MgmtManager()
    monkeypatch.setattr(hooks_ops, "get_hooks_manager", lambda: manager)
    payload = {"name": "c", "topic": "orders", "tool": "notify"}
    response = await hooks.register_hook(cast(Request, _JsonReq(payload)))
    assert response.status_code == 200
    assert _body(response)["data"] == {"registered": True, "name": "c"}
    assert len(manager.registered) == 1
    assert manager.registered[0].name == "c"


async def test_register_hook_rejects_non_object_body(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _MgmtManager()
    monkeypatch.setattr(hooks_ops, "get_hooks_manager", lambda: manager)
    response = await hooks.register_hook(cast(Request, _JsonReq([1, 2])))
    assert response.status_code == 400
    assert manager.registered == []


async def test_register_hook_rejects_invalid_params(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _MgmtManager()
    monkeypatch.setattr(hooks_ops, "get_hooks_manager", lambda: manager)
    # Missing the required ``tool`` field.
    response = await hooks.register_hook(cast(Request, _JsonReq({"name": "c", "topic": "orders"})))
    assert response.status_code == 400
    assert "invalid hook params" in _body(response)["error"]
    assert manager.registered == []


async def test_register_hook_rejects_bad_json(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _MgmtManager()
    monkeypatch.setattr(hooks_ops, "get_hooks_manager", lambda: manager)
    response = await hooks.register_hook(cast(Request, _JsonReq(ValueError("bad"))))
    assert response.status_code == 400
    assert manager.registered == []


async def test_register_hook_maps_manager_jq_error_to_400(monkeypatch: pytest.MonkeyPatch) -> None:
    class _RaisingManager(_MgmtManager):
        async def register(self, params: HookParams) -> bool:
            raise ValueError(f"hook {params.name!r}: condition is not valid jq")

    manager = _RaisingManager()
    monkeypatch.setattr(hooks_ops, "get_hooks_manager", lambda: manager)
    payload = {"name": "c", "topic": "orders", "tool": "notify", "condition": "{{bad"}
    response = await hooks.register_hook(cast(Request, _JsonReq(payload)))
    assert response.status_code == 400
    assert "not valid jq" in _body(response)["error"]


async def test_unregister_hook_removes(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _MgmtManager(_hooks_fixture())
    monkeypatch.setattr(hooks_ops, "get_hooks_manager", lambda: manager)
    response = await hooks.unregister_hook(cast(Request, _DelReq("a")))
    assert response.status_code == 200
    assert _body(response)["data"] == {"removed": True, "name": "a"}
    assert manager.unregistered == ["a"]


async def test_unregister_hook_missing_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _MgmtManager(_hooks_fixture())
    monkeypatch.setattr(hooks_ops, "get_hooks_manager", lambda: manager)
    response = await hooks.unregister_hook(cast(Request, _DelReq("nope")))
    assert response.status_code == 404
    assert "not found" in _body(response)["error"]


# -- Per-topic verifier bindings + ingress hardening -------------------------


class _PutReq:
    def __init__(self, topic: str, body: object) -> None:
        self.path_params = {"topic": topic}
        self._body = body

    async def json(self) -> object:
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class _TopicDelReq:
    def __init__(self, topic: str) -> None:
        self.path_params = {"topic": topic}


class _FakeVerifier:
    """A controllable verifier: records each ``verify`` call, optionally raises,
    and declares its ``post_only`` delivery constraint."""

    def __init__(self, *, post_only: bool = False, raise_exc: Exception | None = None, order: list | None = None):
        self.post_only = post_only
        self._raise = raise_exc
        self._order = order
        self.calls: list = []

    async def verify(self, body, headers, config):
        if self._order is not None:
            self._order.append("verify")
        self.calls.append((body, dict(headers), config))
        if self._raise is not None:
            raise self._raise


@pytest.fixture
def registry():
    """The process app's webhook-verifier registry, with ``tai42_app`` bound to that
    same app so the route resolves verifiers from it; cleared after the test."""
    from tai42_contract.app import tai42_app

    from tai42_skeleton.app.instance import build_app

    app = build_app()
    tai42_app.bind(app)
    reg = app._webhook_verifier_registry
    try:
        yield reg
    finally:
        reg.reset()


async def test_put_binding_sets_and_get_reflects(monkeypatch: pytest.MonkeyPatch, registry) -> None:
    registry.register("prov", _FakeVerifier())
    manager = _MgmtManager()
    monkeypatch.setattr(hooks_ops, "get_hooks_manager", lambda: manager)

    resp = await hooks.set_topic_verifier(cast(Request, _PutReq("orders", {"verifier": "prov", "config": {"k": "v"}})))
    assert resp.status_code == 200
    assert _body(resp)["data"] == {"topic": "orders", "verifier": "prov"}
    # GET now reflects it under topic_verifiers.
    listed = await hooks.list_hooks(cast(Request, _GetReq()))
    assert _body(listed)["data"]["topic_verifiers"] == {"orders": {"verifier": "prov", "config": {"k": "v"}}}


async def test_put_binding_replaces(monkeypatch: pytest.MonkeyPatch, registry) -> None:
    registry.register("prov", _FakeVerifier())
    manager = _MgmtManager(verifiers={"orders": {"verifier": "prov", "config": {"old": 1}}})
    monkeypatch.setattr(hooks_ops, "get_hooks_manager", lambda: manager)
    await hooks.set_topic_verifier(cast(Request, _PutReq("orders", {"verifier": "prov", "config": {"new": 2}})))
    assert manager._verifiers["orders"] == {"verifier": "prov", "config": {"new": 2}}


async def test_put_binding_unknown_verifier_rejected_at_bind_time(monkeypatch: pytest.MonkeyPatch, registry) -> None:
    manager = _MgmtManager()
    monkeypatch.setattr(hooks_ops, "get_hooks_manager", lambda: manager)
    resp = await hooks.set_topic_verifier(cast(Request, _PutReq("orders", {"verifier": "nope", "config": {}})))
    assert resp.status_code == 400
    assert "unknown webhook verifier" in _body(resp)["error"]
    assert manager._verifiers == {}


async def test_put_binding_malformed_body_400(monkeypatch: pytest.MonkeyPatch, registry) -> None:
    manager = _MgmtManager()
    monkeypatch.setattr(hooks_ops, "get_hooks_manager", lambda: manager)
    for bad in ([1, 2], {"config": {}}, {"verifier": "prov", "config": "x"}, ValueError("bad json")):
        resp = await hooks.set_topic_verifier(cast(Request, _PutReq("orders", bad)))
        assert resp.status_code == 400
    assert manager._verifiers == {}


async def test_put_binding_empty_verifier_400_not_500(monkeypatch: pytest.MonkeyPatch, registry) -> None:
    # An empty ``verifier`` name is a clean 400 (the router's non-empty check),
    # never a 500 — nothing is stored. The contract model's ``min_length=1`` is
    # the backing guarantee; the router surfaces it as a client error here.
    manager = _MgmtManager()
    monkeypatch.setattr(hooks_ops, "get_hooks_manager", lambda: manager)
    resp = await hooks.set_topic_verifier(cast(Request, _PutReq("orders", {"verifier": "", "config": {}})))
    assert resp.status_code == 400
    assert "non-empty 'verifier'" in _body(resp)["error"]
    assert manager._verifiers == {}


async def test_delete_binding_removes_and_missing_404(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _MgmtManager(verifiers={"orders": {"verifier": "prov", "config": {}}})
    monkeypatch.setattr(hooks_ops, "get_hooks_manager", lambda: manager)
    resp = await hooks.delete_topic_verifier(cast(Request, _TopicDelReq("orders")))
    assert resp.status_code == 200
    assert _body(resp)["data"]["removed"] is True
    missing = await hooks.delete_topic_verifier(cast(Request, _TopicDelReq("orders")))
    assert missing.status_code == 404


async def test_list_verifiers_returns_sorted_names(registry) -> None:
    # Two verifiers registered on the real registry — the catalog is their names,
    # sorted, and nothing else (no verifier objects, no config).
    registry.register("zebra", _FakeVerifier())
    registry.register("alpha", _FakeVerifier())
    resp = await hooks.list_verifiers(cast(Request, _GetReq()))
    assert resp.status_code == 200
    assert _body(resp)["data"] == ["alpha", "zebra"]


async def test_list_verifiers_empty_registry_returns_empty_list(registry) -> None:
    # No verifier lifecycle module loaded is a valid state — an empty list, not an error.
    resp = await hooks.list_verifiers(cast(Request, _GetReq()))
    assert resp.status_code == 200
    assert _body(resp)["data"] == []


async def test_unbound_topic_ingress_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _FakeManager(verifier_binding=None)

    async def fake_parse(request, include_query=True):
        return {"hello": "world"}

    monkeypatch.setattr(hooks, "parse_any_payload", fake_parse)
    monkeypatch.setattr(hooks, "get_hooks_manager", lambda: manager)
    resp = await hooks.universal_webhook(cast(Request, _FakeRequest("orders", body=b'{"hello":"world"}')))
    assert resp.status_code == 200
    assert _body(resp) == {"status": "accepted", "topic": "orders"}
    # nosniff + no-store on the ingress response.
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.headers["Cache-Control"] == "no-store"


async def test_bound_topic_verifies_before_parse(monkeypatch: pytest.MonkeyPatch, registry) -> None:
    order: list[str] = []
    registry.register("prov", _FakeVerifier(order=order))
    manager = _FakeManager(verifier_binding={"verifier": "prov", "config": {"c": 1}})

    async def fake_parse(request, include_query=True):
        order.append("parse")
        return {"hello": "world"}

    monkeypatch.setattr(hooks, "parse_any_payload", fake_parse)
    monkeypatch.setattr(hooks, "get_hooks_manager", lambda: manager)
    resp = await hooks.universal_webhook(cast(Request, _FakeRequest("orders", body=b'{"hello":"world"}')))
    assert resp.status_code == 200
    # Verify ran, and it ran BEFORE the parse.
    assert order == ["verify", "parse"]


async def test_bound_topic_verify_failure_401_nothing_dispatched(monkeypatch: pytest.MonkeyPatch, registry) -> None:
    from tai42_contract.webhooks import WebhookVerificationError

    registry.register("prov", _FakeVerifier(raise_exc=WebhookVerificationError("bad sig")))
    manager = _FakeManager(verifier_binding={"verifier": "prov", "config": {}})
    parsed = {"called": False}

    async def fake_parse(request, include_query=True):
        parsed["called"] = True
        return {}

    monkeypatch.setattr(hooks, "parse_any_payload", fake_parse)
    monkeypatch.setattr(hooks, "get_hooks_manager", lambda: manager)
    resp = await hooks.universal_webhook(cast(Request, _FakeRequest("orders", body=b"x")))
    assert resp.status_code == 401
    assert _body(resp)["error"] == "webhook verification failed"
    # Nothing parsed, nothing dispatched (no background task).
    assert parsed["called"] is False
    assert resp.background is None


async def test_bound_post_only_verifier_strips_query_from_payload(monkeypatch: pytest.MonkeyPatch, registry) -> None:
    # A body-signature (post_only) verifier authenticates the raw body only, so the
    # dispatched payload must exclude the unauthenticated query string (else a
    # captured signed delivery replays with attacker-appended ?key=val params).
    registry.register("body-sig", _FakeVerifier(post_only=True))
    manager = _FakeManager(verifier_binding={"verifier": "body-sig", "config": {}})
    seen: dict = {}

    async def fake_parse(request, include_query=True):
        seen["include_query"] = include_query
        return {"hello": "world"}

    monkeypatch.setattr(hooks, "parse_any_payload", fake_parse)
    monkeypatch.setattr(hooks, "get_hooks_manager", lambda: manager)
    resp = await hooks.universal_webhook(cast(Request, _FakeRequest("orders", body=b'{"hello":"world"}')))
    assert resp.status_code == 200
    assert seen["include_query"] is False


async def test_bound_header_verifier_keeps_query_in_payload(monkeypatch: pytest.MonkeyPatch, registry) -> None:
    # A header verifier (not post_only) does not sign the body; a legitimate GET/POST
    # may carry payload in the query string, so the query is NOT stripped.
    registry.register("hdr", _FakeVerifier(post_only=False))
    manager = _FakeManager(verifier_binding={"verifier": "hdr", "config": {}})
    seen: dict = {}

    async def fake_parse(request, include_query=True):
        seen["include_query"] = include_query
        return {}

    monkeypatch.setattr(hooks, "parse_any_payload", fake_parse)
    monkeypatch.setattr(hooks, "get_hooks_manager", lambda: manager)
    resp = await hooks.universal_webhook(cast(Request, _FakeRequest("orders", body=b"x")))
    assert resp.status_code == 200
    assert seen["include_query"] is True


async def test_bound_post_only_verifier_rejects_get(monkeypatch: pytest.MonkeyPatch, registry) -> None:
    registry.register("body-sig", _FakeVerifier(post_only=True))
    manager = _FakeManager(verifier_binding={"verifier": "body-sig", "config": {}})
    monkeypatch.setattr(hooks, "get_hooks_manager", lambda: manager)
    resp = await hooks.universal_webhook(cast(Request, _FakeRequest("orders", method="GET")))
    assert resp.status_code == 405


async def test_bound_missing_secret_fails_closed_500(monkeypatch: pytest.MonkeyPatch, registry) -> None:
    registry.register("prov", _FakeVerifier(raise_exc=KeyError("SECRET_ENV")))
    manager = _FakeManager(verifier_binding={"verifier": "prov", "config": {"secret_env": "SECRET_ENV"}})
    parsed = {"called": False}

    async def fake_parse(request, include_query=True):
        parsed["called"] = True
        return {}

    monkeypatch.setattr(hooks, "parse_any_payload", fake_parse)
    monkeypatch.setattr(hooks, "get_hooks_manager", lambda: manager)
    resp = await hooks.universal_webhook(cast(Request, _FakeRequest("orders", body=b"x")))
    assert resp.status_code == 500
    assert parsed["called"] is False


async def test_bound_unresolvable_verifier_fails_closed_500(monkeypatch: pytest.MonkeyPatch, registry) -> None:
    # A topic bound to a verifier NAME that does not resolve (its module absent
    # from the manifest) must deny, not dispatch unverified — a loud 500, and nothing
    # parsed or dispatched.
    manager = _FakeManager(verifier_binding={"verifier": "ghost", "config": {}})
    parsed = {"called": False}

    async def fake_parse(request, include_query=True):
        parsed["called"] = True
        return {}

    monkeypatch.setattr(hooks, "parse_any_payload", fake_parse)
    monkeypatch.setattr(hooks, "get_hooks_manager", lambda: manager)
    resp = await hooks.universal_webhook(cast(Request, _FakeRequest("orders", body=b"x")))
    assert resp.status_code == 500
    assert parsed["called"] is False
    assert resp.background is None


async def test_over_cap_body_413(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _FakeManager()
    monkeypatch.setattr(hooks, "get_hooks_manager", lambda: manager)
    monkeypatch.setattr(hooks, "webhook_ingress_settings", lambda: SimpleNamespace(max_body_bytes=8))
    resp = await hooks.universal_webhook(cast(Request, _FakeRequest("orders", body=b"0123456789")))
    assert resp.status_code == 413


async def test_over_cap_query_413(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _FakeManager()
    monkeypatch.setattr(hooks, "get_hooks_manager", lambda: manager)
    monkeypatch.setattr(hooks, "webhook_ingress_settings", lambda: SimpleNamespace(max_body_bytes=8))
    resp = await hooks.universal_webhook(cast(Request, _FakeRequest("orders", query="a=0123456789")))
    assert resp.status_code == 413


# -- Trigger-link resolver (PUBLIC) ------------------------------------------


class _ResolverManager:
    """A hooks manager spied at ``on_event``, recording the override too."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict, dict | None]] = []

    async def on_event(self, topic: str, payload: dict, *, tool_kwargs_override: dict | None = None) -> None:
        self.events.append((topic, payload, tool_kwargs_override))


def _trig_req(token: str = "tok", **kw) -> _FakeRequest:
    req = _FakeRequest(token, **kw)
    req.path_params = {"token": token}
    return req


def _wire_resolver(monkeypatch, manager, *, topic="orders", tool_kwargs=None, resolve_exc=None):
    from tai42_skeleton.hooks.trigger_links import TriggerLinkError

    async def _resolve(token):
        if resolve_exc is not None:
            raise resolve_exc
        return topic, tool_kwargs

    monkeypatch.setattr(hooks, "resolve_trigger_token", _resolve)
    monkeypatch.setattr(hooks, "get_hooks_manager", lambda: manager)
    return TriggerLinkError


async def test_trigger_get_dispatches_payload_and_override(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _ResolverManager()
    _wire_resolver(monkeypatch, manager, topic="orders", tool_kwargs={"a": 1})

    async def fake_parse(request, include_query=True):
        return {"x": "1"}

    monkeypatch.setattr(hooks, "parse_any_payload", fake_parse)
    resp = await hooks.trigger_link(cast(Request, _trig_req("tok", method="GET", query="x=1")))
    assert resp.status_code == 200
    body = _body(resp)
    assert body == {"status": "accepted"}  # NO topic echoed
    assert resp.headers["Cache-Control"] == "no-store"
    assert resp.background is not None
    await resp.background()
    assert manager.events == [("orders", {"x": "1"}, {"a": 1})]


async def test_trigger_none_override_for_paramless_link(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _ResolverManager()
    _wire_resolver(monkeypatch, manager, topic="t", tool_kwargs=None)

    async def fake_parse(request, include_query=True):
        return {}

    monkeypatch.setattr(hooks, "parse_any_payload", fake_parse)
    resp = await hooks.trigger_link(cast(Request, _trig_req("tok", method="GET")))
    assert resp.background is not None
    await resp.background()
    assert manager.events == [("t", {}, None)]


async def test_trigger_post_json_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _ResolverManager()
    _wire_resolver(monkeypatch, manager, topic="t")

    async def fake_parse(request, include_query=True):
        return {"from": "body"}

    monkeypatch.setattr(hooks, "parse_any_payload", fake_parse)
    resp = await hooks.trigger_link(cast(Request, _trig_req("tok", body=b'{"from":"body"}')))
    assert resp.status_code == 200
    assert resp.background is not None
    await resp.background()
    assert manager.events[0][1] == {"from": "body"}


async def test_trigger_miss_uniform_404_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    from tai42_skeleton.hooks.trigger_links import TriggerLinkError

    manager = _ResolverManager()
    _wire_resolver(monkeypatch, manager, resolve_exc=TriggerLinkError(404, "unknown or expired trigger link"))

    async def fake_parse(request, include_query=True):
        return {}

    monkeypatch.setattr(hooks, "parse_any_payload", fake_parse)
    resp = await hooks.trigger_link(cast(Request, _trig_req("tok", method="GET")))
    assert resp.status_code == 404
    assert _body(resp)["error"] == "unknown or expired trigger link"
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.headers["Cache-Control"] == "no-store"
    assert manager.events == []  # nothing dispatched


async def test_trigger_parse_error_400_without_topic(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _ResolverManager()
    _wire_resolver(monkeypatch, manager, topic="secret-topic")

    async def fake_parse(request, include_query=True):
        raise ValueError("bad body")

    monkeypatch.setattr(hooks, "parse_any_payload", fake_parse)
    resp = await hooks.trigger_link(cast(Request, _trig_req("tok", body=b"{bad")))
    assert resp.status_code == 400
    body = _body(resp)
    assert body["status"] == "rejected"
    assert "topic" not in body  # the link hides its topic even on rejection
    assert "secret-topic" not in json.dumps(body)
    assert resp.headers["Cache-Control"] == "no-store"
    assert manager.events == []


async def test_trigger_oversize_query_and_body_413(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _ResolverManager()
    _wire_resolver(monkeypatch, manager)
    monkeypatch.setattr(hooks, "webhook_ingress_settings", lambda: SimpleNamespace(max_body_bytes=8))
    resp_q = await hooks.trigger_link(cast(Request, _trig_req("tok", query="a=0123456789")))
    assert resp_q.status_code == 413
    assert resp_q.headers["Cache-Control"] == "no-store"
    resp_b = await hooks.trigger_link(cast(Request, _trig_req("tok", body=b"0123456789")))
    assert resp_b.status_code == 413


async def test_trigger_disallowed_method_405_no_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    # The door accepts exactly GET|POST; a PUT is a 405 at the route and never
    # reaches the handler, so nothing dispatches.
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    manager = _ResolverManager()
    _wire_resolver(monkeypatch, manager)
    monkeypatch.setattr(hooks, "parse_any_payload", lambda request, include_query=True: {})
    app = Starlette(routes=[Route("/trigger/{token}", hooks.trigger_link, methods=["GET", "POST"])])
    client = TestClient(app)
    assert client.put("/trigger/tok").status_code == 405
    assert manager.events == []


# -- Trigger-link management routes (AUTHED) ---------------------------------


@pytest.fixture
def gate_off_ambient(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(hooks_ops, "access_control_settings", lambda: SimpleNamespace(enable=False))


async def test_create_trigger_link_route_envelope(monkeypatch: pytest.MonkeyPatch, gate_off_ambient) -> None:
    async def _create(**kwargs):
        return {"name": "n", "trigger_path": "/trigger/tok", "token": "tok", "topic": "t", "expires_at": None}

    monkeypatch.setattr(hooks_ops.trigger_links, "create_trigger_link", _create)
    resp = await hooks.create_trigger_link(cast(Request, _JsonReq({"topic": "t", "ttl_seconds": None})))
    assert resp.status_code == 200
    assert _body(resp)["data"]["trigger_path"] == "/trigger/tok"


async def test_create_trigger_link_route_ttl_absent_400(monkeypatch: pytest.MonkeyPatch, gate_off_ambient) -> None:
    # The ttl_seconds KEY absent from the body → 400 (the request model requires it).
    resp = await hooks.create_trigger_link(cast(Request, _JsonReq({"topic": "t"})))
    assert resp.status_code == 400


@pytest.mark.parametrize("bad_ttl", ["3600", 3600.0, 3600.5, True])
async def test_create_trigger_link_route_wrong_type_ttl_400(
    monkeypatch: pytest.MonkeyPatch, gate_off_ambient, bad_ttl
) -> None:
    resp = await hooks.create_trigger_link(cast(Request, _JsonReq({"topic": "t", "ttl_seconds": bad_ttl})))
    assert resp.status_code == 400


async def test_create_trigger_link_route_status_mapping(monkeypatch: pytest.MonkeyPatch, gate_off_ambient) -> None:
    from tai42_skeleton.hooks.trigger_links import TriggerLinkError

    async def _raise(**kwargs):
        raise TriggerLinkError(409, "trigger link name already exists")

    monkeypatch.setattr(hooks_ops.trigger_links, "create_trigger_link", _raise)
    resp = await hooks.create_trigger_link(cast(Request, _JsonReq({"topic": "t", "name": "dup", "ttl_seconds": None})))
    assert resp.status_code == 409


async def test_trigger_link_route_roundtrip_and_token_not_listed(
    monkeypatch: pytest.MonkeyPatch, gate_off_ambient
) -> None:
    # A stateful fake store behind the routes: create → list → delete, and the
    # create reply's token never appears in the list output.
    store: dict[str, dict] = {}

    async def _create(*, topic, name, ttl_seconds, tool_kwargs, created_by):
        link_name = name or "trg-link-deadbeef"
        store[link_name] = {"name": link_name, "topic": topic, "token_hash_prefix": "abc123abc123", "expires_at": None}
        return {
            "name": link_name,
            "trigger_path": "/trigger/SECRETTOKEN",
            "token": "SECRETTOKEN",
            "topic": topic,
            "expires_at": None,
        }

    async def _list():
        return {"items": list(store.values()), "total": len(store)}

    async def _revoke(name):
        store.pop(name)

    monkeypatch.setattr(hooks_ops.trigger_links, "create_trigger_link", _create)
    monkeypatch.setattr(hooks_ops.trigger_links, "list_trigger_links", _list)
    monkeypatch.setattr(hooks_ops.trigger_links, "revoke_trigger_link", _revoke)

    created = _body(
        await hooks.create_trigger_link(cast(Request, _JsonReq({"topic": "t", "name": "n", "ttl_seconds": None})))
    )["data"]
    listed = _body(await hooks.list_trigger_links(cast(Request, _GetReq())))["data"]
    assert {item["name"] for item in listed["items"]} == {"n"}
    assert "SECRETTOKEN" not in json.dumps(listed)  # the token is never in list output
    deleted = _body(await hooks.delete_trigger_link(cast(Request, _DelReq("n"))))["data"]
    assert deleted == {"removed": True, "name": "n"}
    assert _body(await hooks.list_trigger_links(cast(Request, _GetReq())))["data"]["total"] == 0
    assert created["token"] == "SECRETTOKEN"


async def test_topic_with_newline_does_not_break_log(monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    manager = _FakeManager()

    async def fake_parse(request, include_query=True):
        return {}

    monkeypatch.setattr(hooks, "parse_any_payload", fake_parse)
    monkeypatch.setattr(hooks, "get_hooks_manager", lambda: manager)
    with caplog.at_level("INFO"):
        await hooks.universal_webhook(cast(Request, _FakeRequest("orders\r\ninjected", body=b"{}")))
    # The CR/LF are stripped from the interpolated topic in the log line.
    logged = "".join(rec.getMessage() for rec in caplog.records)
    assert "ordersinjected" in logged
    assert "\n" not in logged.split("INCOMING EVENT ON TOPIC:")[-1].split("---")[0]
