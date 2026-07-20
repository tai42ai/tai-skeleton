"""Agents router tests — the ``/api/agents`` list door and the
``/api/agents/{name}/runs`` SSE run door.

Handlers are driven directly (the router-test pattern): a FAKE agent implementing
the contract ``Agent`` ABC yields a scripted event sequence, and the module's
agent registry seam is monkeypatched to expose it. No concrete agent
implementation is referenced — the surface binds to the contract only.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from pydantic import BaseModel
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

from tai_skeleton.agent import (
    Agent,
    MessageDelta,
    MessageFinal,
    ReasoningStep,
    RunUsage,
    ToolCallStep,
    ToolResultStep,
)
from tai_skeleton.operations import agents as agent_ops
from tai_skeleton.routers import agents as router
from tai_skeleton.routers import interactions as interactions_router
from tai_skeleton.tools.adapters.lc_tool_to_func import build_signature


class _NonJsonPayload:
    """A live, non-JSON-native object of the kind a tool result carries (e.g. a
    ``ToolMessage.content``). It has no pydantic serializer, so a bare
    ``model_dump_json`` would raise — ``fallback=str`` renders it via ``str``."""

    def __init__(self, label: str) -> None:
        self.label = label

    def __str__(self) -> str:
        return f"non-json:{self.label}"


class _FakeInput(BaseModel):
    prompt: str
    count: int = 1


class _FakeAgent(Agent):
    """A streaming agent that replays a scripted event sequence. It records the
    kwargs its ``astream`` received (to pin the ``from_tool_input`` mapping) and
    sets ``cancelled`` when a disconnect cancels it mid-run."""

    tool_name = "faker"
    tool_description = "A fake streaming agent."
    ToolInput = _FakeInput

    def __init__(
        self,
        events: list[Any] | None = None,
        *,
        raise_after: Exception | None = None,
        block: bool = False,
    ) -> None:
        self._events = events or []
        self._raise_after = raise_after
        self._block = block
        self.cancelled = False
        self.received_kwargs: dict[str, Any] | None = None

    async def run(self, **kwargs: Any) -> Any:
        return await self._drain(self.astream(**kwargs))

    async def astream(self, **kwargs: Any):  # type: ignore[override]
        self.received_kwargs = kwargs
        try:
            for event in self._events:
                yield event
            if self._raise_after is not None:
                raise self._raise_after
            if self._block:
                # Park until cancelled — models an agent still working when the
                # client disconnects.
                await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise


class _RenamingAgent(_FakeAgent):
    """Overrides ``from_tool_input`` to rename a field, so a raw pass-through
    would send the wrong kwargs — pins that the route maps through it."""

    @classmethod
    def from_tool_input(cls, validated: BaseModel) -> dict[str, Any]:
        data = {name: getattr(validated, name) for name in validated.model_fields_set}
        if "prompt" in data:
            data["user_message"] = data.pop("prompt")
        return data


# -- request builders --------------------------------------------------------


def _make_get_request() -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/api/agents",
        "headers": [],
        "query_string": b"",
        "client": ("1.2.3.4", 1),
        "path_params": {},
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request(scope, receive)


def _make_run_request(name: str, body: bytes, *, disconnect: bool = False) -> Request:
    scripted: list[dict] = [{"type": "http.request", "body": body, "more_body": False}]
    if disconnect:
        scripted.append({"type": "http.disconnect"})
    idx = {"i": 0}

    async def receive():
        i = idx["i"]
        if i < len(scripted):
            idx["i"] += 1
            return scripted[i]
        # Past the scripted messages: a disconnected client keeps reporting
        # disconnect; a live one reports a benign (non-disconnect) frame so the
        # monitor's ``is_disconnected`` stays False.
        return {"type": "http.disconnect"} if disconnect else {"type": "http.request", "body": b"", "more_body": False}

    scope = {
        "type": "http",
        "method": "POST",
        "path": f"/api/agents/{name}/runs",
        "headers": [(b"content-type", b"application/json")],
        "query_string": b"",
        "client": ("1.2.3.4", 1),
        "path_params": {"name": name},
    }
    return Request(scope, receive)


async def _collect(response: Response) -> list[str]:
    assert isinstance(response, StreamingResponse)
    out: list[str] = []
    async for chunk in response.body_iterator:
        out.append(chunk if isinstance(chunk, str) else bytes(chunk).decode())
    return out


def _data_frames(frames: list[str]) -> list[dict]:
    """Parse the JSON ``data:`` payloads (dropping keep-alive comments)."""
    out: list[dict] = []
    for frame in frames:
        if frame.startswith(":"):
            continue
        assert frame.startswith("data: "), frame
        out.append(json.loads(frame[len("data: ") :].strip()))
    return out


@pytest.fixture
def one_agent(monkeypatch):
    """Register a single fake agent under ``faker`` on the router's registry
    seam and return a factory that swaps the agent's scripted behavior."""
    holder: dict[str, Agent] = {}

    def _install(agent: Agent) -> Agent:
        holder["faker"] = agent
        return agent

    # The registry seam lives in ``operations.agents``: the list ops call it
    # there, and the still-handler SSE run doors reach it through the same op
    # module — so one patch drives both surfaces.
    monkeypatch.setattr(agent_ops, "_agents_registry", lambda: dict(holder))
    return _install


# -- list route --------------------------------------------------------------


async def test_list_shape_and_schema_equals_run_tool_schema(one_agent):
    agent = one_agent(_FakeAgent())
    resp = await router.list_agents(_make_get_request())
    payload = json.loads(bytes(resp.body))
    assert payload["data"]["total"] == 1
    item = payload["data"]["items"][0]
    assert item["name"] == "faker"
    assert item["description"] == "A fake streaming agent."
    assert item["tool_name"] == "faker"
    # One schema source: the binding synthesizes the run tool from this exact
    # ``ToolInput`` model via ``build_signature`` (the same call the binding
    # makes), and the route publishes that same model's schema. Assert the
    # route's schema against the model read from the agent, and that the binding's
    # own signature derivation carries the identical field contract (names +
    # required set) — so a route that diverged from the model the binding builds
    # the tool from is caught.
    input_model = agent.ToolInput
    assert item["input_schema"] == input_model.model_json_schema()
    sig = build_signature(input_model, return_annotation=Any)
    assert set(sig.parameters) == set(item["input_schema"]["properties"])
    required_params = {name for name, p in sig.parameters.items() if p.default is p.empty}
    assert required_params == set(item["input_schema"].get("required", []))


# -- run route: event streaming ----------------------------------------------


async def test_run_streams_full_sequence_in_order_and_parses_back(one_agent):
    events = [
        ReasoningStep(text="thinking"),
        ToolCallStep(tool="search", args={"q": "x"}, call_id="c1"),
        ToolResultStep(tool="search", call_id="c1", result={"hits": 2}),
        MessageDelta(text="hel"),
        MessageDelta(text="lo"),
        RunUsage(input_tokens=3, output_tokens=1, total_tokens=4, model="m"),
        MessageFinal(text="hello"),
    ]
    one_agent(_FakeAgent(events))
    resp = await router.run_agent(_make_run_request("faker", b'{"prompt":"hi"}'))
    frames = _data_frames(await _collect(resp))

    # Every scripted event arrives in order and parses back into its contract
    # model; a terminal stream.end settles the run.
    assert [f["type"] for f in frames] == [
        "reasoning_step",
        "tool_call_step",
        "tool_result_step",
        "message_delta",
        "message_delta",
        "run_usage",
        "message_final",
        "stream.end",
    ]
    assert ReasoningStep.model_validate(frames[0]).text == "thinking"
    assert ToolResultStep.model_validate(frames[2]).result == {"hits": 2}
    assert MessageFinal.model_validate(frames[6]).text == "hello"


async def test_run_serializes_non_json_native_payload(one_agent):
    # A tool-result whose payload is a live non-JSON object serializes via
    # fallback=str instead of crashing the stream.
    events = [ToolResultStep(tool="t", call_id="c1", result=_NonJsonPayload("blob"))]
    one_agent(_FakeAgent(events))
    resp = await router.run_agent(_make_run_request("faker", b'{"prompt":"hi"}'))
    frames = _data_frames(await _collect(resp))
    assert frames[0]["type"] == "tool_result_step"
    assert frames[0]["result"] == "non-json:blob"
    assert frames[-1] == {"type": "stream.end"}


async def test_run_terminates_with_stream_end(one_agent):
    one_agent(_FakeAgent([MessageFinal(text="done")]))
    resp = await router.run_agent(_make_run_request("faker", b'{"prompt":"hi"}'))
    frames = _data_frames(await _collect(resp))
    assert frames[-1] == {"type": "stream.end"}


async def test_run_applies_from_tool_input_mapping(one_agent):
    agent = _RenamingAgent([MessageFinal(text="ok")])
    one_agent(agent)
    resp = await router.run_agent(_make_run_request("faker", b'{"prompt":"hey","count":2}'))
    await _collect(resp)
    # The route mapped the validated input through from_tool_input: prompt was
    # renamed to user_message, count passed through — not a raw body pass-through.
    assert agent.received_kwargs == {"user_message": "hey", "count": 2}


async def test_run_response_headers_mirror_interactions_stream(one_agent):
    one_agent(_FakeAgent([MessageFinal(text="ok")]))
    resp = await router.run_agent(_make_run_request("faker", b'{"prompt":"hi"}'))
    assert isinstance(resp, StreamingResponse)
    # The agents stream mirrors the interactions stream route's headers rather
    # than re-listing literals: derive the reference from the interactions stream
    # response itself (its headers are set at construction — reading them does not
    # start the body iterator, so no backing store is touched) and assert the same
    # media type and no-cache / keep-alive / no-buffering header values.
    reference = await interactions_router.stream(_make_get_request())
    assert isinstance(reference, StreamingResponse)
    assert resp.media_type == reference.media_type == "text/event-stream"
    for header in ("cache-control", "connection", "x-accel-buffering"):
        assert resp.headers[header] == reference.headers[header]
    await _collect(resp)  # drain so the run task finishes


async def test_run_agent_exception_yields_stream_error(one_agent):
    one_agent(_FakeAgent([ReasoningStep(text="oops")], raise_after=RuntimeError("boom")))
    resp = await router.run_agent(_make_run_request("faker", b'{"prompt":"hi"}'))
    frames = _data_frames(await _collect(resp))
    assert frames[0]["type"] == "reasoning_step"
    assert frames[-1] == {"type": "stream.error", "message": "boom"}
    # No stream.end after an error — the error frame is the terminal one.
    assert not any(f["type"] == "stream.end" for f in frames)


async def test_run_disconnect_cancels_underlying_run(one_agent):
    agent = _FakeAgent(block=True)
    one_agent(agent)
    resp = await router.run_agent(_make_run_request("faker", b'{"prompt":"hi"}', disconnect=True))
    await _collect(resp)
    # The disconnect monitor fired; the producer was cancelled, propagating
    # cancellation into astream — the abandoned run stopped.
    assert agent.cancelled is True


# -- run route: input / lookup errors ----------------------------------------


async def test_run_invalid_input_400(one_agent):
    one_agent(_FakeAgent())
    # Missing the required ``prompt`` field.
    resp = await router.run_agent(_make_run_request("faker", b'{"count":2}'))
    assert resp.status_code == 400
    assert "invalid agent input" in json.loads(bytes(resp.body))["error"]


async def test_run_invalid_json_body_400(one_agent):
    one_agent(_FakeAgent())
    resp = await router.run_agent(_make_run_request("faker", b"not json"))
    assert resp.status_code == 400
    assert json.loads(bytes(resp.body))["error"] == "invalid JSON body"


async def test_run_non_object_body_400(one_agent):
    one_agent(_FakeAgent())
    resp = await router.run_agent(_make_run_request("faker", b'"scalar"'))
    assert resp.status_code == 400


async def test_run_unknown_agent_404(one_agent):
    one_agent(_FakeAgent())
    resp = await router.run_agent(_make_run_request("ghost", b'{"prompt":"hi"}'))
    assert resp.status_code == 404
    assert "no such agent" in json.loads(bytes(resp.body))["error"]


async def test_run_rejects_unknown_input_field(one_agent):
    # A body key that is not a ``ToolInput`` field is a loud 400 naming it, never a
    # silent drop (pydantic's default ``extra="ignore"``).
    one_agent(_FakeAgent([MessageFinal(text="x")]))
    resp = await router.run_agent(_make_run_request("faker", b'{"prompt":"hi","bogus":1}'))
    assert resp.status_code == 400
    assert "unknown agent input field" in json.loads(bytes(resp.body))["error"]
    assert "bogus" in json.loads(bytes(resp.body))["error"]


# -- run route: SSE keep-alive ------------------------------------------------


async def test_run_emits_keepalive_between_idle_events(one_agent, monkeypatch):
    # An idle gap between two events emits a ``: keepalive`` comment frame (so a
    # proxy does not drop the connection) WITHOUT dropping either event at the
    # cancellation boundary — the persistent get-task invariant.
    monkeypatch.setattr(router, "_KEEPALIVE_SECONDS", 0.02)

    class _GatedAgent(_FakeAgent):
        async def astream(self, **kwargs: Any):  # type: ignore[override]
            self.received_kwargs = kwargs
            yield MessageDelta(text="first")
            # Idle longer than the keep-alive cadence so a keepalive frame fires
            # between the two events.
            await asyncio.sleep(0.12)
            yield MessageFinal(text="second")

    one_agent(_GatedAgent())
    resp = await router.run_agent(_make_run_request("faker", b'{"prompt":"hi"}'))
    raw = await _collect(resp)

    assert any(frame.startswith(":") and "keepalive" in frame for frame in raw)
    frames = _data_frames(raw)
    assert [f["type"] for f in frames] == ["message_delta", "message_final", "stream.end"]
    assert frames[0]["text"] == "first"
    assert frames[1]["text"] == "second"


# -- run route: from_tool_input ValueError -----------------------------------


class _ConflictAgent(_FakeAgent):
    """``from_tool_input`` raises a plain ``ValueError`` on a conflicting input — the
    loud-400 path distinct from a pydantic ``ValidationError``."""

    @classmethod
    def from_tool_input(cls, validated: BaseModel) -> dict[str, Any]:
        raise ValueError("conflicting fields map to the same run kwarg")


async def test_run_from_tool_input_valueerror_400(one_agent):
    one_agent(_ConflictAgent([MessageFinal(text="ok")]))
    # The input validates against ``ToolInput``, but the mapping step rejects the
    # combination as a loud 400 rather than a silent drop.
    resp = await router.run_agent(_make_run_request("faker", b'{"prompt":"hi"}'))
    assert resp.status_code == 400
    assert "invalid agent input" in json.loads(bytes(resp.body))["error"]
