"""The typed registry client: envelope unwrap, error mapping, and param shaping.

The kit ``HttpxClient`` is faked at the ``marketplace.client.client_ctx`` seam
(the same seam the access-control store tests monkeypatch); the fake records each
request and answers a canned :class:`httpx.Response` (or raises a transport
error), so the client's mapping is exercised against controlled upstreams with no
network.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest

from tai42_skeleton.marketplace import client as client_module
from tai42_skeleton.marketplace.client import RegistryClient
from tai42_skeleton.marketplace.errors import (
    ListingNotFoundError,
    RegistryResponseError,
    RegistryUnreachableError,
    VersionRefusedError,
)

_BASE = "https://registry.example"

Handler = Callable[[str, str, Any, Any], httpx.Response]


class _FakeHttp:
    """Stand-in for the kit ``HttpxClient``: records requests, answers via the
    handler. A handler that raises simulates a transport failure."""

    def __init__(self, handler: Handler) -> None:
        self._handler = handler
        self.calls: list[dict[str, Any]] = []

    async def request(self, method: str, url: str, *, params: Any = None, json: Any = None) -> httpx.Response:
        # Build a real httpx.Request so the fully-encoded query string is captured
        # exactly as httpx would send it (proving list values repeat, never join).
        request = httpx.Request(method, url, params=params, json=json)
        self.calls.append({"method": method, "url": url, "params": params, "json": json, "request": request})
        return self._handler(method, url, params, json)


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch):
    """Install a fake ``client_ctx`` built from a per-test handler; return the fake
    http object so the test can inspect the recorded requests."""

    def _install(handler: Handler) -> _FakeHttp:
        fake = _FakeHttp(handler)

        @asynccontextmanager
        async def fake_ctx(client_cls, *args, **kwargs):
            yield fake

        monkeypatch.setattr(client_module, "client_ctx", fake_ctx)
        return fake

    return _install


def _ok(payload: Any) -> httpx.Response:
    return httpx.Response(200, json={"data": payload})


def _err(status: int, message: str) -> httpx.Response:
    return httpx.Response(status, json={"error": message})


# -- envelope unwrap ---------------------------------------------------------


async def test_plugin_unwraps_the_data_envelope(wire) -> None:
    wire(lambda m, u, p, j: _ok({"ref": "tai42/toolbox", "display_name": "Toolbox"}))
    client = RegistryClient(_BASE)
    listing = await client.plugin("tai42", "toolbox")
    assert listing == {"ref": "tai42/toolbox", "display_name": "Toolbox"}


async def test_versions_unwraps_the_inner_wrapper(wire) -> None:
    # The registry double-wraps: {"data": {"versions": [...]}}.
    wire(lambda m, u, p, j: _ok({"versions": [{"version": "1.0.0"}, {"version": "0.9.0"}]}))
    rows = await RegistryClient(_BASE).versions("tai42", "toolbox")
    assert [r["version"] for r in rows] == ["1.0.0", "0.9.0"]


async def test_versions_missing_wrapper_is_a_response_error(wire) -> None:
    wire(lambda m, u, p, j: _ok([{"version": "1.0.0"}]))  # bare list, not the {"versions": ...} wrapper
    with pytest.raises(RegistryResponseError, match="versions"):
        await RegistryClient(_BASE).versions("tai42", "toolbox")


async def test_advisories_unwraps_the_inner_wrapper(wire) -> None:
    # The registry double-wraps: {"data": {"advisories": [...]}}.
    wire(lambda m, u, p, j: _ok({"advisories": [{"summary": "CVE-1"}, {"summary": "CVE-2"}]}))
    rows = await RegistryClient(_BASE).advisories()
    assert [a["summary"] for a in rows] == ["CVE-1", "CVE-2"]


async def test_advisories_missing_wrapper_is_a_response_error(wire) -> None:
    wire(lambda m, u, p, j: _ok([{"summary": "CVE-1"}]))  # bare list, not the {"advisories": ...} wrapper
    with pytest.raises(RegistryResponseError, match="advisories"):
        await RegistryClient(_BASE).advisories()


async def test_categories_unwraps_the_inner_wrapper(wire) -> None:
    fake = wire(lambda m, u, p, j: _ok({"categories": ["dev", "data"]}))
    cats = await RegistryClient(_BASE).categories()
    assert cats == ["dev", "data"]
    assert fake.calls[0]["url"].endswith("/api/v1/categories")


async def test_categories_missing_wrapper_is_a_response_error(wire) -> None:
    wire(lambda m, u, p, j: _ok(["dev"]))  # bare list, not the {"categories": ...} wrapper
    with pytest.raises(RegistryResponseError, match="categories"):
        await RegistryClient(_BASE).categories()


# -- error mapping -----------------------------------------------------------


async def test_transport_failure_maps_to_unreachable_naming_url(wire) -> None:
    def _boom(m, u, p, j):
        raise httpx.ConnectError("connection refused")

    wire(_boom)
    with pytest.raises(RegistryUnreachableError, match=re.escape(_BASE)):
        await RegistryClient(_BASE).search({})


async def test_404_on_a_ref_call_maps_to_listing_not_found(wire) -> None:
    wire(lambda m, u, p, j: _err(404, "no such listing"))
    with pytest.raises(ListingNotFoundError, match="tai42/toolbox"):
        await RegistryClient(_BASE).plugin("tai42", "toolbox")


async def test_404_on_listing_filtered_advisories_maps_to_listing_not_found(wire) -> None:
    # The listing-filtered advisories call carries a ref, so a vanished listing is
    # the typed per-ref not-found (the advisory refresh's per-ref skip depends on it).
    wire(lambda m, u, p, j: _err(404, "gone"))
    with pytest.raises(ListingNotFoundError, match="tai42/toolbox"):
        await RegistryClient(_BASE).advisories(listing="tai42/toolbox")


async def test_non_2xx_maps_to_response_error_carrying_status(wire) -> None:
    wire(lambda m, u, p, j: _err(500, "registry boom"))
    with pytest.raises(RegistryResponseError) as exc:
        await RegistryClient(_BASE).search({})
    assert exc.value.status == 500
    assert "registry boom" in str(exc.value)


async def test_2xx_without_data_key_is_a_response_error(wire) -> None:
    wire(lambda m, u, p, j: httpx.Response(200, json={"unexpected": 1}))
    with pytest.raises(RegistryResponseError, match="data"):
        await RegistryClient(_BASE).search({})


# -- search param forwarding -------------------------------------------------


async def test_search_drops_none_and_forwards_given_params(wire) -> None:
    fake = wire(lambda m, u, p, j: _ok({"items": []}))
    # ``kind`` is None and must be dropped before the request is issued.
    given: dict[str, Any] = {"q": "uuid", "kind": None, "tier": "official"}
    await RegistryClient(_BASE).search(given)
    params = fake.calls[0]["params"]
    assert params == {"q": "uuid", "tier": "official"}
    assert "kind" not in params


async def test_search_tags_list_encodes_as_repeated_params(wire) -> None:
    fake = wire(lambda m, u, p, j: _ok({"items": []}))
    await RegistryClient(_BASE).search({"tags": ["a", "b"]})
    # httpx repeats a list value; it is never comma-joined nor first-only.
    query = fake.calls[0]["request"].url.query.decode()
    assert query == "tags=a&tags=b"


# -- resolve -----------------------------------------------------------------


async def test_resolve_posts_the_version_body(wire) -> None:
    fake = wire(lambda m, u, p, j: _ok({"version": "1.2.3"}))
    await RegistryClient(_BASE).resolve("tai42", "toolbox", "1.2.3")
    call = fake.calls[0]
    assert call["method"] == "POST"
    assert call["url"].endswith("/api/v1/plugins/tai42/toolbox/resolve")
    assert call["json"] == {"version": "1.2.3"}


async def test_resolve_omits_the_version_key_when_none(wire) -> None:
    fake = wire(lambda m, u, p, j: _ok({"version": "1.0.0"}))
    await RegistryClient(_BASE).resolve("tai42", "toolbox")
    assert fake.calls[0]["json"] == {}


async def test_resolve_409_remaps_to_version_refused(wire) -> None:
    wire(lambda m, u, p, j: _err(409, "version is killed"))
    with pytest.raises(VersionRefusedError, match="killed"):
        await RegistryClient(_BASE).resolve("tai42", "toolbox", "1.0.0")


async def test_resolve_404_remaps_to_listing_not_found_naming_version(wire) -> None:
    wire(lambda m, u, p, j: _err(404, "no such version"))
    with pytest.raises(ListingNotFoundError, match=r"tai42/toolbox@2\.0\.0"):
        await RegistryClient(_BASE).resolve("tai42", "toolbox", "2.0.0")


async def test_base_url_trailing_slash_is_stripped(wire) -> None:
    fake = wire(lambda m, u, p, j: _ok({"items": []}))
    await RegistryClient(_BASE + "/").search({})
    assert fake.calls[0]["url"] == f"{_BASE}/api/v1/search"


# -- element-shape validation at the registry boundary -----------------------


async def test_advisories_non_dict_element_is_a_response_error(wire) -> None:
    # A non-dict element in the advisories list is garbled registry data: the
    # advisory refresh iterates each element with ``.get``, so it is refused HERE
    # as a typed 502, never surfacing an AttributeError-driven 500.
    wire(lambda m, u, p, j: _ok({"advisories": [{"summary": "ok"}, None]}))
    with pytest.raises(RegistryResponseError, match="non-object element"):
        await RegistryClient(_BASE).advisories(listing="tai42/toolbox")


@pytest.mark.parametrize(
    "field", ["version", "contract_range", "source", "artifact_ref", "sha256", "repository_url", "tag"]
)
async def test_resolve_non_string_typed_field_is_a_response_error(wire, field: str) -> None:
    # Every string-contracted resolve field the installer/pip/store consume is
    # type-checked at this boundary: a truthy non-string value (the registry is
    # untrusted) is a typed 502 HERE, never an AttributeError/TypeError/psycopg
    # crash escaping downstream as an untyped 500.
    payload = {"version": "1.0.0", field: 123}
    wire(lambda m, u, p, j: _ok(payload))
    with pytest.raises(RegistryResponseError, match=re.escape(f"{field!r} is not a string")):
        await RegistryClient(_BASE).resolve("tai42", "toolbox")


async def test_resolve_non_object_spec_is_a_response_error(wire) -> None:
    wire(lambda m, u, p, j: _ok({"version": "1.0.0", "spec": "not-an-object"}))
    with pytest.raises(RegistryResponseError, match=re.escape("'spec' is not an object")):
        await RegistryClient(_BASE).resolve("tai42", "toolbox")


async def test_resolve_non_list_advisories_is_a_response_error(wire) -> None:
    wire(lambda m, u, p, j: _ok({"version": "1.0.0", "advisories": "not-a-list"}))
    with pytest.raises(RegistryResponseError, match=re.escape("'advisories' is not a list")):
        await RegistryClient(_BASE).resolve("tai42", "toolbox")


async def test_resolve_null_and_absent_typed_fields_pass(wire) -> None:
    # The boundary types fields only WHEN PRESENT: JSON null and absence both
    # pass (a pypi resolve legitimately carries no github provenance) —
    # per-source presence is the installer's policy, not the boundary's.
    payload = {"version": "1.0.0", "repository_url": None, "tag": None}
    wire(lambda m, u, p, j: _ok(payload))
    resolved = await RegistryClient(_BASE).resolve("tai42", "toolbox")
    assert resolved == payload


async def test_resolve_non_dict_advisory_element_is_a_response_error(wire) -> None:
    # The installer's critical-advisory gate iterates ``resolved["advisories"]``,
    # so a non-dict advisory element in the resolve payload is validated at this
    # boundary as a typed 502 — the install gate never trips an untyped 500.
    wire(lambda m, u, p, j: _ok({"version": "1.0.0", "advisories": [{}, "not-a-dict"]}))
    with pytest.raises(RegistryResponseError, match="non-object element"):
        await RegistryClient(_BASE).resolve("tai42", "toolbox")


# -- path-segment encoding ---------------------------------------------------


async def test_path_segments_with_dot_dot_are_encoded_not_rerouted(wire) -> None:
    # A ``..`` name must not collapse via httpx dot-segment normalization to a
    # different endpoint: the segment is percent-encoded and stays on the wire.
    fake = wire(lambda m, u, p, j: _ok({"ref": "x"}))
    await RegistryClient(_BASE).plugin("tai42", "..")
    raw_path = fake.calls[0]["request"].url.raw_path
    assert raw_path == b"/api/v1/plugins/tai42/%2E%2E"


async def test_path_segments_with_query_char_are_encoded_not_injected(wire) -> None:
    # A name carrying ``?`` and ``/`` must not inject a query or split the path:
    # both are percent-encoded, leaving one opaque path component and no query.
    fake = wire(lambda m, u, p, j: _ok({"ref": "x"}))
    await RegistryClient(_BASE).plugin("tai42", "na?me/evil")
    url = fake.calls[0]["request"].url
    assert url.query == b""
    assert url.raw_path == b"/api/v1/plugins/tai42/na%3Fme%2Fevil"


# -- response-shape fallbacks ------------------------------------------------


async def test_2xx_non_json_body_is_a_response_error(wire) -> None:
    wire(lambda m, u, p, j: httpx.Response(200, content=b"<html>not json</html>"))
    with pytest.raises(RegistryResponseError, match="non-JSON success body"):
        await RegistryClient(_BASE).search({})


async def test_non_json_error_body_falls_back_to_status_message(wire) -> None:
    # A non-JSON error body has no enveloped message, so the status-based fallback
    # is used and carried on the typed error.
    wire(lambda m, u, p, j: httpx.Response(503, content=b"<html>gateway</html>"))
    with pytest.raises(RegistryResponseError) as exc:
        await RegistryClient(_BASE).search({})
    assert exc.value.status == 503
    assert "status 503" in str(exc.value)


@pytest.mark.parametrize("payload", [["not", "a", "dict"], {"error": 123}, {"error": ""}])
async def test_non_enveloped_error_payload_falls_back_to_status_message(wire, payload: Any) -> None:
    # A non-dict payload, a non-string ``error`` value, and an empty ``error``
    # string all miss the enveloped-message shape → the status-based fallback.
    wire(lambda m, u, p, j: httpx.Response(500, json=payload))
    with pytest.raises(RegistryResponseError, match="status 500"):
        await RegistryClient(_BASE).search({})


async def test_data_that_is_not_an_object_is_a_response_error(wire) -> None:
    # ``plugin`` requires an object under ``data``; a list there is garbled shape.
    wire(lambda m, u, p, j: _ok(["not", "an", "object"]))
    with pytest.raises(RegistryResponseError, match="not an object"):
        await RegistryClient(_BASE).plugin("tai42", "toolbox")


async def test_wrapper_inner_value_that_is_not_a_list_is_a_response_error(wire) -> None:
    # The wrapper key is present but its value is not a list ({"versions": "oops"}).
    wire(lambda m, u, p, j: _ok({"versions": "oops"}))
    with pytest.raises(RegistryResponseError, match="not a list"):
        await RegistryClient(_BASE).versions("tai42", "toolbox")
