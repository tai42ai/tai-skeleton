"""``parse_any_payload`` across content types: query merge, JSON object + non-
object, XML, form, raw text/binary fallback, and the malformed-body raises.
"""

from __future__ import annotations

import base64

import pytest
from starlette.requests import Request

from tai42_skeleton.hooks.payload_parser import parse_any_payload


def _request(body: bytes = b"", content_type: str | None = None, query: str = "") -> Request:
    headers = []
    if content_type is not None:
        headers.append((b"content-type", content_type.encode()))
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/webhook",
        "query_string": query.encode(),
        "headers": headers,
    }

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(scope, receive)


async def test_query_params_are_merged():
    data = await parse_any_payload(_request(query="a=1&b=two"))
    assert data == {"a": "1", "b": "two"}


async def test_json_object_body():
    data = await parse_any_payload(_request(body=b'{"order": 7, "ok": true}', content_type="application/json"))
    assert data == {"order": 7, "ok": True}


async def test_json_non_object_carried_under_body():
    data = await parse_any_payload(_request(body=b"[1, 2, 3]", content_type="application/json"))
    assert data == {"body": [1, 2, 3]}


async def test_malformed_json_raises():
    with pytest.raises(ValueError, match="malformed JSON"):
        await parse_any_payload(_request(body=b"{not json", content_type="application/json"))


async def test_xml_body_parsed_to_dict():
    data = await parse_any_payload(_request(body=b"<root><a>1</a></root>", content_type="application/xml"))
    assert data == {"root": {"a": "1"}}


async def test_xml_empty_body_is_skipped():
    data = await parse_any_payload(_request(body=b"", content_type="text/xml"))
    assert data == {}


async def test_malformed_xml_raises():
    with pytest.raises(ValueError, match="malformed XML"):
        await parse_any_payload(_request(body=b"<root>", content_type="application/xml"))


async def test_billion_laughs_xml_rejected_not_expanded():
    # A crafted internal-entity ("billion laughs") body must be rejected, never
    # expanded unbounded — entity resolution is disabled in the parser.
    bomb = (
        b'<?xml version="1.0"?>'
        b'<!DOCTYPE lolz [<!ENTITY lol "lol">'
        b'<!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">'
        b'<!ENTITY lol3 "&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;">]>'
        b"<lolz>&lol3;</lolz>"
    )
    with pytest.raises(ValueError, match="malformed XML"):
        await parse_any_payload(_request(body=bomb, content_type="application/xml"))


async def test_form_urlencoded_body():
    data = await parse_any_payload(_request(body=b"a=1&b=2", content_type="application/x-www-form-urlencoded"))
    assert data == {"a": "1", "b": "2"}


async def test_malformed_form_raises():
    with pytest.raises(ValueError, match="malformed form"):
        await parse_any_payload(
            _request(
                body=b"garbage-not-multipart",
                content_type="multipart/form-data; boundary=xyz",
            )
        )


async def test_raw_text_fallback():
    data = await parse_any_payload(_request(body=b"hello world"))
    assert data == {"raw_body": "hello world"}


async def test_raw_binary_fallback_is_base64():
    body = b"\xff\xfe\x00\x01"
    data = await parse_any_payload(_request(body=body))
    assert data == {"raw_body_base64": base64.b64encode(body).decode("ascii")}


async def test_empty_unknown_body_yields_nothing():
    data = await parse_any_payload(_request(body=b""))
    assert data == {}
