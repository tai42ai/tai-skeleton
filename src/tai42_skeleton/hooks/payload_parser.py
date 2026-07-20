import base64
import logging
from typing import Any

import xmltodict
from starlette.requests import Request

logger = logging.getLogger(__name__)


async def parse_any_payload(request: Request, include_query: bool = True) -> dict[str, Any]:
    """Parse a webhook request into the payload dict hooks run on.

    A body that contradicts its declared Content-Type raises ``ValueError`` —
    firing hooks on a partially-salvaged payload would run flows on wrong
    data. Bodies with no declared type pass through explicitly: text as
    ``raw_body``, binary as ``raw_body_base64`` — never dropped.

    ``include_query=False`` drops the query string from the payload. A
    body-signature verifier authenticates the raw body only; folding the
    unauthenticated query string into the dispatched payload would let a captured
    signed delivery be replayed with attacker-appended ``?key=val`` params, so a
    caller that verified such a topic parses the body alone.
    """
    data = {}

    # 1. Parse Query Params
    if include_query and request.query_params:
        data.update(dict(request.query_params))

    content_type = request.headers.get("Content-Type", "").lower()

    # 2. Handle JSON
    if "application/json" in content_type:
        try:
            json_data = await request.json()
        except Exception as e:
            raise ValueError(f"malformed JSON body: {e}") from e
        if isinstance(json_data, dict):
            data.update(json_data)
        else:
            # A valid non-object JSON body (array, scalar) is carried whole.
            data["body"] = json_data

    # 3. Handle XML
    elif "application/xml" in content_type or "text/xml" in content_type:
        body_bytes = await request.body()
        if body_bytes:
            try:
                # xmltodict converts XML structure to a standard Python Dict.
                # ``disable_entities=True`` turns off entity resolution so a
                # hostile body (billion-laughs / entity-expansion) cannot expand
                # unbounded — a crafted internal-entity payload is rejected, not
                # blown up in memory.
                xml_data = xmltodict.parse(
                    body_bytes,
                    dict_constructor=dict,  # Ensure standard dicts, not OrderedDicts
                    disable_entities=True,
                )
            except Exception as e:
                raise ValueError(f"malformed XML body: {e}") from e
            # Usually XML has a root key, we merge that into data
            data.update(xml_data)

    # 4. Handle Form Data
    elif "application/x-www-form-urlencoded" in content_type or "multipart/form-data" in content_type:
        try:
            form_data = await request.form()
        except Exception as e:
            raise ValueError(f"malformed form body: {e}") from e
        data.update(dict(form_data))

    # 5. Fallback: Raw Body
    else:
        body_bytes = await request.body()
        if body_bytes:
            try:
                data["raw_body"] = body_bytes.decode("utf-8")
            except UnicodeDecodeError:
                data["raw_body_base64"] = base64.b64encode(body_bytes).decode("ascii")

    return data
