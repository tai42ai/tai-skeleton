"""The typed async client for the marketplace registry's public read API.

A thin door over the registry's ``{"data": …}`` / ``{"error": …}`` envelope,
riding the app-pooled :class:`HttpxClient` through ``client_ctx``. No auth (the
read API is public), no retries, and no caching here — one ``_request`` maps
every failure to a typed :mod:`.errors` exception and returns the unwrapped
``data``.

``client_ctx`` and ``HttpxClient`` are imported at module level so tests can
monkeypatch ``client.client_ctx`` to a fake transport.

The kit ``HttpxClient`` sets ``trust_env=False``, so operator proxy env vars do
NOT apply to these registry calls; the pip subprocess the installer runs DOES
inherit the environment — an intentional, documented asymmetry.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

import httpx
from tai42_kit.clients import client_ctx
from tai42_kit.clients.impl.http import HttpxClient

from tai42_skeleton.marketplace.errors import (
    ListingNotFoundError,
    RegistryResponseError,
    RegistryUnreachableError,
    VersionRefusedError,
)
from tai42_skeleton.marketplace.settings import marketplace_settings


class RegistryClient:
    """Typed client for the marketplace registry's public read API."""

    def __init__(self, base_url: str | None = None) -> None:
        # Resolve the base URL now, not per call: a client instance pins one
        # endpoint, and one trailing slash is stripped so ``{base}{path}``
        # composes cleanly (the settings value is left unmutated).
        self._base_url = (base_url or marketplace_settings().url).rstrip("/")

    async def search(self, params: Mapping[str, str | list[str]]) -> dict[str, Any]:
        """Proxy the registry search. ``params`` carries the whitelisted query
        names the route forwards; a ``list[str]`` value (``tags``) is encoded as
        repeated query params, never comma-joined."""
        query = {key: value for key, value in params.items() if value is not None}
        data = await self._request("GET", "/api/v1/search", params=query)
        return _as_dict(data, "search")

    async def plugin(self, namespace: str, name: str) -> dict[str, Any]:
        """The listing detail (listing + latest version + items)."""
        ref = f"{namespace}/{name}"
        data = await self._request("GET", f"/api/v1/plugins/{_seg(namespace)}/{_seg(name)}", ref=ref)
        return _as_dict(data, "plugin")

    async def versions(self, namespace: str, name: str) -> list[dict[str, Any]]:
        """The listing's version rows, unwrapped from the registry's
        ``{"versions": [...]}`` wrapper nested under ``data``."""
        ref = f"{namespace}/{name}"
        data = await self._request("GET", f"/api/v1/plugins/{_seg(namespace)}/{_seg(name)}/versions", ref=ref)
        return _unwrap_list(data, "versions", "versions")

    async def categories(self) -> list[str]:
        """The registry's controlled category vocabulary — a bare list of
        category names, unwrapped from the registry's ``{"categories": [...]}``
        wrapper so the route re-envelopes a plain array."""
        data = await self._request("GET", "/api/v1/categories")
        return _unwrap_list(data, "categories", "categories")

    async def advisories(self, *, listing: str | None = None, since: str | None = None) -> list[dict[str, Any]]:
        """Advisory rows, unwrapped from the registry's ``{"advisories": [...]}``
        wrapper nested under ``data``, optionally filtered to one listing and/or a
        since cursor. A 404 on the listing-filtered call surfaces as
        :class:`ListingNotFoundError` naming that ref."""
        params: dict[str, str] = {}
        if listing is not None:
            params["listing"] = listing
        if since is not None:
            params["since"] = since
        data = await self._request("GET", "/api/v1/advisories", params=params, ref=listing)
        rows = _unwrap_list(data, "advisories", "advisories")
        _require_dict_elements(rows, "advisories")
        return rows

    async def resolve(self, namespace: str, name: str, version: str | None = None) -> dict[str, Any]:
        """The single install-time pinning call: the registry pins the version
        (given, or latest published), returns the artifact pointer + stored
        PluginSpec + matching advisories, and counts the download.

        This call passes no ref to ``_request``, so a registry refusal reaches
        here as a :class:`RegistryResponseError` and is remapped: a 409 (the
        registry's refusal of a killed/unpublished version) →
        :class:`VersionRefusedError`, a 404 → :class:`ListingNotFoundError`
        naming the ref and the requested version.
        """
        ref = f"{namespace}/{name}"
        body: dict[str, str] = {}
        if version is not None:
            body["version"] = version
        try:
            data = await self._request("POST", f"/api/v1/plugins/{_seg(namespace)}/{_seg(name)}/resolve", json=body)
        except RegistryResponseError as exc:
            if exc.status == 409:
                raise VersionRefusedError(str(exc)) from exc
            if exc.status == 404:
                target = f"{ref}@{version}" if version is not None else ref
                raise ListingNotFoundError(f"marketplace listing not found: {target}") from exc
            raise
        resolved = _as_dict(data, "resolve")
        _validate_resolve_field_types(resolved)
        # The installer's critical-advisory gate iterates ``resolved["advisories"]``
        # calling ``.get`` on each element, so a non-dict element is validated HERE
        # at the registry boundary — a garbled shape is a 502, uniform with every
        # other malformed-response case, never an untyped 500 downstream.
        advisories = resolved.get("advisories")
        if advisories is not None:
            _require_dict_elements(advisories, "resolve advisories")
        return resolved

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, str | list[str]] | None = None,
        json: dict[str, Any] | None = None,
        ref: str | None = None,
    ) -> Any:
        """Issue one request and return the unwrapped ``data``.

        A transport failure → :class:`RegistryUnreachableError`. A non-2xx with a
        404 for a ref-carrying call → :class:`ListingNotFoundError` naming the
        ref, any other non-2xx → :class:`RegistryResponseError` carrying the
        status. A 2xx whose body is non-JSON or lacks a ``"data"`` key →
        :class:`RegistryResponseError`.
        """
        url = f"{self._base_url}{path}"
        try:
            async with client_ctx(HttpxClient, timeout=marketplace_settings().request_timeout_s) as http:
                response = await http.request(method, url, params=params, json=json)
        except httpx.HTTPError as exc:
            raise RegistryUnreachableError(f"marketplace registry unreachable at {self._base_url}: {exc}") from exc

        if not response.is_success:
            message = _error_message(response)
            if response.status_code == 404 and ref is not None:
                raise ListingNotFoundError(f"marketplace listing not found: {ref}")
            raise RegistryResponseError(message, status=response.status_code)

        try:
            body = response.json()
        except ValueError as exc:
            raise RegistryResponseError(
                f"marketplace registry returned a non-JSON success body for {path}",
                status=None,
            ) from exc
        if not isinstance(body, dict) or "data" not in body:
            raise RegistryResponseError(
                f"marketplace registry success body for {path} is missing the 'data' key",
                status=None,
            )
        return body["data"]


def _error_message(response: httpx.Response) -> str:
    """The registry's enveloped error message, or a status-based fallback when
    the body is not the expected ``{"error": …}`` shape."""
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, str) and error:
            return error
    return f"marketplace registry returned status {response.status_code}"


def _seg(value: str) -> str:
    """Percent-encode one path segment so a namespace/name value cannot inject a
    query, a fragment, or a dot-segment that reroutes the request.

    ``quote(safe="")`` encodes ``/``, ``?``, and ``#`` so the value stays exactly
    one path component. It leaves ``.`` alone, though, and a bare ``..`` (or ``.``)
    segment is collapsed by httpx's dot-segment normalization to a DIFFERENT
    endpoint — so ``.`` is percent-encoded too, keeping the literal segment on the
    wire (the registry decodes it back)."""
    return quote(value, safe="").replace(".", "%2E")


# The resolve response's field-type contract: every field the installer, the
# pip pin composer, and the attribution store consume, mapped to the JSON type
# it must carry WHEN PRESENT. Presence is per-source policy and stays with the
# installer; the boundary owns only the TYPE of whatever the registry did send.
# Consumers apply str-only operations to these fields (``Version()``,
# ``SpecifierSet()``, ``.startswith``, ``.lower()``, psycopg text params), so a
# truthy non-string here would otherwise escape as an untyped
# AttributeError/TypeError-driven 500 — validated at this trust boundary, it is
# a typed :class:`RegistryResponseError` (a 502) like every other
# malformed-response case. Other endpoints (``plugin``/``versions``/
# ``categories``) pass their payloads through as opaque display data; a
# consumer that starts PARSING a field from those responses owns the typed
# guard at its own extraction point.
_RESOLVE_FIELD_TYPES: dict[str, tuple[type, str]] = {
    "version": (str, "a string"),
    "contract_range": (str, "a string"),
    "source": (str, "a string"),
    "artifact_ref": (str, "a string"),
    "sha256": (str, "a string"),
    "repository_url": (str, "a string"),
    "tag": (str, "a string"),
    "spec": (dict, "an object"),
    "advisories": (list, "a list"),
}


def _validate_resolve_field_types(resolved: dict[str, Any]) -> None:
    """Assert each present resolve-response field carries its contracted JSON
    type (see :data:`_RESOLVE_FIELD_TYPES`). Absent and ``null`` fields pass —
    per-source presence is the installer's check, not the boundary's."""
    for field, (expected, label) in _RESOLVE_FIELD_TYPES.items():
        value = resolved.get(field)
        if value is not None and not isinstance(value, expected):
            raise RegistryResponseError(
                f"marketplace registry resolve response field {field!r} is not {label}",
                status=None,
            )


def _require_dict_elements(rows: list[Any], what: str) -> None:
    """Assert every element of a registry-supplied list is a JSON object.

    Advisory lists are consumed element-by-element with ``.get`` at the trust
    boundary's callers, so a non-dict element is garbled registry data → a typed
    :class:`RegistryResponseError` (a 502), uniform with every other
    malformed-response case, never an ``AttributeError``-driven 500 downstream."""
    for element in rows:
        if not isinstance(element, dict):
            raise RegistryResponseError(
                f"marketplace registry {what} response contains a non-object element", status=None
            )


def _as_dict(data: Any, what: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise RegistryResponseError(f"marketplace registry {what} response data is not an object", status=None)
    return data


def _as_list(data: Any, what: str) -> list[Any]:
    if not isinstance(data, list):
        raise RegistryResponseError(f"marketplace registry {what} response data is not a list", status=None)
    return data


def _unwrap_list(data: Any, key: str, what: str) -> list[Any]:
    """The inner array of the ``{key: [...]}`` wrapper the registry nests under
    ``data`` (its list endpoints double-wrap: ``{"data": {"versions": [...]}}``),
    or a typed fault when the wrapper key is absent or its value is not a list."""
    if not isinstance(data, dict) or key not in data:
        raise RegistryResponseError(
            f"marketplace registry {what} response is missing the {key!r} key",
            status=None,
        )
    return _as_list(data[key], what)
