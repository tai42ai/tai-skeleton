"""Shared helpers for the remote command groups.

Every ``tai`` remote command is a thin wrapper over one or more of the skeleton's
``/api/*`` routes — it resolves a configured :class:`ApiClient` from the invocation
context, calls the route, and renders the result (or lets the typed error surface
loudly). This module holds the small pieces every group reuses:

* :func:`app_context` — recover the :class:`AppContext` the root callback stashed
  on the Typer context;
* :func:`covers` — record the ``/api/*`` route(s) a command invokes, so the
  CLI↔route parity gate can prove every registered route has a command;
* argument parsers for the JSON / ``key=value`` inputs the write commands accept;
* the record/result/stream rendering helpers that honor the global ``--json``.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import typer
from tai42_contract.manifest import ExtensionElement

from tai42_skeleton.cli.context import AppContext
from tai42_skeleton.cli.render import print_json, print_records, print_result, strip_control

# The set of ``(method, path)`` routes every imported remote command declares it
# invokes. The parity gate reads this after importing the command modules and
# asserts every registered ``/api/*`` route (minus its allowlist) appears here.
COVERED_ROUTES: set[tuple[str, str]] = set()


def covers(*routes: tuple[str, str]):
    """Record the ``(method, path)`` route(s) a command invokes for the parity gate.

    Each entry is ``("GET", "/api/tools")``-shaped, matching the registry's own
    ``path`` and one of its ``methods``. A command that fans out to more than one
    route (a list plus its download variant, say) declares them all.
    """

    def decorator(func):
        for method, path in routes:
            COVERED_ROUTES.add((method.upper(), path))
        return func

    return decorator


def app_context(ctx: typer.Context) -> AppContext:
    """The :class:`AppContext` the root callback placed on the Typer context."""
    obj = ctx.obj
    if not isinstance(obj, AppContext):
        raise RuntimeError("CLI invocation context is not initialized")
    return obj


def parse_json_object(value: str, *, param_hint: str) -> dict[str, Any]:
    """Parse ``value`` as a JSON object, raising a usage error otherwise."""
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"must be valid JSON: {exc}", param_hint=param_hint) from exc
    if not isinstance(parsed, dict):
        raise typer.BadParameter("must be a JSON object", param_hint=param_hint)
    return parsed


def parse_json_value(value: str, *, param_hint: str) -> Any:
    """Parse ``value`` as any JSON value, raising a usage error otherwise."""
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"must be valid JSON: {exc}", param_hint=param_hint) from exc


def parse_extension_element(element: Any, *, param_hint: str) -> ExtensionElement:
    """One combo element in the lossless extension-combo syntax, structurally
    validated: a non-empty extension NAME (a bare JSON string), or a
    ``{"name": <non-empty str>, "config": <object>}`` object binding author config
    to it (``config`` is REQUIRED — the config-less selection is the bare-string
    form, so a config-free object is malformed) with no other keys.

    This mirrors the ``ExtensionElement`` contract the
    ``/api/tools/{name}/extensions`` and ``/api/presets`` doors accept, so a combo
    authored on the CLI round-trips losslessly to the wire. Anything else raises a
    usage error — a malformed element is never silently dropped.
    """
    if isinstance(element, str):
        if not element:
            raise typer.BadParameter("an extension name must be a non-empty string", param_hint=param_hint)
        return element
    if isinstance(element, dict):
        name = element.get("name")
        if not isinstance(name, str) or not name:
            raise typer.BadParameter("an extension element must have a non-empty string 'name'", param_hint=param_hint)
        config = element.get("config")
        if not isinstance(config, dict):
            raise typer.BadParameter(f"extension element {name!r} must carry a 'config' mapping", param_hint=param_hint)
        extra = set(element) - {"name", "config"}
        if extra:
            raise typer.BadParameter(
                f"extension element {name!r} has unexpected keys: {sorted(extra)!r}", param_hint=param_hint
            )
        return {"name": name, "config": dict(config)}
    raise typer.BadParameter(
        "each combo element must be an extension name or a {'name', 'config'} object", param_hint=param_hint
    )


def parse_extension_combo(value: str, *, param_hint: str) -> list[ExtensionElement]:
    """Parse ONE extension combo — a JSON array of combo elements (see
    :func:`parse_extension_element`). An empty combo is rejected: a combo names at
    least one extension."""
    parsed = parse_json_value(value, param_hint=param_hint)
    if not isinstance(parsed, list) or not parsed:
        raise typer.BadParameter("a combo must be a non-empty JSON array of extension elements", param_hint=param_hint)
    return [parse_extension_element(element, param_hint=param_hint) for element in parsed]


def parse_extension_combos(value: str, *, param_hint: str) -> list[list[ExtensionElement]]:
    """Parse a full extension spec — a JSON array of combos, each itself a
    non-empty array of combo elements (see :func:`parse_extension_element`). An
    empty top-level array is legal (it clears the tool's/preset's extensions); an
    empty inner combo is rejected, mirroring the doors' own shape rule."""
    parsed = parse_json_value(value, param_hint=param_hint)
    if not isinstance(parsed, list):
        raise typer.BadParameter("must be a JSON array of extension combos", param_hint=param_hint)
    result: list[list[ExtensionElement]] = []
    for combo in parsed:
        if not isinstance(combo, list) or not combo:
            raise typer.BadParameter(
                "each combo must be a non-empty JSON array of extension elements", param_hint=param_hint
            )
        result.append([parse_extension_element(element, param_hint=param_hint) for element in combo])
    return result


def parse_kwargs(kwargs_json: str | None, kw: Sequence[str] | None) -> dict[str, Any]:
    """Build a kwargs mapping from an optional ``--kwargs`` JSON object plus any
    repeated ``--kw key=value`` pairs (each value parsed as JSON, falling back to
    the literal string). A ``--kw`` pair overrides the same key from ``--kwargs``.
    """
    result: dict[str, Any] = {}
    if kwargs_json is not None:
        result.update(parse_json_object(kwargs_json, param_hint="--kwargs"))
    for pair in kw or []:
        key, sep, raw = pair.partition("=")
        if not sep or not key:
            raise typer.BadParameter(f"expected key=value, got {pair!r}", param_hint="--kw")
        try:
            result[key] = json.loads(raw)
        except json.JSONDecodeError:
            result[key] = raw
    return result


def emit_records(
    ctx_obj: AppContext,
    data: Any,
    columns: Sequence[str],
    *,
    items_key: str | None = None,
) -> None:
    """Render a list result as a table, or the raw payload under ``--json``.

    ``items_key`` pulls the row list out of an ``{items, total}``-style envelope for
    the table while the ``--json`` form still emits the whole payload. A row that is
    a bare scalar (a list of tool names) is wrapped under the first column.
    """
    if ctx_obj.json_output:
        print_json(data)
        return
    rows: Any = data.get(items_key, []) if items_key is not None and isinstance(data, Mapping) else data
    normalized = [row if isinstance(row, Mapping) else {columns[0]: row} for row in rows]
    print_records(normalized, columns, json_output=False)


def emit_result(ctx_obj: AppContext, data: Any) -> None:
    """Render a single result, honoring ``--json``."""
    print_result(data, json_output=ctx_obj.json_output)


def stream_frames(
    ctx_obj: AppContext,
    method: str,
    path: str,
    *,
    json_body: Any | None = None,
    params: Mapping[str, Any] | None = None,
    until_empty: bool = False,
) -> None:
    """Stream an SSE run to stdout frame by frame — never buffering the whole run.

    Each server frame is written on its own line and flushed as it arrives. A frame
    that carries an out-of-band ``event:`` type (the interactions stream) is rendered
    as ``{"event": <type>, "data": <payload>}`` so the operator can tell, say, an
    answered interaction from a removed one; a frame with no event type (the runs
    stream, whose type rides inside its JSON) is rendered as its ``data`` payload
    unchanged. With ``until_empty`` the stream is consumed only up to the first
    empty-object frame (the interactions backlog's ``backlog_done`` marker) and
    then closed, so a backlog listing returns instead of tailing forever.

    A frame is the MOST attacker-influenced data the CLI prints, so its control
    characters (terminal escapes) are stripped before it reaches the terminal — the
    rendered line carries no newline and JSON ``\\uXXXX`` escapes stay literal, so
    legitimate payloads are unaffected.
    """
    with ctx_obj.client() as client:
        for line in _iter_stream(client.stream(method, path, json=json_body, params=params), until_empty=until_empty):
            print(strip_control(line), flush=True)


def _render_frame(event: str | None, data: str) -> str:
    """Render one SSE frame for display, surfacing an out-of-band ``event:`` type.

    A frame with no event type prints its ``data`` payload exactly as received. A
    frame that carries an event type prints ``{"event": <type>, "data": <payload>}``
    with the payload parsed as JSON (as these routes always send), falling back to an
    event-prefixed line if the payload is not JSON — never dropping the type.
    """
    if event is None:
        return data
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return f"event: {event}, data: {data}"
    return json.dumps({"event": event, "data": payload})


def _iter_stream(frames: Iterator[tuple[str | None, str]], *, until_empty: bool) -> Iterator[str]:
    for event, data in frames:
        if until_empty:
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, Mapping) and not payload:
                return
        yield _render_frame(event, data)


def fetch_download(
    ctx_obj: AppContext,
    method: str,
    path: str,
    *,
    json_body: Any | None = None,
    params: Mapping[str, Any] | None = None,
) -> str:
    """Fetch the raw body text of a DOWNLOAD route (backup / observability exports).

    A few routes answer a file download — a bare backup document or a CSV/JSON
    export — outside the ``{"data": ...}`` envelope the enveloped client helpers
    unwrap. The client's :meth:`ApiClient.request_raw` performs the auth'd request
    and applies the typed-error mapping without unwrapping an envelope, so error
    handling stays owned by the one client module.
    """
    with ctx_obj.client() as client:
        response = client.request_raw(method, path, json=json_body, params=params)
        return response.text


def echo_stderr(message: str) -> None:
    """Write an informational line to stderr, keeping stdout clean for data."""
    print(message, file=sys.stderr)


def validate_manifest_file(path: str) -> None:
    """Validate a manifest file against the in-repo :class:`Manifest` model, OFFLINE.

    Loads the YAML (expanding ``!ENV`` tags exactly as the runtime read does) and
    runs ``Manifest.model_validate``, raising a usage error carrying the model's
    message on any failure. No server, database, or Redis is touched.
    """
    from pyaml_env import parse_config
    from pydantic import ValidationError

    from tai42_skeleton.manifest import Manifest

    try:
        raw = parse_config(path=path) or {}
    except Exception as exc:
        raise typer.BadParameter(f"could not read manifest YAML: {exc}", param_hint="FILE") from exc
    if not isinstance(raw, dict):
        raise typer.BadParameter("manifest must be a YAML mapping", param_hint="FILE")
    try:
        Manifest.model_validate(raw)
    except ValidationError as exc:
        raise typer.BadParameter(f"invalid manifest: {exc}", param_hint="FILE") from exc
