"""Redact connector secrets from MCP request logs via the log-record factory.

Managed calls put secrets into the request struct: the OAuth access token via
the JSON-RPC ``_meta`` field (stdio) or an ``Authorization`` header (http), and a
no-auth connection's client config into the ``headers`` (http) or ``env`` (stdio)
of the transport config. No local logger surfaces these under normal config, but
a DEBUG bump of ``mcp.shared.session``, a future fastmcp release, or a wrapper
layer could.

Redaction rides ``logging.setLogRecordFactory``: every ``LogRecord`` the process
creates passes through a wrapping factory that masks the ``_meta`` token value
and every value inside a logged ``headers``/``env`` object — in the message, and
in any attached exception/stack text. Because the record is scrubbed at creation,
before any handler sees it, every in-scope downstream sink emits the redacted text
no matter how logging is configured: named uvicorn loggers, late-added handlers,
``propagate=False`` loggers with their own handler, and even
``logging.lastResort`` (the stderr fallback used when a record reaches no
handler). A cheap substring guard keeps a token-free record almost free — no
``%``-formatting and no exception rendering unless a marker is actually present.

The wrapper applies to a scope, so an embedded tai app does not scrub a host's
own log records:

* ``scope="tai"`` (the default, installed from ``build_app``): only records the
  tai runtime's operation feeds are scrubbed — the tai logger family (the module
  ``__name__`` of a tai package: ``tai_skeleton``, ``tai_kit``, ``tai_contract``,
  ``tai_backend_*``, ``tai_connector_*``, …, or a dotted child of one) plus the
  ``mcp`` / ``fastmcp`` library trees the runtime drives. A host app's own
  records pass through untouched.
* ``scope="process"`` (installed from the CLI entrypoints, which own their
  process): every record in the process is scrubbed.

The scope only ever widens: a ``scope="process"`` install upgrades the predicate
even over an already-installed wrapper, and a later ``scope="tai"`` install never
narrows it back.
"""

from __future__ import annotations

import logging
import re
from typing import Literal

from tai_skeleton.connectors.settings import connector_adapter_settings

_REDACTION = "**********"

Scope = Literal["tai", "process"]

_VALID_SCOPES = ("tai", "process")

# Which records the installed factory scrubs. Starts at the embed default (the tai
# logger family only) and only ever widens to the whole process; a live wrapper
# reads this at record-creation time, so a scope upgrade takes effect immediately.
_SCOPE: Scope = "tai"

# The fail-closed replacement when redacting a record itself raises: the original
# renderable text is dropped so a token it failed to scrub can never reach a handler.
_REDACTOR_FAILED = "[meta-log-redactor error: record suppressed]"

# Start of a ``"headers"``/``"env"`` object in the logged config (JSON or python
# repr). The matching close brace is found by a quote-aware balanced scan, NOT a
# brace-free char class — a client secret value can legitimately contain ``{`` or
# ``}``, which would otherwise truncate the object body and leak the rest.
_HEADERS_ENV_START_RE = re.compile(r'["\'](?:headers|env)["\']\s*:\s*\{')

_OPEN_TO_CLOSE = {"[": "]", "{": "}", "(": ")"}


def _build_redactor_regex(meta_key: str) -> re.Pattern[str]:
    # Match the key, then its quoted value in the JSON the emitter logs
    # (``"<key>": "<value>"``). The meta token is a base64url string (no quotes),
    # so a simple value class is safe here.
    quoted_key = re.escape(meta_key)
    return re.compile(
        rf'["\']?{quoted_key}["\']?\s*:\s*(?P<quote>["\'])(?P<value>[^"\']*)(?P=quote)',
    )


def _read_string(text: str, i: int) -> int:
    """Index just past the closing quote of the string starting at ``text[i]``
    (a quote char), honouring ``\\`` escapes. ``len(text)`` if unterminated."""
    quote = text[i]
    j = i + 1
    while j < len(text):
        ch = text[j]
        if ch == "\\":
            j += 2
            continue
        if ch == quote:
            return j + 1
        j += 1
    return len(text)


def _find_object_end(text: str, open_brace: int) -> int:
    """Index of the ``}`` matching the ``{`` at ``open_brace``, skipping braces
    inside quoted strings. Returns ``len(text)`` if unterminated."""
    depth = 0
    i = open_brace
    while i < len(text):
        ch = text[i]
        if ch in "\"'":
            i = _read_string(text, i)
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return len(text)


def _skip_value(text: str, i: int) -> int:
    """Index just past the value starting at ``text[i]`` — a quoted string, a
    bracketed collection (``[]``/``{}``/``()``, quote- and nesting-aware), or a
    bare scalar up to the next top-level ``,``/``}``. Consuming the whole value
    (not stopping at the first comma/brace) is what stops a collection or a
    brace-bearing secret from leaking its tail."""
    n = len(text)
    if i >= n:
        return i
    ch = text[i]
    if ch in "\"'":
        return _read_string(text, i)
    if ch in _OPEN_TO_CLOSE:
        depth = 0
        j = i
        while j < n:
            c = text[j]
            if c in "\"'":
                j = _read_string(text, j)
                continue
            if c in "[{(":
                depth += 1
            elif c in "]})":
                depth -= 1
                if depth == 0:
                    return j + 1
            j += 1
        return n
    j = i
    while j < n and text[j] not in ",}":
        j += 1
    return j


def _mask_object_body(body: str) -> str:
    """Mask every value in a flat/nested ``key: value`` object body. Keys (always
    quoted) are kept; each value (any shape) is replaced with the redaction."""
    out: list[str] = []
    i = 0
    n = len(body)
    while i < n:
        ch = body[i]
        if ch in " \t\r\n,":
            out.append(ch)
            i += 1
            continue
        if ch in "\"'":
            key_end = _read_string(body, i)
            out.append(body[i:key_end])
            i = key_end
            while i < n and body[i] in " \t":
                out.append(body[i])
                i += 1
            if i < n and body[i] == ":":
                out.append(":")
                i += 1
                while i < n and body[i] in " \t":
                    out.append(body[i])
                    i += 1
                value_end = _skip_value(body, i)
                out.append(f'"{_REDACTION}"')
                i = value_end
            continue
        # A bare/unquoted run where a key was expected (a well-formed dict[str,str]
        # never produces this). Never echo it through — consume to the next
        # top-level delimiter and mask, so no unrecognized content passes in clear.
        value_end = _skip_value(body, i)
        out.append(f'"{_REDACTION}"')
        i = value_end
    return "".join(out)


def _mask_headers_env(msg: str) -> str:
    """Mask every value inside any ``headers``/``env`` object in ``msg``."""
    pos = 0
    while True:
        start = _HEADERS_ENV_START_RE.search(msg, pos)
        if start is None:
            return msg
        open_brace = start.end() - 1
        end = _find_object_end(msg, open_brace)
        masked = _mask_object_body(msg[open_brace + 1 : end])
        msg = msg[: open_brace + 1] + masked + msg[end:]
        pos = open_brace + 1 + len(masked) + 1


# Renders an ``exc_info`` tuple to the same text a default handler would emit, so
# the token check and redaction see exactly what would otherwise reach the sink.
_EXC_FORMATTER = logging.Formatter()

# Substring markers of a logged ``headers``/``env`` object, in JSON or python repr.
_HEADERS_ENV_MARKERS = ('"headers"', "'headers'", '"env"', "'env'")


def _redact_meta_match(m: re.Match[str]) -> str:
    """Replace the matched token value with the redaction. An empty value has
    nothing to hide and would garble the text via ``str.replace("", …)`` (which
    injects the redaction between every character), so it is left as-is."""
    value = m.group("value")
    if not value:
        return m.group(0)
    return m.group(0).replace(value, _REDACTION)


def _has_marker(text: str, meta_key: str) -> bool:
    """Cheap guard: does ``text`` carry the meta token key or a headers/env object?"""
    return meta_key in text or any(marker in text for marker in _HEADERS_ENV_MARKERS)


def _redact_text(text: str, meta_key: str, pattern: re.Pattern[str]) -> str:
    """Mask the meta token value and every headers/env value present in ``text``."""
    if meta_key in text:
        text = pattern.sub(_redact_meta_match, text)
    if any(marker in text for marker in _HEADERS_ENV_MARKERS):
        text = _mask_headers_env(text)
    return text


def _redact_record(record: logging.LogRecord, meta_key: str, pattern: re.Pattern[str]) -> None:
    """Scrub secrets from ``record`` in place: its message, and any attached
    exception/stack text.

    The marker check inspects the raw ``msg`` (stringified — a non-``str`` ``msg``
    whose ``str()`` carries a token is caught too) and a string form of ``args``; the
    full ``%``-render (:meth:`~logging.LogRecord.getMessage`) and the redaction subs
    run only when a marker is present, and an exception is rendered only when the
    record carries one. So a token-free record pays only the cheap marker scan, not
    the full format + regex."""
    raw_msg = record.msg if isinstance(record.msg, str) else str(record.msg)
    args_text = str(record.args) if record.args else ""
    if _has_marker(raw_msg, meta_key) or _has_marker(args_text, meta_key):
        # Render (msg % args), redact, and clear args so downstream formatters
        # emit the redacted text and never re-interpolate the un-redacted source.
        record.msg = _redact_text(record.getMessage(), meta_key, pattern)
        record.args = None

    # Exceptions are rare, so pay the render cost only when one is attached. Set
    # ``exc_text`` to the redacted render; ``Formatter.format`` reuses a non-empty
    # ``exc_text`` verbatim rather than re-rendering the raw traceback.
    if record.exc_info or record.exc_text:
        exc_text = record.exc_text or _EXC_FORMATTER.formatException(record.exc_info)  # type: ignore[arg-type]
        if _has_marker(exc_text, meta_key):
            record.exc_text = _redact_text(exc_text, meta_key, pattern)

    if record.stack_info and _has_marker(record.stack_info, meta_key):
        record.stack_info = _redact_text(record.stack_info, meta_key, pattern)


# Logger-name roots the ``"tai"`` scope covers: the tai package family plus the
# MCP client/server libraries the tai runtime itself drives (``mcp`` /
# ``fastmcp``) — their session layers log request/response content carrying
# connector ``_meta``, the primary token leak path, and those records are
# produced by tai's own operation even though the logger names are third-party.
_TAI_SCOPE_EXACT = ("tai", "mcp", "fastmcp")
_TAI_SCOPE_PREFIXES = ("tai_", "tai.", "mcp.", "fastmcp.")


def _is_tai_logger(name: str) -> bool:
    """Whether ``name`` belongs to the loggers tai's operation feeds — the module
    ``__name__`` of a tai package (``tai_skeleton``, ``tai_kit``, ``tai_contract``,
    ``tai_backend_*``, ``tai_connector_*``, …), a dotted child of one, or the
    ``mcp`` / ``fastmcp`` library trees the runtime drives."""
    return name in _TAI_SCOPE_EXACT or name.startswith(_TAI_SCOPE_PREFIXES)


def install_meta_log_redactor(*, meta_key: str | None = None, scope: Scope = "tai") -> None:
    """Install the process-global redacting log-record factory.

    Chains the current ``logging.getLogRecordFactory()`` (rather than replacing
    it), so any factory another component installed still runs. Every in-scope
    record the process creates is then scrubbed by :func:`_redact_record` before any
    handler sees it, which redacts leaks regardless of handler timing, logger
    propagation, or the ``logging.lastResort`` stderr fallback — no leaf logger
    names need enumerating.

    Args:
        meta_key: The connector-meta token key to redact; defaults to the configured
            key. A malformed key fails loudly at install time.
        scope: ``"tai"`` scrubs only the loggers tai's operation feeds — the tai
            logger family plus the ``mcp`` / ``fastmcp`` trees (the embed
            default, so a host app's own records pass through untouched);
            ``"process"`` scrubs
            every record in the process (installed from the CLI entrypoints, which
            own their process). The scope only widens: a ``"process"`` install
            upgrades the predicate even over an already-installed wrapper, and a
            later ``"tai"`` install never narrows it back.

    Idempotent: the wrapping factory is tagged, and a second call detects the tag
    and returns without stacking a duplicate wrapper — after applying any scope
    widening.
    """
    global _SCOPE

    if scope not in _VALID_SCOPES:
        raise ValueError(f"scope must be one of {', '.join(map(repr, _VALID_SCOPES))}; got {scope!r}.")

    if meta_key is None:
        meta_key = connector_adapter_settings().meta_token_key

    # Compiled here (not per record): a malformed meta key fails loudly at install
    # time, never by silently passing tokens at log time.
    pattern = _build_redactor_regex(meta_key)

    # Widen the scope monotonically, before the idempotency check so a process-scope
    # install upgrades an already-installed wrapper; a tai-scope install never
    # narrows an existing process scope.
    if scope == "process":
        _SCOPE = "process"

    previous_factory = logging.getLogRecordFactory()
    if getattr(previous_factory, "_is_meta_log_redactor", False):
        return

    def redacting_factory(*args: object, **kwargs: object) -> logging.LogRecord:
        record = previous_factory(*args, **kwargs)  # type: ignore[arg-type]
        # Out-of-scope records pass through untouched: under the tai scope a host
        # app's own records are left as they are; the live ``_SCOPE`` read makes a
        # later process-scope upgrade take effect on this same wrapper.
        if _SCOPE != "process" and not _is_tai_logger(record.name):
            return record
        try:
            _redact_record(record, meta_key, pattern)
        except Exception:
            # A record-factory exception propagates to the caller's ``log`` call (it
            # runs before ``Handler.handle``'s ``handleError`` guard), so a redactor
            # bug — or a benign logging mistake like a bad ``%`` arg count or a raising
            # ``__repr__`` — must never crash the request path, and must never leak a
            # token it failed to scrub. Fail closed: blank the record's renderable
            # fields to a marker. Re-logging here would re-enter the factory, so the
            # failure is deliberately not re-emitted.
            record.msg = _REDACTOR_FAILED
            record.args = None
            record.exc_info = None
            record.exc_text = None
            record.stack_info = None
        return record

    redacting_factory._is_meta_log_redactor = True  # type: ignore[attr-defined]
    logging.setLogRecordFactory(redacting_factory)
