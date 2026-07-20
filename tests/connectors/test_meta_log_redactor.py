"""Tests for the record-factory redactor that scrubs connector secrets from logs."""

from __future__ import annotations

import logging
import sys

import pytest

from tai42_skeleton.connectors import meta_log_redactor
from tai42_skeleton.connectors.meta_log_redactor import (
    _REDACTOR_FAILED,
    _build_redactor_regex,
    _find_object_end,
    _is_tai_logger,
    _mask_headers_env,
    _mask_object_body,
    _read_string,
    _redact_record,
    _skip_value,
    install_meta_log_redactor,
)

_META = "tai_hub.access_token"
_REDACT = "**********"
_SECRET = "AT-12345-supersecret"
_PATTERN = _build_redactor_regex(_META)


@pytest.fixture(autouse=True)
def _restore_log_record_factory():
    """Reset the process-global record factory to the stock ``logging.LogRecord``
    for the duration of every test, then restore whatever was there before.

    Resetting to the stock factory (rather than merely saving/restoring) gives each
    test a clean baseline that is immune to a redactor another test module already
    installed on the global factory (building the app installs it process-wide); the
    restore afterward keeps this module from leaking its own installs onward. The
    monotonic redaction scope is reset to the embed default the same way, so a
    process-scope install in one test cannot widen another's baseline.
    """
    saved = logging.getLogRecordFactory()
    saved_scope = meta_log_redactor._SCOPE
    logging.setLogRecordFactory(logging.LogRecord)
    meta_log_redactor._SCOPE = "tai"
    try:
        yield
    finally:
        logging.setLogRecordFactory(saved)
        meta_log_redactor._SCOPE = saved_scope


@pytest.fixture
def bare_root():
    """A root logger with NO handlers and DEBUG level — mirrors the uvicorn factory
    serving path, where nothing is attached to the root logger. Restores the root's
    handlers and level afterwards."""
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    root.handlers[:] = []
    root.setLevel(logging.DEBUG)
    try:
        yield root
    finally:
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)


def _record(msg: str) -> logging.LogRecord:
    return logging.LogRecord("t", logging.INFO, __file__, 1, msg, None, None)


def _apply(rec: logging.LogRecord) -> logging.LogRecord:
    """Run the in-place record redaction the factory applies to every record."""
    _redact_record(rec, _META, _PATTERN)
    return rec


def _capture_handler() -> tuple[logging.Handler, list[str]]:
    """A handler that records the (post-factory) message of every record it emits."""
    captured: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record.getMessage())

    handler = _Capture()
    handler.setLevel(logging.DEBUG)
    return handler, captured


def _format_capture_handler() -> tuple[logging.Handler, list[str]]:
    """A handler that records the FULLY formatted output (message + exception text)
    of every record — the text a real sink would write."""
    formatted: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            formatted.append(self.format(record))

    handler = _Capture()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(message)s"))
    return handler, formatted


# -- scanning helpers --------------------------------------------------------


def test_read_string_honours_escapes():
    text = r'"a\"b" rest'
    assert text[: _read_string(text, 0)] == r'"a\"b"'


def test_read_string_unterminated():
    text = '"abc'
    assert _read_string(text, 0) == len(text)


def test_find_object_end_balances_braces():
    text = 'X{ "k": "v{nested}" }Y'
    end = _find_object_end(text, 1)
    assert text[end] == "}"
    # the brace inside the quoted value is not mistaken for the close
    assert text[end + 1] == "Y"


def test_find_object_end_unterminated():
    text = "{ unterminated"
    assert _find_object_end(text, 0) == len(text)


def test_skip_value_quoted():
    text = '"value", next'
    assert text[: _skip_value(text, 0)] == '"value"'


def test_skip_value_collection():
    text = '["a", "b"], next'
    assert text[: _skip_value(text, 0)] == '["a", "b"]'


def test_skip_value_bare_scalar():
    text = "123, next"
    assert text[: _skip_value(text, 0)] == "123"


def test_skip_value_at_end():
    assert _skip_value("", 0) == 0


# -- object-body masking -----------------------------------------------------


def test_mask_object_body_masks_values_keeps_keys():
    out = _mask_object_body('"api_key": "secret", "url": "https://x"')
    assert '"api_key"' in out
    assert "secret" not in out
    assert "https://x" not in out
    assert _REDACT in out


def test_mask_object_body_masks_brace_bearing_secret_fully():
    out = _mask_object_body('"k": "abc{def}ghi"')
    assert "abc" not in out
    assert "ghi" not in out


def test_mask_object_body_masks_bare_run():
    # A non-quoted token where a key was expected is consumed and masked.
    out = _mask_object_body("bareword")
    assert "bareword" not in out
    assert _REDACT in out


# -- headers/env masking -----------------------------------------------------


def test_mask_headers_env_masks_header_values():
    msg = 'config={"headers": {"Authorization": "Bearer sk-123"}}'
    out = _mask_headers_env(msg)
    assert "sk-123" not in out
    assert '"Authorization"' in out


def test_mask_headers_env_masks_env_values():
    msg = 'x "env": {"API_KEY": "topsecret"} y'
    out = _mask_headers_env(msg)
    assert "topsecret" not in out


def test_mask_headers_env_no_match_returns_unchanged():
    msg = "nothing sensitive here"
    assert _mask_headers_env(msg) == msg


# -- record redaction: message shapes + no-auth secret masking ---------------


def test_redacts_meta_token():
    rec = _apply(_record(f'{{"{_META}": "supersecret-token"}}'))
    assert "supersecret-token" not in rec.getMessage()
    assert _REDACT in rec.msg


def test_empty_meta_token_value_not_garbled():
    # An empty token value carries no secret; ``str.replace("", …)`` would inject
    # the redaction between every character, so the empty case is left unchanged.
    rec = _apply(_record(f'{{"{_META}": ""}}'))
    assert rec.msg == f'{{"{_META}": ""}}'


def test_redacts_headers():
    rec = _apply(_record('sending {"headers": {"Authorization": "Bearer leak"}}'))
    assert "leak" not in rec.msg


def test_passes_through_clean_message():
    rec = _apply(_record("nothing to redact"))
    assert rec.msg == "nothing to redact"


def test_clears_args_after_redaction():
    rec = logging.LogRecord("t", logging.INFO, __file__, 1, f'{{"{_META}": "%s"}}', ("tok",), None)
    _apply(rec)
    assert rec.args is None
    assert "tok" not in rec.getMessage()


def test_lazy_args_untouched_on_clean_message():
    # No marker in msg/args -> the record's lazy formatting is preserved (args kept,
    # no %-render forced by the redactor).
    rec = logging.LogRecord("t", logging.INFO, __file__, 1, "value=%s", ("plain",), None)
    _apply(rec)
    assert rec.args == ("plain",)


def test_redacts_token_carried_only_in_args():
    # The token lives in args, not the format string; the guard checks str(args).
    rec = logging.LogRecord("t", logging.INFO, __file__, 1, "sending %s", (f'{{"{_META}": "{_SECRET}"}}',), None)
    _apply(rec)
    assert _SECRET not in rec.getMessage()
    assert _REDACT in rec.getMessage()


def test_redacts_json_shape_double_quoted():
    rec = _apply(_record(f'sending: {{"{_META}": "{_SECRET}"}}'))
    assert _SECRET not in rec.getMessage()
    assert _REDACT in rec.getMessage()


def test_redacts_python_repr_shape():
    rec = _apply(_record(f"meta={{'{_META}': '{_SECRET}'}}"))
    assert _SECRET not in rec.getMessage()


def test_noop_when_key_absent():
    rec = _record("normal log line about tool x")
    original = rec.getMessage()
    _apply(rec)
    assert rec.getMessage() == original


def test_regex_compiles_for_keys_with_dots():
    # The default meta key is ``tai_hub.access_token`` — a literal dot in the
    # regex would match any char. Confirm ``re.escape`` is applied so the dot is
    # escaped.
    pattern = _build_redactor_regex(_META)
    assert pattern.search(f'"{_META}": "x"') is not None
    # A key that DIFFERS only in the dot position must NOT match.
    assert pattern.search('"tai_hubXaccess_token": "x"') is None


def test_redacts_no_auth_http_header_values():
    rec = _apply(_record('config={"headers": {"x_api_key": "secret-key-123", "authorization": "Bearer tok-abc"}}'))
    out = rec.getMessage()
    assert "secret-key-123" not in out
    assert "tok-abc" not in out
    assert _REDACT in out
    # The key names stay (only values masked).
    assert "x_api_key" in out


def test_redacts_no_auth_stdio_env_values():
    rec = _apply(_record("launch config={'env': {'API_KEY': 'env-secret-xyz', 'PATH': '/usr/bin'}}"))
    out = rec.getMessage()
    assert "env-secret-xyz" not in out
    assert "/usr/bin" not in out
    assert _REDACT in out


def test_noop_when_no_headers_env_or_meta():
    rec = _record("plain message with url=https://x.test/mcp and title=foo")
    original = rec.getMessage()
    _apply(rec)
    assert rec.getMessage() == original


def test_no_leak_when_secret_value_contains_braces():
    # A no-auth secret legitimately containing { or } must NOT truncate the object
    # body and leak the co-located Authorization header.
    rec = _apply(_record('cfg={"headers": {"x_api_key": "a}b{c", "authorization": "Bearer LEAK-ME"}}'))
    out = rec.getMessage()
    assert "LEAK-ME" not in out
    assert "a}b{c" not in out
    assert _REDACT in out


def test_no_leak_when_secret_value_contains_escaped_quote():
    rec = _apply(_record(r'cfg={"headers": {"x_api_key": "ab\"cd", "z": "tok-SECRET"}}'))
    out = rec.getMessage()
    assert "tok-SECRET" not in out
    assert "cd" not in out  # no leaked tail of the escaped-quote value
    assert _REDACT in out


def test_redacts_unquoted_non_string_value():
    # env values are typed Any; a non-string secret serializes unquoted and must
    # still be masked (the quoted-only matcher would have leaked it).
    rec = _apply(_record("{'env': {'SECRET_PIN': 837465, 'TOKEN': 'tok-abc'}}"))
    out = rec.getMessage()
    assert "837465" not in out
    assert "tok-abc" not in out
    assert _REDACT in out


def test_redacts_collection_and_nested_values():
    # A collection-valued or nested env/headers value must be masked as one unit,
    # not leak elements past the first comma.
    rec = _apply(_record("{'env': {'list': ['a', 'SEC-1'], 'nested': {'inner': 'SEC-2'}, 'plain': 'SEC-3'}}"))
    out = rec.getMessage()
    for secret in ("SEC-1", "SEC-2", "SEC-3"):
        assert secret not in out, out
    assert _REDACT in out


def test_masks_unexpected_bare_run_never_echoes():
    # A bare/unquoted token where a key is expected must be masked, never echoed in
    # clear (no-silent-passthrough). Not producible by dict[str, str], but the
    # redactor must not leak it if it ever appears.
    rec = _apply(_record("{'env': {'A': 'x', BARESECRET, 'B': 'tok'}}"))
    out = rec.getMessage()
    assert "BARESECRET" not in out
    assert "tok" not in out
    assert _REDACT in out


# -- exception / stack redaction ---------------------------------------------


def _record_with_exception(exc_message: str) -> logging.LogRecord:
    try:
        raise ValueError(exc_message)
    except ValueError:
        return logging.LogRecord("t", logging.ERROR, __file__, 1, "request failed", None, sys.exc_info())


def test_redacts_token_inside_exception_text():
    # A token echoed by an exception's str() (e.g. a pydantic ValidationError over
    # the request _meta) must be redacted in the rendered exception text.
    rec = _record_with_exception(f'bad input {{"{_META}": "{_SECRET}"}}')
    _apply(rec)
    assert rec.exc_text is not None
    assert _SECRET not in rec.exc_text
    assert _REDACT in rec.exc_text


def test_redacts_headers_inside_exception_text():
    rec = _record_with_exception('cfg={"headers": {"Authorization": "Bearer EXC-LEAK"}}')
    _apply(rec)
    assert rec.exc_text is not None
    assert "EXC-LEAK" not in rec.exc_text


def test_exception_without_token_leaves_exc_text_unrendered():
    # No marker in the exception -> the redactor does not pre-render/cache exc_text,
    # leaving normal handler rendering intact.
    rec = _record_with_exception("ordinary failure, nothing secret")
    _apply(rec)
    assert rec.exc_text is None


def test_redacts_token_inside_stack_info():
    rec = _record(f'op {{"{_META}": "{_SECRET}"}}')
    rec.stack_info = f'Stack (most recent call last):\n  cfg = {{"{_META}": "{_SECRET}"}}'
    _apply(rec)
    assert rec.stack_info is not None
    assert _SECRET not in rec.stack_info


# -- install (record-factory redaction) --------------------------------------


def test_install_redacts_when_root_had_no_handler_at_install(bare_root):
    """LEAK-CLOSURE: simulate the uvicorn factory path where NOTHING is on the root
    logger when install runs. The record-factory install scrubs each in-scope record
    at creation, so a handler added AFTER install (and lastResort) still emits
    redacted text. Process scope (the CLI-owned path) covers the arbitrary leaf
    logger below.
    """
    install_meta_log_redactor(meta_key=_META, scope="process")
    handler, captured = _capture_handler()
    bare_root.addHandler(handler)  # added AFTER install, as uvicorn/late config does

    logging.getLogger("some.unlisted.leaf.logger").warning(f'sending {{"{_META}": "{_SECRET}"}}')

    assert captured
    assert all(_SECRET not in m for m in captured)
    assert any(_REDACT in m for m in captured)


def test_install_redacts_propagated_child_logger(bare_root):
    """The most likely leak path — a propagated WARNING from ``mcp.shared.session``
    with no own handler — is redacted under process scope without enumerating the
    leaf logger's name."""
    install_meta_log_redactor(meta_key=_META, scope="process")
    handler, captured = _capture_handler()
    bare_root.addHandler(handler)

    logging.getLogger("mcp.shared.session").warning(f'{{"{_META}": "{_SECRET}"}}')

    assert captured
    assert all(_SECRET not in m for m in captured)
    assert any(_REDACT in m for m in captured)


def test_install_redacts_logged_exception_end_to_end(bare_root):
    """A token inside an exception logged via ``logger.exception`` is redacted in the
    fully formatted sink output (exc_text) under process scope."""
    install_meta_log_redactor(meta_key=_META, scope="process")
    handler, formatted = _format_capture_handler()
    bare_root.addHandler(handler)

    logger = logging.getLogger("mcp.shared.session")
    try:
        raise ValueError(f'validation error over {{"{_META}": "{_SECRET}"}}')
    except ValueError:
        logger.exception("request failed")

    joined = "\n".join(formatted)
    assert _SECRET not in joined
    assert _REDACT in joined


def test_install_is_idempotent():
    original = logging.getLogRecordFactory()
    install_meta_log_redactor(meta_key=_META)
    first = logging.getLogRecordFactory()
    install_meta_log_redactor(meta_key=_META)
    second = logging.getLogRecordFactory()
    assert first is not original  # installed
    assert first is second  # not double-wrapped


def test_install_chains_prior_factory():
    prior = logging.getLogRecordFactory()

    def custom_factory(*args: object, **kwargs: object) -> logging.LogRecord:
        record = prior(*args, **kwargs)  # type: ignore[arg-type]
        record.custom_marker = True  # type: ignore[attr-defined]
        return record

    logging.setLogRecordFactory(custom_factory)
    install_meta_log_redactor(meta_key=_META, scope="process")

    factory = logging.getLogRecordFactory()
    rec = factory("t", logging.INFO, __file__, 1, f'{{"{_META}": "{_SECRET}"}}', None, None)
    assert getattr(rec, "custom_marker", False) is True  # the chained prior factory still runs
    assert _SECRET not in rec.getMessage()  # and redaction runs on top


def test_install_defaults_meta_key_from_settings():
    # No meta_key -> settings default (tai_hub.access_token). Process scope so the
    # bare-named record below is in scope regardless of the logger family.
    install_meta_log_redactor(scope="process")
    factory = logging.getLogRecordFactory()
    rec = factory("t", logging.INFO, __file__, 1, '{"tai_hub.access_token": "SECRETVAL"}', None, None)
    assert "SECRETVAL" not in rec.getMessage()
    assert _REDACT in rec.getMessage()


def test_factory_fails_closed_when_redaction_raises():
    # A marker-bearing message with a mismatched ``%`` arg count makes ``getMessage()``
    # raise ``TypeError`` inside the redactor. A record-factory exception would
    # otherwise propagate to the caller's log call (a DoS); the factory must instead
    # fail closed — blanking the record — never crash and never leak the token.
    install_meta_log_redactor(meta_key=_META, scope="process")
    factory = logging.getLogRecordFactory()
    rec = factory("t", logging.INFO, __file__, 1, f"{_META}={_SECRET} %s %s", ("only-one",), None)
    assert rec.msg == _REDACTOR_FAILED
    assert rec.args is None
    assert _SECRET not in rec.getMessage()


def test_redacts_token_in_non_str_msg():
    # ``logger.debug(obj)`` logs a non-str msg; a token in its ``str()`` must still be
    # redacted (the guard stringifies a non-str msg for the marker scan).
    class _Obj:
        def __str__(self) -> str:
            return f'{{"{_META}": "{_SECRET}"}}'

    rec = logging.LogRecord("t", logging.INFO, __file__, 1, _Obj(), None, None)
    _redact_record(rec, _META, _PATTERN)
    assert _SECRET not in rec.getMessage()
    assert _REDACT in rec.getMessage()


# -- redaction scope (tai family vs whole process) ---------------------------


def _make_record(logger_name: str) -> logging.LogRecord:
    """A secret-bearing record made through the installed factory under ``logger_name``."""
    factory = logging.getLogRecordFactory()
    return factory(logger_name, logging.INFO, __file__, 1, f'{{"{_META}": "{_SECRET}"}}', None, None)


@pytest.mark.parametrize(
    "name",
    [
        "tai",
        "tai42_skeleton",
        "tai42_kit.logging",
        "tai.child.logger",
        "tai42_connector_hub",
        # The MCP libraries the runtime drives: their session layers log
        # request/response content carrying connector ``_meta`` — the primary
        # token leak path — so the tai scope covers them.
        "mcp",
        "mcp.shared.session",
        "fastmcp",
        "fastmcp.server.http",
    ],
)
def test_is_tai_logger_accepts_family(name: str) -> None:
    assert _is_tai_logger(name) is True


@pytest.mark.parametrize("name", ["myhost.app", "taint", "uvicorn.error", "tailscale", "mcpx", "fastmcpx.app"])
def test_is_tai_logger_rejects_non_family(name: str) -> None:
    assert _is_tai_logger(name) is False


def test_default_scope_redacts_tai_logger_record():
    # Embed default (tai scope): a record from the tai logger family is scrubbed.
    install_meta_log_redactor(meta_key=_META)
    rec = _make_record("tai42_skeleton.connectors")
    assert _SECRET not in rec.getMessage()
    assert _REDACT in rec.getMessage()


def test_default_scope_passes_host_logger_record_untouched():
    # Embed default (tai scope): a host app's own logger record passes through
    # unredacted — the wrapper never touches records outside the tai family.
    install_meta_log_redactor(meta_key=_META)
    rec = _make_record("myhost.app")
    assert _SECRET in rec.getMessage()
    assert _REDACT not in rec.getMessage()


def test_process_scope_redacts_host_logger_record():
    # Widening to process scope (the CLI path) upgrades the already-installed
    # wrapper, so the same host logger record is now scrubbed.
    install_meta_log_redactor(meta_key=_META)  # embed default first
    install_meta_log_redactor(meta_key=_META, scope="process")  # widen
    rec = _make_record("myhost.app")
    assert _SECRET not in rec.getMessage()
    assert _REDACT in rec.getMessage()


def test_scope_upgrade_is_one_way():
    # A process-scope install cannot be narrowed back by a later tai-scope install:
    # the host logger record stays scrubbed.
    install_meta_log_redactor(meta_key=_META, scope="process")
    install_meta_log_redactor(meta_key=_META)  # default tai scope — must not downgrade
    rec = _make_record("myhost.app")
    assert _SECRET not in rec.getMessage()
    assert _REDACT in rec.getMessage()


def test_invalid_scope_raises():
    with pytest.raises(ValueError, match="scope must be one of"):
        install_meta_log_redactor(meta_key=_META, scope="bogus")  # type: ignore[arg-type]
