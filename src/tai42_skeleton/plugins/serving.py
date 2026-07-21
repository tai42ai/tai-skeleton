"""Serving primitives for the Studio SPA host: per-response CSP nonce, the
security-header set, the asset content-type map, and the import-map injection
(HTML-escaped so a hostile manifest string can never break out of the inline
script into the PUBLIC index.html).
"""

from __future__ import annotations

import json
import secrets

from tai42_skeleton.plugins.registry import StudioPluginError, StudioPluginRegistry

# The Studio's source index.html carries this token at the TOP of <head>; Vite
# preserves it ahead of its injected module scripts. The server replaces it with
# the nonce-stamped importmap <script>, which MUST precede the shell's module
# entry (the shell resolves react through it).
IMPORTMAP_ANCHOR = "<!--tai-importmap-->"

# Explicit content-type mapping — never OS mimetype guessing (a wrong module MIME
# silently breaks ESM loading). Every UNMAPPED extension serves as
# application/octet-stream; this route must NEVER emit text/html.
_CONTENT_TYPES: dict[str, str] = {
    ".js": "text/javascript",
    ".mjs": "text/javascript",
    ".css": "text/css",
    ".map": "application/json",
    ".wasm": "application/wasm",
    # Static image assets a built SPA serves (favicon / touch icon). Mapped so the
    # browser gets the right image type rather than application/octet-stream.
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
}
_OCTET_STREAM = "application/octet-stream"

# The HTML content-type for the SPA document responses (index.html + static
# pages). Kept here so the asset route (which must never emit it) and the HTML
# branch can't diverge.
HTML_CONTENT_TYPE = "text/html; charset=utf-8"


def asset_content_type(filename: str) -> str:
    """Content-type for a served plugin/static asset by extension. Unmapped ->
    octet-stream; never text/html."""
    lower = filename.lower()
    for suffix, ctype in _CONTENT_TYPES.items():
        if lower.endswith(suffix):
            return ctype
    return _OCTET_STREAM


def generate_nonce() -> str:
    """A fresh CSPRNG nonce per response (>=128 bits). Never a counter/timestamp —
    a predictable nonce defeats the inline-script gate."""
    return secrets.token_urlsafe(16)


def security_headers(nonce: str) -> dict[str, str]:
    """The app CSP designed for the NATIVE plugin-loading model (no shim, so no
    ``blob:``/``unsafe-eval``). The nonce authorizes the ONE first-party inline
    script (the import map); ``wasm-unsafe-eval`` permits WebAssembly instantiation
    (CSP-gated separately from JS eval) for rich wasm-using plugins; the three
    non-inheriting directives (``base-uri``/``object-src``/``form-action``) are set
    explicitly because they do NOT fall back to ``default-src``. ``img-src`` admits
    ``https:`` so a question's media images render directly from remote https
    origins (``http:`` stays excluded — mixed content); images cannot execute
    script, so the ``script-src`` posture is unaffected.
    """
    csp = "; ".join(
        [
            "default-src 'none'",
            f"script-src 'self' 'nonce-{nonce}' 'wasm-unsafe-eval'",
            "style-src 'self' 'unsafe-inline'",
            "img-src 'self' data: https:",
            "connect-src 'self'",
            "font-src 'self'",
            "frame-ancestors 'none'",
            "object-src 'none'",
            "base-uri 'none'",
            "form-action 'self'",
        ]
    )
    # ``nosniff`` rides every HTML document response (the SPA shell + the OAuth pages):
    # the browser must honor the declared ``text/html`` type and never MIME-sniff a
    # served body into a script, complementing the CSP ``script-src`` gate. The
    # ``frame-ancestors 'none'`` directive above is the clickjacking (frame) policy.
    return {"content-security-policy": csp, "x-content-type-options": "nosniff"}


def _escape_json_for_html(payload: str) -> str:
    """Render ``<``, ``>``, ``&``, ``/`` as ``\\u`` escapes so a manifest string
    can never close the <script> block (``</script>``) or open a new tag in the
    PUBLIC index.html. The browser's JSON parser decodes the escapes back, so URL
    values stay correct."""
    return payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026").replace("/", "\\u002f")


def render_importmap_script(registry: StudioPluginRegistry, nonce: str) -> str:
    payload = json.dumps(registry.import_map(), separators=(",", ":"), sort_keys=True)
    return f'<script type="importmap" nonce="{nonce}">{_escape_json_for_html(payload)}</script>'


def inject_importmap(html: str, registry: StudioPluginRegistry, nonce: str) -> str:
    """Replace the anchor token with the nonce-stamped importmap script. A missing
    anchor is LOUD — shipping index.html without the map would leave the app dead
    on unresolved bare specifiers."""
    if IMPORTMAP_ANCHOR not in html:
        raise StudioPluginError(f"served index.html is missing the {IMPORTMAP_ANCHOR!r} import-map anchor")
    return html.replace(IMPORTMAP_ANCHOR, render_importmap_script(registry, nonce), 1)
