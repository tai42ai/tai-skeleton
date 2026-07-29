"""Studio SPA serving primitives: CSP nonce, security headers, content-type map,
and the injection-safe import-map rendering."""

from __future__ import annotations

import json

import pytest

import tai42_skeleton.plugins.registry as reg
from tai42_skeleton.plugins.registry import StudioPluginError, StudioPluginRegistry, build_registry
from tai42_skeleton.plugins.serving import (
    IMPORTMAP_ANCHOR,
    asset_content_type,
    generate_nonce,
    inject_importmap,
    render_importmap_script,
    security_headers,
)

# -- Content-type map --------------------------------------------------------


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("index-a1b2.js", "text/javascript"),
        ("chunk.mjs", "text/javascript"),
        ("style.css", "text/css"),
        ("bundle.js.map", "application/json"),
        ("codec.wasm", "application/wasm"),
        ("data.bin", "application/octet-stream"),
        ("evil.html", "application/octet-stream"),  # never text/html on the asset route
    ],
)
def test_asset_content_type(filename, expected):
    assert asset_content_type(filename) == expected


# -- Nonce -------------------------------------------------------------------


def test_nonce_is_fresh_each_call():
    assert generate_nonce() != generate_nonce()


def test_nonce_has_entropy():
    # token_urlsafe(16) -> >=128 bits, url-safe (no chars needing HTML escaping).
    nonce = generate_nonce()
    assert len(nonce) >= 20
    assert all(c.isalnum() or c in "-_" for c in nonce)


# -- Security headers --------------------------------------------------------


def test_security_headers_csp_directives():
    csp = security_headers("NONCEVALUE")["content-security-policy"]
    assert "default-src 'none'" in csp
    assert "script-src 'self' 'nonce-NONCEVALUE' 'wasm-unsafe-eval'" in csp
    assert "frame-ancestors 'none'" in csp
    assert "object-src 'none'" in csp
    assert "base-uri 'none'" in csp
    assert "form-action 'self'" in csp
    assert "style-src 'self' 'unsafe-inline'" in csp
    # Remote https media images render directly; http: stays excluded (mixed content).
    assert "img-src 'self' data: https:" in csp


# -- Import-map injection ----------------------------------------------------


def _registry() -> StudioPluginRegistry:
    return StudioPluginRegistry(
        plugins={},
        vendor_integrity_by_url={
            "/vendor/react.js": "sha384-" + "A" * 64,
            "/vendor/studio-sdk-host.js": "sha384-" + "C" * 64,
        },
    )


def test_import_map_includes_studio_sdk_host(tmp_path):
    """The host registry API is a distinct import-map specifier resolving to its
    own served vendor asset, with an SRI hash derived from that file's real bytes.
    Fails if ``/host`` were dropped from VENDOR_MODULES (KeyError) or hashed from
    the wrong file (mismatch)."""
    # Give each vendor file distinct bytes so the host integrity is uniquely tied
    # to its own on-disk file.
    for rel in reg.VENDOR_MODULES.values():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(f"export const m = {rel!r};\n", encoding="utf-8")
    imap = build_registry([], str(tmp_path)).import_map()
    assert imap["imports"]["@tai42/studio-sdk"] == "/vendor/studio-sdk.js"
    assert imap["imports"]["@tai42/studio-sdk/host"] == "/vendor/studio-sdk-host.js"
    expected = reg._hash_file(tmp_path / "vendor" / "studio-sdk-host.js")
    assert imap["integrity"]["/vendor/studio-sdk-host.js"] == expected


def test_render_importmap_is_valid_json_after_unescape():
    script = render_importmap_script(_registry(), "N0NCE")
    assert script.startswith('<script type="importmap" nonce="N0NCE">')
    body = script[script.index(">") + 1 : script.rindex("<")]
    # The browser JSON parser decodes the \u escapes; the payload stays valid JSON
    # with correct URL values.
    parsed = json.loads(body)
    assert parsed["imports"]["react"] == "/vendor/react.js"


def test_importmap_escapes_script_breakout():
    """A hostile manifest string containing </script> must NOT break out of the
    inline script block into the PUBLIC index.html."""
    registry = StudioPluginRegistry(
        plugins={},
        vendor_integrity_by_url={"/vendor/x.js</script><script>alert(1)</script>": "sha384-" + "B" * 64},
    )
    script = render_importmap_script(registry, "N")
    # Isolate the JSON body (between the opening > and the wrapper's closing <).
    body = script[script.index(">") + 1 : script.rindex("<")]
    # The raw breakout sequence must not appear in the body; < and / are \u-escaped.
    assert "</script>" not in body
    assert "<" not in body
    assert "\\u003c" in body
    assert "\\u002f" in body


def test_inject_importmap_replaces_anchor_before_module_scripts():
    html = f'<head>{IMPORTMAP_ANCHOR}<script type="module" src="/app.js"></script></head>'
    out = inject_importmap(html, _registry(), "N")
    assert out.count('type="importmap"') == 1
    assert out.index('type="importmap"') < out.index('type="module"')
    assert IMPORTMAP_ANCHOR not in out


def test_inject_importmap_missing_anchor_is_loud():
    with pytest.raises(StudioPluginError, match="import-map anchor"):
        inject_importmap("<head></head>", _registry(), "N")
