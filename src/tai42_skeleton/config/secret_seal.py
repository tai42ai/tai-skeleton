"""Manifest secret seal — retag resolved ``!ENV`` values back to their markers and
refuse any stranded resolved secret before a manifest document persists.

A manifest's only read surface for the ``mcp`` section is the RESOLVED view
(``!ENV`` markers materialized), so the natural round-trip — read the resolved view,
edit it, post it back — carries resolved secret values (an auth header, a stdio
``env``) that would otherwise bake to ``manifest.yml`` as plaintext, destroying the
operator's ``!ENV`` marker. :func:`seal_resolved_secrets` closes that hole for every
feature writer and every config mode: it walks the outgoing document against the
CURRENT manifest's PRESERVED (markers) and EXPANDED (resolved) views, restores the
``!ENV`` marker wherever an outgoing leaf still equals a currently-resolved secret,
and then raises if any resolved secret with no marker origin in the current document
would still reach disk.

The retag pairs list elements by a stable shared identity key rather than by
position, so a reorder / insert / delete in a marker-bearing list does not misalign
indices and restore the wrong element's marker. The leak net then scans exactly the
subtrees the retag could not descend into (a new / renamed key, an unmatched or
ambiguous list element, a changed or restructured former-marker leaf), so an
unchanged resolved secret can never be stranded as plaintext while a value the
operator genuinely composed is left untouched.
"""

from __future__ import annotations

from typing import Any

# The prefix of an ``!ENV`` marker string in the preserved manifest view (kept in
# sync with the config manager's preserved-view convention).
_ENV_MARKER_PREFIX = "!ENV "


class ResolvedSecretError(ValueError):
    """A resolved ``!ENV`` secret would persist to the manifest as plaintext.

    A :class:`ValueError` subclass so the operations layer's ``except ValueError``
    maps it to a 400 — a client that posts a resolved secret with no marker origin
    is a bad request, not a server fault.
    """


def seal_resolved_secrets(document: dict[str, Any], preserved: dict[str, Any], expanded: dict[str, Any]) -> None:
    """Retag *document* in place against the current manifest, then refuse a leak.

    *document* is the outgoing PRESERVED-view document about to persist; *preserved*
    is the CURRENT persisted manifest in its preserved view (``!ENV`` markers kept as
    ``"!ENV <expr>"`` marker strings) and *expanded* is that same current manifest
    with its markers RESOLVED — the caller resolves it through the config layer's one
    ``parse_config`` / ``dump_manifest`` resolution, so this helper stays a pure
    structural walk. With the two views:

    * RETAG — every leaf in *document* whose value equals a currently-resolved secret
      is put back as the corresponding ``!ENV`` marker, so a resolved round-trip
      silently preserves the operator's markers. A leaf the operator genuinely
      changed keeps its new literal.
    * LEAK NET — after the retag, if any leaf still equals a resolved secret that has
      NO marker origin at its position in the current document (a genuinely stranded
      resolved secret), raise :class:`ResolvedSecretError` naming the key path and
      refuse to persist. A pure in-place mutation whose leaves already carry markers
      is a retag no-op and passes the net untouched.

    *document* is mutated in place (markers restored); the caller persists it.
    """
    unmatched: list[tuple[str, Any]] = []
    _retag_env_markers(document, preserved, expanded, unmatched, path="")
    secrets = _resolved_secrets(preserved, expanded)
    if not secrets:
        return
    for path, subtree in unmatched:
        if _contains_secret(subtree, secrets):
            raise ResolvedSecretError(
                f"refusing to persist manifest: a resolved !ENV secret would be written as "
                f"plaintext at {path or '<root>'!r} (a secret-bearing key or entry was likely "
                f"renamed, added, or restructured); re-apply the !ENV marker for that key or entry"
            )


def _retag_env_markers(
    incoming: Any, preserved: Any, expanded: Any, unmatched: list[tuple[str, Any]], *, path: str
) -> Any:
    """Restore ``!ENV`` markers in *incoming* against the current manifest.

    Walks *incoming*, *preserved* (markers kept as ``!ENV <expr>`` strings), and
    *expanded* (markers resolved) in parallel — recursing dicts AND lists — and, for
    every leaf the operator left unchanged (its incoming value still equals the
    resolved value), puts the marker back; a leaf the operator edited keeps its new
    literal value. List elements are paired by a stable shared identity key (the
    first of ``title``, ``name``, ``module``, ``id`` present in both entries and
    unambiguous among the unused candidates), else by deep equality, so a reorder /
    insert / delete does not misalign indices.

    Every subtree the retag returns WITHOUT having descended into and re-tagged is
    appended to *unmatched* with its key path — the complete set of places an
    unchanged secret can strand as plaintext: a new / renamed key, an unmatched (or
    ambiguous duplicate-identity) list element, a changed or restructured
    former-marker leaf, and any fallthrough leaf whose value differs from the resolved
    value. A corresponded position the retag fully descends is never appended, so a
    leaf correctly re-tagged to its marker never false-positives. Mutates the
    *incoming* structure in place and returns it.
    """
    if isinstance(preserved, str) and preserved.startswith(_ENV_MARKER_PREFIX):
        if expanded == incoming:
            # Unchanged secret leaf: restore the operator's ``!ENV`` marker.
            return preserved
        # The former-marker leaf changed or was restructured, so the retag cannot
        # re-tag it. Record it for the leak scan in case its new value is an unchanged
        # secret carried over from another (identity-swapped) entry.
        unmatched.append((path, incoming))
        return incoming
    if isinstance(incoming, dict) and isinstance(preserved, dict) and isinstance(expanded, dict):
        for key in list(incoming):
            child = f"{path}.{key}" if path else str(key)
            if key in preserved and key in expanded:
                incoming[key] = _retag_env_markers(incoming[key], preserved[key], expanded[key], unmatched, path=child)
            else:
                # New or renamed key: absent from the current view, so the retag has no
                # marker to descend into. Record the whole subtree for the leak scan.
                unmatched.append((child, incoming[key]))
        return incoming
    if isinstance(incoming, list) and isinstance(preserved, list) and isinstance(expanded, list):
        used: set[int] = set()
        for i, element in enumerate(incoming):
            child = f"{path}[{i}]"
            j = _match_list_element(element, expanded, used)
            if j is None:
                unmatched.append((child, element))
                continue
            used.add(j)
            incoming[i] = _retag_env_markers(element, preserved[j], expanded[j], unmatched, path=child)
        return incoming
    # Fallthrough: no dict/list/marker branch could descend here. Append *incoming*
    # unless it still equals the resolved value — an UNCHANGED value (``incoming ==
    # expanded``) strands no secret, while a CHANGED or mis-aligned value is appended
    # and scanned; the scan raises only if that value equals a resolved secret.
    if incoming != expanded:
        unmatched.append((path, incoming))
    return incoming


def _identity_key(incoming: dict[str, Any], candidate: dict[str, Any]) -> str | None:
    """The first shared identity key of two dicts, or ``None`` if they share none."""
    for key in ("title", "name", "module", "id"):
        if key in candidate and key in incoming:
            return key
    return None


def _match_list_element(element: Any, expanded: list[Any], used: set[int]) -> int | None:
    """Index in *expanded* that identifies *element*, or ``None`` if unmatched.

    Prefers a dict-to-dict match on a shared identity key, but only when that identity
    value is UNAMBIGUOUS — exactly one still-unused candidate dict shares it. A
    duplicate identity is rejected so a marker is never restored against the wrong
    element's secret; it then falls through to the deep-equality check, which still
    pairs an exactly-equal / unchanged element safely. Skips consumed indices so each
    is paired at most once; returns ``None`` when nothing matches.
    """
    if isinstance(element, dict):
        identity_matches = [
            j
            for j, candidate in enumerate(expanded)
            if j not in used
            and isinstance(candidate, dict)
            and (key := _identity_key(element, candidate)) is not None
            and element[key] == candidate[key]
        ]
        if len(identity_matches) == 1:
            return identity_matches[0]
    for j, candidate in enumerate(expanded):
        if j not in used and candidate == element:
            return j
    return None


def _resolved_secrets(preserved: Any, expanded: Any) -> set[str]:
    """The set of resolved secret strings in the current manifest.

    Walks *preserved* (markers) and *expanded* (resolved) in parallel — recursing
    dicts and lists (paired by index, since only the value set is gathered) — and
    collects every *expanded* leaf whose *preserved* leaf is an ``!ENV`` marker
    string. Empty strings are excluded so they never trip the plaintext-leak net.
    """
    secrets: set[str] = set()
    if isinstance(preserved, str) and preserved.startswith(_ENV_MARKER_PREFIX):
        if isinstance(expanded, str) and expanded:
            secrets.add(expanded)
        return secrets
    if isinstance(preserved, dict) and isinstance(expanded, dict):
        for key in preserved:
            if key in expanded:
                secrets |= _resolved_secrets(preserved[key], expanded[key])
    elif isinstance(preserved, list) and isinstance(expanded, list):
        # Both are parses of the SAME manifest, so their lists are always equal
        # length — a mismatch is a broken invariant and raises loudly.
        for pre, exp in zip(preserved, expanded, strict=True):
            secrets |= _resolved_secrets(pre, exp)
    return secrets


def _contains_secret(node: Any, secrets: set[str]) -> bool:
    """True if any string leaf in *node* is one of the resolved *secrets*.

    Recurses dicts, lists, and strings. A secret matches only as an EXACT string leaf
    — not as a substring, and not as a value coerced to a non-``str`` — because
    ``!ENV`` only tags scalar strings, so a preserved secret always reappears as a
    whole string leaf when it is the thing that leaked.
    """
    if isinstance(node, str):
        return node in secrets
    if isinstance(node, dict):
        return any(_contains_secret(value, secrets) for value in node.values())
    if isinstance(node, list):
        return any(_contains_secret(item, secrets) for item in node)
    return False
