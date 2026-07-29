"""File-based configuration manager.

Implements :class:`~tai42_contract.config.manager.ConfigManager` for the ``file``
config mode.  Reads and writes ``.env`` files and ``manifest.yml`` on the
local filesystem.
"""

import copy
import io
import logging
import os
import re
import sys
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, cast

from dotenv import dotenv_values
from pyaml_env import parse_config
from ruamel.yaml.comments import CommentedMap
from tai42_contract.config.manager import ConfigManager
from tai42_kit.utils.data import (
    load_manifest,
    merge_and_dump_manifest,
)

logger = logging.getLogger(__name__)


def _acquire_exclusive_lock(fd: int) -> None:
    """Take a blocking exclusive advisory lock on *fd*.

    POSIX uses ``fcntl.flock(LOCK_EX)`` — a genuine, blocking, cross-process
    mutex. ``fcntl`` is POSIX-only and this module is reachable from the
    always-imported ``config`` package, so the import is done here (never at module
    top level) to keep the package importable on Windows.

    On Windows ``fcntl`` is unavailable, so a config write cannot be serialized
    against a concurrent writer. Rather than proceed unlocked and risk a lost
    update, the write is refused with a :class:`RuntimeError` naming the platform
    limitation — a missing lock is a fail-loud condition, not a warning.
    """
    if sys.platform == "win32":
        raise RuntimeError(
            "config file locking requires fcntl, which is POSIX-only and "
            "unavailable on Windows; a serialized config write cannot be "
            "guaranteed here, so the write is refused (run on a POSIX host)."
        )
    import fcntl

    fcntl.flock(fd, fcntl.LOCK_EX)


# An env key is a shell-identifier: a letter/underscore then letters/digits/
# underscores. Anything else (a newline, ``=``, a space) could inject a second
# assignment or corrupt the parse, so it is rejected loudly.
_ENV_KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# The prefix of an ``!ENV`` marker string in the preserved manifest view.
_ENV_MARKER_PREFIX = "!ENV "


def _dotenv_serialize_value(value: str) -> str:
    """Serialize *value* as a double-quoted ``.env`` literal that ``dotenv_values``
    parses back to the exact string — the write side of :meth:`read_env`.

    Backslash is escaped first (so real backslashes survive), then the double
    quote and newline/carriage-return that would otherwise break out of the quote
    or split the line. Every other character rides through the quotes verbatim.
    """
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r")
    return f'"{escaped}"'


class FileConfigManager(ConfigManager):
    """Config backend that reads/writes local ``.env`` and manifest files.

    ``config_dir_path`` resolves to the constructor arg, then
    ``TAI_CONFIG_DIR_PATH``, then ``"/app"``; every anchored path (env, manifest,
    defaults) is rooted there. ``TAI_MANIFEST_PATH`` overrides the manifest
    path.
    """

    def __init__(self, config_dir_path: str | None = None) -> None:
        self._config_dir_path = config_dir_path or os.environ.get("TAI_CONFIG_DIR_PATH", "").strip() or "/app"

    @property
    def _env_path(self) -> str:
        return os.path.join(self._config_dir_path, ".env")

    @property
    def _manifest_path(self) -> str:
        return os.getenv("TAI_MANIFEST_PATH") or os.path.join(self._config_dir_path, "manifest.yml")

    @property
    def _defaults_manifest_path(self) -> str:
        return os.path.join(self._config_dir_path, "templates", "manifest.yml")

    @contextmanager
    def _file_lock(self, path: str) -> Iterator[None]:
        """Serialize a read-modify-write on *path* across processes.

        Takes a blocking exclusive POSIX ``flock`` on a SIDECAR lock file
        (``<path>.lock``, created ``0600``) held for the whole read → modify →
        atomic-write span the caller wraps, so a concurrent worker's RMW cannot
        interleave and lose an update. The lock subject is the sidecar, never
        *path* itself: the atomic write replaces *path* with ``os.replace``, which
        swaps the target's inode out from under any lock held on it, making a lock
        on the target useless.

        This is the lock primitive; :meth:`_manifest_transaction` builds the
        manifest read-modify-write span on top of it, and :meth:`write_env` uses it
        for the env file.

        On Windows ``fcntl`` is unavailable, so acquiring the lock raises rather
        than proceeding unserialized. An ``OSError`` from opening the lock file or
        acquiring the lock propagates loudly; there is no timeout or retry loop (the
        critical section is a sub-millisecond atomic write).
        """
        fd = os.open(f"{path}.lock", os.O_RDWR | os.O_CREAT, 0o600)
        try:
            _acquire_exclusive_lock(fd)
            yield
        finally:
            # Closing the descriptor releases the flock.
            os.close(fd)

    @contextmanager
    def _manifest_transaction(self) -> Iterator[None]:
        """Hold the manifest sidecar lock across a whole read → mutate → write span.

        :meth:`write_manifest`, :meth:`mutate_manifest`, and
        :meth:`replace_manifest` run their ENTIRE read-modify-write under this one
        lock, so a concurrent worker cannot interleave between another writer's read
        and write and lose an update.
        """
        with self._file_lock(self._manifest_path):
            yield

    # -- Environment configuration -------------------------------------------

    def read_env(self) -> dict[str, str]:
        """Parse the ``.env`` file into a ``{key: value}`` dict.

        Delegates to :func:`dotenv.dotenv_values`, which handles ``export``,
        inline comments, quoting, and escapes. Interpolation is disabled, so
        values are returned literally — a ``$`` or ``${VAR}`` in a value is not
        POSIX-expanded, making the double-quoted serializer a true inverse.
        """
        path = self._env_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"Env file not found: {path}")
        try:
            parsed = dotenv_values(path, interpolate=False)
        except OSError:
            logger.error("Failed to read env file: %s", path, exc_info=True)
            raise
        # Drop valueless keys (``KEY=`` with nothing after) — None values.
        return {k: v for k, v in parsed.items() if v is not None}

    def write_env(self, config: dict[str, str]) -> None:
        """Merge *config* into the ``.env`` file.

        Every incoming key is validated against a shell-identifier charset
        (raising :class:`ValueError` otherwise), and every value is serialized as
        a double-quoted literal so a write followed by :meth:`read_env` is an exact
        round-trip — a newline, ``#``, quote, or leading/trailing space survives
        instead of injecting a key or being silently truncated. Existing keys not
        present in *config* are preserved; empty / ``None`` values are dropped.

        Before the file is written, the serialized content is re-parsed with the
        same parser as :meth:`read_env` (``dotenv_values`` with
        ``interpolate=False``, here reading from a ``StringIO`` stream rather than
        the file path) and every written value is compared to its parsed result; a
        value the parser cannot round-trip (which would silently drop that key and
        every key after it on the next reload) raises :class:`ValueError` naming the
        offending key, and no write happens.
        """
        for key in config:
            if not _ENV_KEY_RE.fullmatch(key):
                raise ValueError(f"invalid env key {key!r}: must match [A-Za-z_][A-Za-z0-9_]*")
        path = self._env_path
        # The whole read-modify-write (re-read existing → merge → atomic write) runs
        # under the env sidecar lock so a concurrent worker's write cannot merge
        # against a stale base and drop this write's keys.
        with self._file_lock(path):
            existing: dict[str, str] = {}
            if os.path.exists(path):
                existing = self.read_env()
            merged = {**existing, **config}
            written = {k: v for k, v in merged.items() if v is not None and v != ""}
            lines = [f"{k}={_dotenv_serialize_value(v)}" for k, v in written.items()]
            content = "\n".join(lines) + ("\n" if lines else "")
            # Re-parse the serialized content through the same parser as read_env
            # (``dotenv_values`` with ``interpolate=False``, over a StringIO stream)
            # and confirm every written key round-trips. python-dotenv exposes no
            # double-quoted representation for some values (e.g. a value ending in an
            # odd number of backslashes): its parser pairs the trailing backslash with
            # the closing quote and drops that key AND every key after it. A key
            # serialized last hides the fault until the next write appends a key after
            # it, so the content is probed with a trailing sentinel binding to exercise
            # that position now — any value that cannot round-trip fails loudly here
            # rather than silently corrupting the store on a later reload.
            # The sentinel key is grown with ``_`` until it collides with no written
            # key, so a caller that legitimately writes a key literally named
            # ``_TAI_ENV_ROUNDTRIP_PROBE`` is not shadowed by the probe (dotenv is
            # last-wins) and thereby falsely flagged.
            probe_key = "_TAI_ENV_ROUNDTRIP_PROBE"
            while probe_key in written:
                probe_key += "_"
            probe = f'{content}{probe_key}="0"\n'
            reparsed = dotenv_values(stream=io.StringIO(probe), interpolate=False)
            corrupted = [key for key, value in written.items() if reparsed.get(key) != value]
            if corrupted:
                names = ", ".join(repr(key) for key in corrupted)
                raise ValueError(f"env value for {names} cannot be round-tripped through the .env format")
            try:
                self._atomic_write(path, content)
            except OSError:
                logger.error("Failed to write env file: %s", path, exc_info=True)
                raise

    # -- Manifest configuration ----------------------------------------------

    # Two YAML views of the manifest: the runtime expands ``!ENV`` tags to their
    # env values; the write-merge preserves them as tags so the operator-authored
    # placeholder survives a round-trip (instead of baking the secret to disk).

    def _load_yaml_expanded(self, path: str) -> dict:
        """``!ENV`` tags expanded to their resolved values — runtime view."""
        with open(path) as fh:
            return parse_config(data=fh.read()) or {}

    def _load_yaml_preserved(self, path: str) -> CommentedMap:
        """``!ENV`` tags preserved as ``"!ENV <expr>"`` marker strings — round-trip
        view. Comments, key ordering, and formatting are kept for a later dump."""
        with open(path) as fh:
            return load_manifest(fh.read())

    def read_manifest(self) -> dict:
        """Read ``manifest.yml`` with ``!ENV`` tags EXPANDED (runtime view)."""
        path = self._manifest_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"Manifest not found: {path}")
        return self._load_yaml_expanded(path)

    def read_manifest_preserved(self) -> dict:
        """Read ``manifest.yml`` with ``!ENV`` tags PRESERVED as ``"!ENV <expr>"``
        marker strings (round-trip view) — no secret values are resolved."""
        path = self._manifest_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"Manifest not found: {path}")
        return self._load_yaml_preserved(path)

    def read_defaults_manifest(self) -> dict:
        """Read template defaults with ``!ENV`` tags EXPANDED (runtime view).

        A missing defaults file is optional and yields ``{}``; a malformed one
        raises so the broken template surfaces instead of silently dropping the
        defaults from the three-way write merge.
        """
        path = self._defaults_manifest_path
        if not os.path.exists(path):
            return {}
        return self._load_yaml_expanded(path)

    def _retag_env_markers(self, incoming, preserved, expanded, unmatched: list):
        """Restore ``!ENV`` markers in *incoming* against the current manifest.

        Callers read the EXPANDED manifest, mutate it, and hand the whole dict
        back; without this the resolved secret would be dumped verbatim and the
        operator's placeholder destroyed. Walking *incoming*, *preserved* (tags
        kept as ``!ENV <expr>`` markers), and *expanded* (tags resolved) in
        parallel — recursing through nested dicts AND lists — every leaf the
        operator left unchanged (its incoming value still equals the resolved
        value) is put back as the marker; a leaf the operator edited keeps its new
        literal value.

        List elements are paired by stable identity rather than position, so a
        reorder / insert / delete in an ``!ENV``-bearing list does not misalign
        indices and bake another element's secret to disk. For each incoming
        element the matching preserved/expanded pair is found by an UNAMBIGUOUS
        shared identity key (the first of ``title``, ``name``, ``module``, ``id``
        present in BOTH the incoming and current entries, whose value is unique
        among the still-unused candidates) when both are dicts, else by deep
        equality of an untouched element; each preserved/expanded index is consumed
        at most once, and an incoming element with no match (operator-added /
        unrecognizable / ambiguous duplicate identity) is left unchanged since it
        carries no marker to restore.

        Every subtree retag returns WITHOUT having descended into and re-tagged its
        structure is appended to *unmatched* — the complete set of places an
        unchanged ``!ENV`` secret can be stranded as plaintext: an incoming dict key
        absent from the current view (a new or renamed key); an incoming list element
        not matched to a current entry (including an ambiguous duplicate-identity
        element); a former-``!ENV`` leaf whose incoming value changed or was
        restructured away from the marker; and any other fallthrough leaf whose value
        differs from the resolved value. The fallthrough applies one symmetric rule:
        append ``incoming`` unless ``incoming == expanded``. An UNCHANGED value
        (``incoming == expanded``) strands no secret — it is either a marker already
        handled above or an unchanged plain literal — and is not appended; a CHANGED or
        mis-aligned value (``incoming != expanded``) is appended and scanned. This one
        rule subsumes both restructure directions (a scalar↔container change makes the
        two differ) AND an identity-collision mis-match where ``_match_list_element``
        paired an entry against the wrong current entry, so at a leaf the unchanged
        secret differs from the mis-paired position's value and is caught.
        ``write_manifest`` scans exactly those subtrees for a leaked secret. A
        CORRESPONDED position retag fully descends — a key present in both views, a
        matched list element, or an unchanged ``!ENV`` leaf restored to its marker — is
        never appended: retag reaches every leaf under it, so nothing there can strand.
        The sole residual boundary is a plain field whose literal value is exactly a
        resolved secret's value at a corresponded position — a coincidence treated as
        the operator's own value to preserve precision, so it never false-positives.

        Mutates the *incoming* structure it is given in place and returns it;
        ``write_manifest`` passes a deep copy so the caller's dict is left
        untouched.
        """
        if isinstance(preserved, str) and preserved.startswith(_ENV_MARKER_PREFIX):
            if expanded == incoming:
                # Unchanged secret leaf: restore the operator's ``!ENV`` marker.
                return preserved
            # The former-marker leaf changed or was restructured, so retag cannot
            # re-tag it. Record it for the leak scan in case its new value is an
            # unchanged secret carried over from another (identity-swapped) entry.
            unmatched.append(incoming)
            return incoming
        if isinstance(incoming, dict) and isinstance(preserved, dict) and isinstance(expanded, dict):
            for key in list(incoming):
                if key in preserved and key in expanded:
                    incoming[key] = self._retag_env_markers(incoming[key], preserved[key], expanded[key], unmatched)
                else:
                    # New or renamed key: absent from the current view, so retag has
                    # no marker to descend into. Record the whole subtree for the leak
                    # scan and do not recurse it.
                    unmatched.append(incoming[key])
            return incoming
        if isinstance(incoming, list) and isinstance(preserved, list) and isinstance(expanded, list):
            used: set[int] = set()
            for i, element in enumerate(incoming):
                j = self._match_list_element(element, expanded, used)
                if j is None:
                    unmatched.append(element)
                    continue
                used.add(j)
                incoming[i] = self._retag_env_markers(element, preserved[j], expanded[j], unmatched)
            return incoming
        # Fallthrough: no dict/list/marker branch could descend here. Symmetric with
        # the marker branch above — append ``incoming`` unless it still equals the
        # resolved value. When ``incoming == expanded`` the value is UNCHANGED: either
        # a marker already handled above, or an unchanged plain literal that strands no
        # secret, so it is not appended. When ``incoming != expanded`` the value was
        # CHANGED or mis-aligned — an operator restructure (a scalar↔container change
        # makes the two differ), or an identity-collision mis-match that paired this
        # position against the wrong current entry — so it is appended and scanned; the
        # scan raises only if the changed value equals a resolved secret.
        if incoming != expanded:
            unmatched.append(incoming)
        return incoming

    @staticmethod
    def _identity_key(incoming: dict, candidate: dict) -> str | None:
        """The first shared identity key of two dicts, or ``None`` if they share none."""
        for key in ("title", "name", "module", "id"):
            if key in candidate and key in incoming:
                return key
        return None

    @staticmethod
    def _match_list_element(element, expanded: list, used: set[int]) -> int | None:
        """Index in *expanded* that identifies *element*, or ``None`` if unmatched.

        Prefers a dict-to-dict match on a shared identity key, but only when that
        identity value is UNAMBIGUOUS — exactly one still-unused candidate dict
        shares it. If two or more unused candidates carry the same identity value
        (a duplicate identity), the match is ambiguous and rejected so a marker is
        never restored against the wrong element's secret; it then falls through to
        the deep-equality check, which still pairs an exactly-equal / unchanged
        element safely. Skips indices already consumed by an earlier element so each
        is paired at most once; returns ``None`` when nothing matches (the element
        is then recorded as uncorresponded and scanned for a stranded secret).
        """
        if isinstance(element, dict):
            identity_matches = [
                j
                for j, candidate in enumerate(expanded)
                if j not in used
                and isinstance(candidate, dict)
                and (key := FileConfigManager._identity_key(element, candidate)) is not None
                and element[key] == candidate[key]
            ]
            if len(identity_matches) == 1:
                return identity_matches[0]
        for j, candidate in enumerate(expanded):
            if j not in used and candidate == element:
                return j
        return None

    @staticmethod
    def _resolved_secrets(preserved, expanded) -> set[str]:
        """The set of resolved secret strings in the current manifest.

        Walks *preserved* (``!ENV`` tags kept as markers) and *expanded* (tags
        resolved) in parallel — recursing dicts and lists (paired by index, since
        only the value set is gathered, not a mapping) — and collects every
        *expanded* leaf whose *preserved* leaf is an ``!ENV`` marker string. Empty
        strings are excluded so they never trip the plaintext-leak safety net.

        The set is collected from the CURRENT manifest's ``!ENV`` markers only; the
        defaults template contributes only preserved markers to the dumped output
        (never plaintext), so it cannot introduce a leak the net must scan for.
        """
        secrets: set[str] = set()
        if isinstance(preserved, str) and preserved.startswith(_ENV_MARKER_PREFIX):
            if isinstance(expanded, str) and expanded:
                secrets.add(expanded)
            return secrets
        if isinstance(preserved, dict) and isinstance(expanded, dict):
            for key in preserved:
                if key in expanded:
                    secrets |= FileConfigManager._resolved_secrets(preserved[key], expanded[key])
        elif isinstance(preserved, list) and isinstance(expanded, list):
            # Strict pairing asserts the two views share the file's shape: both are
            # parses of the SAME manifest, so their lists are always equal length —
            # a mismatch is a broken invariant and raises loudly rather than being
            # silently truncated.
            for pre, exp in zip(preserved, expanded, strict=True):
                secrets |= FileConfigManager._resolved_secrets(pre, exp)
        return secrets

    @staticmethod
    def _contains_secret(node, secrets: set[str]) -> bool:
        """True if any string leaf in *node* is one of the resolved *secrets*.

        Recurses dicts, lists, and strings so a resolved secret left un-re-tagged
        anywhere in the to-dump structure is detected before it reaches disk.

        A secret matches only as an EXACT string leaf — not as a substring, and not
        as a value coerced to a non-``str``. This boundary is intentional: an
        operator who deliberately embeds a secret substring into a new value is
        composing their own value, and ``!ENV`` only tags scalar strings, so a
        preserved secret always reappears as a whole string leaf when it is the
        thing that leaked.
        """
        if isinstance(node, str):
            return node in secrets
        if isinstance(node, dict):
            return any(FileConfigManager._contains_secret(value, secrets) for value in node.values())
        if isinstance(node, list):
            return any(FileConfigManager._contains_secret(item, secrets) for item in node)
        return False

    def write_manifest(self, manifest: dict) -> None:
        """Three-way merge (defaults + current + new) and write ``manifest.yml``.

        Defaults and current are read with ``!ENV`` tags preserved so the dump
        round-trips them. Because every caller hands back the EXPANDED manifest
        (``read_manifest`` → mutate → ``write_manifest``), a deep copy of the
        incoming dict is re-tagged against the current preserved/expanded views
        first, so an unchanged ``!ENV`` secret is written back as its placeholder
        rather than baked to disk as plaintext. The caller's dict is never
        mutated.

        The input contract is the current-expanded manifest view produced by
        ``read_manifest`` (not the expanded-defaults view from
        ``read_defaults_manifest``): the retag and leak net are computed against the
        CURRENT manifest, and defaults are merged only as preserved markers that
        never contribute plaintext in that flow.

        The whole read-modify-write (defaults/current preserved reads → three-way
        merge → atomic write) runs under the shared manifest transaction lock so a
        concurrent worker's write cannot merge against a stale ``current`` and drop
        this write's top-level keys.
        """
        with self._manifest_transaction():
            defaults: dict = {}
            if os.path.exists(self._defaults_manifest_path):
                defaults = self._load_yaml_preserved(self._defaults_manifest_path)

            current: CommentedMap = CommentedMap()
            # No current manifest means there is no retag to run: nothing mutates the
            # caller's dict, and ``merge_and_dump_manifest`` only reads it, so dumping
            # ``manifest`` directly (no deep copy) still leaves the caller's dict
            # untouched.
            to_dump: dict = manifest
            if os.path.exists(self._manifest_path):
                current = self._load_yaml_preserved(self._manifest_path)
                current_expanded = self._load_yaml_expanded(self._manifest_path)
                # Retag a deep copy in place so the caller's dict is never mutated,
                # collecting every uncorresponded subtree that retag could not descend
                # into (see ``_retag_env_markers``).
                to_dump = copy.deepcopy(manifest)
                unmatched: list = []
                self._retag_env_markers(to_dump, current, current_expanded, unmatched)
                # Safety net: retag re-tags only positions it can descend into, so a
                # structural edit or an identity-collision mis-match can strand an unchanged
                # ``!ENV`` leaf un-re-tagged and bake its resolved secret to disk as
                # plaintext. The scan covers exactly the subtrees ``_retag_env_markers``
                # returned WITHOUT fully re-tagging — incoming dict keys absent from the
                # current view (new/renamed keys), list elements not matched to a current
                # entry (including ambiguous duplicate-identity elements), a
                # changed/restructured former-``!ENV`` leaf, and any fallthrough leaf whose
                # value differs from the resolved value. The fallthrough follows one
                # symmetric rule — append unless ``incoming == expanded`` — which subsumes
                # both restructure directions (a scalar↔container change makes the two
                # differ) AND an identity-collision mis-match (the unchanged secret differs
                # from the mis-paired position's value). That is the complete set of places
                # a secret can strand (an unchanged secret can only ride through a subtree
                # retag did not descend), so an unchanged ``!ENV`` secret can never reach
                # disk as plaintext. Precision is preserved: a corresponded position retag
                # fully re-tags is never scanned, so a leaf correctly re-tagged to its
                # marker never false-positives; the sole residual boundary is a plain field
                # whose literal value is exactly a resolved secret's value at a corresponded
                # position (coincidental — treated as the operator's own value); and though
                # a former-secret leaf edited to a NON-secret literal (or a restructured
                # non-secret field) is scanned, it only raises if its value equals a
                # resolved secret (fail-closed) — the common edit-to-a-different-literal
                # drops the value and does not trip. Raise before any write so nothing
                # reaches disk.
                secrets = self._resolved_secrets(current, current_expanded)
                if secrets and any(self._contains_secret(sub, secrets) for sub in unmatched):
                    raise ValueError(
                        "refusing to write manifest: an !ENV secret would be written as "
                        "plaintext (a secret-bearing key or entry was likely renamed or "
                        "added); re-apply the !ENV marker for that key or entry"
                    )
            content = merge_and_dump_manifest(defaults, current, to_dump)
            path = self._manifest_path
            try:
                self._atomic_write(path, content)
            except OSError:
                logger.error("Failed to write manifest file: %s", path, exc_info=True)
                raise

    def mutate_manifest(self, mutator: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        """Atomically read-modify-write the manifest under the transaction lock.

        Inside the shared manifest transaction: the current manifest is read in the
        PRESERVED view (every ``!ENV <expr>`` kept as its literal marker string, so
        no secret is resolved), ``mutator`` edits that round-trip document IN PLACE,
        and the edited document is dumped (defaults backfilling only keys it is
        missing) and written atomically. Untouched keys keep their values, comments,
        and ordering; no other writer's change can interleave between the read and
        the write. Returns the persisted preserved-view document.

        Because the mutator only ever sees marker strings, a resolved secret can
        never be present to bake to disk — the ``!ENV`` preservation is structural
        here, not a post-hoc scan. A ``mutator`` exception aborts the transaction
        with nothing written and propagates unchanged. Validation of the mutated
        document against the resolved view is the caller's concern.
        """
        with self._manifest_transaction():
            document: CommentedMap = CommentedMap()
            if os.path.exists(self._manifest_path):
                document = self._load_yaml_preserved(self._manifest_path)
            # A mutator exception propagates here, before any write — nothing lands.
            mutator(document)
            defaults: dict = {}
            if os.path.exists(self._defaults_manifest_path):
                defaults = self._load_yaml_preserved(self._defaults_manifest_path)
            content = merge_and_dump_manifest(defaults, document, {})
            path = self._manifest_path
            try:
                self._atomic_write(path, content)
            except OSError:
                logger.error("Failed to write manifest file: %s", path, exc_info=True)
                raise
        return document

    def replace_manifest(self, document: dict[str, Any]) -> dict[str, Any]:
        """Atomically replace the whole persisted manifest under the transaction lock.

        Inside the shared manifest transaction: *document* becomes the entire stored
        manifest — a key absent from *document* is DELETED, nothing from the old
        manifest survives uninvited (defaults still backfill keys *document* is
        missing). The document is dumped verbatim and written atomically; it is not
        read or re-preserved, so building it from the preserved view (``!ENV`` marker
        strings, never resolved secrets) is the caller's obligation. Returns the
        persisted document (a copy of *document* with defaults backfilled); the
        caller's dict is left untouched.
        """
        with self._manifest_transaction():
            defaults: dict = {}
            if os.path.exists(self._defaults_manifest_path):
                defaults = self._load_yaml_preserved(self._defaults_manifest_path)
            persisted = cast("CommentedMap", copy.deepcopy(document))
            content = merge_and_dump_manifest(defaults, persisted, {})
            path = self._manifest_path
            try:
                self._atomic_write(path, content)
            except OSError:
                logger.error("Failed to write manifest file: %s", path, exc_info=True)
                raise
        return persisted

    @staticmethod
    def _atomic_write(path: str, content: str) -> None:
        """Write *content* to *path* atomically with ``0600`` permissions.

        A uniquely-named temp file is created ``0600`` in the target's own
        directory (so ``os.replace`` stays on one filesystem), flushed + fsynced,
        then renamed over the target — atomic on POSIX, so a crash mid-write never
        leaves a truncated secret store. The unique name avoids colliding with a
        temp left by a previously crashed write. Failures propagate loudly after
        the orphan temp is removed. After the rename, the target directory is
        fsynced so the rename itself is durable across a crash.
        """
        directory = os.path.dirname(path) or "."
        fd, tmp = tempfile.mkstemp(dir=directory)
        try:
            with os.fdopen(fd, "w") as fh:
                fh.write(content)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except OSError:
            try:
                os.unlink(tmp)
            except OSError:
                logger.error("Failed to remove temp file after a failed write: %s", tmp, exc_info=True)
            raise
        dir_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)


def build_config_manager() -> ConfigManager:
    """Provider entry point for the ``file`` config mode (the factory convention)."""
    return FileConfigManager()
