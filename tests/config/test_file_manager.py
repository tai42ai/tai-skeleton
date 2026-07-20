"""FileConfigManager read/write coverage over a faked filesystem (``tmp_path``).

Exercises the path resolution (constructor arg / ``TAI_CONFIG_DIR_PATH`` / ``/app``
default and the ``TAI_MANIFEST_PATH`` override), the ``.env`` read/write merge
(preserve + drop-empty), and the manifest read/write (``!ENV`` expansion vs
preservation, missing-file behavior, three-way write merge)."""

from __future__ import annotations

import copy
import multiprocessing
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
import yaml

import tai42_skeleton.config.file_manager as fm
from tai42_skeleton.config.file_manager import FileConfigManager

# --- cross-process lock helpers (module level so ``spawn`` can import them) --
#
# The RMW window (re-read current → merge → atomic write) is widened by a small
# sleep so two processes race deterministically; the ``_Unlocked*`` variants
# disable the flock to prove an update is lost WITHOUT it. Both variants exercise
# the real ``write_env`` / ``write_manifest`` bodies — only the read is slowed and
# (for the unlocked variants) the lock is neutered.

_LOCK_RACE_DELAY = 0.05


class _SlowEnvManager(FileConfigManager):
    def read_env(self) -> dict[str, str]:
        result = super().read_env()
        time.sleep(_LOCK_RACE_DELAY)
        return result


class _UnlockedSlowEnvManager(_SlowEnvManager):
    @contextmanager
    def _file_lock(self, path: str) -> Iterator[None]:
        yield


class _SlowManifestManager(FileConfigManager):
    def _load_yaml_expanded(self, path: str) -> dict:
        result = super()._load_yaml_expanded(path)
        time.sleep(_LOCK_RACE_DELAY)
        return result


class _UnlockedSlowManifestManager(_SlowManifestManager):
    @contextmanager
    def _file_lock(self, path: str) -> Iterator[None]:
        yield


def _env_writer(config_dir: str, prefix: str, count: int, use_lock: bool, barrier) -> None:
    mgr = _SlowEnvManager(config_dir) if use_lock else _UnlockedSlowEnvManager(config_dir)
    barrier.wait()
    for i in range(count):
        mgr.write_env({f"{prefix}{i}": str(i)})


def _manifest_writer(config_dir: str, prefix: str, count: int, use_lock: bool, barrier) -> None:
    mgr = _SlowManifestManager(config_dir) if use_lock else _UnlockedSlowManifestManager(config_dir)
    barrier.wait()
    for i in range(count):
        mgr.write_manifest({f"{prefix}{i}": i})


class _SlowPreservedManager(FileConfigManager):
    def _load_yaml_preserved(self, path: str):
        result = super()._load_yaml_preserved(path)
        time.sleep(_LOCK_RACE_DELAY)
        return result


class _UnlockedSlowPreservedManager(_SlowPreservedManager):
    @contextmanager
    def _file_lock(self, path: str) -> Iterator[None]:
        yield


def _mutate_writer(config_dir: str, key: str, use_lock: bool, barrier) -> None:
    mgr = _SlowPreservedManager(config_dir) if use_lock else _UnlockedSlowPreservedManager(config_dir)
    barrier.wait()
    mgr.mutate_manifest(lambda doc: doc.__setitem__(key, key))


def _single_env_write(config_dir: str, key: str, value: str) -> None:
    FileConfigManager(config_dir_path=config_dir).write_env({key: value})


def _run_two_writers(target, args_a: tuple, args_b: tuple) -> None:
    """Run ``target`` in two spawned processes, released together by a barrier."""
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(2)
    proc_a = ctx.Process(target=target, args=(*args_a, barrier))
    proc_b = ctx.Process(target=target, args=(*args_b, barrier))
    proc_a.start()
    proc_b.start()
    proc_a.join(timeout=60)
    proc_b.join(timeout=60)
    assert proc_a.exitcode == 0, "writer A did not exit cleanly"
    assert proc_b.exitcode == 0, "writer B did not exit cleanly"


# --- path resolution --------------------------------------------------------


def test_default_dir_is_app(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TAI_CONFIG_DIR_PATH", raising=False)
    monkeypatch.delenv("TAI_MANIFEST_PATH", raising=False)
    mgr = FileConfigManager()
    assert mgr._env_path == os.path.join("/app", ".env")
    assert mgr._manifest_path == os.path.join("/app", "manifest.yml")
    assert mgr._defaults_manifest_path == os.path.join("/app", "templates", "manifest.yml")


def test_config_dir_path_env_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("TAI_CONFIG_DIR_PATH", str(tmp_path))
    mgr = FileConfigManager()
    assert mgr._env_path == os.path.join(str(tmp_path), ".env")


def test_constructor_arg_wins(tmp_path) -> None:
    mgr = FileConfigManager(config_dir_path=str(tmp_path))
    assert mgr._env_path.startswith(str(tmp_path))


def test_tai_manifest_path_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    override = str(tmp_path / "custom.yml")
    monkeypatch.setenv("TAI_MANIFEST_PATH", override)
    mgr = FileConfigManager(config_dir_path=str(tmp_path))
    assert mgr._manifest_path == override


# --- env read/write ---------------------------------------------------------


def test_read_env_missing_raises(tmp_path) -> None:
    mgr = FileConfigManager(config_dir_path=str(tmp_path))
    with pytest.raises(FileNotFoundError, match="Env file not found"):
        mgr.read_env()


def test_read_env_parses_and_drops_valueless(tmp_path) -> None:
    # ``BARE`` (no ``=``) parses to None; ``EMPTY=`` parses to "".
    (tmp_path / ".env").write_text("A=1\nB=two\nEMPTY=\nBARE\n", encoding="utf-8")
    mgr = FileConfigManager(config_dir_path=str(tmp_path))
    parsed = mgr.read_env()
    assert parsed["A"] == "1"
    assert parsed["B"] == "two"
    # Valueless (None) keys are filtered out; an explicit empty string is kept.
    assert "BARE" not in parsed
    assert parsed["EMPTY"] == ""


def test_write_env_creates_file_when_absent(tmp_path) -> None:
    mgr = FileConfigManager(config_dir_path=str(tmp_path))
    mgr.write_env({"A": "1", "B": "2"})
    # Values are written as double-quoted dotenv literals so they round-trip.
    text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert text == 'A="1"\nB="2"\n'
    assert mgr.read_env() == {"A": "1", "B": "2"}


def test_write_env_round_trips_tricky_values(tmp_path) -> None:
    """A newline, ``#``, quote, and leading/trailing space survive a write→read
    cycle instead of injecting a key or being silently truncated."""
    mgr = FileConfigManager(config_dir_path=str(tmp_path))
    tricky = {
        "MULTILINE": "value\nINJECTED=1",
        "HASH": "hello #world",
        "QUOTED": 'has "quotes"',
        "SPACED": "  padded  ",
        "PEM": "-----BEGIN-----\nkeybody\n-----END-----",
    }
    mgr.write_env(tricky)
    assert mgr.read_env() == tricky


def test_write_env_rejects_malformed_key(tmp_path) -> None:
    mgr = FileConfigManager(config_dir_path=str(tmp_path))
    with pytest.raises(ValueError, match="invalid env key"):
        mgr.write_env({"BAD KEY": "1"})
    with pytest.raises(ValueError, match="invalid env key"):
        mgr.write_env({"A\nB": "1"})


def test_write_env_round_trips_dollar_values_without_interpolation(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Values containing ``$``, ``$var`` and ``${OTHER}`` survive a write→read
    cycle byte-identically — ``read_env`` must NOT POSIX-interpolate them, even
    when the referenced var exists in the process environment."""
    # Set OTHER in the environment to prove it is NOT substituted on read.
    monkeypatch.setenv("OTHER", "LEAKED")
    mgr = FileConfigManager(config_dir_path=str(tmp_path))
    values = {
        "DOLLAR": "a$b",
        "SHELL_VAR": "prefix-$var-suffix",
        "BRACED": "before ${OTHER} after",
    }
    mgr.write_env(values)
    assert mgr.read_env() == values


def test_write_env_rejects_value_that_cannot_round_trip(tmp_path) -> None:
    """A value ending in a single backslash cannot be represented as a
    double-quoted ``.env`` literal that the parser reads back — ``write_env``
    raises ``ValueError`` naming the key and leaves any existing store untouched
    instead of silently dropping that key (and every key after it) on reload."""
    mgr = FileConfigManager(config_dir_path=str(tmp_path))
    # Pre-existing store that must survive the rejected write.
    mgr.write_env({"KEEP": "safe"})
    before = (tmp_path / ".env").read_text(encoding="utf-8")

    with pytest.raises(ValueError, match=r"WINPATH.*cannot be round-tripped"):
        mgr.write_env({"WINPATH": "C:\\data\\"})

    # The write did not happen: the store is byte-for-byte unchanged.
    assert (tmp_path / ".env").read_text(encoding="utf-8") == before
    assert mgr.read_env() == {"KEEP": "safe"}


def test_write_env_rejects_value_that_cannot_round_trip_leaves_no_file(tmp_path) -> None:
    """When there is no pre-existing store, a value that cannot round-trip raises
    and no ``.env`` file is created."""
    mgr = FileConfigManager(config_dir_path=str(tmp_path))
    with pytest.raises(ValueError, match="cannot be round-tripped"):
        mgr.write_env({"WINPATH": "C:\\data\\"})
    assert not (tmp_path / ".env").exists()


def test_write_env_rejects_bad_value_in_last_position_via_sentinel(tmp_path) -> None:
    """A value that cannot round-trip is caught even when it is the LAST key written
    (its trailing backslash would otherwise pair with the closing quote only to be
    exposed by the next appended key). The trailing sentinel binding exercises that
    last position now, so the write raises whether the bad value is last or not."""
    mgr = FileConfigManager(config_dir_path=str(tmp_path))
    # Bad value in the LAST position: only the appended sentinel sits after it, so
    # the sentinel is what surfaces the fault.
    with pytest.raises(ValueError, match=r"WINPATH.*cannot be round-tripped"):
        mgr.write_env({"GOOD": "ok", "WINPATH": "C:\\data\\"})
    assert not (tmp_path / ".env").exists()

    # Same bad value NOT last (a good key follows it) also raises.
    with pytest.raises(ValueError, match=r"WINPATH.*cannot be round-tripped"):
        mgr.write_env({"WINPATH": "C:\\data\\", "GOOD": "ok"})
    assert not (tmp_path / ".env").exists()


def test_write_env_round_trips_even_backslash_and_tricky_values(tmp_path) -> None:
    """The round-trip guard is not over-strict: a value with an even number of
    internal backslashes plus the usual tricky characters still write and read
    back byte-identically."""
    mgr = FileConfigManager(config_dir_path=str(tmp_path))
    values = {
        "EVEN_BACKSLASH": "a\\b",
        "MULTILINE": "line\nINJECTED=1",
        "QUOTED": 'has "quotes"',
        "HASH": "trailing #comment",
        "DOLLAR": "a$b ${OTHER}",
        "SPACED": "  padded  ",
    }
    mgr.write_env(values)
    assert mgr.read_env() == values


def test_write_env_rejects_key_with_trailing_newline(tmp_path) -> None:
    """A key ending in a newline (or containing a space) is rejected before it can
    inject a stray line into ``.env``."""
    mgr = FileConfigManager(config_dir_path=str(tmp_path))
    with pytest.raises(ValueError, match="invalid env key"):
        mgr.write_env({"A\n": "1"})
    with pytest.raises(ValueError, match="invalid env key"):
        mgr.write_env({"A B": "1"})


def test_write_env_preserves_key_order_across_updates(tmp_path) -> None:
    """Updating an existing key rewrites its value in place instead of moving it
    to the top, so the on-disk key order stays stable."""
    mgr = FileConfigManager(config_dir_path=str(tmp_path))
    mgr.write_env({"A": "1", "B": "2", "C": "3"})
    mgr.write_env({"B": "new"})
    text = (tmp_path / ".env").read_text(encoding="utf-8")
    keys = [line.split("=", 1)[0] for line in text.splitlines() if line]
    assert keys == ["A", "B", "C"]
    assert mgr.read_env() == {"A": "1", "B": "new", "C": "3"}


def test_write_env_is_mode_0600(tmp_path) -> None:
    mgr = FileConfigManager(config_dir_path=str(tmp_path))
    mgr.write_env({"A": "1"})
    assert (os.stat(tmp_path / ".env").st_mode & 0o777) == 0o600


def test_write_env_preserves_existing_unrelated_keys(tmp_path) -> None:
    (tmp_path / ".env").write_text("KEEP=old\nA=stale\n", encoding="utf-8")
    mgr = FileConfigManager(config_dir_path=str(tmp_path))
    mgr.write_env({"A": "new"})
    parsed = mgr.read_env()
    # Provided key overwritten; unrelated key preserved.
    assert parsed["A"] == "new"
    assert parsed["KEEP"] == "old"


def test_write_env_drops_empty_and_none_values(tmp_path) -> None:
    mgr = FileConfigManager(config_dir_path=str(tmp_path))
    mgr.write_env({"A": "1", "B": "", "C": None})  # type: ignore[dict-item]
    parsed = mgr.read_env()
    assert parsed == {"A": "1"}


def test_write_env_empty_merge_writes_no_trailing_newline(tmp_path) -> None:
    mgr = FileConfigManager(config_dir_path=str(tmp_path))
    mgr.write_env({})
    assert (tmp_path / ".env").read_text(encoding="utf-8") == ""


# --- manifest read ----------------------------------------------------------


def test_read_manifest_missing_raises(tmp_path) -> None:
    mgr = FileConfigManager(config_dir_path=str(tmp_path))
    with pytest.raises(FileNotFoundError, match="Manifest not found"):
        mgr.read_manifest()


def test_read_manifest_expands_env_tags(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("MY_SECRET", "resolved-value")
    (tmp_path / "manifest.yml").write_text("token: !ENV ${MY_SECRET}\n", encoding="utf-8")
    mgr = FileConfigManager(config_dir_path=str(tmp_path))
    assert mgr.read_manifest()["token"] == "resolved-value"


def test_read_manifest_preserved_keeps_env_marker_and_hides_secret(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    # The preserved-tag view keeps each ``!ENV`` node as its literal marker string —
    # JSON-safe, and the resolved secret is never read.
    monkeypatch.setenv("SOME_VAR", "super-secret-value")
    (tmp_path / "manifest.yml").write_text("key: !ENV ${SOME_VAR}\n", encoding="utf-8")
    mgr = FileConfigManager(config_dir_path=str(tmp_path))
    preserved = mgr.read_manifest_preserved()
    assert preserved == {"key": "!ENV ${SOME_VAR}"}
    import json

    assert "super-secret-value" not in json.dumps(preserved)


def test_read_manifest_preserved_missing_raises(tmp_path) -> None:
    mgr = FileConfigManager(config_dir_path=str(tmp_path))
    with pytest.raises(FileNotFoundError, match="Manifest not found"):
        mgr.read_manifest_preserved()


def test_manifest_preserved_export_import_fresh_host_round_trip(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    # Fresh-host restore: the preserved view exported from one host imports into an
    # EMPTY config dir as a raw ``!ENV`` tag (not the baked secret), and the runtime
    # read then resolves it from env — the secret never touches disk.
    monkeypatch.setenv("SOME_VAR", "super-secret-value")
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "manifest.yml").write_text("key: !ENV ${SOME_VAR}\n", encoding="utf-8")
    source = FileConfigManager(config_dir_path=str(source_dir))
    exported = source.read_manifest_preserved()

    # A brand-new host with no existing manifest.
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    target = FileConfigManager(config_dir_path=str(target_dir))
    target.write_manifest(exported)

    raw = (target_dir / "manifest.yml").read_text(encoding="utf-8")
    assert "!ENV" in raw
    assert "super-secret-value" not in raw
    # The runtime view resolves the placeholder from env.
    assert target.read_manifest()["key"] == "super-secret-value"


def test_read_defaults_manifest_absent_returns_empty(tmp_path) -> None:
    mgr = FileConfigManager(config_dir_path=str(tmp_path))
    assert mgr.read_defaults_manifest() == {}


def test_read_defaults_manifest_expands(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("DEF_VAL", "xyz")
    templates = tmp_path / "templates"
    templates.mkdir()
    (templates / "manifest.yml").write_text("k: !ENV ${DEF_VAL}\n", encoding="utf-8")
    mgr = FileConfigManager(config_dir_path=str(tmp_path))
    assert mgr.read_defaults_manifest() == {"k": "xyz"}


def test_read_defaults_manifest_bad_yaml_raises(tmp_path) -> None:
    """A malformed defaults file raises rather than degrading to ``{}`` — a
    broken template must surface, not silently drop defaults from the write merge."""
    templates = tmp_path / "templates"
    templates.mkdir()
    (templates / "manifest.yml").write_text("a: [unterminated\n", encoding="utf-8")
    mgr = FileConfigManager(config_dir_path=str(tmp_path))
    with pytest.raises(yaml.YAMLError):
        mgr.read_defaults_manifest()


# --- manifest write ---------------------------------------------------------


def test_write_manifest_writes_merged_content(tmp_path) -> None:
    mgr = FileConfigManager(config_dir_path=str(tmp_path))
    mgr.write_manifest({"tools": [{"title": "t", "module": "m"}]})
    written = (tmp_path / "manifest.yml").read_text(encoding="utf-8")
    assert "tools" in written
    # Round-trips back through the expanded reader.
    assert mgr.read_manifest()["tools"][0]["module"] == "m"


def test_read_env_os_error_propagates(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """A read failure from the parser is logged and re-raised, never swallowed
    into an empty config."""
    (tmp_path / ".env").write_text("A=1\n", encoding="utf-8")  # exists() True
    import tai42_skeleton.config.file_manager as fm

    def boom(path, interpolate=True):
        raise OSError("disk error")

    monkeypatch.setattr(fm, "dotenv_values", boom)
    mgr = FileConfigManager(config_dir_path=str(tmp_path))
    with pytest.raises(OSError, match="disk error"):
        mgr.read_env()


def test_write_env_os_error_propagates(tmp_path) -> None:
    """A write failure (parent dir missing) is logged and re-raised."""
    missing = tmp_path / "nope"  # never created -> open() raises FileNotFoundError
    mgr = FileConfigManager(config_dir_path=str(missing))
    with pytest.raises(FileNotFoundError):
        mgr.write_env({"A": "1"})


def test_write_manifest_os_error_propagates(tmp_path) -> None:
    """A manifest write failure (parent dir missing) is logged and re-raised."""
    missing = tmp_path / "nope"
    mgr = FileConfigManager(config_dir_path=str(missing))
    with pytest.raises(FileNotFoundError):
        mgr.write_manifest({"tools": []})


def test_write_manifest_is_mode_0600(tmp_path) -> None:
    mgr = FileConfigManager(config_dir_path=str(tmp_path))
    mgr.write_manifest({"tools": []})
    assert (os.stat(tmp_path / "manifest.yml").st_mode & 0o777) == 0o600


def test_write_manifest_preserves_env_markers_through_edit_cycle(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """A read-expanded → mutate → write cycle must NOT bake a resolved ``!ENV``
    secret to disk or destroy the placeholder, including inside a nested mcp
    header map (a list-of-dicts)."""
    monkeypatch.setenv("MY_TOKEN", "super-secret")
    (tmp_path / "manifest.yml").write_text(
        "static_dir: /app/ui\n"
        "mcp:\n"
        "- title: remote\n"
        "  config:\n"
        "    url: http://localhost:9000/mcp\n"
        "    headers:\n"
        "      Authorization: !ENV ${MY_TOKEN}\n",
        encoding="utf-8",
    )
    mgr = FileConfigManager(config_dir_path=str(tmp_path))

    # Real callers read the EXPANDED view, mutate an unrelated key, and write back.
    manifest = mgr.read_manifest()
    assert manifest["mcp"][0]["config"]["headers"]["Authorization"] == "super-secret"
    manifest["static_dir"] = "/app/ui2"
    mgr.write_manifest(manifest)

    # The secret is not on disk and the placeholder tag survives (the dumper may
    # quote the tagged scalar, e.g. ``!ENV '${MY_TOKEN}'``).
    raw = (tmp_path / "manifest.yml").read_text(encoding="utf-8")
    assert "super-secret" not in raw
    assert "!ENV" in raw
    assert "MY_TOKEN" in raw
    # The unrelated edit landed, and the tag still expands on read.
    reread = mgr.read_manifest()
    assert reread["static_dir"] == "/app/ui2"
    assert reread["mcp"][0]["config"]["headers"]["Authorization"] == "super-secret"


def test_write_manifest_does_not_mutate_caller_dict(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """``write_manifest`` retags a deep copy, so the caller's dict is unchanged
    after the call (callers read → mutate → write and may reuse the dict)."""
    monkeypatch.setenv("MY_TOKEN", "super-secret")
    (tmp_path / "manifest.yml").write_text("token: !ENV ${MY_TOKEN}\n", encoding="utf-8")
    mgr = FileConfigManager(config_dir_path=str(tmp_path))
    # An expanded manifest with the resolved secret still in place (unchanged leaf).
    manifest = mgr.read_manifest()
    before = copy.deepcopy(manifest)
    mgr.write_manifest(manifest)
    assert manifest == before


def test_write_manifest_does_not_mutate_caller_dict_when_no_current_manifest(tmp_path) -> None:
    """With NO existing manifest file, ``write_manifest`` runs no retag and dumps the
    caller's dict directly, but still must not mutate it — the caller may reuse the
    dict after the call. (Complements the with-current-manifest non-mutation test.)"""
    mgr = FileConfigManager(config_dir_path=str(tmp_path))
    assert not (tmp_path / "manifest.yml").exists()
    manifest = {"tools": [{"title": "t", "module": "m"}], "static_dir": "/app/ui"}
    before = copy.deepcopy(manifest)
    mgr.write_manifest(manifest)
    assert manifest == before


def test_write_manifest_keeps_operator_edited_secret_value(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """When the operator actually CHANGES a previously-``!ENV`` leaf to a new
    literal, the new value is written (the marker is only restored for unchanged
    leaves)."""
    monkeypatch.setenv("MY_TOKEN", "super-secret")
    (tmp_path / "manifest.yml").write_text("token: !ENV ${MY_TOKEN}\n", encoding="utf-8")
    mgr = FileConfigManager(config_dir_path=str(tmp_path))
    manifest = mgr.read_manifest()
    manifest["token"] = "explicitly-new"
    mgr.write_manifest(manifest)
    raw = (tmp_path / "manifest.yml").read_text(encoding="utf-8")
    assert "explicitly-new" in raw
    assert "!ENV" not in raw


def test_write_manifest_allows_env_value_that_coincides_with_sibling_literal(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A resolved ``!ENV`` value that happens to equal an unrelated plain literal
    elsewhere in the manifest must NOT trip the leak net. The ``!ENV`` leaf is
    correctly re-tagged to its marker and the coincidental sibling literal lives in
    no unmatched subtree, so the scoped scan never sees it — the round-trip
    succeeds and the marker survives on disk."""
    # A low-entropy value is a realistic collision: an env-provided region equal to
    # a plain sibling ``default_region`` literal.
    monkeypatch.setenv("REGION", "us-east-1")
    (tmp_path / "manifest.yml").write_text(
        "region: !ENV ${REGION}\ndefault_region: us-east-1\nstatic_dir: /app/ui\n",
        encoding="utf-8",
    )
    mgr = FileConfigManager(config_dir_path=str(tmp_path))

    manifest = mgr.read_manifest()
    assert manifest["region"] == "us-east-1"
    assert manifest["default_region"] == "us-east-1"
    # Mutate an unrelated key and write back; the sibling collision must not raise.
    manifest["static_dir"] = "/app/ui2"
    mgr.write_manifest(manifest)

    raw = (tmp_path / "manifest.yml").read_text(encoding="utf-8")
    # ``region`` is written back as its ``!ENV`` marker, not a baked literal.
    assert "!ENV" in raw
    assert "REGION" in raw
    reread = mgr.read_manifest()
    assert reread["region"] == "us-east-1"
    assert reread["default_region"] == "us-east-1"
    assert reread["static_dir"] == "/app/ui2"


def _write_two_mcp_manifest(tmp_path) -> None:
    """Write a manifest with two mcp entries, each holding a distinct ``!ENV``
    secret in ``config.headers.Authorization`` under a distinct ``title``."""
    (tmp_path / "manifest.yml").write_text(
        "static_dir: /app/ui\n"
        "mcp:\n"
        "- title: alpha\n"
        "  config:\n"
        "    url: http://localhost:9001/mcp\n"
        "    headers:\n"
        "      Authorization: !ENV ${ALPHA_TOKEN}\n"
        "- title: beta\n"
        "  config:\n"
        "    url: http://localhost:9002/mcp\n"
        "    headers:\n"
        "      Authorization: !ENV ${BETA_TOKEN}\n",
        encoding="utf-8",
    )


def test_write_manifest_reorder_mcp_list_keeps_secrets_tagged(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Swapping two ``!ENV``-bearing mcp entries (and editing an unrelated key)
    must not bake either resolved secret to disk — identity matching keeps each
    marker aligned with its own entry despite the index change."""
    monkeypatch.setenv("ALPHA_TOKEN", "alpha-secret")
    monkeypatch.setenv("BETA_TOKEN", "beta-secret")
    _write_two_mcp_manifest(tmp_path)
    mgr = FileConfigManager(config_dir_path=str(tmp_path))

    manifest = mgr.read_manifest()
    manifest["mcp"] = [manifest["mcp"][1], manifest["mcp"][0]]  # swap
    manifest["static_dir"] = "/app/ui2"
    mgr.write_manifest(manifest)

    raw = (tmp_path / "manifest.yml").read_text(encoding="utf-8")
    assert "alpha-secret" not in raw
    assert "beta-secret" not in raw
    assert raw.count("!ENV") == 2
    reread = mgr.read_manifest()
    by_title = {entry["title"]: entry for entry in reread["mcp"]}
    assert by_title["alpha"]["config"]["headers"]["Authorization"] == "alpha-secret"
    assert by_title["beta"]["config"]["headers"]["Authorization"] == "beta-secret"


def test_write_manifest_delete_mcp_entry_keeps_other_secret_tagged(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Deleting the first mcp entry must not bake the remaining entry's secret to
    plaintext — its marker survives via identity matching."""
    monkeypatch.setenv("ALPHA_TOKEN", "alpha-secret")
    monkeypatch.setenv("BETA_TOKEN", "beta-secret")
    _write_two_mcp_manifest(tmp_path)
    mgr = FileConfigManager(config_dir_path=str(tmp_path))

    manifest = mgr.read_manifest()
    del manifest["mcp"][0]  # drop alpha, keep beta
    mgr.write_manifest(manifest)

    raw = (tmp_path / "manifest.yml").read_text(encoding="utf-8")
    assert "beta-secret" not in raw
    assert "!ENV" in raw
    reread = mgr.read_manifest()
    assert reread["mcp"][0]["title"] == "beta"
    assert reread["mcp"][0]["config"]["headers"]["Authorization"] == "beta-secret"


def test_write_manifest_insert_mcp_entry_keeps_secrets_tagged(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Inserting a new literal-header entry at the front must not bake the existing
    entries' secrets, and the new entry is written literally (it carries no
    marker to restore)."""
    monkeypatch.setenv("ALPHA_TOKEN", "alpha-secret")
    monkeypatch.setenv("BETA_TOKEN", "beta-secret")
    _write_two_mcp_manifest(tmp_path)
    mgr = FileConfigManager(config_dir_path=str(tmp_path))

    manifest = mgr.read_manifest()
    new_entry = {
        "title": "gamma",
        "config": {"url": "http://localhost:9003/mcp", "headers": {"Authorization": "plain-literal"}},
    }
    manifest["mcp"].insert(0, new_entry)
    mgr.write_manifest(manifest)

    raw = (tmp_path / "manifest.yml").read_text(encoding="utf-8")
    assert "alpha-secret" not in raw
    assert "beta-secret" not in raw
    assert "plain-literal" in raw
    reread = mgr.read_manifest()
    by_title = {entry["title"]: entry for entry in reread["mcp"]}
    assert by_title["gamma"]["config"]["headers"]["Authorization"] == "plain-literal"
    assert by_title["alpha"]["config"]["headers"]["Authorization"] == "alpha-secret"
    assert by_title["beta"]["config"]["headers"]["Authorization"] == "beta-secret"


def test_write_manifest_partial_edit_keeps_untouched_secret_tagged(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Editing one entry's ``config.url`` while leaving its ``!ENV`` header
    untouched must not bake that entry's secret — the marker is restored via the
    identity-matched recursion into the entry."""
    monkeypatch.setenv("ALPHA_TOKEN", "alpha-secret")
    monkeypatch.setenv("BETA_TOKEN", "beta-secret")
    _write_two_mcp_manifest(tmp_path)
    mgr = FileConfigManager(config_dir_path=str(tmp_path))

    manifest = mgr.read_manifest()
    manifest["mcp"][0]["config"]["url"] = "http://localhost:9999/mcp"
    mgr.write_manifest(manifest)

    raw = (tmp_path / "manifest.yml").read_text(encoding="utf-8")
    assert "alpha-secret" not in raw
    assert "beta-secret" not in raw
    assert raw.count("!ENV") == 2
    reread = mgr.read_manifest()
    by_title = {entry["title"]: entry for entry in reread["mcp"]}
    assert by_title["alpha"]["config"]["url"] == "http://localhost:9999/mcp"
    assert by_title["alpha"]["config"]["headers"]["Authorization"] == "alpha-secret"
    assert by_title["beta"]["config"]["headers"]["Authorization"] == "beta-secret"


def test_write_manifest_refuses_to_leak_secret_on_identity_rename(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Renaming an entry's IDENTITY field (its ``title``) while keeping its
    unchanged ``!ENV`` secret leaves the marker unmatched, so the resolved secret
    would bake to disk as plaintext. The safety net raises ``ValueError`` and
    refuses the write, leaving the on-disk manifest untouched (marker intact)."""
    monkeypatch.setenv("MY_TOKEN", "super-secret")
    (tmp_path / "manifest.yml").write_text(
        "mcp:\n"
        "- title: remote\n"
        "  config:\n"
        "    url: http://localhost:9000/mcp\n"
        "    headers:\n"
        "      Authorization: !ENV ${MY_TOKEN}\n",
        encoding="utf-8",
    )
    mgr = FileConfigManager(config_dir_path=str(tmp_path))
    before = (tmp_path / "manifest.yml").read_text(encoding="utf-8")

    manifest = mgr.read_manifest()
    # Rename the identity field but keep the resolved secret value in place.
    manifest["mcp"][0]["title"] = "renamed"
    with pytest.raises(ValueError, match="!ENV secret would be written as plaintext"):
        mgr.write_manifest(manifest)

    # The write was refused: the marker survives and the plaintext secret is absent.
    raw = (tmp_path / "manifest.yml").read_text(encoding="utf-8")
    assert raw == before
    assert "!ENV" in raw
    assert "super-secret" not in raw


def test_write_manifest_reorder_name_identity_list_keeps_secret_tagged(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A list whose entries carry no ``title`` but share a ``name`` identity is
    matched on ``name`` (the non-title identity branch). Reordering it keeps each
    ``!ENV`` marker aligned with its own entry, so no secret bakes to disk."""
    monkeypatch.setenv("ALPHA_TOKEN", "alpha-secret")
    monkeypatch.setenv("BETA_TOKEN", "beta-secret")
    (tmp_path / "manifest.yml").write_text(
        "mcp:\n"
        "- name: alpha\n"
        "  config:\n"
        "    headers:\n"
        "      Authorization: !ENV ${ALPHA_TOKEN}\n"
        "- name: beta\n"
        "  config:\n"
        "    headers:\n"
        "      Authorization: !ENV ${BETA_TOKEN}\n",
        encoding="utf-8",
    )
    mgr = FileConfigManager(config_dir_path=str(tmp_path))

    manifest = mgr.read_manifest()
    manifest["mcp"] = [manifest["mcp"][1], manifest["mcp"][0]]  # swap
    mgr.write_manifest(manifest)

    raw = (tmp_path / "manifest.yml").read_text(encoding="utf-8")
    assert "alpha-secret" not in raw
    assert "beta-secret" not in raw
    assert raw.count("!ENV") == 2
    reread = mgr.read_manifest()
    by_name = {entry["name"]: entry for entry in reread["mcp"]}
    assert by_name["alpha"]["config"]["headers"]["Authorization"] == "alpha-secret"
    assert by_name["beta"]["config"]["headers"]["Authorization"] == "beta-secret"


def test_identity_key_matches_first_key_shared_by_both() -> None:
    """``_identity_key`` returns the first key present in BOTH dicts, skipping a
    key that only the candidate has: a candidate with ``title`` and ``name`` and
    an incoming with only ``name`` matches on ``name`` (not ``None``)."""
    incoming = {"name": "shared", "config": {}}
    candidate = {"title": "only-on-candidate", "name": "shared", "config": {}}
    assert FileConfigManager._identity_key(incoming, candidate) == "name"


def test_write_env_allows_probe_named_key(tmp_path) -> None:
    """A caller may write a key literally named ``_TAI_ENV_ROUNDTRIP_PROBE``: the
    round-trip guard derives a non-colliding probe key, so the value is not shadowed
    by the sentinel and falsely flagged. It writes and reads back intact."""
    mgr = FileConfigManager(config_dir_path=str(tmp_path))
    mgr.write_env({"_TAI_ENV_ROUNDTRIP_PROBE": "real-value", "OTHER": "x"})
    parsed = mgr.read_env()
    assert parsed["_TAI_ENV_ROUNDTRIP_PROBE"] == "real-value"
    assert parsed["OTHER"] == "x"


def test_write_manifest_empty_env_secret_in_uncorresponded_subtree_does_not_raise(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """An ``!ENV`` marker that resolves to an EMPTY string must not falsely trip the
    plaintext-leak safety net even when it sits in an UNCORRESPONDED subtree the scan
    actually visits: renaming its dict key strands the leaf, so the scan sees it, but
    empty resolved values are excluded from the secret set and the write SUCCEEDS.

    This genuinely guards the empty-string exclusion in ``_resolved_secrets``: the
    stranded leaf reaches the scan (unlike a corresponded leaf, which is never
    scanned), and because ``_resolved_secrets`` excludes ``""`` from the secret set
    the scan finds nothing to flag and the write succeeds. The control below shows a
    non-empty secret in the same stranded position is caught, proving the scan does
    reach this subtree and only the empty value is spared."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    monkeypatch.setenv("EMPTY_TOKEN", "")
    (empty_dir / "manifest.yml").write_text(
        "token: !ENV ${EMPTY_TOKEN}\n",
        encoding="utf-8",
    )
    mgr = FileConfigManager(config_dir_path=str(empty_dir))

    manifest = mgr.read_manifest()
    assert manifest["token"] == ""
    # Rename the secret-bearing key so its (empty) resolved value lands in an
    # uncorresponded subtree that the leak scan visits.
    manifest["renamed_token"] = manifest.pop("token")
    # Must not raise: the empty resolved value is excluded from the secret set.
    mgr.write_manifest(manifest)
    assert (empty_dir / "manifest.yml").exists()

    # Control: a NON-empty ``!ENV`` secret in the very same uncorresponded position
    # DOES raise — proving the scan reaches this subtree and only the empty value is
    # spared.
    nonempty_dir = tmp_path / "nonempty"
    nonempty_dir.mkdir()
    monkeypatch.setenv("REAL_TOKEN", "super-secret")
    (nonempty_dir / "manifest.yml").write_text(
        "token: !ENV ${REAL_TOKEN}\n",
        encoding="utf-8",
    )
    mgr2 = FileConfigManager(config_dir_path=str(nonempty_dir))
    before = (nonempty_dir / "manifest.yml").read_text(encoding="utf-8")

    manifest2 = mgr2.read_manifest()
    assert manifest2["token"] == "super-secret"
    manifest2["renamed_token"] = manifest2.pop("token")
    with pytest.raises(ValueError, match="!ENV secret would be written as plaintext"):
        mgr2.write_manifest(manifest2)
    # The write was refused: the marker survives and the plaintext secret is absent.
    raw = (nonempty_dir / "manifest.yml").read_text(encoding="utf-8")
    assert raw == before
    assert "super-secret" not in raw


def test_write_manifest_refuses_leak_on_name_identity_rename(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """The safety net fires for a NON-title identity rename: entries identified by
    ``name`` whose ``name`` is changed while keeping the unchanged ``!ENV`` secret
    leave the marker unmatched, so the resolved secret would bake to disk. The net
    raises ``ValueError`` and no plaintext secret reaches disk."""
    monkeypatch.setenv("ALPHA_TOKEN", "alpha-secret")
    monkeypatch.setenv("BETA_TOKEN", "beta-secret")
    (tmp_path / "manifest.yml").write_text(
        "mcp:\n"
        "- name: alpha\n"
        "  config:\n"
        "    headers:\n"
        "      Authorization: !ENV ${ALPHA_TOKEN}\n"
        "- name: beta\n"
        "  config:\n"
        "    headers:\n"
        "      Authorization: !ENV ${BETA_TOKEN}\n",
        encoding="utf-8",
    )
    mgr = FileConfigManager(config_dir_path=str(tmp_path))
    before = (tmp_path / "manifest.yml").read_text(encoding="utf-8")

    manifest = mgr.read_manifest()
    # Rename the identity field while keeping the resolved secret in place.
    manifest["mcp"][0]["name"] = "renamed"
    with pytest.raises(ValueError, match="!ENV secret would be written as plaintext"):
        mgr.write_manifest(manifest)

    raw = (tmp_path / "manifest.yml").read_text(encoding="utf-8")
    assert raw == before
    assert "alpha-secret" not in raw


def test_write_manifest_reorder_module_identity_list_keeps_secret_tagged(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A list whose entries carry neither ``title`` nor ``name`` but share a
    ``module`` identity is matched on ``module`` (the ``module`` leg of
    ``_identity_key``). Reordering it keeps each ``!ENV`` marker aligned with its
    own entry, so no secret bakes to disk."""
    monkeypatch.setenv("ALPHA_TOKEN", "alpha-secret")
    monkeypatch.setenv("BETA_TOKEN", "beta-secret")
    (tmp_path / "manifest.yml").write_text(
        "tools:\n"
        "- module: alpha\n"
        "  config:\n"
        "    headers:\n"
        "      Authorization: !ENV ${ALPHA_TOKEN}\n"
        "- module: beta\n"
        "  config:\n"
        "    headers:\n"
        "      Authorization: !ENV ${BETA_TOKEN}\n",
        encoding="utf-8",
    )
    mgr = FileConfigManager(config_dir_path=str(tmp_path))

    manifest = mgr.read_manifest()
    manifest["tools"] = [manifest["tools"][1], manifest["tools"][0]]  # swap
    mgr.write_manifest(manifest)

    raw = (tmp_path / "manifest.yml").read_text(encoding="utf-8")
    assert "alpha-secret" not in raw
    assert "beta-secret" not in raw
    assert raw.count("!ENV") == 2
    reread = mgr.read_manifest()
    by_module = {entry["module"]: entry for entry in reread["tools"]}
    assert by_module["alpha"]["config"]["headers"]["Authorization"] == "alpha-secret"
    assert by_module["beta"]["config"]["headers"]["Authorization"] == "beta-secret"


def test_write_manifest_refuses_leak_on_dict_key_rename(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Renaming a secret-bearing DICT KEY (not a list identity) while keeping its
    unchanged resolved ``!ENV`` value strands the secret under a NEW key absent from
    the current view, so retag cannot descend into it and the resolved secret would
    bake to disk as plaintext. The scan covers uncorresponded dict-key subtrees too,
    so it raises ``ValueError`` and refuses the write (the on-disk marker survives)."""
    monkeypatch.setenv("TOK", "super-secret")
    (tmp_path / "manifest.yml").write_text(
        "mcp:\n- title: remote\n  env:\n    API_KEY: !ENV ${TOK}\n",
        encoding="utf-8",
    )
    mgr = FileConfigManager(config_dir_path=str(tmp_path))
    before = (tmp_path / "manifest.yml").read_text(encoding="utf-8")

    manifest = mgr.read_manifest()
    assert manifest["mcp"][0]["env"]["API_KEY"] == "super-secret"
    # Rename the secret-bearing dict key, keeping its resolved value.
    env = manifest["mcp"][0]["env"]
    env["APIKEY"] = env.pop("API_KEY")
    with pytest.raises(ValueError, match="!ENV secret would be written as plaintext"):
        mgr.write_manifest(manifest)

    # The write was refused: the marker survives and the plaintext secret is absent.
    raw = (tmp_path / "manifest.yml").read_text(encoding="utf-8")
    assert raw == before
    assert "!ENV" in raw
    assert "super-secret" not in raw


def test_write_manifest_new_plain_dict_key_does_not_false_positive(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Adding an ordinary NEW dict key with a non-secret value lands that subtree in
    the uncorresponded set the scan visits, but its value is not a resolved secret,
    so the write SUCCEEDS — the scan does not false-positive on ordinary new keys."""
    monkeypatch.setenv("MY_TOKEN", "super-secret")
    (tmp_path / "manifest.yml").write_text("token: !ENV ${MY_TOKEN}\n", encoding="utf-8")
    mgr = FileConfigManager(config_dir_path=str(tmp_path))

    manifest = mgr.read_manifest()
    manifest["new_key"] = "ordinary-value"
    mgr.write_manifest(manifest)

    raw = (tmp_path / "manifest.yml").read_text(encoding="utf-8")
    assert "ordinary-value" in raw
    # The untouched secret is still a preserved marker, not plaintext.
    assert "!ENV" in raw
    assert "super-secret" not in raw
    reread = mgr.read_manifest()
    assert reread["new_key"] == "ordinary-value"
    assert reread["token"] == "super-secret"


def test_write_manifest_refuses_leak_on_duplicate_identity_mismatch(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Two list entries sharing the same ``title`` identity but each holding a
    DIFFERENT ``!ENV`` secret are ambiguous: an identity match could restore the
    wrong entry's marker over the other's resolved secret. Identity matching is
    therefore rejected for a duplicate identity; when an entry is also edited (so
    deep equality cannot rescue it), it is left uncorresponded and its secret would
    bake to disk. The scan catches it and the write is refused (fail-closed).

    The manifest is written to disk and read through ``read_manifest`` directly, so
    the duplicate-``title`` list is not rejected by higher-level validation first."""
    monkeypatch.setenv("ALPHA_TOKEN", "alpha-secret")
    monkeypatch.setenv("BETA_TOKEN", "beta-secret")
    (tmp_path / "manifest.yml").write_text(
        "mcp:\n"
        "- title: dup\n"
        "  config:\n"
        "    url: http://localhost:9001/mcp\n"
        "    headers:\n"
        "      Authorization: !ENV ${ALPHA_TOKEN}\n"
        "- title: dup\n"
        "  config:\n"
        "    url: http://localhost:9002/mcp\n"
        "    headers:\n"
        "      Authorization: !ENV ${BETA_TOKEN}\n",
        encoding="utf-8",
    )
    mgr = FileConfigManager(config_dir_path=str(tmp_path))
    before = (tmp_path / "manifest.yml").read_text(encoding="utf-8")

    manifest = mgr.read_manifest()
    # Reorder the two same-title entries and edit the (now-first) entry's url so it
    # is no longer deep-equal to any current entry — leaving only the ambiguous
    # identity, which is not matched.
    manifest["mcp"] = [manifest["mcp"][1], manifest["mcp"][0]]  # swap
    manifest["mcp"][0]["config"]["url"] = "http://localhost:9999/mcp"
    with pytest.raises(ValueError, match="!ENV secret would be written as plaintext"):
        mgr.write_manifest(manifest)

    # The write was refused: neither plaintext secret reaches disk.
    raw = (tmp_path / "manifest.yml").read_text(encoding="utf-8")
    assert raw == before
    assert "alpha-secret" not in raw
    assert "beta-secret" not in raw


def test_write_manifest_reorder_duplicate_identity_equal_entries_is_safe(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A pure reorder of two same-``title`` entries with different secrets is still
    safe: identity is ambiguous, but each unchanged entry is paired by deep equality
    to its own current entry, so both markers are restored and neither secret bakes
    to disk (no false raise for an unedited reorder)."""
    monkeypatch.setenv("ALPHA_TOKEN", "alpha-secret")
    monkeypatch.setenv("BETA_TOKEN", "beta-secret")
    (tmp_path / "manifest.yml").write_text(
        "mcp:\n"
        "- title: dup\n"
        "  config:\n"
        "    headers:\n"
        "      Authorization: !ENV ${ALPHA_TOKEN}\n"
        "- title: dup\n"
        "  config:\n"
        "    headers:\n"
        "      Authorization: !ENV ${BETA_TOKEN}\n",
        encoding="utf-8",
    )
    mgr = FileConfigManager(config_dir_path=str(tmp_path))

    manifest = mgr.read_manifest()
    manifest["mcp"] = [manifest["mcp"][1], manifest["mcp"][0]]  # swap, no edit
    mgr.write_manifest(manifest)

    raw = (tmp_path / "manifest.yml").read_text(encoding="utf-8")
    assert "alpha-secret" not in raw
    assert "beta-secret" not in raw
    assert raw.count("!ENV") == 2


def test_atomic_write_cleans_up_temp_on_replace_failure(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """When ``os.replace`` fails mid-write, the error propagates loudly AND the
    orphan temp file is removed, so no leftover temp lingers in the config dir
    (the ``except OSError: os.unlink(tmp)`` branch ran)."""
    import tai42_skeleton.config.file_manager as fm

    def boom(src, dst):
        raise OSError("replace failed")

    monkeypatch.setattr(fm.os, "replace", boom)
    mgr = FileConfigManager(config_dir_path=str(tmp_path))
    with pytest.raises(OSError, match="replace failed"):
        mgr.write_env({"A": "1"})

    # The error propagated and no orphan temp file remains in the config dir. Only
    # the persistent flock sidecar (``.env.lock``) may remain — it is the lock
    # subject, not a temp, and is never unlinked.
    assert not (tmp_path / ".env").exists()
    assert [p.name for p in tmp_path.iterdir()] == [".env.lock"]


def test_write_manifest_write_failure_propagates(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """A manifest write failure inside the locked section (here a failing
    ``os.replace``) is logged and re-raised, not swallowed — the lock is held over a
    genuinely fallible atomic write."""
    import tai42_skeleton.config.file_manager as fm

    def boom(src, dst):
        raise OSError("replace failed")

    monkeypatch.setattr(fm.os, "replace", boom)
    mgr = FileConfigManager(config_dir_path=str(tmp_path))
    with pytest.raises(OSError, match="replace failed"):
        mgr.write_manifest({"tools": []})
    assert not (tmp_path / "manifest.yml").exists()


def test_write_manifest_refuses_leak_on_identity_swap(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Swapping only the ``title``s of two ``!ENV``-bearing entries (A↔B) while each
    entry keeps its own resolved secret makes every entry identity-match the WRONG
    current entry: its unchanged secret leaf then differs from its paired current
    value, so the marker cannot be restored and the resolved secret would bake to
    disk. The safety net raises ``ValueError`` and neither plaintext secret reaches
    disk (the on-disk markers survive)."""
    monkeypatch.setenv("ALPHA_TOKEN", "alpha-secret")
    monkeypatch.setenv("BETA_TOKEN", "beta-secret")
    _write_two_mcp_manifest(tmp_path)
    mgr = FileConfigManager(config_dir_path=str(tmp_path))
    before = (tmp_path / "manifest.yml").read_text(encoding="utf-8")

    manifest = mgr.read_manifest()
    # Swap the identity fields only; each entry retains its own resolved secret.
    manifest["mcp"][0]["title"] = "beta"
    manifest["mcp"][1]["title"] = "alpha"
    with pytest.raises(ValueError, match="!ENV secret would be written as plaintext"):
        mgr.write_manifest(manifest)

    # The write was refused: both markers survive and neither plaintext secret is on disk.
    raw = (tmp_path / "manifest.yml").read_text(encoding="utf-8")
    assert raw == before
    assert raw.count("!ENV") == 2
    assert "alpha-secret" not in raw
    assert "beta-secret" not in raw


def test_write_manifest_refuses_leak_on_identity_collision_mismatch(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Renaming a secret-bearing entry's IDENTITY to COLLIDE with a different,
    plain-valued entry mis-pairs the renamed entry against the plain entry: retag
    then descends into the WRONG preserved subtree, reaching the secret leaf as
    ``both plain scalars`` at the fallthrough (the paired ``token`` is a plain literal,
    not a marker). The symmetric fallthrough rule (append unless ``incoming ==
    expanded``) records the unchanged secret because it differs from the mis-paired
    plain value, so the scan raises ``ValueError`` and the plaintext secret never
    reaches disk (the on-disk marker survives)."""
    monkeypatch.setenv("TOK", "super-secret")
    (tmp_path / "manifest.yml").write_text(
        'services:\n- name: alpha\n  token: !ENV ${TOK}\n- name: beta\n  token: "plainbeta"\n',
        encoding="utf-8",
    )
    mgr = FileConfigManager(config_dir_path=str(tmp_path))
    before = (tmp_path / "manifest.yml").read_text(encoding="utf-8")

    manifest = mgr.read_manifest()
    assert manifest["services"][0]["token"] == "super-secret"
    assert manifest["services"][1]["token"] == "plainbeta"
    # Rename alpha's identity to collide with beta, keeping its resolved secret token.
    manifest["services"][0]["name"] = "beta"
    with pytest.raises(ValueError, match="!ENV secret would be written as plaintext"):
        mgr.write_manifest(manifest)

    # The write was refused: the marker survives and the plaintext secret is absent.
    raw = (tmp_path / "manifest.yml").read_text(encoding="utf-8")
    assert raw == before
    assert "!ENV" in raw
    assert "super-secret" not in raw


def test_write_manifest_refuses_leak_on_type_divergent_restructure(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """A secret held under a key as a LIST that the operator restructures into a
    dict still carrying the resolved secret diverges in container type from the
    current view, so retag cannot descend into it and the secret would bake to disk.
    The safety net raises ``ValueError`` and no plaintext secret reaches disk (the
    on-disk marker survives)."""
    monkeypatch.setenv("TOK", "super-secret")
    (tmp_path / "manifest.yml").write_text(
        "config:\n- !ENV ${TOK}\n",
        encoding="utf-8",
    )
    mgr = FileConfigManager(config_dir_path=str(tmp_path))
    before = (tmp_path / "manifest.yml").read_text(encoding="utf-8")

    manifest = mgr.read_manifest()
    assert manifest["config"] == ["super-secret"]
    # Restructure the list into a dict that still carries the resolved secret.
    manifest["config"] = {"item": manifest["config"][0]}
    with pytest.raises(ValueError, match="!ENV secret would be written as plaintext"):
        mgr.write_manifest(manifest)

    # The write was refused: the marker survives and the plaintext secret is absent.
    raw = (tmp_path / "manifest.yml").read_text(encoding="utf-8")
    assert raw == before
    assert "!ENV" in raw
    assert "super-secret" not in raw


def test_write_manifest_non_secret_restructure_does_not_false_positive(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Restructuring a NON-secret field (a plain scalar reshaped into a container)
    lands that type-divergent subtree in the scanned set, but its value is not a
    resolved secret, so the write SUCCEEDS — the untouched ``!ENV`` secret keeps its
    marker and the restructured plain field is written literally."""
    monkeypatch.setenv("MY_TOKEN", "super-secret")
    (tmp_path / "manifest.yml").write_text(
        "token: !ENV ${MY_TOKEN}\nplain: hello\n",
        encoding="utf-8",
    )
    mgr = FileConfigManager(config_dir_path=str(tmp_path))

    manifest = mgr.read_manifest()
    assert manifest["plain"] == "hello"
    # Reshape the non-secret scalar into a container carrying only non-secret values.
    manifest["plain"] = {"nested": "world"}
    mgr.write_manifest(manifest)

    raw = (tmp_path / "manifest.yml").read_text(encoding="utf-8")
    assert "world" in raw
    # The untouched secret is still a preserved marker, not plaintext.
    assert "!ENV" in raw
    assert "super-secret" not in raw
    reread = mgr.read_manifest()
    assert reread["plain"] == {"nested": "world"}
    assert reread["token"] == "super-secret"


def test_write_manifest_refuses_leak_on_container_collapse_to_scalar(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """A secret held inside a nested CONTAINER that the operator collapses to the
    resolved scalar diverges in container type from the current view (incoming is a
    scalar where the current view is a dict), so retag cannot descend and re-tag it.
    The fallthrough records it because the current side is a container, and the scan
    raises ``ValueError`` so the plaintext secret never reaches disk (the on-disk
    marker survives)."""
    monkeypatch.setenv("TOK", "super-secret")
    (tmp_path / "manifest.yml").write_text(
        "a:\n  b: !ENV ${TOK}\n",
        encoding="utf-8",
    )
    mgr = FileConfigManager(config_dir_path=str(tmp_path))
    before = (tmp_path / "manifest.yml").read_text(encoding="utf-8")

    manifest = mgr.read_manifest()
    assert manifest["a"] == {"b": "super-secret"}
    # Collapse the container that held the marker down to the resolved scalar.
    manifest["a"] = manifest["a"]["b"]
    assert manifest["a"] == "super-secret"
    with pytest.raises(ValueError, match="!ENV secret would be written as plaintext"):
        mgr.write_manifest(manifest)

    # The write was refused: the marker survives and the plaintext secret is absent.
    raw = (tmp_path / "manifest.yml").read_text(encoding="utf-8")
    assert raw == before
    assert "!ENV" in raw
    assert "super-secret" not in raw


def test_write_manifest_non_secret_collapse_to_scalar_does_not_false_positive(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Collapsing a NON-secret container to a scalar lands that type-divergent
    position in the scanned set, but its value is not a resolved secret, so the write
    SUCCEEDS — an untouched ``!ENV`` secret elsewhere keeps its marker (so the secret
    set is non-empty and the scan genuinely runs) and the collapsed plain field is
    written literally."""
    monkeypatch.setenv("MY_TOKEN", "super-secret")
    (tmp_path / "manifest.yml").write_text(
        "token: !ENV ${MY_TOKEN}\nplain:\n  nested: hello\n",
        encoding="utf-8",
    )
    mgr = FileConfigManager(config_dir_path=str(tmp_path))

    manifest = mgr.read_manifest()
    assert manifest["plain"] == {"nested": "hello"}
    # Collapse the non-secret container down to its scalar value.
    manifest["plain"] = manifest["plain"]["nested"]
    mgr.write_manifest(manifest)

    raw = (tmp_path / "manifest.yml").read_text(encoding="utf-8")
    assert "hello" in raw
    # The untouched secret is still a preserved marker, not plaintext.
    assert "!ENV" in raw
    assert "super-secret" not in raw
    reread = mgr.read_manifest()
    assert reread["plain"] == "hello"
    assert reread["token"] == "super-secret"


def test_write_manifest_secret_substring_in_new_leaf_does_not_false_positive(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A new leaf whose value merely EMBEDS a resolved secret as a substring (not the
    whole leaf) does not falsely trip the leak net: ``_contains_secret`` matches only
    exact string leaves. The new key is uncorresponded, so the scan genuinely runs
    over its value, and because that value is not exactly a resolved secret the write
    SUCCEEDS — proving the exact-leaf boundary rather than a skipped scan."""
    monkeypatch.setenv("MY_TOKEN", "super-secret")
    (tmp_path / "manifest.yml").write_text("token: !ENV ${MY_TOKEN}\n", encoding="utf-8")
    mgr = FileConfigManager(config_dir_path=str(tmp_path))

    manifest = mgr.read_manifest()
    # A new leaf the operator composed themselves, embedding the secret as a substring
    # of a larger value (not equal to the secret).
    manifest["composed"] = "prefix-super-secret-suffix"
    mgr.write_manifest(manifest)

    raw = (tmp_path / "manifest.yml").read_text(encoding="utf-8")
    assert "prefix-super-secret-suffix" in raw
    # The untouched secret leaf is still a preserved marker, not plaintext.
    assert "!ENV" in raw
    reread = mgr.read_manifest()
    assert reread["composed"] == "prefix-super-secret-suffix"
    assert reread["token"] == "super-secret"


def test_write_manifest_three_way_merges_defaults_and_current(tmp_path) -> None:
    templates = tmp_path / "templates"
    templates.mkdir()
    (templates / "manifest.yml").write_text("from_defaults: 1\n", encoding="utf-8")
    (tmp_path / "manifest.yml").write_text("from_current: 2\n", encoding="utf-8")
    mgr = FileConfigManager(config_dir_path=str(tmp_path))
    mgr.write_manifest({"from_new": 3})
    merged = mgr.read_manifest()
    assert merged["from_defaults"] == 1
    assert merged["from_current"] == 2
    assert merged["from_new"] == 3


# --- mutate_manifest seam ---------------------------------------------------


def test_mutate_manifest_edits_in_place_and_persists(tmp_path) -> None:
    (tmp_path / "manifest.yml").write_text("a: 1\nb: 2\n", encoding="utf-8")
    mgr = FileConfigManager(config_dir_path=str(tmp_path))

    def mutator(doc: dict) -> None:
        doc["a"] = 99
        doc["c"] = 3

    result = mgr.mutate_manifest(mutator)
    assert result["a"] == 99
    assert result["c"] == 3
    reread = mgr.read_manifest()
    assert reread == {"a": 99, "b": 2, "c": 3}


def test_mutate_manifest_preserves_comments(tmp_path) -> None:
    """The round-trip read/dump keeps hand-authored comments on untouched keys."""
    source = "# top comment\na: 1  # inline on a\nb: 2  # inline on b\n"
    (tmp_path / "manifest.yml").write_text(source, encoding="utf-8")
    mgr = FileConfigManager(config_dir_path=str(tmp_path))
    mgr.mutate_manifest(lambda doc: doc.__setitem__("a", 5))
    raw = (tmp_path / "manifest.yml").read_text(encoding="utf-8")
    assert "# top comment" in raw
    assert "# inline on b" in raw
    assert "a: 5" in raw


def test_mutate_manifest_creates_manifest_when_absent(tmp_path) -> None:
    mgr = FileConfigManager(config_dir_path=str(tmp_path))
    mgr.mutate_manifest(lambda doc: doc.__setitem__("fresh", 1))
    assert mgr.read_manifest() == {"fresh": 1}


def test_mutate_manifest_mutator_exception_aborts_with_nothing_written(tmp_path) -> None:
    (tmp_path / "manifest.yml").write_text("a: 1\n", encoding="utf-8")
    mgr = FileConfigManager(config_dir_path=str(tmp_path))

    def boom(doc: dict) -> None:
        doc["a"] = 2
        raise ValueError("mutator failed")

    with pytest.raises(ValueError, match="mutator failed"):
        mgr.mutate_manifest(boom)
    # Nothing was written: the original content is intact.
    assert (tmp_path / "manifest.yml").read_text(encoding="utf-8") == "a: 1\n"


def test_mutate_manifest_reads_preserved_view_and_keeps_secret_tagged(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """The mutator sees the ``!ENV`` marker (never the resolved secret), and an
    untouched marker is written back as its placeholder, not baked to disk."""
    monkeypatch.setenv("SOME_VAR", "super-secret-value")
    (tmp_path / "manifest.yml").write_text("token: !ENV ${SOME_VAR}\nother: 1\n", encoding="utf-8")
    mgr = FileConfigManager(config_dir_path=str(tmp_path))

    seen: dict = {}

    def mutator(doc: dict) -> None:
        seen["token"] = doc["token"]
        doc["other"] = 2

    result = mgr.mutate_manifest(mutator)
    assert seen["token"] == "!ENV ${SOME_VAR}"
    assert result["token"] == "!ENV ${SOME_VAR}"
    raw = (tmp_path / "manifest.yml").read_text(encoding="utf-8")
    assert "!ENV" in raw
    assert "super-secret-value" not in raw
    # The runtime view still resolves the untouched placeholder from env.
    assert mgr.read_manifest()["token"] == "super-secret-value"


def test_mutate_manifest_backfills_defaults(tmp_path) -> None:
    templates = tmp_path / "templates"
    templates.mkdir()
    (templates / "manifest.yml").write_text("from_defaults: 1\n", encoding="utf-8")
    (tmp_path / "manifest.yml").write_text("from_current: 2\n", encoding="utf-8")
    mgr = FileConfigManager(config_dir_path=str(tmp_path))
    mgr.mutate_manifest(lambda doc: doc.__setitem__("added", 3))
    merged = mgr.read_manifest()
    assert merged == {"from_defaults": 1, "from_current": 2, "added": 3}


# --- replace_manifest seam --------------------------------------------------


def test_replace_manifest_deletes_absent_keys(tmp_path) -> None:
    """A replace is a true replace: a key absent from the document is DELETED
    (unlike the three-way write merge, which backfills old keys)."""
    (tmp_path / "manifest.yml").write_text("keep: 1\ndrop: 2\n", encoding="utf-8")
    mgr = FileConfigManager(config_dir_path=str(tmp_path))
    result = mgr.replace_manifest({"keep": 10})
    assert result == {"keep": 10}
    assert mgr.read_manifest() == {"keep": 10}


def test_replace_manifest_backfills_defaults_only(tmp_path) -> None:
    templates = tmp_path / "templates"
    templates.mkdir()
    (templates / "manifest.yml").write_text("from_defaults: 1\n", encoding="utf-8")
    (tmp_path / "manifest.yml").write_text("old: 9\n", encoding="utf-8")
    mgr = FileConfigManager(config_dir_path=str(tmp_path))
    mgr.replace_manifest({"new": 2})
    merged = mgr.read_manifest()
    # Defaults backfill the missing key; the old document's key is gone.
    assert merged == {"from_defaults": 1, "new": 2}


def test_replace_manifest_does_not_mutate_caller_dict(tmp_path) -> None:
    templates = tmp_path / "templates"
    templates.mkdir()
    (templates / "manifest.yml").write_text("from_defaults: 1\n", encoding="utf-8")
    mgr = FileConfigManager(config_dir_path=str(tmp_path))
    document = {"new": 2}
    mgr.replace_manifest(document)
    # The backfilled default landed in the persisted copy, not the caller's dict.
    assert document == {"new": 2}


def test_replace_manifest_persists_env_marker_verbatim(tmp_path) -> None:
    """The caller builds the document from the preserved view; a marker string is
    dumped back as a genuine ``!ENV`` tag, so no secret bakes to disk."""
    mgr = FileConfigManager(config_dir_path=str(tmp_path))
    mgr.replace_manifest({"token": "!ENV ${SOME_VAR}"})
    raw = (tmp_path / "manifest.yml").read_text(encoding="utf-8")
    assert "!ENV ${SOME_VAR}" in raw
    assert mgr.read_manifest_preserved() == {"token": "!ENV ${SOME_VAR}"}


def test_replace_manifest_preserves_comments_from_round_trip_document(tmp_path) -> None:
    """A comment-bearing round-trip document (the preserved-read product) is
    persisted verbatim: comments and ``!ENV`` markers survive the replace, so the
    caller-owned round-trip view is not silently flattened to a plain mapping."""
    (tmp_path / "manifest.yml").write_text(
        "# leading comment\ntoken: !ENV ${SOME_VAR}  # inline comment\n",
        encoding="utf-8",
    )
    mgr = FileConfigManager(config_dir_path=str(tmp_path))
    preserved = mgr.read_manifest_preserved()
    mgr.replace_manifest(preserved)
    raw = (tmp_path / "manifest.yml").read_text(encoding="utf-8")
    assert "# leading comment" in raw
    assert "# inline comment" in raw
    assert "!ENV ${SOME_VAR}" in raw


# --- cross-process flock: concurrent lost-update guard ----------------------


def test_write_env_concurrent_processes_keep_every_key_with_lock(tmp_path) -> None:
    """Two spawned processes each ``write_env`` disjoint key batches against the same
    dir. Under the flock the RMWs serialize, so every key from both survives."""
    (tmp_path / ".env").write_text("", encoding="utf-8")  # exists so read_env re-reads
    _run_two_writers(_env_writer, (str(tmp_path), "a", 5, True), (str(tmp_path), "b", 5, True))
    result = FileConfigManager(config_dir_path=str(tmp_path)).read_env()
    expected = {f"a{i}": str(i) for i in range(5)} | {f"b{i}": str(i) for i in range(5)}
    assert result == expected


def test_write_env_concurrent_processes_lose_update_without_lock(tmp_path) -> None:
    """The SAME race without the flock loses at least one update — proving the race is
    real and the lock in the companion test is what saves it."""
    (tmp_path / ".env").write_text("", encoding="utf-8")
    _run_two_writers(_env_writer, (str(tmp_path), "a", 5, False), (str(tmp_path), "b", 5, False))
    result = FileConfigManager(config_dir_path=str(tmp_path)).read_env()
    expected = {f"a{i}": str(i) for i in range(5)} | {f"b{i}": str(i) for i in range(5)}
    assert result != expected


def test_write_manifest_concurrent_processes_keep_every_key_with_lock(tmp_path) -> None:
    """Two spawned processes each ``write_manifest`` disjoint top-level keys. Under the
    flock every key survives the three-way merge."""
    (tmp_path / "manifest.yml").write_text("seed: 0\n", encoding="utf-8")
    _run_two_writers(_manifest_writer, (str(tmp_path), "alpha", 5, True), (str(tmp_path), "beta", 5, True))
    result = FileConfigManager(config_dir_path=str(tmp_path)).read_manifest()
    keys = {f"alpha{i}" for i in range(5)} | {f"beta{i}" for i in range(5)}
    assert keys <= set(result)


def test_write_manifest_concurrent_processes_lose_update_without_lock(tmp_path) -> None:
    """The SAME manifest race without the flock drops at least one top-level key."""
    (tmp_path / "manifest.yml").write_text("seed: 0\n", encoding="utf-8")
    _run_two_writers(_manifest_writer, (str(tmp_path), "alpha", 5, False), (str(tmp_path), "beta", 5, False))
    result = FileConfigManager(config_dir_path=str(tmp_path)).read_manifest()
    keys = {f"alpha{i}" for i in range(5)} | {f"beta{i}" for i in range(5)}
    assert not (keys <= set(result))


def test_mutate_manifest_concurrent_processes_keep_both_edits_with_lock(tmp_path) -> None:
    """Two spawned processes each ``mutate_manifest`` a disjoint key. The widened
    transaction lock spans each read → mutate → write, so both edits land."""
    (tmp_path / "manifest.yml").write_text("seed: 0\n", encoding="utf-8")
    _run_two_writers(_mutate_writer, (str(tmp_path), "alpha", True), (str(tmp_path), "beta", True))
    result = FileConfigManager(config_dir_path=str(tmp_path)).read_manifest()
    assert result["alpha"] == "alpha"
    assert result["beta"] == "beta"


def test_mutate_manifest_concurrent_processes_lose_update_without_lock(tmp_path) -> None:
    """The SAME mutate race without the lock loses one edit — proving the widened
    transaction span is what fixes the lost update the seam exists to close."""
    (tmp_path / "manifest.yml").write_text("seed: 0\n", encoding="utf-8")
    _run_two_writers(_mutate_writer, (str(tmp_path), "alpha", False), (str(tmp_path), "beta", False))
    result = FileConfigManager(config_dir_path=str(tmp_path)).read_manifest()
    assert not ({"alpha", "beta"} <= set(result))


# --- cross-process flock: sidecar mechanics ---------------------------------


def test_lock_sidecar_created_0600_next_to_target(tmp_path) -> None:
    """The lock sidecar is created ``0600`` beside the target, and the target's own
    content is unaffected by the sidecar's presence."""
    mgr = FileConfigManager(config_dir_path=str(tmp_path))
    mgr.write_env({"A": "1"})
    lock = tmp_path / ".env.lock"
    assert lock.exists()
    assert (os.stat(lock).st_mode & 0o777) == 0o600
    assert (tmp_path / ".env").read_text(encoding="utf-8") == 'A="1"\n'


def test_write_env_blocks_on_held_sidecar_lock(tmp_path) -> None:
    """A ``write_env`` in a child process blocks while the SIDECAR lock is held by the
    test, and completes only after it is released — proving the write serializes on
    the sidecar lock."""
    import fcntl

    mgr = FileConfigManager(config_dir_path=str(tmp_path))
    fd = os.open(f"{mgr._env_path}.lock", os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)
    ctx = multiprocessing.get_context("spawn")
    child = ctx.Process(target=_single_env_write, args=(str(tmp_path), "A", "1"))
    child.start()
    try:
        child.join(timeout=2.0)
        # While the sidecar lock is held the child cannot acquire it — it is still
        # running and has written nothing.
        assert child.is_alive()
        assert not (tmp_path / ".env").exists()
        # Release the lock; the child now acquires it and completes the write.
        fcntl.flock(fd, fcntl.LOCK_UN)
        child.join(timeout=30)
        assert child.exitcode == 0
        assert FileConfigManager(config_dir_path=str(tmp_path)).read_env() == {"A": "1"}
    finally:
        if child.is_alive():
            child.terminate()
            child.join()
        os.close(fd)


def test_write_env_locks_the_sidecar_not_the_target(tmp_path) -> None:
    """Holding a lock on the TARGET ``.env`` file itself does NOT block a child's
    ``write_env`` — the manager locks the sidecar, never the target (a lock on the
    target would be swapped out from under by ``os.replace`` anyway)."""
    import fcntl

    mgr = FileConfigManager(config_dir_path=str(tmp_path))
    mgr.write_env({"SEED": "0"})  # create the target so it can be locked
    fd = os.open(mgr._env_path, os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX)
    ctx = multiprocessing.get_context("spawn")
    child = ctx.Process(target=_single_env_write, args=(str(tmp_path), "A", "1"))
    child.start()
    try:
        child.join(timeout=30)
        # The child locked the SIDECAR (which we do not hold), so it completed despite
        # our lock on the target.
        assert child.exitcode == 0
        assert FileConfigManager(config_dir_path=str(tmp_path)).read_env()["A"] == "1"
    finally:
        if child.is_alive():
            child.terminate()
            child.join()
        os.close(fd)


def test_windows_lock_path_refuses_the_write(monkeypatch, tmp_path) -> None:
    """On Windows ``fcntl`` is unavailable, so a serialized write cannot be
    guaranteed and the write is REFUSED with a clear error (not a warn-and-proceed
    no-op). Simulated by faking the module's ``sys.platform`` (the real lock path is
    POSIX-only and untestable on Linux)."""
    import types

    monkeypatch.setattr(fm, "sys", types.SimpleNamespace(platform="win32"))
    mgr = FileConfigManager(config_dir_path=str(tmp_path))
    with pytest.raises(RuntimeError, match="POSIX"):
        mgr.write_env({"A": "1"})
    # The write is refused before anything lands: no file is created.
    assert not (tmp_path / ".env").exists()


def test_windows_lock_path_refuses_manifest_seams(monkeypatch, tmp_path) -> None:
    """The Windows refusal covers every manifest writer — ``write_manifest``,
    ``mutate_manifest``, and ``replace_manifest`` all raise before writing."""
    import types

    monkeypatch.setattr(fm, "sys", types.SimpleNamespace(platform="win32"))
    mgr = FileConfigManager(config_dir_path=str(tmp_path))
    with pytest.raises(RuntimeError, match="POSIX"):
        mgr.write_manifest({"a": 1})
    with pytest.raises(RuntimeError, match="POSIX"):
        mgr.mutate_manifest(lambda doc: doc.__setitem__("a", 1))
    with pytest.raises(RuntimeError, match="POSIX"):
        mgr.replace_manifest({"a": 1})
    assert not (tmp_path / "manifest.yml").exists()
