"""Argv-injection guard for synthesized stdio launch specs."""

from __future__ import annotations

import pytest

from tai42_skeleton.connectors.stdio.launcher import reject_leading_dash


def test_accepts_normal_value():
    reject_leading_dash("tai-mcp-widgets", field="entry_point")  # no raise


def test_rejects_leading_dash():
    with pytest.raises(ValueError, match="must not start with '-'"):
        reject_leading_dash("--inject-evil", field="entry_point")


def test_error_names_the_field():
    with pytest.raises(ValueError, match="pkg_version"):
        reject_leading_dash("-1.0", field="pkg_version")
