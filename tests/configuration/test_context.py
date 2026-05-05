"""Tests for ConfigContext path expansion helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from neuralls.platform.config.context import expand_config_glob, expand_config_path
from neuralls.platform.config.settings import NeurallsSettings


def test_expand_neuralls_raw_dir(config_context, neuralls_settings: NeurallsSettings) -> None:
    """`${NEURALLS_RAW_DIR}` expands to the settings raw root."""
    value = expand_config_path("${NEURALLS_RAW_DIR}/matrix.mtx", config_context)
    assert value == str((neuralls_settings.raw_dir / "matrix.mtx").resolve())


def test_expand_neuralls_output_dir(
    config_context,
    neuralls_settings: NeurallsSettings,
) -> None:
    """`${NEURALLS_OUTPUT_DIR}` expands to the settings output root."""
    value = expand_config_path("${NEURALLS_OUTPUT_DIR}/reports", config_context)
    assert value == str((neuralls_settings.output_dir / "reports").resolve())


def test_expand_graph_cg_raises_with_migration_hint(config_context) -> None:
    """Stale placeholders fail with a NEURALLS migration hint."""
    with pytest.raises(ValueError, match="NEURALLS"):
        expand_config_path("${GRAPH_CG_RAW_DIR}/matrix.mtx", config_context)


def test_expand_unknown_neuralls_raises(config_context) -> None:
    """Unknown placeholders are rejected."""
    with pytest.raises(ValueError, match="RAW_DIR, PROCESSED_DIR, OUTPUT_DIR"):
        expand_config_path("${NEURALLS_UNKNOWN}/matrix.mtx", config_context)


def test_relative_path_anchored_to_config_dir(config_context) -> None:
    """Relative paths are anchored to the config directory."""
    value = expand_config_path("relative/path.mtx", config_context)
    assert value == str((config_context.config_dir / "relative/path.mtx").resolve())


def test_absolute_path_unchanged(config_context, tmp_path: Path) -> None:
    """Absolute paths pass through unchanged after resolution."""
    absolute = tmp_path / "matrix.mtx"
    value = expand_config_path(str(absolute), config_context)
    assert value == str(absolute.resolve())


def test_tilde_expansion(config_context, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Tilde expansion uses the current home directory."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    value = expand_config_path("~/matrix.mtx", config_context)
    assert value == str((home / "matrix.mtx").resolve())


def test_glob_wildcard_preserved(
    config_context,
    neuralls_settings: NeurallsSettings,
) -> None:
    """Glob suffixes are preserved after prefix resolution."""
    value = expand_config_glob("${NEURALLS_RAW_DIR}/*.mtx", config_context)
    assert value == str(neuralls_settings.raw_dir / "*.mtx")
