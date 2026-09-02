"""Tests for typed TOML loader functions."""

from __future__ import annotations

import tomllib
from pathlib import Path, PureWindowsPath

import pytest
from pydantic import ValidationError

from neuralls.platform.config.loaders import (
    load_case_config,
    load_comparison_config,
    load_data_config,
    load_raw_toml,
)
from neuralls.platform.config.models.experiments import CaseConfig
from neuralls.platform.config.settings import NeurallsSettings


def test_load_data_config_resolves_paths(
    minimal_data_config_toml: Path,
    neuralls_settings: NeurallsSettings,
) -> None:
    """Dataset loader expands NEURALLS placeholders into absolute paths."""
    config = load_data_config(minimal_data_config_toml, neuralls_settings)
    assert config.id == "test-dataset"
    assert config.source.matrix_path == str(
        (neuralls_settings.processed_dir / "matrix.mtx").resolve()
    )


def test_load_data_config_requires_id(
    tmp_path: Path,
    neuralls_settings: NeurallsSettings,
) -> None:
    """Dataset configs without an id fail validation."""
    config_file = tmp_path / "data.toml"
    config_file.write_text(
        """
[source]
matrix_path = "matrix.txt"
"""
    )
    with pytest.raises(ValidationError, match="id"):
        load_data_config(config_file, neuralls_settings)


def test_load_data_config_rejects_graph_cg_placeholder(
    tmp_path: Path,
    neuralls_settings: NeurallsSettings,
) -> None:
    """Stale GRAPH_CG placeholders are rejected with migration text."""
    config_file = tmp_path / "data.toml"
    config_file.write_text(
        """
id = "test"

[source]
matrix_path = "${GRAPH_CG_RAW_DIR}/matrix.txt"
"""
    )
    with pytest.raises(ValueError, match="NEURALLS"):
        load_data_config(config_file, neuralls_settings)


def test_load_comparison_config_resolves_paths(
    tmp_path: Path,
    neuralls_settings: NeurallsSettings,
) -> None:
    """Comparison loader expands matrix and rhs paths."""
    config_file = tmp_path / "comparison.toml"
    config_file.write_text(
        """
[general]

[general.params]
rtol = 1e-6

[general.data]
matrix_path = "${NEURALLS_PROCESSED_DIR}/matrix.npy"
rhs_path = "${NEURALLS_PROCESSED_DIR}/rhs.npy"

[[preconditioners]]
name = "jacobi"
type = "jacobi"
"""
    )
    config = load_comparison_config(config_file, neuralls_settings)
    assert (
        config.general.data.matrix_path
        == (neuralls_settings.processed_dir / "matrix.npy").resolve()
    )
    assert config.general.data.rhs_path == (neuralls_settings.processed_dir / "rhs.npy").resolve()


def test_load_case_config_returns_typed_model(
    tmp_path: Path,
    neuralls_settings: NeurallsSettings,
) -> None:
    """Case loader returns CaseConfig directly."""
    dataset_cfg = tmp_path / "dataset.toml"
    dataset_cfg.write_text(
        'id = "dataset"\n[source]\nmatrix_path = "${NEURALLS_PROCESSED_DIR}/matrix.mtx"\n'
    )
    config_file = tmp_path / "experiments.toml"
    config_file.write_text(
        f"""
[[datasets]]
id = "dataset"
path = "{dataset_cfg.as_posix()}"
"""
    )
    config = load_case_config(config_file, neuralls_settings)
    assert isinstance(config, CaseConfig)
    assert config.datasets[0].path == dataset_cfg.resolve()


def test_load_case_config_fills_missing_dataset_id_from_dataset_config(
    tmp_path: Path,
    neuralls_settings: NeurallsSettings,
) -> None:
    """A [[datasets]] entry without an id defaults to the dataset config's own id."""
    dataset_cfg = tmp_path / "dataset.toml"
    dataset_cfg.write_text(
        'id = "dataset-own-id"\n[source]\nmatrix_path = "${NEURALLS_PROCESSED_DIR}/matrix.mtx"\n'
    )
    config_file = tmp_path / "experiments.toml"
    config_file.write_text(
        f"""
[[datasets]]
path = "{dataset_cfg.as_posix()}"
"""
    )
    config = load_case_config(config_file, neuralls_settings)
    assert config.datasets[0].id == "dataset-own-id"


def test_load_case_config_explicit_dataset_id_wins(
    tmp_path: Path,
    neuralls_settings: NeurallsSettings,
) -> None:
    """An explicit [[datasets]] id is never overridden by the dataset config's own id."""
    dataset_cfg = tmp_path / "dataset.toml"
    dataset_cfg.write_text(
        'id = "dataset-own-id"\n[source]\nmatrix_path = "${NEURALLS_PROCESSED_DIR}/matrix.mtx"\n'
    )
    config_file = tmp_path / "experiments.toml"
    config_file.write_text(
        f"""
[[datasets]]
id = "case-local-alias"
path = "{dataset_cfg.as_posix()}"
"""
    )
    config = load_case_config(config_file, neuralls_settings)
    assert config.datasets[0].id == "case-local-alias"


def test_load_case_config_requires_dataset_id_when_neither_has_one(
    tmp_path: Path,
    neuralls_settings: NeurallsSettings,
) -> None:
    """A missing id on both the case entry and the dataset config is a clear error."""
    dataset_cfg = tmp_path / "dataset.toml"
    dataset_cfg.write_text('[source]\nmatrix_path = "${NEURALLS_PROCESSED_DIR}/matrix.mtx"\n')
    config_file = tmp_path / "experiments.toml"
    config_file.write_text(
        f"""
[[datasets]]
path = "{dataset_cfg.as_posix()}"
"""
    )
    with pytest.raises(ValueError, match="has no 'id'"):
        load_case_config(config_file, neuralls_settings)


def test_load_raw_toml_success(tmp_path: Path) -> None:
    """Raw TOML loading returns a plain dict."""
    config_file = tmp_path / "test.toml"
    config_file.write_text(
        """
[section]
key = "value"
number = 42
"""
    )
    data = load_raw_toml(config_file)
    assert isinstance(data, dict)
    assert data["section"]["key"] == "value"
    assert data["section"]["number"] == 42


def test_load_raw_toml_invalid_syntax(tmp_path: Path) -> None:
    """Invalid TOML raises the stdlib decode error."""
    config_file = tmp_path / "test.toml"
    config_file.write_text("invalid [ syntax")
    with pytest.raises(tomllib.TOMLDecodeError):
        load_raw_toml(config_file)


def test_load_raw_toml_rejects_unescaped_windows_path(tmp_path: Path) -> None:
    """Raw Windows backslashes inside TOML basic strings trigger stdlib parsing errors."""
    config_file = tmp_path / "test.toml"
    config_file.write_text(
        r"""
[source]
matrix_path = "C:\Users\runneradmin\AppData\Local\Temp\matrix.txt"
"""
    )
    with pytest.raises(tomllib.TOMLDecodeError, match="Invalid hex value"):
        load_raw_toml(config_file)


def test_load_raw_toml_accepts_normalized_windows_path(tmp_path: Path) -> None:
    """POSIX-normalized Windows paths remain valid TOML literals everywhere."""
    config_file = tmp_path / "test.toml"
    matrix_path = PureWindowsPath(r"C:\Users\runneradmin\AppData\Local\Temp\matrix.txt")
    config_file.write_text(f'[source]\nmatrix_path = "{matrix_path.as_posix()}"\n')
    data = load_raw_toml(config_file)
    assert data["source"]["matrix_path"] == matrix_path.as_posix()
