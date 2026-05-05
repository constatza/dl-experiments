"""Tests for path context construction."""

from __future__ import annotations

from pathlib import Path

import pytest

from neuralls.platform.config.mlflow import build_sqlite_tracking_uri
from neuralls.platform.config.models.data_models import DataConfigFile, OutputConfig
from neuralls.platform.config.paths import PathContext, build_path_context
from neuralls.platform.config.settings import NeurallsSettings
from neuralls.shared.constants import DEFAULT_PROJECT_ROOT


def test_path_context_creation(tmp_path: Path) -> None:
    """PathContext is a frozen value object."""
    ctx = PathContext(
        project_root=tmp_path / "project",
        output_root=tmp_path / "output",
        processed_root=tmp_path / "processed",
    )
    assert ctx.project_root == tmp_path / "project"
    assert ctx.output_root == tmp_path / "output"
    assert ctx.processed_root == tmp_path / "processed"


def test_path_context_is_frozen(tmp_path: Path) -> None:
    """PathContext is immutable."""
    ctx = PathContext(
        project_root=tmp_path,
        output_root=tmp_path,
        processed_root=tmp_path,
    )
    with pytest.raises(AttributeError):
        ctx.project_root = tmp_path / "new"


def test_path_context_mlflow_properties(tmp_path: Path) -> None:
    """MLflow paths are derived from output_root."""
    output_root = tmp_path / "output"
    ctx = PathContext(
        project_root=tmp_path,
        output_root=output_root,
        processed_root=tmp_path,
    )
    assert ctx.mlflow_tracking_uri == build_sqlite_tracking_uri(output_root / "mlruns" / "mlflow.db")
    assert ctx.mlflow_artifact_location == str((output_root / "mlartifacts").resolve())


def test_build_path_context_uses_settings_defaults(
    neuralls_settings: NeurallsSettings,
) -> None:
    """Default roots come from NeurallsSettings."""
    data_cfg = DataConfigFile(id="test", output=OutputConfig())
    ctx = build_path_context(data_cfg, neuralls_settings)
    assert ctx.project_root == DEFAULT_PROJECT_ROOT
    assert ctx.output_root == neuralls_settings.output_dir
    assert ctx.processed_root == neuralls_settings.processed_dir


def test_build_path_context_honours_output_override(
    neuralls_settings: NeurallsSettings,
    tmp_path: Path,
) -> None:
    """Explicit output overrides beat settings.output_dir."""
    custom_output = tmp_path / "custom-output"
    custom_output.mkdir()
    data_cfg = DataConfigFile(id="test", output=OutputConfig())
    ctx = build_path_context(data_cfg, neuralls_settings, output_override=custom_output)
    assert ctx.output_root == custom_output.resolve()


def test_build_path_context_honours_data_dir_override(
    neuralls_settings: NeurallsSettings,
    tmp_path: Path,
) -> None:
    """Dataset-level data_dir overrides settings.processed_dir."""
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    data_cfg = DataConfigFile(id="test", output=OutputConfig(data_dir=processed_dir))
    ctx = build_path_context(data_cfg, neuralls_settings)
    assert ctx.processed_root == processed_dir.resolve()
