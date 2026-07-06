"""Tests for comparison tracking setup helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from neuralls.platform.config.models.experiments import ExperimentNamesConfig
from neuralls.platform.config.resolution import MlflowPaths
from neuralls.platform.tracking.comparison_tracking import setup_comparison_tracking


def test_setup_comparison_tracking_delegates_to_ensure_experiment(tmp_path: Path) -> None:
    """Comparison tracking setup ensures the experiment and selects it as current.

    Experiment get-or-create behavior itself is covered by
    ``tests/test_mlflow_utils.py`` for the shared ``ensure_experiment`` helper.
    """
    tracking_uri = f"sqlite:///{(tmp_path / 'mlruns' / 'mlflow.db').as_posix()}"
    artifact_dir = str(tmp_path / "artifacts")
    with (
        patch("neuralls.platform.tracking.comparison_tracking.mlflow") as mock_mlflow,
        patch("neuralls.platform.tracking.comparison_tracking.ensure_experiment") as mock_ensure,
    ):
        setup_comparison_tracking(
            tracking_uri,
            artifact_dir,
            experiment_name="CustomComparison",
        )

    mock_ensure.assert_called_once_with(
        "CustomComparison",
        MlflowPaths(tracking_uri=tracking_uri, artifact_uri=artifact_dir),
    )
    mock_mlflow.set_experiment.assert_called_once_with("CustomComparison")


def test_setup_comparison_tracking_uses_model_default_when_name_missing(tmp_path: Path) -> None:
    """Missing comparison names fall back to the config-model default."""
    tracking_uri = f"sqlite:///{(tmp_path / 'mlruns' / 'mlflow.db').as_posix()}"
    artifact_dir = str(tmp_path / "artifacts")
    with (
        patch("neuralls.platform.tracking.comparison_tracking.mlflow") as mock_mlflow,
        patch("neuralls.platform.tracking.comparison_tracking.ensure_experiment") as mock_ensure,
    ):
        setup_comparison_tracking(tracking_uri, artifact_dir)

    mock_ensure.assert_called_once_with(
        ExperimentNamesConfig().comparison,
        MlflowPaths(tracking_uri=tracking_uri, artifact_uri=artifact_dir),
    )
    mock_mlflow.set_experiment.assert_called_once_with(ExperimentNamesConfig().comparison)
