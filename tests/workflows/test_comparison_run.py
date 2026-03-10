"""Tests for comparison tracking setup helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from neuralls.workflows.comparison_run import setup_comparison_tracking


def test_setup_comparison_tracking_creates_missing_experiment(tmp_path: Path) -> None:
    """A missing comparison experiment should be created and selected."""
    tracking_uri = f"sqlite:///{(tmp_path / 'mlruns' / 'mlflow.db').as_posix()}"
    artifact_dir = str(tmp_path / "artifacts")
    with (
        patch("neuralls.workflows.comparison_run.mlflow") as mock_mlflow,
        patch("neuralls.workflows.comparison_run.MlflowClient") as mock_client_cls,
    ):
        mock_client = mock_client_cls.return_value
        mock_client.get_experiment_by_name.return_value = None
        setup_comparison_tracking(tracking_uri, artifact_dir)

    mock_mlflow.set_tracking_uri.assert_called_once_with(tracking_uri)
    mock_client.create_experiment.assert_called_once_with(
        "Comparisons",
        artifact_location=artifact_dir,
    )
    mock_mlflow.set_experiment.assert_called_once_with("Comparisons")


def test_setup_comparison_tracking_reuses_existing_experiment(tmp_path: Path) -> None:
    """Existing comparison experiments should not be recreated."""
    tracking_uri = f"sqlite:///{(tmp_path / 'mlruns' / 'mlflow.db').as_posix()}"
    artifact_dir = str(tmp_path / "artifacts")
    with (
        patch("neuralls.workflows.comparison_run.mlflow") as mock_mlflow,
        patch("neuralls.workflows.comparison_run.MlflowClient") as mock_client_cls,
    ):
        mock_client = mock_client_cls.return_value
        mock_client.get_experiment_by_name.return_value = object()
        setup_comparison_tracking(tracking_uri, artifact_dir)

    mock_client.create_experiment.assert_not_called()
    mock_mlflow.set_experiment.assert_called_once_with("Comparisons")
