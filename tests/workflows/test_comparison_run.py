"""Tests for comparison tracking setup helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from dlkit.infrastructure.io import url_resolver

from neuralls.platform.config.models.experiments import ExperimentNamesConfig
from neuralls.platform.tracking.comparison_tracking import setup_comparison_tracking


def test_setup_comparison_tracking_creates_missing_experiment(tmp_path: Path) -> None:
    """A missing comparison experiment should be created and selected."""
    tracking_uri = f"sqlite:///{(tmp_path / 'mlruns' / 'mlflow.db').as_posix()}"
    artifact_dir = str(tmp_path / "artifacts")
    artifact_location = url_resolver.build_uri((tmp_path / "artifacts").resolve(), scheme="file")
    with (
        patch("neuralls.platform.tracking.comparison_tracking.mlflow") as mock_mlflow,
        patch("neuralls.platform.tracking.comparison_tracking.MlflowClient") as mock_client_cls,
    ):
        mock_client = mock_client_cls.return_value
        mock_client.get_experiment_by_name.return_value = None
        setup_comparison_tracking(
            tracking_uri,
            artifact_dir,
            experiment_name="CustomComparison",
        )

    mock_mlflow.set_tracking_uri.assert_called_once_with(tracking_uri)
    mock_client.create_experiment.assert_called_once_with(
        "CustomComparison",
        artifact_location=artifact_location,
    )
    mock_mlflow.set_experiment.assert_called_once_with("CustomComparison")


def test_setup_comparison_tracking_uses_model_default_when_name_missing(tmp_path: Path) -> None:
    """Missing comparison names fall back to the config-model default."""
    tracking_uri = f"sqlite:///{(tmp_path / 'mlruns' / 'mlflow.db').as_posix()}"
    artifact_dir = str(tmp_path / "artifacts")
    with (
        patch("neuralls.platform.tracking.comparison_tracking.mlflow") as mock_mlflow,
        patch("neuralls.platform.tracking.comparison_tracking.MlflowClient") as mock_client_cls,
    ):
        mock_client = mock_client_cls.return_value
        mock_client.get_experiment_by_name.return_value = object()
        setup_comparison_tracking(tracking_uri, artifact_dir)

    mock_client.create_experiment.assert_not_called()
    mock_mlflow.set_experiment.assert_called_once_with(ExperimentNamesConfig().comparison)
