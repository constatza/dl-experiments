"""MLflow setup helpers for comparison workflows."""

from __future__ import annotations

import mlflow
from mlflow.tracking import MlflowClient

from neuralls.platform.config.models.experiments import ExperimentNamesConfig
from neuralls.platform.config.resolution import to_mlflow_artifact_location


def setup_comparison_tracking(
    tracking_uri: str,
    artifact_location: str | None = None,
    experiment_name: str | None = None,
) -> None:
    """Configure MLflow tracking for comparison runs."""
    resolved_experiment_name = experiment_name or ExperimentNamesConfig().comparison
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    if client.get_experiment_by_name(resolved_experiment_name) is None:
        resolved_artifact_location = (
            to_mlflow_artifact_location(artifact_location) if artifact_location else None
        )
        client.create_experiment(
            resolved_experiment_name,
            artifact_location=resolved_artifact_location,
        )
    mlflow.set_experiment(resolved_experiment_name)
