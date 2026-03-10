"""MLflow setup helpers for comparison workflows."""

from __future__ import annotations

import mlflow
from mlflow.tracking import MlflowClient

_COMPARISON_EXPERIMENT_NAME = "Comparisons"


def setup_comparison_tracking(
    tracking_uri: str,
    artifact_location: str | None = None,
    experiment_name: str = _COMPARISON_EXPERIMENT_NAME,
) -> None:
    """Configure MLflow tracking for comparison runs."""
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    if client.get_experiment_by_name(experiment_name) is None:
        client.create_experiment(experiment_name, artifact_location=artifact_location)
    mlflow.set_experiment(experiment_name)
