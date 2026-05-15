"""MLflow client utilities for training workflows.

Pure side-effecting helpers that wrap mlflow / MlflowClient calls.
No training logic — kept separate so training.py and multi_training.py
stay free of low-level MLflow plumbing.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


_WORKSPACE_ARTIFACT_DIRS: tuple[str, ...] = (
    "config",
    "figures",
    "metrics",
    "predictions",
)


def fetch_mlflow_metrics(run_id: str, tracking_uri: str) -> dict[str, float]:
    """Query MLflow for all metrics belonging to a run.

    Args:
        run_id: MLflow run UUID.
        tracking_uri: SQLite or HTTP tracking URI.

    Returns:
        Flat dict of ``{metric_key: value}`` for the run.
    """
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)
    return dict(run.data.metrics)


def log_artifacts_to_mlflow(
    tracking_uri: str,
    run_id: str,
    workspace_root: Path,
) -> None:
    """Upload the staged workspace artifacts to an existing MLflow run.

    Args:
        tracking_uri: MLflow tracking URI.
        run_id: Existing MLflow run to upload into.
        workspace_root: Local directory whose contents are uploaded.
    """
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)
    for artifact_dir in _WORKSPACE_ARTIFACT_DIRS:
        path = workspace_root / artifact_dir
        if not path.exists():
            continue
        client.log_artifacts(run_id, str(path))


def find_mlflow_run(
    *,
    tracking_uri: str,
    experiment_name: str,
    run_name: str | None,
) -> tuple[str, str] | None:
    """Look up the most recent training run by experiment + run name.

    Args:
        tracking_uri: MLflow tracking URI.
        experiment_name: Experiment name to search in.
        run_name: Optional run name filter.

    Returns:
        ``(experiment_id, run_id)`` when found, otherwise ``None``.
    """
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None

    filter_string = f"attributes.run_name = '{run_name}'" if run_name else ""
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_string,
        max_results=1,
        order_by=["attributes.start_time DESC"],
    )
    if not runs:
        return None
    return experiment.experiment_id, runs[0].info.run_id


@contextmanager
def parent_run_context(parent_run_id: str | None) -> Iterator[None]:
    """Temporarily set MLFLOW_PARENT_RUN_ID for nested MLflow runs.

    When ``parent_run_id`` is ``None`` this is a no-op context manager.

    Args:
        parent_run_id: Optional MLflow parent run UUID. When set, injects
            ``MLFLOW_PARENT_RUN_ID`` into the environment for the duration
            of the context and restores the previous value on exit.

    Yields:
        None
    """
    if parent_run_id is None:
        yield
        return

    previous = os.environ.get("MLFLOW_PARENT_RUN_ID")
    os.environ["MLFLOW_PARENT_RUN_ID"] = parent_run_id
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("MLFLOW_PARENT_RUN_ID", None)
        else:
            os.environ["MLFLOW_PARENT_RUN_ID"] = previous


def log_diagnostics_to_mlflow(
    tracking_uri: str,
    run_id: str,
    diagnostics: object,
    figure_path: Path,
) -> None:
    """Log diagnostics metrics and figure to an existing MLflow run.

    Args:
        tracking_uri: MLflow tracking URI (HTTP or SQLite).
        run_id: Existing MLflow run ID to reopen.
        diagnostics: Diagnostics result with ``metrics`` dict attribute.
        figure_path: Path to the figure file to upload.
    """
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)
    metrics = getattr(diagnostics, "metrics", {})
    for key, value in metrics.items():
        client.log_metric(run_id, key, float(value))
    if figure_path.exists():
        client.log_artifact(run_id, str(figure_path), artifact_path="figures")
