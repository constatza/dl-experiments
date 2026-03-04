"""Typed handshake between training and comparison phases.

``ComparisonRun`` is the shared identity produced by ``train_batch()`` and
consumed by ``run_comparison()``.  The MLflow run_id IS the comparison
identifier — no custom UUID management.

Serialisable to/from JSON so CLI phases can be invoked independently.

Key Functions:
    - ``make_comparison_run()``: Pure constructor from training results.
    - ``make_comparison_run_from_checkpoints()``: Create from existing checkpoints (no retraining).
    - ``resolve_checkpoint_from_run()``: Pure lookup of checkpoint by experiment ID.
    - ``save_comparison_run()``: Write to JSON (side effect).
    - ``load_comparison_run()``: Read from JSON (side effect).
    - ``load_comparison_run_from_mlflow()``: Reconstruct from MLflow run (side effect).
    - ``setup_comparison_tracking()``: Configure MLflow for a comparison run (side effect).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import mlflow
from loguru import logger
from mlflow.tracking import MlflowClient

if TYPE_CHECKING:
    from neuralls.workflows.multi_training import TrainingRunResult

_COMPARISON_EXPERIMENT_NAME = "neuralls-comparisons"


@dataclass(frozen=True)
class ComparisonRun:
    """Immutable handshake between training and comparison phases.

    Produced by ``train_batch()``; consumed by ``run_comparison()``.
    The MLflow ``mlflow_run_id`` IS the comparison identifier — no custom UUID.

    Attributes:
        mlflow_run_id: MLflow batch run UUID (the comparison identifier).
        mlflow_experiment_id: MLflow experiment ID for the batch run.
        tracking_uri: MLflow tracking URI used for both phases.
        artifact_location: Explicit artifact directory from config.
        checkpoint_map: Maps experiment_id → checkpoint_path for all trained models.
        artifact_uri: MLflow artifact URI for the batch run (MLflow-managed path).
    """

    mlflow_run_id: str
    mlflow_experiment_id: str
    tracking_uri: str
    artifact_location: str
    checkpoint_map: dict[str, Path]
    artifact_uri: str


# ---------------------------------------------------------------------------
# Side-effect MLflow setup helper
# ---------------------------------------------------------------------------


def setup_comparison_tracking(
    tracking_uri: str,
    artifact_location: str,
    experiment_name: str = _COMPARISON_EXPERIMENT_NAME,
) -> None:
    """Configure MLflow for a comparison run. Idempotent.

    Sets tracking URI, creates the comparisons experiment with a fixed
    artifact_location if it does not exist, and activates it.
    Must precede every mlflow.start_run() targeting the comparisons DB.

    Args:
        tracking_uri: Explicit SQLite URI from config (never derived).
        artifact_location: Explicit artifact directory from config (never derived).
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    if client.get_experiment_by_name(experiment_name) is None:
        client.create_experiment(
            experiment_name, artifact_location=artifact_location
        )
    mlflow.set_experiment(experiment_name)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _artifact_uri_to_local_path(uri: str) -> Path:
    """Convert an MLflow artifact URI to a local filesystem Path.

    Handles both plain paths (e.g. ``mlartifacts/0/run-id/artifacts``) and
    ``file://`` URIs (e.g. ``file:///abs/path/mlruns/0/run-id/artifacts``).

    Args:
        uri: MLflow artifact URI string from ``mlflow.get_artifact_uri()``.

    Returns:
        Local filesystem ``Path`` for the artifact directory.
    """
    parsed = urlparse(uri)
    return (Path(parsed.path) if parsed.scheme == "file" else Path(uri)).resolve()


def make_comparison_run(
    parent_run_id: str,
    tracking_uri: str,
    results: list[TrainingRunResult],
    artifact_uri: str,
    mlflow_experiment_id: str = "",
    artifact_location: str = "",
) -> ComparisonRun:
    """Pure: build a ComparisonRun from a list of training results.

    Args:
        parent_run_id: MLflow batch run UUID opened by ``train_batch()``.
        tracking_uri: MLflow tracking URI.
        results: Ordered list of ``TrainingRunResult`` from ``train_batch()``.
        artifact_uri: MLflow artifact URI for the batch run.
        mlflow_experiment_id: MLflow experiment ID for the batch run.
        artifact_location: Explicit artifact directory from config.

    Returns:
        Frozen ``ComparisonRun`` with ``checkpoint_map`` populated from results.
    """
    checkpoint_map = {r.experiment_id: r.checkpoint_path for r in results}
    return ComparisonRun(
        mlflow_run_id=parent_run_id,
        mlflow_experiment_id=mlflow_experiment_id,
        tracking_uri=tracking_uri,
        artifact_location=artifact_location,
        checkpoint_map=checkpoint_map,
        artifact_uri=artifact_uri,
    )


def resolve_checkpoint_from_run(spec: Any, comparison_run: ComparisonRun) -> Path:
    """Pure: map an experiment reference in a solver spec to a checkpoint path.

    Args:
        spec: Solver spec with a ``.experiment`` attribute (experiment ID string).
        comparison_run: ComparisonRun produced by the training phase.

    Returns:
        Path to the checkpoint for the referenced experiment.

    Raises:
        KeyError: If ``spec.experiment`` is not in ``comparison_run.checkpoint_map``.
    """
    exp_id = spec.experiment
    if exp_id not in comparison_run.checkpoint_map:
        raise KeyError(
            f"Experiment '{exp_id}' not found in ComparisonRun. "
            f"Available: {list(comparison_run.checkpoint_map)}"
        )
    return comparison_run.checkpoint_map[exp_id]


# ---------------------------------------------------------------------------
# Side-effect I/O helpers (explicitly named)
# ---------------------------------------------------------------------------


def save_comparison_run(run: ComparisonRun, path: Path) -> None:
    """Write a ComparisonRun to a JSON file.

    Args:
        run: Frozen ComparisonRun to serialise.
        path: Destination path (parent directory must exist).
    """
    payload = {
        "mlflow_run_id": run.mlflow_run_id,
        "mlflow_experiment_id": run.mlflow_experiment_id,
        "tracking_uri": run.tracking_uri,
        "artifact_location": run.artifact_location,
        "checkpoint_map": {k: str(v) for k, v in run.checkpoint_map.items()},
        "artifact_uri": run.artifact_uri,
    }
    path.write_text(json.dumps(payload, indent=2))


def load_comparison_run(path: Path) -> ComparisonRun:
    """Read a ComparisonRun from a JSON file produced by ``save_comparison_run()``.

    Args:
        path: Path to ``comparison_run.json``.

    Returns:
        Reconstructed ``ComparisonRun``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        KeyError: If the JSON is missing required fields.
    """
    raw: dict[str, Any] = json.loads(path.read_text())
    return ComparisonRun(
        mlflow_run_id=raw["mlflow_run_id"],
        mlflow_experiment_id=raw.get("mlflow_experiment_id", ""),
        tracking_uri=raw["tracking_uri"],
        artifact_location=raw.get("artifact_location", ""),
        checkpoint_map={k: Path(v) for k, v in raw["checkpoint_map"].items()},
        artifact_uri=raw.get("artifact_uri", ""),
    )


def make_comparison_run_from_checkpoints(
    checkpoint_map: dict[str, Path],
    tracking_uri: str,
    artifact_location: str,
    run_name: str = "batch-comparison",
) -> ComparisonRun:
    """Create a ComparisonRun from existing checkpoints without retraining.

    Calls ``setup_comparison_tracking`` to configure MLflow, opens a new run,
    logs comparison_run.json as an artifact, and returns the handshake.
    Use this to reuse existing trained models for new comparisons.

    Args:
        checkpoint_map: Maps experiment_id → checkpoint_path for all models.
        tracking_uri: MLflow tracking URI.
        artifact_location: Explicit artifact directory from config.
        run_name: Display name for the new MLflow run.

    Returns:
        Frozen ``ComparisonRun`` ready for the comparison phase.
    """
    setup_comparison_tracking(tracking_uri, artifact_location)
    with mlflow.start_run(run_name=run_name, tags={"phase": "training_batch"}) as run:
        experiment_id = run.info.experiment_id
        run_id = run.info.run_id
        artifact_uri = mlflow.get_artifact_uri()
        mlflow.log_param("artifact_uri", artifact_uri)
        mlflow.log_param("artifact_location", artifact_location)
        for exp_id, ckpt in checkpoint_map.items():
            mlflow.log_param(f"checkpoint_{exp_id}", str(ckpt))

        cr = ComparisonRun(
            mlflow_run_id=run_id,
            mlflow_experiment_id=experiment_id,
            tracking_uri=tracking_uri,
            artifact_location=artifact_location,
            checkpoint_map=checkpoint_map,
            artifact_uri=artifact_uri,
        )
        payload = {
            "mlflow_run_id": run_id,
            "mlflow_experiment_id": experiment_id,
            "tracking_uri": tracking_uri,
            "artifact_location": artifact_location,
            "checkpoint_map": {k: str(v) for k, v in checkpoint_map.items()},
            "artifact_uri": artifact_uri,
        }
        mlflow.log_dict(payload, "comparison_run.json")
        logger.info(f"Opened MLflow run for checkpoint-based comparison: {run_id}")

    return cr


def load_comparison_run_from_mlflow(run_id: str, tracking_uri: str) -> ComparisonRun:
    """Reconstruct a ComparisonRun from an existing MLflow batch run.

    Reads params from the batch run to rebuild ``checkpoint_map``.  Each entry
    must have been logged as ``checkpoint_{experiment_id}`` by the batch run.

    Args:
        run_id: MLflow batch run UUID.
        tracking_uri: MLflow tracking URI.

    Returns:
        Reconstructed ``ComparisonRun``.

    Raises:
        ValueError: If the run or expected params are missing.
    """
    client = MlflowClient(tracking_uri=tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)

    batch_run = client.get_run(run_id)
    experiment_id = batch_run.info.experiment_id
    artifact_uri = batch_run.data.params.get(
        "artifact_uri", batch_run.info.artifact_uri or ""
    )
    artifact_location = batch_run.data.params.get("artifact_location", "")

    checkpoint_map: dict[str, Path] = {}
    for key, value in batch_run.data.params.items():
        if key.startswith("checkpoint_"):
            exp_id = key[len("checkpoint_"):]
            checkpoint_map[exp_id] = Path(value)

    return ComparisonRun(
        mlflow_run_id=run_id,
        mlflow_experiment_id=experiment_id,
        tracking_uri=tracking_uri,
        artifact_location=artifact_location,
        checkpoint_map=checkpoint_map,
        artifact_uri=artifact_uri,
    )
