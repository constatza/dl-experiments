"""Output saving and MLflow logging for inference workflow.

This module provides functions for saving inference outputs to disk and
optionally logging them to MLflow for experiment tracking.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
from loguru import logger

from neuralls.domain.inference import InferenceOutputs, InferencePredictions
from neuralls.platform.config.models.workspace import ExperimentWorkspace
from neuralls.platform.reporting.prediction_csv import save_prediction_samples_to_csv
from neuralls.platform.reporting.synthetic import save_synthetic_results
from neuralls.platform.reporting.plots import plot_parity_and_residuals, plot_prediction_diagnostics
from neuralls.platform.storage.filesystem import derive_model_identifier, sanitize_identifier
from neuralls.platform.tracking.mlflow import build_run_config, finalize_run, open_run


PREDICTION_ARTIFACTS: tuple[str, ...] = ("figures", "predictions")


def get_session_name(settings: Any) -> str | None:
    """Extract session name from settings.

    Args:
        settings: DLKit GeneralSettings object

    Returns:
        Session name or None if not configured
    """
    session = getattr(settings, "SESSION", None)
    name = getattr(session, "name", None) if session else None
    return name if isinstance(name, str) and name else None


def derive_run_identifier(
    settings: Any,
    workspace: ExperimentWorkspace,
    checkpoint_path: Path,
    config_path: Path,
) -> str:
    """Derive identifier for this inference run.

    Priority:
    1. Checkpoint parent directory (if named appropriately)
    2. Checkpoint stem
    3. Workspace run_id (if not default)
    4. Model identifier from settings

    Args:
        settings: DLKit GeneralSettings object
        workspace: Experiment workspace
        checkpoint_path: Path to model checkpoint
        config_path: Path to config file

    Returns:
        Sanitized run identifier string
    """
    # Try checkpoint path
    if checkpoint_path:
        parent = checkpoint_path.parent
        if parent.name.lower() == "checkpoints" and parent.parent.name:
            return parent.parent.name
        if checkpoint_path.stem:
            return checkpoint_path.stem

    # Try workspace run_id
    run_id = workspace.run_id
    if isinstance(run_id, str) and run_id and not run_id.lower().startswith("dlkit-session"):
        return run_id

    # Fallback to model identifier
    return derive_model_identifier(settings, workspace, config_path)


def start_mlflow_run(
    settings: Any,
    workspace: ExperimentWorkspace,
    dataset_id: str,
    enable_mlflow: bool,
) -> Any:
    """Start MLflow run for inference tracking.

    Args:
        settings: DLKit GeneralSettings object
        workspace: Experiment workspace
        dataset_id: Dataset identifier
        enable_mlflow: Whether to enable MLflow logging

    Returns:
        MLflow run state or None if disabled/unavailable
    """
    config = build_run_config(
        settings=settings,
        workspace_root=workspace.root_dir,
        dataset_id=dataset_id,
        model_name=workspace.run_id,
        session_name=get_session_name(settings),
        enabled=enable_mlflow,
    )
    if config is None:
        return None
    try:
        return open_run(config)
    except ModuleNotFoundError:
        logger.info("MLflow not installed; skipping MLflow logging.")
        return None


def save_predictions_csv(
    predictions: InferencePredictions,
    output_dir: Path,
    identifier: str,
) -> list[Path]:
    """Save predictions to CSV files.

    Args:
        predictions: Inference predictions with targets
        output_dir: Directory for saving CSVs
        identifier: Filename prefix/identifier

    Returns:
        List of saved CSV paths
    """
    y_true = predictions.targets.get("y_true", np.array([]))
    y_pred = predictions.predictions.get("y_pred", np.array([]))

    csv_paths = save_prediction_samples_to_csv(
        y_true=y_true,
        y_pred=y_pred,
        output_dir=output_dir,
        filename_prefix=identifier,
    )

    if csv_paths:
        logger.info(f"Saved prediction samples to CSV: {[str(p) for p in csv_paths]}")

    return csv_paths or []


def save_synthetic_predictions(
    predictions: InferencePredictions,
    output_dir: Path,
    identifier: str,
) -> None:
    """Save synthetic benchmark predictions.

    Args:
        predictions: Inference predictions with synthetic targets
        output_dir: Directory for saving results
        identifier: Filename prefix/identifier
    """
    y_true = predictions.targets.get("y_true", np.array([]))
    y_pred = predictions.predictions.get("y_pred", np.array([]))

    save_synthetic_results(
        output_dir,
        y_true=y_true,
        y_pred=y_pred,
        identifier=identifier,
    )


def create_diagnostic_plots(
    predictions: InferencePredictions,
    figures_dir: Path,
    identifier: str,
) -> dict[str, Path]:
    """Generate diagnostic plots for predictions.

    Creates two plots:
    1. Parity and residuals plot
    2. Detailed diagnostics plot

    Args:
        predictions: Inference predictions with targets
        figures_dir: Directory for saving plots
        identifier: Filename prefix/identifier

    Returns:
        Dictionary mapping plot types to saved paths
    """
    figures_dir.mkdir(parents=True, exist_ok=True)

    y_true = predictions.targets.get("y_true", np.array([]))
    y_pred = predictions.predictions.get("y_pred", np.array([]))

    plot_paths = {}

    # Parity and residuals plot
    parity_path = figures_dir / f"parity_residuals_{identifier}.png"
    plot_parity_and_residuals(y_pred, y_true, sample=0, save_path=parity_path, show=False)
    plot_paths["parity"] = parity_path

    # Diagnostics plot
    diagnostic_path = figures_dir / f"diagnostics_{identifier}.png"
    plot_prediction_diagnostics(y_pred, y_true, sample=0, save_path=diagnostic_path, show=False)
    plot_paths["diagnostics"] = diagnostic_path

    logger.info(f"Saved diagnostic plots to {figures_dir}")

    return plot_paths


def save_inference_outputs(
    predictions: InferencePredictions,
    workspace: ExperimentWorkspace,
    settings: Any,
    checkpoint_path: Path,
    config_path: Path,
    dataset_id: str,
    save_plots: bool = True,
    figures_dir: Path | None = None,
) -> InferenceOutputs:
    """Save all inference outputs to workspace.

    Args:
        predictions: Inference predictions with targets
        workspace: Experiment workspace
        settings: DLKit GeneralSettings object
        checkpoint_path: Path to model checkpoint
        config_path: Path to config file
        dataset_id: Dataset identifier
        save_plots: Whether to generate diagnostic plots
        figures_dir: Custom figures directory (optional)

    Returns:
        InferenceOutputs with saved paths and metrics
    """
    # Derive identifier
    run_identifier = sanitize_identifier(
        derive_run_identifier(settings, workspace, checkpoint_path, config_path)
    )
    dataset_slug = sanitize_identifier(str(dataset_id))
    identifier = f"{dataset_slug}-{run_identifier}"

    # Save CSV predictions
    csv_paths = save_predictions_csv(predictions, workspace.predictions_dir, identifier)

    # Save plots if enabled
    plot_paths_dict = {}
    if save_plots:
        output_figures_dir = Path(figures_dir) if figures_dir is not None else workspace.figures_dir
        plot_paths_dict = create_diagnostic_plots(predictions, output_figures_dir, identifier)

    # Extract metrics
    duration = predictions.metadata.get("duration_seconds", 0.0)
    metrics = {"duration_seconds": float(duration)}

    return InferenceOutputs(
        csv_paths=csv_paths,
        plot_paths=list(plot_paths_dict.values()),
        metrics=metrics,
    )


def write_mlflow_sidecar(
    path: Path,
    *,
    run_id: str,
    experiment_id: str,
    tracking_uri: str,
    artifacts_destination: str,
) -> None:
    """Write MLflow run metadata sidecar JSON next to checkpoint.

    Args:
        path: Destination path (e.g. checkpoint_dir / "mlflow_run.json")
        run_id: MLflow run ID
        experiment_id: MLflow experiment ID
        tracking_uri: SQLite or HTTP tracking URI
        artifacts_destination: Local filesystem path for MLflow artifacts
    """
    data = {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "tracking_uri": tracking_uri,
        "artifacts_destination": artifacts_destination,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    logger.debug(f"Wrote MLflow sidecar: {path}")


def read_mlflow_sidecar(path: Path) -> dict[str, str] | None:
    """Read MLflow run metadata sidecar JSON.

    Args:
        path: Path to mlflow_run.json sidecar

    Returns:
        Parsed sidecar dict, or None if file not found
    """
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Invalid MLflow sidecar payload at {path}: expected object")
    return {str(key): str(value) for key, value in cast(dict[object, object], data).items()}


def finalize_mlflow_run(
    mlflow_state: Any,
    metrics: dict[str, float],
    workspace: ExperimentWorkspace,
) -> None:
    """Finalize MLflow run with metrics and artifacts.

    Args:
        mlflow_state: MLflow run state
        metrics: Metrics to log
        workspace: Experiment workspace for artifacts
    """
    if mlflow_state is None:
        return

    finalize_run(
        mlflow_state,
        metrics=metrics,
        workspace_root=workspace.root_dir,
        allowlist=PREDICTION_ARTIFACTS,
    )
