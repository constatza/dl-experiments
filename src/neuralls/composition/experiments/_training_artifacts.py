"""MLflow run setup, artifact staging, and checkpoint resolution for the training workflow."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from mlflow.tracking import MlflowClient

from neuralls.composition.experiments.runtime_dataset_contract import RuntimeDatasetContract
from neuralls.composition.experiments.model_resolution import _find_single_checkpoint
from neuralls.composition.tracking.run_specs import build_training_run_spec, format_run_timestamp
from neuralls.platform.config.models.experiments import ExperimentEntry, ExperimentNamesConfig
from neuralls.platform.storage.checkpoints import get_latest_checkpoint
from neuralls.platform.storage.training_artifacts import (
    coerce_jsonable,
    save_training_predictions,
)
from neuralls.platform.tracking.mlflow import MlflowRunConfig, runtime_paths_from_env
from neuralls.platform.tracking.mlflow_client import (
    find_mlflow_run,
    log_diagnostics_to_mlflow,
)
from neuralls.platform.reporting.training_diagnostics import (
    compute_diagnostics,
    write_diagnostics_figure,
)


def _resolve_training_experiment_name(mlflow_experiment_name: str | None) -> str:
    """Resolve the training MLflow experiment name from caller input or config defaults.

    Args:
        mlflow_experiment_name: Caller-supplied name, or None to use the config default.

    Returns:
        Resolved experiment name.
    """
    if mlflow_experiment_name is not None:
        return mlflow_experiment_name
    return ExperimentNamesConfig().training


def _build_training_run_config(
    *,
    experiment_id: str | None,
    experiment_display_name: str,
    dataset_registry_id: str | None,
    model_registry_id: str | None,
    dataset_display_name: str,
    mlflow_experiment_name: str | None,
    runtime_mlflow_env: Mapping[str, str],
    workspace_root: Path,
    parent_run_id: str | None = None,
) -> MlflowRunConfig:
    """Build the execute()-time MLflow run config for training.

    Args:
        experiment_id: Registry experiment ID, or None for ad-hoc runs.
        experiment_display_name: Human-readable experiment name.
        dataset_registry_id: Registry dataset ID, or None.
        model_registry_id: Registry model ID, or None.
        dataset_display_name: Human-readable dataset name (unused in run name).
        mlflow_experiment_name: Override for the MLflow experiment bucket name.
        runtime_mlflow_env: MLflow environment variable mapping.
        workspace_root: Root directory for the training workspace.
        parent_run_id: Optional parent run UUID for nested tracking.

    Returns:
        Fully configured MlflowRunConfig.
    """
    _ = dataset_display_name
    experiment_name = _resolve_training_experiment_name(mlflow_experiment_name)
    paths = runtime_paths_from_env(runtime_mlflow_env)
    if experiment_id and dataset_registry_id and model_registry_id:
        entry = ExperimentEntry(
            id=experiment_id,
            dataset=dataset_registry_id,
            model=model_registry_id,
            display_name=experiment_display_name,
        )
        return build_training_run_spec(
            entry=entry,
            experiment_name=experiment_name,
            paths=paths,
            workspace_root=workspace_root,
            include_timestamp=parent_run_id is None,
        )
    ts = f" | {format_run_timestamp()}" if parent_run_id is None else ""
    return MlflowRunConfig(
        experiment_name=experiment_name,
        run_name=f"{experiment_display_name}{ts}",
        tags={},
        paths=paths,
        workspace_root=workspace_root,
    )


def _resolve_mlflow_run_ids(
    *,
    training_result: Any,
    fallback_tracking_uri: str | None,
    experiment_name: str,
    run_name: str,
) -> tuple[str, str, str] | None:
    """Resolve MLflow tracking URI, experiment ID, and run ID for a training run.

    Args:
        training_result: DLKit training result object.
        fallback_tracking_uri: Tracking URI to use when not found in result metrics.
        experiment_name: MLflow experiment name for fallback lookup.
        run_name: MLflow run name for fallback lookup.

    Returns:
        (tracking_uri, experiment_id, run_id) tuple, or None if unresolvable.
    """
    metrics = getattr(training_result, "metrics", {}) or {}
    tracking_uri = metrics.get("mlflow_tracking_uri") or fallback_tracking_uri
    experiment_id = metrics.get("mlflow_experiment_id")
    run_id = metrics.get("mlflow_run_id")
    if isinstance(tracking_uri, str) and isinstance(experiment_id, str) and isinstance(run_id, str):
        return tracking_uri, experiment_id, run_id

    if not isinstance(tracking_uri, str):
        return None

    direct_run_id = getattr(training_result, "run_id", None)
    if isinstance(direct_run_id, str) and direct_run_id:
        try:
            resolved_experiment_id = (
                MlflowClient(tracking_uri=tracking_uri).get_run(direct_run_id).info.experiment_id
            )
        except Exception:  # noqa: BLE001
            resolved_experiment_id = None
        if isinstance(resolved_experiment_id, str) and resolved_experiment_id:
            return tracking_uri, resolved_experiment_id, direct_run_id

    found = find_mlflow_run(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        run_name=run_name,
    )
    if found is None:
        return None

    fallback_experiment_id, fallback_run_id = found
    return tracking_uri, fallback_experiment_id, fallback_run_id


def _log_training_context(
    *,
    tracking_uri: str,
    run_id: str,
    experiment_id: str | None,
    experiment_display_name: str | None,
    dataset_id: str,
    dataset_display_name: str,
    dataset_registry_id: str | None,
    model_registry_id: str | None,
    model_display_name: str,
) -> None:
    """Log stable ids and display names to the training MLflow run.

    Args:
        tracking_uri: MLflow tracking URI.
        run_id: Target MLflow run ID.
        experiment_id: Registry experiment ID, or None.
        experiment_display_name: Human-readable experiment name, or None.
        dataset_id: Dataset workspace ID.
        dataset_display_name: Human-readable dataset name.
        dataset_registry_id: Registry dataset ID, or None.
        model_registry_id: Registry model ID, or None.
        model_display_name: Human-readable model name.
    """
    client = MlflowClient(tracking_uri=tracking_uri)
    params: dict[str, str] = {
        "dataset_id": dataset_id,
        "dataset_display_name": dataset_display_name,
        "model_display_name": model_display_name,
    }
    if experiment_id is not None:
        params["experiment_id"] = experiment_id
    if experiment_display_name is not None:
        params["experiment_display_name"] = experiment_display_name
    if dataset_registry_id is not None:
        params["dataset_registry_id"] = dataset_registry_id
    if model_registry_id is not None:
        params["model_registry_id"] = model_registry_id
    for key, value in params.items():
        client.log_param(run_id, key, value)


def _extract_evaluation_arrays(
    all_numpy: Any,
    contract: RuntimeDatasetContract,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract prediction and target arrays from a normalized numpy payload.

    Args:
        all_numpy: Normalized numpy payload dict (predictions + targets).
        contract: Runtime dataset contract for key resolution.

    Returns:
        (y_pred, y_true) arrays, or None if the payload is absent or incomplete.
    """
    if not isinstance(all_numpy, Mapping):
        return None

    predictions_raw = all_numpy.get("predictions")
    targets_raw = all_numpy.get("targets")
    if not isinstance(predictions_raw, Mapping) or not isinstance(targets_raw, Mapping):
        return None

    y_pred_raw = predictions_raw.get(contract.prediction_name)
    y_true_raw = targets_raw.get(contract.target_name)
    if y_pred_raw is None or y_true_raw is None:
        return None

    y_pred = np.asarray(y_pred_raw)
    y_true = np.asarray(y_true_raw)
    if y_pred.size == 0 or y_true.size == 0:
        return None
    return y_pred, y_true


def _normalize_training_numpy_payload(
    all_numpy: Any,
    contract: RuntimeDatasetContract,
) -> Mapping[str, Any] | None:
    """Normalize DLKit prediction payloads into the local runtime contract once.

    Args:
        all_numpy: Raw DLKit numpy payload (may expose 'output' instead of the
            canonical prediction key).
        contract: Runtime dataset contract for key resolution.

    Returns:
        Normalized payload with predictions keyed by contract.prediction_name,
        or None if the payload structure is invalid.

    Raises:
        ValueError: If neither the canonical prediction key nor 'output' is found.
    """
    if not isinstance(all_numpy, Mapping):
        return None

    predictions_raw = all_numpy.get("predictions")
    targets_raw = all_numpy.get("targets")
    if not isinstance(predictions_raw, Mapping) or not isinstance(targets_raw, Mapping):
        return None

    if contract.prediction_name in predictions_raw:
        prediction_value = predictions_raw[contract.prediction_name]
    elif "output" in predictions_raw:
        prediction_value = predictions_raw["output"]
    else:
        raise ValueError(
            "Training prediction payload must expose either the canonical prediction key "
            f"'{contract.prediction_name}' or the DLKit boundary key 'output'."
        )

    normalized = dict(all_numpy)
    normalized["predictions"] = {contract.prediction_name: prediction_value}
    normalized["targets"] = dict(targets_raw)
    return normalized


def _get_normalized_training_numpy_payload(
    training_result: Any,
    contract: RuntimeDatasetContract,
) -> Mapping[str, Any] | None:
    """Read and normalize DLKit numpy payloads when the result exposes them.

    Args:
        training_result: DLKit training result object.
        contract: Runtime dataset contract for key resolution.

    Returns:
        Normalized payload, or None if the result has no to_numpy() method.
    """
    to_numpy = getattr(training_result, "to_numpy", None)
    if not callable(to_numpy):
        return None
    return _normalize_training_numpy_payload(to_numpy(), contract)


def _log_training_evaluation(
    tracking_uri: str,
    run_id: str,
    numpy_payload: Mapping[str, Any] | None,
    figures_dir: Path,
    contract: RuntimeDatasetContract,
) -> None:
    """Compute diagnostics from training predictions and log to an existing MLflow run.

    Uses the predictions and targets already captured by trainer.predict() during
    training. Delegates figure writing and MLflow logging to dedicated helpers.

    Args:
        tracking_uri: MLflow tracking URI (HTTP or SQLite).
        run_id: Existing MLflow run ID to reopen.
        numpy_payload: Normalized DLKit prediction/target payload.
        figures_dir: Directory to write the diagnostics figure.
        contract: Runtime dataset contract for array key resolution.
    """
    selected = _extract_evaluation_arrays(numpy_payload, contract)
    if selected is None:
        logger.warning(
            "Skipping training diagnostics logging: unable to resolve prediction/target arrays."
        )
        return

    y_pred, y_true = selected
    try:
        diagnostics = compute_diagnostics(y_pred, y_true)
        figure_path = write_diagnostics_figure(y_true, y_pred, diagnostics, figures_dir)
        log_diagnostics_to_mlflow(tracking_uri, run_id, diagnostics, figure_path)
        metrics_dir = figures_dir.parent / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / "training_diagnostics.json").write_text(
            json.dumps({k: float(v) for k, v in diagnostics.metrics.items()}, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Skipping MLflow diagnostics logging for run {}: {}",
            run_id,
            exc,
        )


def _stage_training_artifacts(
    *,
    workspace: Any,
    training_result: Any,
    numpy_payload: Mapping[str, Any] | None,
    model_config_path: Path,
    data_config_path: Path | None,
) -> None:
    """Stage full training artifacts into the workspace for MLflow upload.

    Args:
        workspace: Experiment workspace (provides root_dir, predictions_dir).
        training_result: DLKit training result object.
        numpy_payload: Normalized prediction/target payload.
        model_config_path: Path to the model TOML config.
        data_config_path: Path to the dataset TOML config, or None.
    """
    import shutil

    config_dir = workspace.root_dir / "config"
    metrics_dir = workspace.root_dir / "metrics"
    config_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(model_config_path, config_dir / model_config_path.name)
    if data_config_path is not None:
        shutil.copy2(data_config_path, config_dir / data_config_path.name)

    metrics_payload = getattr(training_result, "metrics", {}) or {}
    (metrics_dir / "training_result_metrics.json").write_text(
        json.dumps(coerce_jsonable(metrics_payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    save_training_predictions(
        training_result,
        workspace.predictions_dir,
        numpy_payload=numpy_payload,
    )


def _download_training_checkpoint(
    *,
    tracking_uri: str,
    run_id: str,
    destination: Path,
) -> Path:
    """Download checkpoint artifacts for a completed MLflow run.

    Args:
        tracking_uri: MLflow tracking URI.
        run_id: MLflow run ID whose checkpoints to download.
        destination: Local directory to download artifacts into.

    Returns:
        Path to the single checkpoint file found under the downloaded artifacts.

    Raises:
        RuntimeError: If the MLflow download fails.
    """
    client = MlflowClient(tracking_uri=tracking_uri)
    try:
        downloaded_root = Path(
            client.download_artifacts(
                run_id=run_id,
                path="checkpoints",
                dst_path=str(destination),
            )
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Could not download checkpoints for run '{run_id}' from MLflow."
        ) from exc
    return _find_single_checkpoint(downloaded_root)


def _resolve_training_checkpoint(
    *,
    training_result: Any,
    workspace: Any,
    tracking_uri: str | None,
    run_id: str | None,
) -> Path:
    """Resolve the produced checkpoint from local artifacts or MLflow.

    Tries in order: checkpoint_path attribute, artifacts dict, local checkpoint dir,
    then downloads from MLflow if tracking coordinates are available.

    Args:
        training_result: DLKit training result object.
        workspace: Experiment workspace (provides checkpoint_dir, root_dir).
        tracking_uri: MLflow tracking URI for remote download fallback.
        run_id: MLflow run ID for remote download fallback.

    Returns:
        Path to the resolved checkpoint file.

    Raises:
        RuntimeError: If no checkpoint is found through any mechanism.
    """
    checkpoint_path = getattr(training_result, "checkpoint_path", None)
    if checkpoint_path is not None:
        direct_checkpoint = Path(checkpoint_path)
        if direct_checkpoint.exists():
            return direct_checkpoint

    artifacts = getattr(training_result, "artifacts", {}) or {}
    for key in ("best_checkpoint", "last_checkpoint"):
        candidate = artifacts.get(key)
        if candidate is None:
            continue
        artifact_checkpoint = Path(candidate)
        if artifact_checkpoint.exists():
            return artifact_checkpoint

    local_checkpoint = get_latest_checkpoint(workspace.checkpoint_dir)
    if local_checkpoint is not None and local_checkpoint.exists():
        return local_checkpoint

    if tracking_uri and run_id:
        download_root = workspace.root_dir / "mlflow-downloads"
        logger.info(
            "Checkpoint missing locally for run {}. Downloading from MLflow artifacts into {}.",
            run_id,
            download_root,
        )
        return _download_training_checkpoint(
            tracking_uri=tracking_uri,
            run_id=run_id,
            destination=download_root,
        )

    raise RuntimeError(f"No checkpoint found in {workspace.checkpoint_dir}")
