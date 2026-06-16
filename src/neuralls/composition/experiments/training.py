"""Training orchestration helpers consumed by CLI scripts and workflows."""

from __future__ import annotations


import json
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dlkit.infrastructure.config.core.patching import patch_model
from dlkit.infrastructure.config.data_entries import (
    DataEntry,
    DataRole,
    ValueEntry,
    ZarrEntry,
)
from dlkit.infrastructure.config.transform_settings import TransformSettings
from dlkit.infrastructure.config.dataset_settings import DatasetSettings
from dlkit.infrastructure.config.workflow_configs import (
    OptimizationWorkflowConfig,
    TrainingWorkflowConfig,
)
from dlkit.interfaces.api import execute
from dlkit.interfaces.api.domain.override_types import ExecutionOverrides
from loguru import logger
from mlflow.tracking import MlflowClient
import numpy as np

from neuralls.platform.config.models.workspace import ExperimentWorkspace
from neuralls.composition.experiments.runtime_dataset_contract import (
    RuntimeDatasetContract,
    default_training_dataset_contract,
)
from neuralls.platform.config.models.experiments import ExperimentEntry, ExperimentNamesConfig
from neuralls.platform.config.settings import NeurallsSettings, require_settings
from neuralls.composition.experiments.assembler import load_experiment
from neuralls.platform.storage.checkpoints import get_latest_checkpoint
from neuralls.platform.tracking.environment import scoped_mlflow_environment
from neuralls.platform.tracking.mlflow import (
    MlflowRunConfig,
    build_runtime_environment,
    resolve_runtime_tracking_config,
    runtime_paths_from_env,
)
from neuralls.platform.storage.datasets import load_dense_training_arrays
from neuralls.platform.storage.training_artifacts import (
    TrainingArrays,
    coerce_jsonable,
    load_training_arrays,
    save_training_predictions,
)
from neuralls.platform.tracking.extra_features import log_extra_feature_names_tag
from neuralls.composition.experiments.feature_providers import (
    registered_extra_names,
    resolve_extra_feature,
)
from neuralls.platform.reporting.training_diagnostics import (
    compute_diagnostics,
    write_diagnostics_figure,
)
from neuralls.platform.reporting.predictions import write_mlflow_sidecar
from neuralls.platform.tracking.mlflow_client import (
    find_mlflow_run,
    log_artifacts_to_mlflow,
    log_diagnostics_to_mlflow,
    parent_run_context,
)
from neuralls.composition.experiments.model_resolution import _find_single_checkpoint
from neuralls.composition.tracking.run_specs import build_training_run_spec, format_run_timestamp

GRAPH_DATASET_NAME: str = "GraphDataset"
FLEXIBLE_DATASET_NAME: str = "FlexibleDataset"
type TrainingWorkflowSettings = TrainingWorkflowConfig | OptimizationWorkflowConfig


@dataclass(frozen=True)
class TrainingResult:
    """Immutable training result data.

    Attributes:
        checkpoint_path: Path to saved model checkpoint (.ckpt file)
        experiment_dir: Root experiment directory
        data_dir: Directory containing training data
    """

    checkpoint_path: Path
    experiment_dir: Path
    data_dir: Path


def _resolve_training_experiment_name(mlflow_experiment_name: str | None) -> str:
    """Resolve the training MLflow experiment name from caller input or config defaults."""
    if mlflow_experiment_name is not None:
        return mlflow_experiment_name
    return ExperimentNamesConfig().training


def _extra_feature_names_from_settings(
    settings: TrainingWorkflowSettings,
    contract: RuntimeDatasetContract,
) -> frozenset[str]:
    """Extract extra feature names declared in [[DATASET.features]] beyond x and matrix.

    Args:
        settings: DLKit workflow settings; DATASET.features carries the TOML declarations.
        contract: Provides the base names (primary_input_name, matrix_input_name) to exclude.

    Returns:
        Frozenset of extra feature names the model TOML declares beyond the base entries.
    """
    dataset = settings.DATASET
    match dataset:
        case None:
            return frozenset()
        case _:
            base = {contract.primary_input_name, contract.matrix_input_name}
            return frozenset(
                e.name for e in dataset.features if e.name is not None and e.name not in base
            )


def _load_and_prepare_data(
    settings: TrainingWorkflowSettings,
    workspace: ExperimentWorkspace,
    contract: RuntimeDatasetContract,
) -> tuple[TrainingArrays, list[DataEntry], list[DataEntry]]:
    """Load training data and create Feature/Target configurations.

    This function:
    1. Resolves file-backed dataset artifacts (paths)
    2. Loads dense numpy arrays from rhs.zarr/ and solutions.zarr/
    3. Creates DLKit in-memory feature entries (ValueEntry/ZarrEntry)
    4. Creates DLKit in-memory target entries (ValueEntry with DataRole.TARGET)

    Args:
        settings: DLKit training or optimization workflow settings.
        workspace: Experiment workspace (provides data_dir path).
        contract: Runtime dataset contract defining canonical entry names.

    Returns:
        Tuple of (arrays, features, targets) where:
            - arrays: Resolved data artifact paths
            - features: List of in-memory / path-dropped feature configs for DLKit
            - targets: List of in-memory target configs for DLKit
    """
    arrays = load_training_arrays(workspace.data_dir)
    logger.info(
        "Loading training arrays ({} samples) from {}", arrays.sample_count, workspace.data_dir
    )
    rhs_data, solutions_data = load_dense_training_arrays(workspace.data_dir)
    logger.info(
        "Training arrays loaded — rhs {}, solutions {}", rhs_data.shape, solutions_data.shape
    )
    extra_names = _extra_feature_names_from_settings(settings, contract)
    features = _create_feature_configs(arrays, rhs_data, contract, extra_names)
    targets = _create_target_configs(solutions_data, contract)
    return arrays, features, targets


def _create_feature_configs(
    arrays: TrainingArrays,
    rhs_data: np.ndarray,
    contract: RuntimeDatasetContract,
    declared_extra_names: frozenset[str],
) -> list[DataEntry]:
    """Create in-memory Feature configs from dataset artifacts.

    Args:
        arrays: Training data artifact paths (used for matrix path).
        rhs_data: RHS array loaded from rhs.zarr.
        contract: Runtime dataset entry name contract.
        declared_extra_names: Extra feature names declared in [[DATASET.features]]
            beyond the base ``x`` and ``matrix`` entries.

    Returns:
        List of feature entries: base entries (rhs, matrix) plus any extras
        resolved from the feature provider registry.
    """
    base: list[DataEntry] = [
        ValueEntry(name=contract.primary_input_name, value=rhs_data),
        ZarrEntry(name=contract.matrix_input_name, path=arrays.matrix_zarr, model_input=False),
    ]
    extras = [resolve_extra_feature(name, arrays) for name in sorted(declared_extra_names)]
    return [*base, *extras]


def _create_target_configs(
    solutions_data: np.ndarray,
    contract: RuntimeDatasetContract,
) -> list[DataEntry]:
    """Create in-memory Target configs from dataset artifacts.

    Args:
        solutions_data: Solutions array loaded from solutions.zarr.
        contract: Runtime dataset entry name contract.

    Returns:
        The canonical supervised target entry.
    """
    return [ValueEntry(name=contract.target_name, value=solutions_data, data_role=DataRole.TARGET)]


def _find_duplicate_entry_names(entries: tuple[DataEntry, ...]) -> set[str]:
    """Return duplicated non-null dataset entry names."""
    seen: set[str] = set()
    duplicates: set[str] = set()
    for entry in entries:
        if entry.name is None:
            continue
        if entry.name in seen:
            duplicates.add(entry.name)
        seen.add(entry.name)
    return duplicates


def _validate_runtime_dataset_contract(
    settings: TrainingWorkflowSettings,
    contract: RuntimeDatasetContract,
) -> None:
    """Validate the local runtime dataset-entry contract for training.

    This training bridge keeps storage artifact names separate from runtime
    entry names. Runtime placeholder entries and loss routing must resolve
    through the caller-supplied contract.
    """
    _validate_dataset_section(settings)
    dataset = settings.DATASET or DatasetSettings()

    duplicate_features = _find_duplicate_entry_names(dataset.features)
    if duplicate_features:
        raise ValueError(
            f"Duplicate DATASET feature entry names are not allowed: {sorted(duplicate_features)}"
        )

    feature_names = {entry.name for entry in dataset.features if entry.name is not None}
    allowed_names = {
        contract.primary_input_name,
        contract.matrix_input_name,
    } | registered_extra_names()
    unsupported_feature_names = sorted(name for name in feature_names if name not in allowed_names)
    if unsupported_feature_names:
        raise ValueError(
            "DATASET feature placeholders must use only the resolved runtime "
            f"feature names '{contract.primary_input_name}', optional "
            f"'{contract.matrix_input_name}', or registered extras "
            f"{sorted(registered_extra_names())}, got {unsupported_feature_names}."
        )

    duplicate_targets = _find_duplicate_entry_names(dataset.targets)
    if duplicate_targets:
        raise ValueError(
            f"Duplicate DATASET target entry names are not allowed: {sorted(duplicate_targets)}"
        )

    target_names = {entry.name for entry in dataset.targets if entry.name is not None}
    unsupported_target_names = sorted(name for name in target_names if name != contract.target_name)
    if unsupported_target_names:
        raise ValueError(
            "DATASET target placeholders must use only the resolved supervised "
            f"target name '{contract.target_name}', got {unsupported_target_names}."
        )

    training_cfg = settings.TRAINING
    loss_function = getattr(training_cfg, "loss_function", None) if training_cfg else None
    target_key = getattr(loss_function, "target_key", None)
    if target_key is not None and target_key != contract.loss_target_key:
        raise ValueError(
            "TRAINING.loss_function.target_key must resolve to the runtime "
            f"supervised target '{contract.loss_target_key}', got '{target_key}'."
        )


def _merge_entry_transforms[T: DataEntry](
    entries: list[T],
    config_entries: tuple[DataEntry, ...],
) -> list[T]:
    """Inject transforms from model-config placeholder entries into file-backed entries.

    Args:
        entries: File-backed Feature/Target objects built from disk artifacts.
        config_entries: Placeholder entries parsed from [[DATASET.features]] or
            [[DATASET.targets]] in the model TOML (name + transforms, no path).

    Returns:
        New list where each entry whose name matches a config placeholder has
        that placeholder's transforms applied via patch_model; all others unchanged.
    """
    transforms_by_name: dict[str, list[TransformSettings]] = {
        e.name: list(e.transforms) for e in config_entries if e.name is not None and e.transforms
    }
    if not transforms_by_name:
        return entries
    result: list[T] = []
    consumed: set[str] = set()
    for entry in entries:
        name = entry.name
        if name is not None and name in transforms_by_name:
            entry = patch_model(entry, {"transforms": transforms_by_name[name]}, revalidate=False)
            consumed.add(name)
        result.append(entry)
    unmatched = transforms_by_name.keys() - consumed
    if unmatched:
        raise ValueError(
            "DATASET placeholder entries must match injected runtime entry names. "
            f"Unmatched entries: {sorted(unmatched)}"
        )
    return result


def _validate_dataset_section(settings: TrainingWorkflowSettings) -> None:
    """Validate that DATASET section exists in settings.

    Args:
        settings: DLKit workflow settings to validate

    Raises:
        ValueError: If [DATASET] section missing from config
    """
    if settings.DATASET is None:
        raise ValueError("Config is missing [DATASET] section")


def _resolve_dataset(
    settings: TrainingWorkflowSettings,
    features: list[DataEntry],
    targets: list[DataEntry],
    contract: RuntimeDatasetContract,
) -> TrainingWorkflowSettings:
    """Resolve and apply dataset configurations to settings.

    This is a pure transformation that injects file-backed features/targets
    into the DATASET section. Transforms declared as placeholder entries in
    [[DATASET.features]] / [[DATASET.targets]] in the model TOML are merged
    into the programmatic Feature/Target objects by matching on entry name.

    Args:
        settings: Current DLKit settings
        features: Feature configs to inject (from _create_feature_configs)
        targets: Target configs to inject (from _create_target_configs)

    Returns:
        New workflow settings with updated DATASET section

    Raises:
        ValueError: If DATASET section missing from config
    """
    _validate_runtime_dataset_contract(settings, contract)
    base_dataset = settings.DATASET or DatasetSettings()
    features = _merge_entry_transforms(features, base_dataset.features)
    targets = _merge_entry_transforms(targets, base_dataset.targets)
    return patch_model(
        settings,
        {
            "DATASET": {
                "features": features,
                "targets": targets,
                "name": base_dataset.name or "FlexibleDataset",
            }
        },
    )


def _configure_output_paths(
    settings: TrainingWorkflowSettings,
    output_dir: Path,
) -> TrainingWorkflowSettings:
    """Configure training output paths.

    This transformation sets:
    - default_root_dir: Root directory for trainer outputs
    - callback dirpaths: Checkpoint directory location

    Args:
        settings: Current DLKit settings
        output_dir: Workspace root directory (from experiment.workspace.root_dir)

    Returns:
        New workflow settings with updated TRAINING section
    """
    training_cfg = settings.TRAINING
    if training_cfg is None or training_cfg.trainer is None:
        raise ValueError("Training settings require TRAINING.trainer.")

    return patch_model(
        settings,
        {
            "TRAINING": {
                "trainer": {
                    "default_root_dir": str(output_dir),
                    "callbacks": list(training_cfg.trainer.callbacks or []),
                }
            }
        },
    )


def _configure_dataloader_runtime(
    settings: TrainingWorkflowSettings,
) -> TrainingWorkflowSettings:
    """Configure dataloader for reliable dense zarr runtime execution.

    Dense zarr readers and constrained runtime environments are currently
    safer with single-process dataloading.
    """
    # TODO: zarr readers may support num_workers > 0; revisit when multiprocess zarr is stable
    datamodule_cfg = settings.DATAMODULE
    if datamodule_cfg is None or datamodule_cfg.dataloader is None:
        return settings

    return patch_model(
        settings,
        {
            "DATAMODULE": {
                "dataloader": {
                    "num_workers": 0,
                    "persistent_workers": False,
                    "pin_memory": False,
                }
            }
        },
    )


def _configure_mlflow(
    settings: TrainingWorkflowSettings,
    *,
    mlflow_experiment_name: str,
    mlflow_run_name: str,
) -> TrainingWorkflowSettings:
    """Inject concrete MLflow names into training settings."""
    if not settings.MLFLOW:
        return settings
    updated = patch_model(
        settings,
        {
            "MLFLOW": {
                "experiment_name": mlflow_experiment_name,
                "run_name": mlflow_run_name,
            }
        },
    )
    return updated


def _configure_training_pipeline(
    settings: TrainingWorkflowSettings,
    workspace: ExperimentWorkspace,
    features: list[DataEntry],
    targets: list[DataEntry],
    contract: RuntimeDatasetContract,
    mlflow_experiment_name: str,
    mlflow_run_name: str,
) -> tuple[TrainingWorkflowSettings, ExperimentWorkspace]:
    """Apply all configuration transformations to settings.

    This function applies sequential transformations:
    1. Inject features/targets into DATASET section (in-memory arrays)
    2. Configure dataloader runtime (workers, pin_memory)
    3. Configure output paths (checkpoint dir, default root dir)
    4. Inject concrete MLflow experiment/run names into settings

    Each transformation is a pure function that returns new workflow settings.
    The transformations are independent and can be tested in isolation.

    Args:
        settings: Base DLKit settings from config file
        workspace: Experiment workspace with directory paths
        features: Feature configs created from dataset artifacts
        targets: Target configs created from dataset artifacts
        mlflow_experiment_name: Training experiment bucket name.
        mlflow_run_name: Timestamped training run name.

    Returns:
        Tuple of (updated_settings, workspace) ready for DLKit execute()
    """
    settings = _resolve_dataset(settings, features, targets, contract)
    settings = _configure_dataloader_runtime(settings)
    settings = _configure_output_paths(settings, workspace.root_dir)
    settings = _configure_mlflow(
        settings,
        mlflow_experiment_name=mlflow_experiment_name,
        mlflow_run_name=mlflow_run_name,
    )
    return settings, workspace


def _unwrap_execution_result(result: Any) -> Any:
    """Normalize DLKit execute() results to the underlying training result."""
    return getattr(result, "training_result", result)


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
) -> MlflowRunConfig:
    """Build the execute()-time MLflow run config for training."""
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
        )
    return MlflowRunConfig(
        experiment_name=experiment_name,
        run_name=f"{experiment_display_name} | {format_run_timestamp()}",
        tags={},
        paths=paths,
        workspace_root=workspace_root,
    )


def _log_training_evaluation(
    tracking_uri: str,
    run_id: str,
    numpy_payload: Mapping[str, Any] | None,
    figures_dir: Path,
    contract: RuntimeDatasetContract,
) -> None:
    """Compute diagnostics from training predictions and log to existing MLflow run.

    Uses the predictions and targets already captured by ``trainer.predict()``
    during training. Delegates figure writing and MLflow logging to dedicated
    helpers for testability.

    Args:
        tracking_uri: MLflow tracking URI (HTTP or SQLite).
        run_id: Existing MLflow run ID to reopen.
        numpy_payload: Normalized DLKit prediction/target payload.
        figures_dir: Directory to write the diagnostics figure.
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
    workspace: ExperimentWorkspace,
    training_result: Any,
    numpy_payload: Mapping[str, Any] | None,
    model_config_path: Path,
    data_config_path: Path | None,
) -> None:
    """Stage full training artifacts into the workspace for MLflow upload."""
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


def _resolve_mlflow_run_ids(
    *,
    training_result: Any,
    fallback_tracking_uri: str | None,
    experiment_name: str,
    run_name: str,
) -> tuple[str, str, str] | None:
    """Resolve MLflow tracking URI, experiment ID, and run ID for a training run."""
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


def _download_training_checkpoint(
    *,
    tracking_uri: str,
    run_id: str,
    destination: Path,
) -> Path:
    """Download checkpoint artifacts for a completed MLflow run."""
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
    workspace: ExperimentWorkspace,
    tracking_uri: str | None,
    run_id: str | None,
) -> Path:
    """Resolve the produced checkpoint from local artifacts or MLflow."""
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


def _extract_evaluation_arrays(
    all_numpy: Any,
    contract: RuntimeDatasetContract,
) -> tuple[np.ndarray, np.ndarray] | None:
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
    """Normalize DLKit prediction payloads into the local runtime contract once."""
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
    """Read and normalize DLKit numpy payloads when the result exposes them."""
    to_numpy = getattr(training_result, "to_numpy", None)
    if not callable(to_numpy):
        return None
    return _normalize_training_numpy_payload(to_numpy(), contract)


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
    """Log stable ids and display names to the training MLflow run."""
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


def train_model(
    *,
    config_path: str | Path,
    settings: NeurallsSettings | None = None,
    case_config_path: str | Path | None = None,
    data_config_path: str | Path | None = None,
    output_root: Path | str | None = None,
    max_epochs: int | None = None,
    parent_run_id: str | None = None,
    experiment_id: str | None = None,
    experiment_display_name: str | None = None,
    dataset_registry_id: str | None = None,
    dataset_display_name: str | None = None,
    model_registry_id: str | None = None,
    model_display_name: str | None = None,
    mlflow_experiment_name: str | None = None,
) -> Path:
    """Train a DLKit model using resolved data+config context.

    This is the main entry point for model training. It orchestrates:
    1. Load experiment configuration (model + data configs) into a temp dir
    2. Resolve dataset artifacts (rhs/solutions/matrix_zarr)
    3. Build execute()-time MLflow naming and tags
    4. Configure DLKit settings (dataset, paths, MLflow names)
    5. Execute training via DLKit
    6. Copy checkpoint to permanent location under output_root
    7. Return path to saved checkpoint

    The workspace (checkpoints, figures, predictions) is created in a temporary
    directory and deleted after training. Only the final checkpoint and sidecar
    are copied to the permanent ``output_root / "checkpoints" / dataset_id``.

    DLKit handles all MLflow operations including server startup and tracking.

    Args:
        config_path: Path to a model configuration TOML (e.g., /path/to/model.toml)
        data_config_path: Path to a dataset configuration TOML (e.g., /path/to/dataset.toml)
        output_root: Root directory for the permanent checkpoint. Defaults to
            ``DEFAULT_OUTPUT_DIR`` from constants.
        parent_run_id: Optional MLflow parent run UUID. When set, the training run is
            nested as a child of the given parent via ``MLFLOW_PARENT_RUN_ID``.

    Returns:
        Path to saved model checkpoint (.ckpt file) in the permanent location

    Raises:
        RuntimeError: If no checkpoint found after training
        ValueError: If config missing required sections

    Example:
        >>> checkpoint = train_model(
        ...     config_path="/tmp/model.toml",
        ...     data_config_path="/tmp/dataset.toml",
        ... )
        >>> print(checkpoint)
        Path('output/checkpoints/my-dataset/my-model.ckpt')
    """
    resolved_case_config_path = Path(case_config_path) if case_config_path else None
    settings = require_settings(settings, case_config_path=resolved_case_config_path)
    permanent_root = Path(output_root).resolve() if output_root else settings.output_dir
    runtime_mlflow_env = build_runtime_environment(
        output_root,
        default_output_root=settings.output_dir,
    ).env
    if data_config_path is None:
        raise ValueError("data_config_path is required for training.")

    with tempfile.TemporaryDirectory(prefix="neuralls_train_") as _tmp:
        tmp_path = Path(_tmp)
        config_path = Path(config_path)
        resolved_data_config_path = Path(data_config_path)

        # Step 1: Load experiment configuration into temp dir
        experiment = load_experiment(
            config_path,
            resolved_data_config_path,
            settings,
            case_config_path=resolved_case_config_path,
            output_root=tmp_path,
            experiment_id=experiment_id,
            experiment_display_name=experiment_display_name,
            dataset_registry_id=dataset_registry_id,
            dataset_display_name=dataset_display_name,
            model_registry_id=model_registry_id,
            model_display_name=model_display_name,
        )
        workflow_settings = experiment.settings
        workspace = experiment.workspace
        contract = default_training_dataset_contract()
        dataset_id = workspace.dataset_id
        resolved_experiment_display_name = experiment.spec.experiment_display_name
        resolved_dataset_display_name = experiment.spec.dataset_display_name or dataset_id

        # Step 2: Resolve training dataset artifacts
        _, features, targets = _load_and_prepare_data(workflow_settings, workspace, contract)

        # Step 3: Build execute()-time MLflow naming and tags
        run_config = _build_training_run_config(
            experiment_id=experiment.spec.experiment_id,
            experiment_display_name=resolved_experiment_display_name,
            dataset_registry_id=experiment.spec.dataset_registry_id,
            model_registry_id=experiment.spec.model_registry_id,
            dataset_display_name=resolved_dataset_display_name,
            mlflow_experiment_name=mlflow_experiment_name,
            runtime_mlflow_env=runtime_mlflow_env,
            workspace_root=workspace.root_dir,
        )

        # Step 4: Configure DLKit settings (dataset, paths, MLflow names)
        workflow_settings, workspace = _configure_training_pipeline(
            workflow_settings,
            workspace,
            features,
            targets,
            contract,
            run_config.experiment_name,
            run_config.run_name,
        )

        # Step 5: Execute training via DLKit
        if max_epochs is not None:
            workflow_settings = patch_model(
                workflow_settings,
                {"TRAINING": {"trainer": {"max_epochs": max_epochs}}},
            )
        with scoped_mlflow_environment(runtime_mlflow_env):
            with parent_run_context(parent_run_id):
                execution_result = execute(
                    workflow_settings,
                    overrides=ExecutionOverrides(
                        experiment_name=run_config.experiment_name,
                        run_name=run_config.run_name,
                        tags=dict(run_config.tags),
                    ),
                )
                training_result = _unwrap_execution_result(execution_result)

            # Step 6: Resolve MLflow run metadata and retrieve the checkpoint
            tracking_uri, artifacts_destination = resolve_runtime_tracking_config()
            mlflow_coords = _resolve_mlflow_run_ids(
                training_result=training_result,
                fallback_tracking_uri=tracking_uri,
                experiment_name=run_config.experiment_name,
                run_name=run_config.run_name,
            )
            resolved_tracking_uri = tracking_uri
            resolved_run_id = None
            if mlflow_coords is not None:
                resolved_tracking_uri, _, resolved_run_id = mlflow_coords
            else:
                resolved_run_id = getattr(training_result, "mlflow_run_id", None) or getattr(
                    training_result, "run_id", None
                )
            local_checkpoint = _resolve_training_checkpoint(
                training_result=training_result,
                workspace=workspace,
                tracking_uri=resolved_tracking_uri,
                run_id=resolved_run_id,
            )

            # Step 7: Stage artifacts and upload them to MLflow while temp files still exist
            if mlflow_coords is not None:
                tracking_uri, exp_id, run_id = mlflow_coords
                _log_training_context(
                    tracking_uri=tracking_uri,
                    run_id=run_id,
                    experiment_id=experiment.spec.experiment_id,
                    experiment_display_name=resolved_experiment_display_name,
                    dataset_id=dataset_id,
                    dataset_display_name=resolved_dataset_display_name,
                    dataset_registry_id=experiment.spec.dataset_registry_id,
                    model_registry_id=experiment.spec.model_registry_id,
                    model_display_name=experiment.spec.model_display_name or workspace.run_id,
                )
                log_extra_feature_names_tag(
                    tracking_uri,
                    run_id,
                    _extra_feature_names_from_settings(workflow_settings, contract),
                )
                write_mlflow_sidecar(
                    path=workspace.root_dir / "mlflow_run.json",
                    run_id=run_id,
                    experiment_id=exp_id,
                    tracking_uri=tracking_uri,
                    artifacts_destination=artifacts_destination,
                )
                normalized_numpy = _get_normalized_training_numpy_payload(
                    training_result,
                    contract,
                )
                _stage_training_artifacts(
                    workspace=workspace,
                    training_result=training_result,
                    numpy_payload=normalized_numpy,
                    model_config_path=config_path,
                    data_config_path=resolved_data_config_path,
                )
                _log_training_evaluation(
                    tracking_uri,
                    run_id,
                    normalized_numpy,
                    workspace.figures_dir,
                    contract,
                )
                log_artifacts_to_mlflow(
                    tracking_uri=tracking_uri,
                    run_id=run_id,
                    workspace_root=workspace.root_dir,
                )
            else:
                logger.warning(
                    "Training completed without MLflow run metadata; skipping artifact upload."
                )

        # Step 8: Copy checkpoint to permanent location (temp dir deleted after this block)
        permanent_dir = permanent_root / "checkpoints" / dataset_id
        permanent_dir.mkdir(parents=True, exist_ok=True)
        permanent_checkpoint = Path(shutil.copy2(local_checkpoint, permanent_dir))

        # Step 9: Write sidecar next to permanent checkpoint
        if mlflow_coords is not None:
            tracking_uri, exp_id, run_id = mlflow_coords
            write_mlflow_sidecar(
                path=permanent_dir / "mlflow_run.json",
                run_id=run_id,
                experiment_id=exp_id,
                tracking_uri=tracking_uri,
                artifacts_destination=artifacts_destination,
            )

    # temp dir deleted here
    return permanent_checkpoint
