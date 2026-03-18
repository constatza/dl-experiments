"""Training orchestration helpers consumed by CLI scripts and workflows."""

from __future__ import annotations


import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dlkit import GeneralSettings
from dlkit.interfaces.api import execute
from dlkit.tools.config.core.updater import update_settings
from dlkit.tools.config.data_entries import (
    Feature,
    FeatureType,
    SparseFeature,
    Target,
    TargetType,
)
from dlkit.tools.config.dataset_settings import DatasetSettings
from dlkit.tools.io.sparse import PackFiles
from loguru import logger
from mlflow.tracking import MlflowClient
import numpy as np

from neuralls.configuration import ExperimentWorkspace
from neuralls.configuration.experiments import ExperimentEntry
from neuralls.configuration.mlflow_normalization import (
    build_mlflow_environment,
    scoped_mlflow_environment,
)
from neuralls.configuration.loader import load_experiment
from neuralls.constants import DEFAULT_OUTPUT_DIR
from neuralls.io.checkpoints import get_latest_checkpoint
from neuralls.mlflow_utils import MlflowPaths, MlflowRunConfig
from neuralls.workflows.artifact_io import (
    TrainingArrays,
    coerce_jsonable,
    load_training_arrays,
    save_training_predictions,
)
from neuralls.workflows.diagnostics import compute_diagnostics, write_diagnostics_figure
from neuralls.workflows.inference.output import write_mlflow_sidecar
from neuralls.workflows.mlflow_client import (
    find_mlflow_run,
    log_artifacts_to_mlflow,
    log_diagnostics_to_mlflow,
    parent_run_context,
)
from neuralls.workflows.run_specs import build_training_run_spec, format_run_timestamp

GRAPH_DATASET_NAME: str = "GraphDataset"
FLEXIBLE_DATASET_NAME: str = "FlexibleDataset"
TRAINING_EXPERIMENT_NAME: str = "Train"


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


def _load_and_prepare_data(
    settings: GeneralSettings,
    workspace: ExperimentWorkspace,
) -> tuple[TrainingArrays, list[FeatureType], list[TargetType]]:
    """Load training data and create Feature/Target configurations.

    This function:
    1. Resolves file-backed dataset artifacts
    2. Creates DLKit feature entries from file paths
    3. Creates DLKit target entries from file paths

    Args:
        settings: DLKit general settings (used to check dataset type)
        workspace: Experiment workspace (provides data_dir path)

    Returns:
        Tuple of (arrays, features, targets) where:
            - arrays: Resolved data artifact paths
            - features: List of path-based feature configs for DLKit
            - targets: List of path-based target configs for DLKit
    """
    arrays = load_training_arrays(workspace.data_dir)
    dataset_name = settings.DATASET.name if settings.DATASET else None
    features = _create_feature_configs(arrays, dataset_name)
    targets = _create_target_configs(arrays)
    return arrays, features, targets


def _create_feature_configs(
    arrays: TrainingArrays, dataset_name: str | None
) -> list[FeatureType]:
    """Create file-backed Feature configs from dataset artifacts.

    Args:
        arrays: Training data artifact paths
        dataset_name: Name from [DATASET].name in config

    Returns:
        List of file-backed features to inject into DATASET section
    """
    _ = dataset_name
    sparse_feature = SparseFeature(
        name="matrix",
        path=arrays.matrix_pack,
        model_input=False,
    )
    # Bridge config-type mismatch between SparseFeature.files and current dlkit sparse reader.
    object.__setattr__(
        sparse_feature,
        "files",
        PackFiles(
            indices=sparse_feature.files.indices,
            values=sparse_feature.files.values,
            nnz_ptr=sparse_feature.files.nnz_ptr,
        ),
    )
    return [
        Feature(
            name="x",
            path=arrays.rhs,
        ),
        sparse_feature,
    ]


def _create_target_configs(arrays: TrainingArrays) -> list[TargetType]:
    """Create file-backed Target configs from dataset artifacts.

    Args:
        arrays: Training data artifact paths

    Returns:
        Targets with both model-compatible and loss-compatible keys.
    """
    return [
        Target(
            name="y",
            path=arrays.solutions,
        ),
        Target(
            name="solutions",
            path=arrays.solutions,
        )
    ]


def _validate_dataset_section(settings: GeneralSettings) -> None:
    """Validate that DATASET section exists in settings.

    Args:
        settings: DLKit general settings to validate

    Raises:
        ValueError: If [DATASET] section missing from config
    """
    if settings.DATASET is None:
        raise ValueError("Config is missing [DATASET] section")


def _resolve_dataset(
    settings: GeneralSettings,
    features: list[FeatureType],
    targets: list[TargetType],
) -> GeneralSettings:
    """Resolve and apply dataset configurations to settings.

    This is a pure transformation that injects file-backed features/targets
    into the DATASET section.

    Args:
        settings: Current DLKit settings
        features: Feature configs to inject (from _create_feature_configs)
        targets: Target configs to inject (from _create_target_configs)

    Returns:
        New GeneralSettings with updated DATASET section

    Raises:
        ValueError: If DATASET section missing from config
    """
    _validate_dataset_section(settings)
    base_dataset = settings.DATASET or DatasetSettings()
    dataset = base_dataset.update_with(
        {
            "features": features,
            "targets": targets,
            "name": base_dataset.name or "FlexibleDataset",
            # SparseFeature entries are not compatible with current memmap cache path.
            "memmap_cache": False,
        }
    )
    return settings.update_with({"DATASET": dataset})


def _configure_output_paths(
    settings: GeneralSettings,
    output_dir: Path,
) -> GeneralSettings:
    """Configure training output paths.

    This transformation sets:
    - default_root_dir: Root directory for trainer outputs
    - callback dirpaths: Checkpoint directory location

    Args:
        settings: Current DLKit settings
        output_dir: Workspace root directory (from experiment.workspace.root_dir)

    Returns:
        New GeneralSettings with updated TRAINING section
    """
    training_cfg = settings.TRAINING

    trainer_cfg = training_cfg.trainer
    callbacks = list(trainer_cfg.callbacks or [])

    trainer_cfg = trainer_cfg.update_with({"default_root_dir": str(output_dir)})

    trainer_cfg = trainer_cfg.update_with({"callbacks": callbacks})
    training_cfg = training_cfg.update_with({"trainer": trainer_cfg})
    return settings.update_with({"TRAINING": training_cfg})


def _configure_dataloader_runtime(settings: GeneralSettings) -> GeneralSettings:
    """Configure dataloader for reliable sparse runtime execution.

    Sparse pack readers and constrained runtime environments are currently
    safer with single-process dataloading.
    """
    datamodule_cfg = settings.DATAMODULE
    if datamodule_cfg is None or datamodule_cfg.dataloader is None:
        return settings

    dataloader_cfg = datamodule_cfg.dataloader.update_with(
        {
            "num_workers": 0,
            "persistent_workers": False,
            "pin_memory": False,
        }
    )
    datamodule_cfg = datamodule_cfg.update_with({"dataloader": dataloader_cfg})
    return settings.update_with({"DATAMODULE": datamodule_cfg})


def _configure_mlflow(
    settings: GeneralSettings,
    *,
    mlflow_experiment_name: str,
    mlflow_run_name: str,
) -> GeneralSettings:
    """Inject concrete MLflow names into training settings."""
    if not settings.MLFLOW:
        return settings
    updated = update_settings(
        settings,
        {
            "MLFLOW": {
                "experiment_name": mlflow_experiment_name,
                "run_name": mlflow_run_name,
            }
        },
    )
    return updated  # type: ignore[return-value]


def _configure_training_pipeline(
    settings: GeneralSettings,
    workspace: ExperimentWorkspace,
    features: list[FeatureType],
    targets: list[TargetType],
    mlflow_experiment_name: str,
    mlflow_run_name: str,
) -> tuple[GeneralSettings, ExperimentWorkspace]:
    """Apply all configuration transformations to settings.

    This function applies sequential transformations:
    1. Inject features/targets into DATASET section (in-memory arrays)
    2. Configure dataloader runtime (workers, pin_memory)
    3. Configure output paths (checkpoint dir, default root dir)
    4. Inject concrete MLflow experiment/run names into settings

    Each transformation is a pure function that returns new GeneralSettings.
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
    settings = _resolve_dataset(settings, features, targets)
    settings = _configure_dataloader_runtime(settings)
    settings = _configure_output_paths(settings, workspace.root_dir)
    settings = _configure_mlflow(
        settings,
        mlflow_experiment_name=mlflow_experiment_name,
        mlflow_run_name=mlflow_run_name,
    )
    return settings, workspace


def _resolve_mlflow_logging_config() -> tuple[str | None, str]:
    """Resolve tracking URI and artifact destination from runtime env."""
    return os.environ.get("MLFLOW_TRACKING_URI"), os.environ.get("MLFLOW_ARTIFACT_URI", "")


def _build_runtime_mlflow_paths(runtime_mlflow_env: Mapping[str, str]) -> MlflowPaths:
    """Build resolved MLflow paths from the active runtime environment."""
    return MlflowPaths(
        tracking_uri=runtime_mlflow_env["MLFLOW_TRACKING_URI"],
        artifact_uri=runtime_mlflow_env.get("MLFLOW_ARTIFACT_URI", ""),
    )


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
    _ = (mlflow_experiment_name, dataset_display_name)
    experiment_name = TRAINING_EXPERIMENT_NAME
    paths = _build_runtime_mlflow_paths(runtime_mlflow_env)
    if experiment_id and dataset_registry_id and model_registry_id:
        entry = ExperimentEntry(
            id=experiment_id,
            dataset_id=dataset_registry_id,
            model_id=model_registry_id,
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
        run_name=f"{experiment_display_name}-{format_run_timestamp()}",
        tags={},
        paths=paths,
        workspace_root=workspace_root,
    )




def _log_training_evaluation(
    tracking_uri: str,
    run_id: str,
    training_result: Any,
    figures_dir: Path,
) -> None:
    """Compute diagnostics from training predictions and log to existing MLflow run.

    Uses the predictions and targets already captured by ``trainer.predict()``
    during training. Delegates figure writing and MLflow logging to dedicated
    helpers for testability.

    Args:
        tracking_uri: MLflow tracking URI (HTTP or SQLite).
        run_id: Existing MLflow run ID to reopen.
        training_result: DLKit TrainingResult with captured predictions and targets.
        figures_dir: Directory to write the diagnostics figure.
    """
    all_numpy = training_result.to_numpy()
    selected = _extract_evaluation_arrays(all_numpy)
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

    save_training_predictions(training_result, workspace.predictions_dir)


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
            resolved_experiment_id = MlflowClient(tracking_uri=tracking_uri).get_run(
                direct_run_id
            ).info.experiment_id
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


def _find_single_checkpoint(root: Path) -> Path:
    """Return one checkpoint under a downloaded MLflow artifact tree."""
    checkpoints = sorted(root.glob("**/*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found under {root}")
    return checkpoints[0]


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
        logger.info(
            "Checkpoint missing locally for run {}. Downloading from MLflow artifacts.",
            run_id,
        )
        return _download_training_checkpoint(
            tracking_uri=tracking_uri,
            run_id=run_id,
            destination=workspace.checkpoint_dir,
        )

    raise RuntimeError(f"No checkpoint found in {workspace.checkpoint_dir}")


_PREDICTION_KEYS: tuple[str, ...] = (
    "output",
    "predictions",
    "y_pred",
    "y_hat",
    "y",
    "solutions",
)
_TARGET_KEYS: tuple[str, ...] = (
    "y",
    "solutions",
    "targets",
    "target",
    "y_true",
    "output",
)


def _select_mapping_value(
    payload: Mapping[str, Any],
    preferred_keys: tuple[str, ...],
    fallback_keys: list[str],
) -> Any | None:
    for key in preferred_keys:
        value = payload.get(key)
        if value is not None:
            return value
    for key in fallback_keys:
        value = payload.get(key)
        if value is not None:
            return value
    return next((value for value in payload.values() if value is not None), None)


def _extract_evaluation_arrays(
    all_numpy: Any,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not isinstance(all_numpy, Mapping):
        return None

    predictions_raw = all_numpy.get("predictions")
    targets_raw = all_numpy.get("targets")
    if predictions_raw is None or targets_raw is None:
        return None

    prediction_keys = (
        list(targets_raw.keys()) if isinstance(targets_raw, Mapping) else []
    )
    target_keys = (
        list(predictions_raw.keys()) if isinstance(predictions_raw, Mapping) else []
    )

    y_pred_raw = (
        _select_mapping_value(predictions_raw, _PREDICTION_KEYS, prediction_keys)
        if isinstance(predictions_raw, Mapping)
        else predictions_raw
    )
    y_true_raw = (
        _select_mapping_value(targets_raw, _TARGET_KEYS, target_keys)
        if isinstance(targets_raw, Mapping)
        else targets_raw
    )
    if y_pred_raw is None or y_true_raw is None:
        return None

    y_pred = np.asarray(y_pred_raw)
    y_true = np.asarray(y_true_raw)
    if y_pred.size == 0 or y_true.size == 0:
        return None
    return y_pred, y_true


def _resolve_runtime_mlflow_env(output_root: Path | str | None) -> dict[str, str]:
    """Return the MLflow env vars to activate for this training run.

    If the caller already set ``MLFLOW_TRACKING_URI`` in the environment (e.g.
    via ``scoped_mlflow_environment`` in ``train_batch``), honour it verbatim.
    Otherwise fall back to a local SQLite database under ``output_root``.

    Args:
        output_root: Permanent output root; used only for the sqlite fall-back.

    Returns:
        Dict suitable for ``scoped_mlflow_environment``.
    """
    existing_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if existing_uri:
        return build_mlflow_environment(tracking_uri=existing_uri)

    permanent_root = Path(output_root).resolve() if output_root else DEFAULT_OUTPUT_DIR
    return build_mlflow_environment(
        tracking_uri=f"sqlite:///{(permanent_root / 'mlruns' / 'mlflow.db').as_posix()}",
        artifacts_destination=str((permanent_root / "mlartifacts").resolve()),
    )


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
    2. Resolve dataset artifacts (rhs/solutions/matrix_coo)
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
        config_path: Path to model configuration TOML (e.g., configs/linear.toml)
        data_config_path: Path to data configuration TOML (e.g., data-configs/collect-504.toml)
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
        ...     config_path="configs/linear.toml",
        ...     data_config_path="data-configs/collect-504.toml",
        ... )
        >>> print(checkpoint)
        Path('output/checkpoints/collect-504/linear.ckpt')
    """
    permanent_root = Path(output_root).resolve() if output_root else DEFAULT_OUTPUT_DIR
    runtime_mlflow_env = _resolve_runtime_mlflow_env(output_root)

    with tempfile.TemporaryDirectory(prefix="neuralls_train_") as _tmp:
        tmp_path = Path(_tmp)
        config_path = Path(config_path)
        resolved_data_config_path = (
            Path(data_config_path) if data_config_path is not None else None
        )

        # Step 1: Load experiment configuration into temp dir
        experiment = load_experiment(
            config_path,
            resolved_data_config_path,
            output_root=tmp_path,
            experiment_id=experiment_id,
            experiment_display_name=experiment_display_name,
            dataset_registry_id=dataset_registry_id,
            dataset_display_name=dataset_display_name,
            model_registry_id=model_registry_id,
            model_display_name=model_display_name,
        )
        settings = experiment.settings
        workspace = experiment.workspace
        dataset_id = workspace.dataset_id
        resolved_experiment_display_name = experiment.spec.experiment_display_name
        resolved_dataset_display_name = (
            experiment.spec.dataset_display_name or dataset_id
        )

        # Step 2: Resolve training dataset artifacts
        _, features, targets = _load_and_prepare_data(settings, workspace)

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
        settings, workspace = _configure_training_pipeline(
            settings,
            workspace,
            features,
            targets,
            run_config.experiment_name,
            run_config.run_name,
        )

        # Step 5: Execute training via DLKit
        if max_epochs is not None:
            settings = update_settings(settings, {"TRAINING": {"trainer": {"max_epochs": max_epochs}}})
        with scoped_mlflow_environment(runtime_mlflow_env):
            with parent_run_context(parent_run_id):
                training_result = execute(  # type: ignore[arg-type]
                    settings,
                    experiment_name=run_config.experiment_name,
                    run_name=run_config.run_name,
                    tags=dict(run_config.tags) or None,
                )

            # Step 6: Resolve MLflow run metadata and retrieve the checkpoint
            tracking_uri, artifacts_destination = _resolve_mlflow_logging_config()
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
                resolved_run_id = (
                    getattr(training_result, "mlflow_run_id", None)
                    or getattr(training_result, "run_id", None)
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
                write_mlflow_sidecar(
                    path=workspace.root_dir / "mlflow_run.json",
                    run_id=run_id,
                    experiment_id=exp_id,
                    tracking_uri=tracking_uri,
                    artifacts_destination=artifacts_destination,
                )
                _stage_training_artifacts(
                    workspace=workspace,
                    training_result=training_result,
                    model_config_path=config_path,
                    data_config_path=resolved_data_config_path,
                )
                _log_training_evaluation(
                    tracking_uri,
                    run_id,
                    training_result,
                    workspace.figures_dir,
                )
                log_artifacts_to_mlflow(
                    tracking_uri=tracking_uri,
                    run_id=run_id,
                    workspace_root=workspace.root_dir,
                )
            else:
                logger.warning("Training completed without MLflow run metadata; skipping artifact upload.")

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
