"""Training orchestration helpers consumed by CLI scripts and workflows."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from dlkit import GeneralSettings
from dlkit.interfaces.api import execute
from dlkit.tools.config.core.updater import update_settings
from dlkit.tools.config.data_entries import Feature, FeatureType, Target, TargetType
from dlkit.tools.config.dataset_settings import DatasetSettings

from src.configuration import ExperimentWorkspace, load_experiment
from src.mlflow_utils import (
    MlflowRunState,
    build_run_config,
    finalize_run,
    open_run,
)
from src.system_loading import get_latest_checkpoint
from src.validation import validate_directory_writable


@dataclass(frozen=True)
class TrainingArrays:
    """Training data arrays from normalized dataset."""

    rhs: np.ndarray
    solutions: np.ndarray
    matrix: np.ndarray


@dataclass(frozen=True)
class DatasetConfig:
    """Immutable dataset configuration."""

    features_path: Path | None
    targets_path: Path | None


@dataclass(frozen=True)
class SessionConfig:
    """Immutable session configuration."""

    session_name: str | None


@dataclass(frozen=True)
class OutputConfig:
    """Immutable output configuration."""

    output_dir: Path | None
    accelerator: str | None
    checkpoint_filename: str | None


@dataclass(frozen=True)
class TrainingResult:
    """Immutable training result data."""

    checkpoint_path: Path
    experiment_dir: Path
    data_dir: Path


def _load_training_arrays(data_path: Path) -> TrainingArrays:
    """Load training arrays from normalized.npz file."""
    dataset = np.load(data_path)
    return TrainingArrays(
        rhs=dataset["rhs"],
        solutions=dataset["solutions"],
        matrix=dataset["matrix"],
    )


def _load_and_prepare_data(
    settings: GeneralSettings,
    workspace: ExperimentWorkspace,
) -> tuple[TrainingArrays, list[FeatureType], list[TargetType]]:
    """Load training data and create Feature/Target configurations."""
    arrays = _load_training_arrays(workspace.data_dir / "normalized.npz")
    dataset_name = settings.DATASET.name if settings.DATASET else None
    features = _create_feature_configs(arrays, dataset_name)
    targets = _create_target_configs(arrays)
    return arrays, features, targets


def _create_feature_configs(
    arrays: TrainingArrays, dataset_name: str | None
) -> list[FeatureType]:
    """Create Feature configs from training arrays."""
    if dataset_name == "GraphDataset":
        sample_count = arrays.rhs.shape[0]
        matrix = arrays.matrix
        if matrix.ndim == 3:
            if matrix.shape[0] != sample_count:
                raise ValueError("Matrix feature must match number of samples.")
        elif matrix.ndim != 2:
            raise ValueError(
                f"Matrix feature must be 2D or 3D for GraphDataset, got {matrix.shape}."
            )
        return [
            Feature(name="rhs", value=arrays.rhs, path=None),
            Feature(name="matrix", value=matrix, path=None),
        ]

    return [Feature(name="rhs", value=arrays.rhs, path=None)]


def _create_target_configs(arrays: TrainingArrays) -> list[TargetType]:
    """Create Target configs from training arrays."""
    return [Target(name="solutions", value=arrays.solutions, path=None)]


def _validate_dataset_section(settings: GeneralSettings) -> None:
    """Validate that DATASET section exists in settings."""
    if settings.DATASET is None:
        raise ValueError("Config is missing [DATASET] section")


def _build_dataset_config(
    features: list[FeatureType],
    targets: list[TargetType],
) -> dict[str, Any]:
    """Build dataset configuration dict from features and targets."""
    return {"features": features, "targets": targets}


def _update_settings_with_dataset(
    settings: GeneralSettings,
    dataset_updates: dict[str, Any],
) -> GeneralSettings:
    """Update settings with new dataset configuration."""
    base_dataset = settings.DATASET or DatasetSettings()
    dataset_cfg = base_dataset.model_copy(update={"features": [], "targets": []})
    dataset_cfg = dataset_cfg.model_copy(update=dataset_updates)
    return settings.model_copy(update={"DATASET": dataset_cfg})


def _resolve_dataset(
    settings: GeneralSettings,
    features: list[FeatureType],
    targets: list[TargetType],
) -> GeneralSettings:
    """Resolve and apply dataset configurations to settings."""
    _validate_dataset_section(settings)
    base_dataset = settings.DATASET or DatasetSettings()
    dataset = base_dataset.model_copy(
        update={
            "features": features,
            "targets": targets,
            "name": base_dataset.name or "FlexibleDataset",
        }
    )
    return settings.model_copy(update={"DATASET": dataset})


def _configure_session(
    settings: GeneralSettings,
    workspace: ExperimentWorkspace,
    session_name: str | None,
) -> tuple[GeneralSettings, ExperimentWorkspace]:
    """Apply session name override to settings and context."""
    if session_name is None:
        return settings, workspace

    session_cfg = settings.SESSION
    if session_cfg is None:
        raise ValueError("Config is missing [SESSION] section")

    session_cfg = session_cfg.model_copy(update={"name": session_name})
    settings = settings.model_copy(update={"SESSION": session_cfg})
    workspace = replace(workspace, run_id=session_name)

    return settings, workspace


def _configure_callbacks(
    callbacks: list[Any],
    output_dir: Path,
    checkpoint_filename: str | None,
) -> list[Any]:
    """Configure checkpoint callbacks with output directory."""
    checkpoint_dir = output_dir / "checkpoints"
    updated_callbacks = []

    for cb in callbacks:
        if getattr(cb, "name", None) == "ModelCheckpoint":
            updates: dict[str, Any] = {"dirpath": str(checkpoint_dir)}
            if checkpoint_filename is not None:
                updates["filename"] = checkpoint_filename
            cb = cb.model_copy(update=updates)
        updated_callbacks.append(cb)

    return updated_callbacks


def _configure_output_paths(
    settings: GeneralSettings,
    output_dir: Path | None,
    accelerator: str | None,
    checkpoint_filename: str | None,
) -> GeneralSettings:
    """Configure training output paths and accelerator."""
    if output_dir is None and accelerator is None:
        return settings

    training_cfg = settings.TRAINING
    if training_cfg is None:
        raise ValueError("Config is missing [TRAINING] section")

    trainer_cfg = training_cfg.trainer
    callbacks = list(trainer_cfg.callbacks or [])

    if output_dir is not None:
        validated_dir = validate_directory_writable(output_dir, "Output directory")
        trainer_cfg = trainer_cfg.model_copy(
            update={"default_root_dir": str(validated_dir)}
        )
        callbacks = _configure_callbacks(
            callbacks, Path(validated_dir), checkpoint_filename
        )

    if accelerator is not None:
        trainer_cfg = trainer_cfg.model_copy(update={"accelerator": accelerator})

    trainer_cfg = trainer_cfg.model_copy(update={"callbacks": callbacks})
    training_cfg = training_cfg.model_copy(update={"trainer": trainer_cfg})
    return settings.model_copy(update={"TRAINING": training_cfg})


def _configure_mlflow(
    settings: GeneralSettings,
    dataset_id: str,
) -> GeneralSettings:
    """Configure MLflow experiment tracking."""
    if not settings.MLFLOW:
        return settings

    model_cfg = settings.MODEL
    model_name = getattr(model_cfg, "name", None) if model_cfg is not None else None
    if not isinstance(model_name, str) or not model_name:
        raise ValueError(
            "Config is missing [MODEL].name required for MLflow run naming"
        )

    updated = update_settings(
        settings,
        {
            "MLFLOW": {
                "client": {
                    "experiment_name": dataset_id,
                    "run_name": model_name,
                }
            }
        },
    )
    return updated  # type: ignore[return-value]


def _configure_training_pipeline(
    settings: GeneralSettings,
    workspace: ExperimentWorkspace,
    features: list[FeatureType],
    targets: list[TargetType],
    session_name: str | None,
    output_dir: Path | None,
    accelerator: str | None,
    checkpoint_filename: str | None,
    dataset_id: str,
) -> tuple[GeneralSettings, ExperimentWorkspace]:
    """Apply all configuration transformations to settings."""
    settings = _resolve_dataset(settings, features, targets)
    settings, workspace = _configure_session(settings, workspace, session_name)
    settings = _configure_output_paths(
        settings,
        output_dir,
        accelerator,
        checkpoint_filename,
    )
    settings = _configure_mlflow(settings, dataset_id)
    return settings, workspace


def _start_mlflow_run(
    settings: GeneralSettings,
    workspace: ExperimentWorkspace,
    dataset_id: str,
    session_name: str | None,
    enable_mlflow: bool,
) -> MlflowRunState | None:
    """Start MLflow run when enabled and available."""
    config = build_run_config(
        settings=settings,
        workspace_root=workspace.root_dir,
        dataset_id=dataset_id,
        model_name=workspace.run_id,
        session_name=session_name,
        enabled=enable_mlflow,
    )
    if config is None:
        return None
    try:
        return open_run(config)
    except ModuleNotFoundError:
        print("MLflow not installed; skipping MLflow logging.")
        return None


def train_model(
    *,
    config_path: str | Path,
    data_config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    accelerator: str | None = None,
    session_name: str | None = None,
    checkpoint_filename: str | None = None,
    manifest_metadata: dict[str, Any] | None = None,
    enable_mlflow: bool = False,
) -> Path:
    """Train a DLKit model using resolved data+config context."""
    experiment = load_experiment(
        config_path,
        data_config_path,
        output_root=Path(output_dir) if output_dir is not None else None,
    )
    settings = experiment.settings
    workspace = experiment.workspace
    dataset_id = experiment.spec.data_config_path.stem

    _, features, targets = _load_and_prepare_data(settings, workspace)

    output_root = workspace.root_dir
    settings, workspace = _configure_training_pipeline(
        settings,
        workspace,
        features,
        targets,
        session_name,
        output_root,
        accelerator,
        checkpoint_filename,
        dataset_id,
    )

    mlflow_state = _start_mlflow_run(
        settings,
        workspace,
        dataset_id,
        session_name,
        enable_mlflow,
    )

    error: Exception | None = None
    try:
        execute(settings, run_name=workspace.run_id)  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001
        error = exc
        raise
    finally:
        finalize_run(
            mlflow_state,
            workspace_root=workspace.root_dir,
            failed=error is not None,
        )

    checkpoint_dir = workspace.checkpoint_dir
    checkpoint_path = get_latest_checkpoint(checkpoint_dir)
    if checkpoint_path is None:
        raise RuntimeError(f"No checkpoint found in {checkpoint_dir}")

    return checkpoint_path
