"""Training helpers consumed by CLI scripts and workflows.

Functional architecture:
- Pure functions compute configuration transformations (data, pure logic)
- Action functions perform I/O (training execution, file operations, manifest updates)
- train_model orchestrates the pipeline with minimal imperative glue
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from dlkit import GeneralSettings
from dlkit.interfaces.api import execute
from dlkit.tools.config.data_entries import Feature, FeatureType, Target, TargetType
from dlkit.tools.config.core.updater import update_settings

from ..configuration import load_experiment, RunnableExperiment, ExperimentWorkspace
from ..system_loading import get_latest_checkpoint
from ..file_operations import derive_model_identifier, sanitize_identifier
from ..constants import DEFAULT_MLRUNS_DIR
from ..validation import validate_directory_writable
from ..pca_training import (
    fit_pca_from_solutions,
    save_pca_model,
    plot_variance_ratios,
)
from dlkit.tools.config.dataset_settings import DatasetSettings


# Pure data structures for configuration transformations


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


# Action functions (I/O operations - data loading)


def _load_training_arrays(data_path: Path) -> TrainingArrays:
    """Load training arrays from normalized.npz file.

    Action function: performs file I/O.

    Args:
        data_path: Path to normalized.npz file

    Returns:
        TrainingArrays containing rhs, solutions, and matrix

    Raises:
        FileNotFoundError: If data_path does not exist
        KeyError: If required arrays are missing from npz file
    """
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
    """Load training data and create Feature/Target configurations.

    Action function: performs file I/O to load arrays, then creates configs.

    Args:
        settings: Training settings for dataset type resolution
        workspace: Experiment workspace with data paths

    Returns:
        Tuple of (arrays, features, targets)
    """
    arrays = _load_training_arrays(workspace.data_dir / "normalized.npz")
    dataset_name = settings.DATASET.name if settings.DATASET else None
    features = _create_feature_configs(arrays, dataset_name)
    targets = _create_target_configs(arrays)
    return arrays, features, targets


# Pure functions (data transformations, no I/O)


def _create_feature_configs(
    arrays: TrainingArrays, dataset_name: str | None
) -> list[FeatureType]:
    """Create Feature configs from training arrays.

    Pure function: transforms data without side effects.

    Args:
        arrays: Training data arrays
        dataset_name: Dataset type to decide whether matrix is needed

    Returns:
        Tuple of feature configs (rhs always, matrix only for graph datasets)
    """
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
    """Create Target configs from training arrays.

    Pure function: transforms data without side effects.

    Args:
        arrays: Training data arrays

    Returns:
        Tuple containing solutions target
    """
    return [Target(name="solutions", value=arrays.solutions, path=None)]


def _validate_dataset_section(settings: GeneralSettings) -> None:
    """Validate that DATASET section exists in settings.

    Pure function: validation only, no mutation.

    Args:
        settings: Settings to validate

    Raises:
        ValueError: If DATASET section is missing
    """
    if settings.DATASET is None:
        raise ValueError("Config is missing [DATASET] section")


def _build_dataset_config(
    features: list[FeatureType],
    targets: list[TargetType],
) -> dict[str, Any]:
    """Build dataset configuration dict from features and targets.

    Pure function: creates configuration dict.

    Args:
        features: Feature configurations
        targets: Target configurations

    Returns:
        Dict with features and targets keys
    """
    return {"features": features, "targets": targets}


def _update_settings_with_dataset(
    settings: GeneralSettings,
    dataset_updates: dict[str, Any],
) -> GeneralSettings:
    """Update settings with new dataset configuration.

    Pure function: creates new settings with updated dataset.

    Args:
        settings: Original settings
        dataset_updates: Dataset configuration updates

    Returns:
        New settings with updated DATASET section
    """
    base_dataset = settings.DATASET or DatasetSettings()
    # Preserve existing fields but replace features/targets explicitly to avoid placeholders
    dataset_cfg = base_dataset.model_copy(update={"features": [], "targets": []})
    dataset_cfg = dataset_cfg.model_copy(update=dataset_updates)
    return settings.model_copy(update={"DATASET": dataset_cfg})


def _resolve_dataset(
    settings: GeneralSettings,
    features: list[FeatureType],
    targets: list[TargetType],
) -> GeneralSettings:
    """Resolve and apply dataset configurations to settings.

    Pure function: takes immutable inputs, returns new settings without side effects.
    Composes validation, config building, and settings update.

    Args:
        settings: Base settings object
        features: Feature configurations with in-memory arrays
        targets: Target configurations with in-memory arrays

    Returns:
        New settings with updated dataset configuration

    Raises:
        ValueError: If DATASET section is missing
    """
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
    """Apply session name override to settings and context.

    Pure function: computes new configuration without side effects.

    Args:
        settings: Base settings object
        workspace: Experiment workspace
        session_name: Optional session name override

    Returns:
        Tuple of (updated_settings, updated_workspace)

    Raises:
        ValueError: If SESSION section is missing
    """
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
    """Configure checkpoint callbacks with output directory.

    Pure function: transforms callback list without mutation.

    Args:
        callbacks: Original callback list
        output_dir: Validated output directory
        checkpoint_filename: Optional checkpoint filename override

    Returns:
        New list of updated callbacks
    """
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
    """Configure training output paths and accelerator.

    Impure function: validates directory writability (performs I/O check).

    Args:
        settings: Base settings object
        output_dir: Optional output directory override
        accelerator: Optional accelerator override
        checkpoint_filename: Optional checkpoint filename

    Returns:
        New settings with updated training configuration

    Raises:
        ValueError: If TRAINING section is missing
    """
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
    """Configure MLflow experiment tracking.

    Pure function: updates MLflow settings with experiment name from context.

    Args:
        settings: Base settings object
        dataset_id: Identifier for the dataset (MLflow experiment name)

    Returns:
        New settings with MLflow configuration
    """
    if not settings.MLFLOW:
        return settings

    model_cfg = settings.MODEL
    model_name = getattr(model_cfg, "name", None) if model_cfg is not None else None
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("Config is missing [MODEL].name required for MLflow run naming")

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
    # update_settings returns BasicSettings but we need GeneralSettings
    # Safe cast since GeneralSettings extends BasicSettings
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
    """Apply all configuration transformations to settings.

    Impure function: chains configuration steps (includes directory validation I/O).

    Args:
        settings: Base settings
        workspace: Experiment workspace
        features: Feature configurations
        targets: Target configurations
        session_name: Optional session name override
        output_dir: Optional output directory override
        accelerator: Optional accelerator override
        checkpoint_filename: Optional checkpoint filename
        dataset_id: Dataset identifier for MLflow

    Returns:
        Tuple of (configured_settings, updated_workspace)
    """
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


def _log_dataset_debug(dataset: Any) -> None:
    """Log dataset entry resolution for debugging placeholder issues."""
    try:
        entries = list(getattr(dataset, "features", ()) or ()) + list(
            getattr(dataset, "targets", ()) or ()
        )
    except Exception:
        return

    for entry in entries:
        try:
            name = getattr(entry, "name", "unknown")
            has_value = getattr(entry, "has_value", lambda: False)()
            has_path = getattr(entry, "has_path", lambda: False)()
            path = getattr(entry, "path", None)
            value = getattr(entry, "value", None)
            value_shape = getattr(value, "shape", None)
            print(
                f"[dataset-debug] {name}: type={type(entry).__name__} has_value={has_value} "
                f"has_path={has_path} path={path} value_shape={value_shape}"
            )
        except Exception:
            continue


# Action functions (I/O operations)


# Public orchestration functions


def train_model(
    *,
    config_path: str | Path,
    data_config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    accelerator: str | None = None,
    session_name: str | None = None,
    checkpoint_filename: str | None = None,
    manifest_metadata: dict[str, Any] | None = None,
) -> Path:
    """Train a DLKit model using resolved data+config context.

    Orchestrates training pipeline with functional composition:
    1. Load configuration and data (action)
    2. Configure pipeline (pure transformations)
    3. Execute training (action)
    4. Finalize results (action)

    Args:
        config_path: Path to training configuration
        data_config_path: Optional data configuration override
        output_dir: Optional output directory override
        accelerator: Optional accelerator override
        session_name: Optional session name override
        checkpoint_filename: Optional checkpoint filename
        manifest_metadata: Deprecated, ignored for backward compatibility

    Returns:
        Path to trained model checkpoint

    Raises:
        ValueError: If required config sections are missing
        RuntimeError: If no checkpoint is found after training
    """
    # Phase 1: Load (action)
    experiment = load_experiment(
        config_path,
        data_config_path,
        output_root=Path(output_dir) if output_dir is not None else None,
    )
    settings = experiment.settings
    workspace = experiment.workspace
    
    # We need to extract the dataset ID before we lose the spec in helpers
    dataset_id = experiment.spec.data_config_path.stem

    _, features, targets = _load_and_prepare_data(settings, workspace)

    # Phase 2: Configure (pure)
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

    _log_dataset_debug(settings.DATASET)

    # Phase 3: Execute (action)
    execute(settings, run_name=workspace.run_id)  # type: ignore[arg-type]

    # Phase 4: Finalize (action)
    checkpoint_dir = workspace.checkpoint_dir
    checkpoint_path = get_latest_checkpoint(checkpoint_dir)
    if checkpoint_path is None:
        raise RuntimeError(f"No checkpoint found in {checkpoint_dir}")

    return checkpoint_path


def train_pca_preconditioner(
    *,
    config_path: str | Path,
    n_components: int,
    data_config_path: str | Path | None = None,
    solution_samples_path: str | Path | None = None,
    output_path: str | Path | None = None,
    normalize: bool = True,
) -> tuple[Path, Path]:
    """Train PCA preconditioner on solution samples."""
    experiment = load_experiment(config_path, data_config_path)
    settings = experiment.settings
    workspace = experiment.workspace

    solution_samples = (
        Path(solution_samples_path)
        if solution_samples_path is not None
        else workspace.data_dir / "normalized.npz"
    )
    if not solution_samples.exists():
        raise ValueError(
            "No targets path resolved from config; please pass solution_samples_path"
        )

    if output_path is None:
        base_output = workspace.root_dir
        output_path = base_output / "pca" / f"pca_{n_components}comp.pt"
    output_path = Path(output_path)

    pca, stats = fit_pca_from_solutions(
        solution_samples, n_components, normalize=normalize
    )
    save_pca_model(pca, stats, output_path)

    figures_dir = workspace.figures_dir
    figures_dir.mkdir(parents=True, exist_ok=True)
    dataset_slug = sanitize_identifier(experiment.spec.data_config_path.stem)
    model_slug = sanitize_identifier(
        derive_model_identifier(settings, workspace, config_path)
    )
    plot_filename = f"pca_variance_{dataset_slug}-{model_slug}_{n_components}comp.png"
    plot_path = figures_dir / plot_filename
    plot_variance_ratios(stats, plot_path)

    return output_path, plot_path
