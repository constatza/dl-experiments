"""Training orchestration helpers consumed by CLI scripts and workflows.

This module provides functions for orchestrating model training with DLKit:
- Data loading and preparation (features/targets from normalized.npz)
- Configuration transformations (dataset, paths, MLflow)
- Training execution via DLKit execute()

Architecture:
    The training pipeline applies sequential transformations to GeneralSettings:
    1. Resolve dataset (inject features/targets from arrays)
    2. Configure output paths (checkpoint directory, root dir)
    3. Configure MLflow (experiment name, run name)
    4. Execute training via DLKit

Key Functions:
    - `train_model()`: Main entry point for training
    - `_configure_training_pipeline()`: Apply all configuration transforms
    - `_load_and_prepare_data()`: Load arrays and create Feature/Target configs
    - `_resolve_dataset()`: Inject features/targets into DATASET section
    - `_configure_output_paths()`: Set checkpoint directory and root dir
    - `_configure_mlflow()`: Set experiment/run names for tracking
"""

from __future__ import annotations


from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from dlkit import GeneralSettings
from dlkit.interfaces.api import execute
from dlkit.tools.config.core.updater import update_settings
from dlkit.tools.config.data_entries import (
    FeatureType,
    TargetType,
    ValueFeature,
    ValueTarget,
)
from dlkit.tools.config.dataset_settings import DatasetSettings

from neuralls.configuration import ExperimentWorkspace
from neuralls.configuration.loader import load_experiment
from neuralls.io.checkpoints import get_latest_checkpoint


@dataclass(frozen=True)
class TrainingArrays:
    """Training data arrays from normalized dataset.

    Immutable container for the three core arrays needed for training.
    Supports two types of training pairs with identical structure:

    1. **Base pairs** (A@x, x): Direct solution training
       - rhs: A @ x (RHS vectors)
       - solutions: x (solution vectors)

    2. **Trace pairs** (r_k, e_k): Residual-error training from CG iterations
       - rhs: r_k (residual vectors from CG, where r_k = A @ e_k)
       - solutions: e_k (error corrections, where e_k = x* - x_k)

    Mathematical relationship holds for both: rhs = matrix @ solutions

    Attributes:
        rhs: Shape (n_samples, n_dims) - Features (RHS or residuals)
        solutions: Shape (n_samples, n_dims) - Targets (solutions or errors)
        matrix: Shape (n_dims, n_dims) or (n_samples, n_dims, n_dims) - System matrix A
    """

    rhs: np.ndarray
    solutions: np.ndarray
    matrix: np.ndarray


@dataclass(frozen=True)
class DatasetConfig:
    """Immutable dataset configuration.

    Attributes:
        features_path: Optional path to features file (for file-based datasets)
        targets_path: Optional path to targets file (for file-based datasets)
    """

    features_path: Path | None
    targets_path: Path | None


@dataclass(frozen=True)
class SessionConfig:
    """Immutable session configuration.

    Attributes:
        session_name: Optional session name for MLflow tracking
    """

    session_name: str | None


@dataclass(frozen=True)
class OutputConfig:
    """Immutable output configuration.

    Attributes:
        output_dir: Root directory for training outputs
        accelerator: Hardware accelerator type (cpu/gpu/tpu)
        checkpoint_filename: Custom checkpoint filename pattern
    """

    output_dir: Path | None
    accelerator: str | None
    checkpoint_filename: str | None


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


def _load_training_arrays(data_path: Path) -> TrainingArrays:
    """Load training arrays from normalized.npz file.

    Loads either base pairs (A@x, x) or trace pairs (r_k, e_k) depending on
    what the data generation strategy produced. Both use identical file structure.

    Args:
        data_path: Path to normalized.npz file (must contain rhs, solutions, matrix)

    Returns:
        TrainingArrays with:
        - rhs: Either RHS vectors (A@x) or residuals (r_k)
        - solutions: Either solutions (x) or error corrections (e_k)
        - matrix: System matrix A

    Raises:
        FileNotFoundError: If normalized.npz doesn't exist
        KeyError: If required keys missing from .npz file

    Note:
        No conditional logic needed - file structure is identical for both pair types.
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

    This function:
    1. Loads arrays from normalized.npz
    2. Creates DLKit Feature configs (ValueFeature with arrays)
    3. Creates DLKit Target configs (ValueTarget with arrays)

    Args:
        settings: DLKit general settings (used to check dataset type)
        workspace: Experiment workspace (provides data_dir path)

    Returns:
        Tuple of (arrays, features, targets) where:
            - arrays: Loaded numpy arrays
            - features: List of ValueFeature configs for DLKit
            - targets: List of ValueTarget configs for DLKit
    """
    from ..constants import NORMALIZED_DATASET_FILENAME

    arrays = _load_training_arrays(workspace.data_dir / NORMALIZED_DATASET_FILENAME)
    dataset_name = settings.DATASET.name if settings.DATASET else None
    features = _create_feature_configs(arrays, dataset_name)
    targets = _create_target_configs(arrays)
    return arrays, features, targets


def _create_feature_configs(
    arrays: TrainingArrays, dataset_name: str | None
) -> list[FeatureType]:
    """Create Feature configs from training arrays.

    Different datasets require different features:
    - GraphDataset: Includes both rhs and matrix as features
    - FlexibleDataset: Only rhs as feature

    Args:
        arrays: Training arrays (rhs, solutions, matrix)
        dataset_name: Name from [DATASET].name in config

    Returns:
        List of ValueFeature configs to inject into DATASET section

    Raises:
        ValueError: If matrix dimensions incompatible with samples
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
            ValueFeature(
                name="rhs",
                value=arrays.rhs,
            ),
            ValueFeature(
                name="matrix",
                value=matrix,
            ),
        ]

    return [
        ValueFeature(
            name="rhs",
            value=arrays.rhs,
        )
    ]


def _create_target_configs(arrays: TrainingArrays) -> list[TargetType]:
    """Create Target configs from training arrays.

    Args:
        arrays: Training arrays containing solutions

    Returns:
        List with single ValueTarget for solutions
    """
    return [
        ValueTarget(
            name="solutions",
            value=arrays.solutions,
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


def _build_dataset_config(
    features: list[FeatureType],
    targets: list[TargetType],
) -> dict[str, Any]:
    """Build dataset configuration dict from features and targets.

    Args:
        features: Feature configurations (ValueFeature)
        targets: Target configurations (ValueTarget)

    Returns:
        Dictionary ready for model_copy update
    """
    return {"features": features, "targets": targets}


def _update_settings_with_dataset(
    settings: GeneralSettings,
    dataset_updates: dict[str, Any],
) -> GeneralSettings:
    """Update settings with new dataset configuration.

    Args:
        settings: Current settings to update
        dataset_updates: Dict with features/targets to inject

    Returns:
        New GeneralSettings with updated DATASET section
    """
    base_dataset = settings.DATASET or DatasetSettings()
    dataset_cfg = base_dataset.model_copy(update={"features": [], "targets": []})
    dataset_cfg = dataset_cfg.model_copy(update=dataset_updates)
    return settings.model_copy(update={"DATASET": dataset_cfg})


def _resolve_dataset(
    settings: GeneralSettings,
    features: list[FeatureType],
    targets: list[TargetType],
) -> GeneralSettings:
    """Resolve and apply dataset configurations to settings.

    This is a pure transformation that injects features/targets into the
    DATASET section. It preserves existing dataset configuration while
    replacing features/targets with in-memory arrays.

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
    dataset = base_dataset.model_copy(
        update={
            "features": features,
            "targets": targets,
            "name": base_dataset.name or "FlexibleDataset",
        }
    )
    return settings.model_copy(update={"DATASET": dataset})


def _configure_callbacks(
    callbacks: list[Any],
    output_dir: Path,
) -> list[Any]:
    """Configure checkpoint callbacks with output directory.

    Updates ModelCheckpoint callback to save to correct checkpoint directory.

    Args:
        callbacks: List of trainer callbacks from config
        output_dir: Root output directory (checkpoints/ will be subdirectory)

    Returns:
        Updated callbacks list with correct dirpath
    """
    checkpoint_dir = output_dir / "checkpoints"
    updated_callbacks = []

    for cb in callbacks:
        if getattr(cb, "name", None) == "ModelCheckpoint":
            updates: dict[str, Any] = {"dirpath": str(checkpoint_dir)}
            cb = cb.model_copy(update=updates)
        updated_callbacks.append(cb)

    return updated_callbacks


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


def _configure_mlflow(
    settings: GeneralSettings,
    dataset_id: str,
) -> GeneralSettings:
    """Configure MLflow experiment tracking.

    This transformation sets:
    - experiment_name: Dataset ID (groups runs by dataset)
    - run_name: Model name (identifies specific model architecture)

    Args:
        settings: Current DLKit settings
        dataset_id: Dataset identifier from data config filename

    Returns:
        New GeneralSettings with updated MLFLOW section

    Raises:
        ValueError: If [MODEL].name missing from config (required for run naming)
    """
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
    dataset_id: str,
) -> tuple[GeneralSettings, ExperimentWorkspace]:
    """Apply all configuration transformations to settings.

    This function applies three sequential transformations:
    1. Inject features/targets into DATASET section (in-memory arrays)
    2. Configure output paths (checkpoint dir, default root dir)
    3. Configure MLflow tracking (experiment name, run name)

    Each transformation is a pure function that returns new GeneralSettings.
    The transformations are independent and can be tested in isolation.

    Args:
        settings: Base DLKit settings from config file
        workspace: Experiment workspace with directory paths
        features: Feature configs created from normalized.npz
        targets: Target configs created from normalized.npz
        dataset_id: Dataset identifier for MLflow experiment naming

    Returns:
        Tuple of (updated_settings, workspace) ready for DLKit execute()
    """
    # Apply transformations sequentially
    # Each returns new GeneralSettings (immutable updates)
    settings = _resolve_dataset(settings, features, targets)
    settings = _configure_output_paths(settings, workspace.root_dir)
    settings = _configure_mlflow(settings, dataset_id)

    return settings, workspace


def train_model(
    *,
    config_path: str | Path,
    data_config_path: str | Path | None = None,
    session_name: str | None = None,
    output_root: Path | str | None = None,
) -> Path:
    """Train a DLKit model using resolved data+config context.

    This is the main entry point for model training. It orchestrates:
    1. Load experiment configuration (model + data configs)
    2. Load training data from normalized.npz
    3. Configure DLKit settings (dataset, paths, MLflow)
    4. Execute training via DLKit
    5. Return path to saved checkpoint

    DLKit handles all MLflow operations including server startup and tracking.

    Args:
        config_path: Path to model configuration TOML (e.g., configs/linear.toml)
        data_config_path: Path to data configuration TOML (e.g., data-configs/collect-504.toml)
        session_name: Optional session name for MLflow (unused, for compatibility)
        output_root: Optional custom output root (defaults to constants.DEFAULT_OUTPUT_ROOT)

    Returns:
        Path to saved model checkpoint (.ckpt file)

    Raises:
        RuntimeError: If no checkpoint found after training
        ValueError: If config missing required sections

    Example:
        >>> checkpoint = train_model(
        ...     config_path="configs/linear.toml",
        ...     data_config_path="data-configs/collect-504.toml",
        ... )
        >>> print(checkpoint)
        Path('output/collect-504/linear/2025-12-30T15-24-30/checkpoints/epoch=9-step=1000.ckpt')
    """
    # Step 1: Load experiment configuration
    # Creates workspace directories and loads both model + data configs
    experiment = load_experiment(
        config_path,
        data_config_path,
        output_root=output_root,
    )
    settings = experiment.settings
    workspace = experiment.workspace
    dataset_id = experiment.spec.data_config_path.stem

    # Step 2: Load training data from normalized.npz
    # Creates Feature/Target configs with in-memory arrays
    _, features, targets = _load_and_prepare_data(settings, workspace)

    # Step 3: Configure DLKit settings
    # Applies three transformations: dataset, paths, MLflow
    settings, workspace = _configure_training_pipeline(
        settings,
        workspace,
        features,
        targets,
        dataset_id,
    )

    # Step 4: Execute training via DLKit
    # DLKit handles all MLflow operations including server startup and tracking
    execute(settings, run_name=workspace.run_id)  # type: ignore[arg-type]

    # Step 5: Retrieve saved checkpoint
    checkpoint_dir = workspace.checkpoint_dir
    checkpoint_path = get_latest_checkpoint(checkpoint_dir)
    if checkpoint_path is None:
        raise RuntimeError(f"No checkpoint found in {checkpoint_dir}")

    return checkpoint_path
