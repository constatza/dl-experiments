"""Training orchestration helpers consumed by CLI scripts and workflows.

This module provides functions for orchestrating model training with DLKit:
- Data loading and preparation (features/targets from dataset artifacts)
- Configuration transformations (dataset, paths, MLflow)
- Training execution via DLKit execute()

Architecture:
    The training pipeline applies sequential transformations to GeneralSettings:
    1. Resolve dataset (inject features/targets from file-backed entries)
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


import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
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

from neuralls.configuration import ExperimentWorkspace
from neuralls.configuration.loader import load_experiment
from neuralls.constants import DEFAULT_OUTPUT_DIR
from neuralls.io.dataset_storage import (
    load_dense_training_arrays,
    resolve_dataset_paths,
)
from neuralls.io.checkpoints import get_latest_checkpoint
from neuralls.workflows.diagnostics import compute_diagnostics
from neuralls.workflows.inference.output import write_mlflow_sidecar

GRAPH_DATASET_NAME: str = "GraphDataset"
FLEXIBLE_DATASET_NAME: str = "FlexibleDataset"


@dataclass(frozen=True)
class TrainingArrays:
    """Training data artifact paths from dataset storage.

    Attributes:
        rhs: Path to rhs.npy
        solutions: Path to solutions.npy
        matrix_pack: Path to matrix_coo sparse pack directory
        sample_count: Number of samples in rhs/solutions
    """

    rhs: Path
    solutions: Path
    matrix_pack: Path
    sample_count: int


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


def _load_training_arrays(data_dir: Path) -> TrainingArrays:
    """Resolve training artifact paths from dataset directory."""
    paths = resolve_dataset_paths(data_dir)
    rhs, solutions = load_dense_training_arrays(data_dir)
    if rhs.shape[0] != solutions.shape[0]:
        raise ValueError(
            f"RHS and solutions sample counts must match, got {rhs.shape[0]} and {solutions.shape[0]}"
        )
    return TrainingArrays(
        rhs=paths.rhs_path,
        solutions=paths.solutions_path,
        matrix_pack=paths.matrix_pack_dir,
        sample_count=int(rhs.shape[0]),
    )


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
    arrays = _load_training_arrays(workspace.data_dir)
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
    dataset = base_dataset.model_copy(
        update={
            "features": features,
            "targets": targets,
            "name": base_dataset.name or "FlexibleDataset",
            # SparseFeature entries are not compatible with current memmap cache path.
            "memmap_cache": False,
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
        features: Feature configs created from dataset artifacts
        targets: Target configs created from dataset artifacts
        dataset_id: Dataset identifier for MLflow experiment naming

    Returns:
        Tuple of (updated_settings, workspace) ready for DLKit execute()
    """
    # Apply transformations sequentially
    # Each returns new GeneralSettings (immutable updates)
    settings = _resolve_dataset(settings, features, targets)
    settings = _configure_dataloader_runtime(settings)
    settings = _configure_output_paths(settings, workspace.root_dir)
    settings = _configure_mlflow(settings, dataset_id)

    return settings, workspace


@contextmanager
def _parent_run_context(parent_run_id: str | None) -> Iterator[None]:
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


def _log_artifacts_to_mlflow(
    tracking_uri: str,
    artifacts_destination: str,
    dataset_id: str,
    run_name: str,
    model_config_path: Path,
    data_config_path: Path | None,
    checkpoint_path: Path | None,
) -> tuple[str, str] | None:
    """Copy run artifacts into the MLflow run's artifact directory.

    Uses MlflowClient (SQLite) to look up exp_id/run_id, then copies files
    directly to the filesystem. This avoids the HTTP server which is shut down
    by DLKit before this function is called.

    Layout:
        {artifacts_destination}/{exp_id}/{run_id}/artifacts/configs/   <- TOML configs
        {artifacts_destination}/{exp_id}/{run_id}/artifacts/checkpoints/ <- .ckpt

    Args:
        tracking_uri: SQLite tracking URI (backend_store_uri from model config).
        artifacts_destination: Local filesystem path for MLflow artifacts.
        dataset_id: MLflow experiment name.
        run_name: MLflow run name matching workspace.run_id.
        model_config_path: Original model config TOML.
        data_config_path: Original data config TOML (or None).
        checkpoint_path: Saved checkpoint file (or None if training produced none).

    Returns:
        Tuple of (exp_id, run_id), or None if the run was not found.
    """
    import shutil

    import mlflow

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(dataset_id)
    if experiment is None:
        return None

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"attributes.run_name = '{run_name}'",
        max_results=1,
        order_by=["attributes.start_time DESC"],
    )
    if not runs:
        # DLKit can override the run name (for example with MODEL.name).
        # Fall back to the latest run in the target experiment.
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=1,
            order_by=["attributes.start_time DESC"],
        )
    if not runs:
        return None

    exp_id = experiment.experiment_id
    run_id = runs[0].info.run_id
    artifact_root = Path(artifacts_destination) / exp_id / run_id / "artifacts"

    # Copy configs
    configs_dir = artifact_root / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_config_path.resolve(), configs_dir / model_config_path.name)
    if data_config_path is not None:
        shutil.copy2(data_config_path.resolve(), configs_dir / data_config_path.name)

    # Copy checkpoint
    if checkpoint_path is not None:
        ckpt_dir = artifact_root / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint_path, ckpt_dir / checkpoint_path.name)

    return exp_id, run_id


def _write_diagnostics_figure(
    y_true: Any,
    y_pred: Any,
    diagnostics: Any,
    figures_dir: Path,
) -> Path:
    """Write training diagnostics figure to disk.

    Args:
        y_true: True target values.
        y_pred: Predicted values.
        diagnostics: Diagnostics result with ``rel_error`` attribute.
        figures_dir: Directory to write the figure into.

    Returns:
        Path to the saved figure file.
    """
    from neuralls.plotting import plot_parity_and_residuals

    figure_path = figures_dir / "diagnostics_training.png"
    plot_parity_and_residuals(
        y_true.ravel(),
        y_pred.ravel(),
        rel_l2_error=diagnostics.rel_error,
        save_path=figure_path,
    )
    return figure_path


def _log_diagnostics_to_mlflow(
    tracking_uri: str,
    run_id: str,
    exp_id: str,
    artifacts_destination: str,
    diagnostics: Any,
    figure_path: Path,
) -> None:
    """Log diagnostics metrics and figure artifact to an existing MLflow run.

    Args:
        tracking_uri: SQLite tracking URI.
        run_id: Existing MLflow run ID to reopen.
        exp_id: MLflow experiment ID (unused, kept for API compatibility).
        artifacts_destination: Local filesystem path for MLflow artifacts (unused).
        diagnostics: Diagnostics result with ``metrics`` dict.
        figure_path: Path to the figure file to log.
    """
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(diagnostics.metrics)
        mlflow.log_artifact(str(figure_path), artifact_path="figures")


def _log_training_evaluation(
    tracking_uri: str,
    run_id: str,
    exp_id: str,
    artifacts_destination: str,
    training_result: Any,
    figures_dir: Path,
) -> None:
    """Compute diagnostics from training predictions and log to existing MLflow run.

    Uses the predictions and targets already captured by ``trainer.predict()``
    during training. Delegates figure writing and MLflow logging to dedicated
    helpers for testability.

    Args:
        tracking_uri: SQLite tracking URI.
        run_id: Existing MLflow run ID to reopen.
        exp_id: MLflow experiment ID (for filesystem artifact path).
        artifacts_destination: Local filesystem path for MLflow artifacts.
        training_result: DLKit TrainingResult with captured predictions and targets.
        figures_dir: Directory to write the diagnostics figure.
    """
    all_numpy = training_result.to_numpy()
    if all_numpy is None:
        return
    y_pred = all_numpy.get("predictions", {}).get("output")
    targets = all_numpy.get("targets", {})
    if y_pred is None or not targets:
        return
    y_true = next(iter(targets.values()))

    diagnostics = compute_diagnostics(y_pred, y_true)
    figure_path = _write_diagnostics_figure(y_true, y_pred, diagnostics, figures_dir)
    _log_diagnostics_to_mlflow(
        tracking_uri, run_id, exp_id, artifacts_destination, diagnostics, figure_path
    )


def train_model(
    *,
    config_path: str | Path,
    data_config_path: str | Path | None = None,
    session_name: str | None = None,
    output_root: Path | str | None = None,
    max_epochs: int | None = None,
    parent_run_id: str | None = None,
) -> Path:
    """Train a DLKit model using resolved data+config context.

    This is the main entry point for model training. It orchestrates:
    1. Load experiment configuration (model + data configs) into a temp dir
    2. Resolve dataset artifacts (rhs/solutions/matrix_coo)
    3. Configure DLKit settings (dataset, paths, MLflow)
    4. Execute training via DLKit
    5. Copy checkpoint to permanent location under output_root
    6. Return path to saved checkpoint

    The workspace (checkpoints, figures, predictions) is created in a temporary
    directory and deleted after training. Only the final checkpoint and sidecar
    are copied to the permanent ``output_root / "checkpoints" / dataset_id``.

    DLKit handles all MLflow operations including server startup and tracking.

    Args:
        config_path: Path to model configuration TOML (e.g., configs/linear.toml)
        data_config_path: Path to data configuration TOML (e.g., data-configs/collect-504.toml)
        session_name: Optional session name for MLflow (unused, for compatibility)
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
    with tempfile.TemporaryDirectory(prefix="neuralls_train_") as _tmp:
        tmp_path = Path(_tmp)

        # Step 1: Load experiment configuration into temp dir
        experiment = load_experiment(
            config_path,
            data_config_path,
            output_root=tmp_path,
        )
        settings = experiment.settings
        workspace = experiment.workspace
        dataset_id = experiment.spec.data_config_path.stem

        # Step 2: Resolve training dataset artifacts
        _, features, targets = _load_and_prepare_data(settings, workspace)

        # Step 3: Configure DLKit settings (dataset, paths, MLflow)
        settings, workspace = _configure_training_pipeline(
            settings,
            workspace,
            features,
            targets,
            dataset_id,
        )

        # Step 4: Execute training via DLKit
        if max_epochs is not None:
            settings = update_settings(settings, {"TRAINING": {"trainer": {"max_epochs": max_epochs}}})
        with _parent_run_context(parent_run_id):
            training_result = execute(settings, run_name=workspace.run_id)  # type: ignore[arg-type]

        # Step 5: Retrieve checkpoint from temp dir (before it is deleted)
        local_checkpoint = get_latest_checkpoint(workspace.checkpoint_dir)
        if local_checkpoint is None:
            raise RuntimeError(f"No checkpoint found in {workspace.checkpoint_dir}")

        # Step 6: Log artifacts to MLflow (must happen while temp dir still exists)
        mlflow_cfg = getattr(settings, "MLFLOW", None)
        server_cfg = getattr(mlflow_cfg, "server", None)
        tracking_uri = getattr(server_cfg, "backend_store_uri", None) if server_cfg else None
        artifacts_destination = getattr(server_cfg, "artifacts_destination", None) if server_cfg else None

        mlflow_ids: tuple[str, str] | None = None
        if tracking_uri and artifacts_destination:
            mlflow_ids = _log_artifacts_to_mlflow(
                tracking_uri=tracking_uri,
                artifacts_destination=artifacts_destination,
                dataset_id=dataset_id,
                run_name=workspace.run_id,
                model_config_path=Path(config_path),
                data_config_path=Path(data_config_path) if data_config_path else None,
                checkpoint_path=local_checkpoint,
            )
            if mlflow_ids is not None:
                exp_id, run_id = mlflow_ids
                _log_training_evaluation(
                    tracking_uri,
                    run_id,
                    exp_id,
                    artifacts_destination,
                    training_result,
                    workspace.figures_dir,
                )

        # Step 7: Copy checkpoint to permanent location (temp dir deleted after this block)
        permanent_root = Path(output_root).resolve() if output_root else DEFAULT_OUTPUT_DIR
        permanent_dir = permanent_root / "checkpoints" / dataset_id
        permanent_dir.mkdir(parents=True, exist_ok=True)
        permanent_checkpoint = Path(shutil.copy2(local_checkpoint, permanent_dir))

        # Step 8: Write sidecar next to permanent checkpoint
        if mlflow_ids is not None:
            exp_id, run_id = mlflow_ids
            write_mlflow_sidecar(
                path=permanent_dir / "mlflow_run.json",
                run_id=run_id,
                experiment_id=exp_id,
                tracking_uri=tracking_uri,
                artifacts_destination=artifacts_destination,
            )

    # temp dir deleted here
    return permanent_checkpoint
