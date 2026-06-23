"""DLKit settings transformation pipeline for the training workflow.

Each function is a pure transformation that returns new workflow settings.
The transformations are independent and can be tested in isolation.
"""

from __future__ import annotations

from pathlib import Path

from dlkit.infrastructure.config.core.patching import patch_model
from dlkit.infrastructure.config.dataset_settings import DatasetSettings
from dlkit.infrastructure.config.workflow_configs import (
    OptimizationWorkflowConfig,
    TrainingWorkflowConfig,
)

from neuralls.composition.experiments._dataset_assembly import (
    _validate_runtime_dataset_contract,
)
from dlkit.infrastructure.config.data_entries import DataEntry

from neuralls.composition.experiments.runtime_dataset_contract import RuntimeDatasetContract
from neuralls.platform.config.dataset_entries import apply_placeholder_metadata
from neuralls.platform.config.models.workspace import ExperimentWorkspace

type TrainingWorkflowSettings = TrainingWorkflowConfig | OptimizationWorkflowConfig


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
        settings: Current DLKit settings.
        features: Resolved feature entries to inject.
        targets: Resolved target entries to inject.
        contract: Runtime dataset entry name contract.

    Returns:
        New workflow settings with updated DATASET section.

    Raises:
        ValueError: If DATASET section is missing from config.
    """
    _validate_runtime_dataset_contract(settings, contract)
    base_dataset = settings.DATASET or DatasetSettings()
    feature_entries = apply_placeholder_metadata(features, base_dataset.features)
    target_entries = apply_placeholder_metadata(targets, base_dataset.targets)
    return patch_model(
        settings,
        {
            "DATASET": {
                "features": feature_entries,
                "targets": target_entries,
                "name": base_dataset.name or "FlexibleDataset",
            }
        },
    )


def _configure_output_paths(
    settings: TrainingWorkflowSettings,
    output_dir: Path,
) -> TrainingWorkflowSettings:
    """Configure training output paths.

    Sets default_root_dir and callback dirpaths.

    Args:
        settings: Current DLKit settings.
        output_dir: Workspace root directory (from experiment.workspace.root_dir).

    Returns:
        New workflow settings with updated TRAINING section.

    Raises:
        ValueError: If TRAINING.trainer is absent.
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
    """Inject concrete MLflow names into training settings.

    Args:
        settings: Current DLKit settings.
        mlflow_experiment_name: Resolved MLflow experiment bucket name.
        mlflow_run_name: Timestamped training run name.

    Returns:
        New workflow settings with updated MLFLOW section, or unchanged if
        the MLFLOW section is absent.
    """
    if not settings.MLFLOW:
        return settings
    return patch_model(
        settings,
        {
            "MLFLOW": {
                "experiment_name": mlflow_experiment_name,
                "run_name": mlflow_run_name,
            }
        },
    )


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

    Applies sequential transformations:
    1. Inject resolved feature/target specs into DATASET section.
    2. Configure dataloader runtime (workers, pin_memory).
    3. Configure output paths (checkpoint dir, default root dir).
    4. Inject concrete MLflow experiment/run names into settings.

    Args:
        settings: Base DLKit settings from config file.
        workspace: Experiment workspace with directory paths.
        features: Resolved feature specs created from dataset artifacts.
        targets: Resolved target specs created from dataset artifacts.
        contract: Runtime dataset entry name contract.
        mlflow_experiment_name: Training experiment bucket name.
        mlflow_run_name: Timestamped training run name.

    Returns:
        Tuple of (updated_settings, workspace) ready for DLKit execute().
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
