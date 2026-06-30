"""Compose narrow runtime patch stages into runnable DLKit jobs."""

from __future__ import annotations

from dlkit.infrastructure.config.data_entries import DataEntry

from neuralls.composition.experiments._job_types import AnyJobConfig, TrainLikeJobConfig
from neuralls.composition.experiments.runtime_dataset_contract import RuntimeDatasetContract
from neuralls.composition.experiments.runtime_dataset_patcher import patch_runtime_dataset
from neuralls.composition.experiments.runtime_tracking_patcher import (
    ensure_runtime_tracking,
    patch_training_tracking,
)
from neuralls.composition.experiments.runtime_workspace_patcher import (
    patch_dataloader_runtime,
    patch_runtime_workspace,
)
from neuralls.platform.config.models.workspace import ExperimentWorkspace


def materialize_training_job(
    settings: TrainLikeJobConfig,
    *,
    workspace: ExperimentWorkspace,
    features: list[DataEntry],
    targets: list[DataEntry],
    contract: RuntimeDatasetContract,
) -> tuple[TrainLikeJobConfig, ExperimentWorkspace]:
    """Apply runtime dataset, workspace, and tracking stages in order."""
    settings = patch_runtime_dataset(
        settings,
        features=features,
        targets=targets,
        contract=contract,
    )
    settings = patch_dataloader_runtime(settings)
    settings = patch_runtime_workspace(settings, output_dir=workspace.root_dir)
    settings = patch_training_tracking(settings)
    return settings, workspace


def materialize_inference_job(settings: AnyJobConfig) -> AnyJobConfig:
    """Apply inference-safe runtime tracking normalization only."""
    return ensure_runtime_tracking(settings)
