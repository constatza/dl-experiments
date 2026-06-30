"""Training job materialization pipeline for lower-case DLKit jobs."""

from __future__ import annotations

from dlkit.infrastructure.config.data_entries import DataEntry

from neuralls.composition.experiments._job_types import TrainLikeJobConfig
from neuralls.composition.experiments.job_materializer import materialize_training_job
from neuralls.composition.experiments.runtime_dataset_contract import RuntimeDatasetContract
from neuralls.platform.config.models.workspace import ExperimentWorkspace


def _configure_training_pipeline(
    settings: TrainLikeJobConfig,
    workspace: ExperimentWorkspace,
    features: list[DataEntry],
    targets: list[DataEntry],
    contract: RuntimeDatasetContract,
) -> tuple[TrainLikeJobConfig, ExperimentWorkspace]:
    """Materialize one runnable training job through narrow patch stages."""
    return materialize_training_job(
        settings,
        workspace=workspace,
        features=features,
        targets=targets,
        contract=contract,
    )
