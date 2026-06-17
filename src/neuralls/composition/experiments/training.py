"""Training orchestration helpers consumed by CLI scripts and workflows."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from dlkit.infrastructure.config.core.patching import patch_model
from dlkit.infrastructure.config.workflow_configs import (
    OptimizationWorkflowConfig,
    TrainingWorkflowConfig,
)
from dlkit.interfaces.api import execute
from dlkit.interfaces.api.domain.override_types import ExecutionOverrides

from neuralls.composition.experiments._dataset_assembly import (
    _extra_feature_names_from_settings,
    _load_and_prepare_data,
)
from neuralls.composition.experiments._settings_pipeline import _configure_training_pipeline
from neuralls.composition.experiments._training_artifacts import (
    _build_training_run_config,
    _get_normalized_training_numpy_payload,
    _log_training_context,
    _log_training_evaluation,
    _resolve_mlflow_run_ids,
    _resolve_training_checkpoint,
    _stage_training_artifacts,
)
from neuralls.composition.experiments.assembler import load_experiment
from neuralls.composition.experiments.runtime_dataset_contract import (
    default_training_dataset_contract,
)
from neuralls.platform.config.settings import NeurallsSettings, require_settings
from neuralls.platform.reporting.predictions import write_mlflow_sidecar
from neuralls.platform.tracking.environment import scoped_mlflow_environment
from neuralls.platform.tracking.extra_features import log_extra_feature_names_tag
from neuralls.platform.tracking.mlflow import (
    build_runtime_environment,
    resolve_runtime_tracking_config,
)
from neuralls.platform.tracking.mlflow_client import (
    log_artifacts_to_mlflow,
    parent_run_context,
)

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


def _unwrap_execution_result(result: object) -> object:
    """Normalize DLKit execute() results to the underlying training result."""
    return getattr(result, "training_result", result)


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
            parent_run_id=parent_run_id,
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
                from loguru import logger

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
