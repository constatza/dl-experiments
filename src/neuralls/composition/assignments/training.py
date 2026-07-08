"""Training orchestration helpers consumed by CLI scripts and workflows."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from dlkit.infrastructure.config.core.patching import patch_model
from dlkit.interfaces.api import execute
from dlkit.interfaces.api.domain.override_types import ExecutionOverrides

from neuralls.composition.assignments._dataset_assembly import (
    _extra_feature_names_from_settings,
    _load_and_prepare_data,
)
from neuralls.composition.assignments._settings_pipeline import _configure_training_pipeline
from neuralls.composition.assignments._training_artifacts import (
    _build_training_run_config,
    _get_normalized_training_numpy_payload,
    _log_training_context,
    _log_training_evaluation,
    _resolve_mlflow_run_ids,
    _resolve_training_checkpoint,
    _stage_training_artifacts,
    create_fallback_training_run,
)
from neuralls.composition.assignments.assembler import load_assignment
from neuralls.composition.assignments.runtime_dataset_contract import (
    default_training_dataset_contract,
)
from neuralls.platform.dlkit.tracking_hooks import build_parent_link_hooks
from neuralls.platform.config.loaders import load_tracking_config
from neuralls.platform.config.settings import NeurallsSettings, require_settings
from neuralls.platform.tracking.environment import scoped_mlflow_environment
from neuralls.platform.tracking.extra_features import log_extra_feature_names_tag
from neuralls.platform.tracking.mlflow import (
    build_runtime_environment,
    resolve_runtime_tracking_config,
)
from neuralls.platform.tracking.mlflow_client import (
    log_artifacts_to_mlflow,
)
from neuralls.platform.tracking.model_registry import ensure_checkpoint_artifact


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
    assignment_id: str | None = None,
    assignment_display_name: str | None = None,
    dataset_registry_id: str | None = None,
    dataset_display_name: str | None = None,
    job_registry_id: str | None = None,
    job_display_name: str | None = None,
    mlflow_experiment_name: str | None = None,
) -> tuple[str, str]:
    """Train a DLKit model using resolved data+config context.

    This is the main entry point for model training. It orchestrates:
    1. Load assignment configuration (model + data configs) into a temp dir
    2. Resolve dataset artifacts (rhs/solutions/matrix_zarr)
    3. Build execute()-time MLflow naming and tags
    4. Configure DLKit settings (dataset, paths, MLflow names)
    5. Execute training via DLKit
    6. Upload the checkpoint to the MLflow run's artifact store
    7. Return the MLflow run id

    The workspace (checkpoints, figures, predictions) is created in a temporary
    directory and deleted after training. MLflow owns the durable copy of the
    checkpoint — nothing is kept on local disk after this function returns.

    DLKit handles all MLflow operations including server startup and tracking.

    Args:
        config_path: Path to a job configuration TOML (e.g., /path/to/job.toml)
        data_config_path: Path to a dataset configuration TOML (e.g., /path/to/dataset.toml)
        output_root: Root directory for the training workspace. Defaults to
            ``DEFAULT_OUTPUT_DIR`` from constants.
        parent_run_id: Optional MLflow parent run UUID. When set, the run created by
            dlkit is tagged with ``mlflow.parentRunId`` the instant it's created (via
            dlkit's ``on_run_created`` lifecycle hook), nesting it under the parent
            from the start rather than after training completes.

    Returns:
        Tuple of (run_id, tracking_uri) for the MLflow run the checkpoint was
        uploaded to.

    Raises:
        RuntimeError: If no checkpoint found after training
        ValueError: If config missing required sections
    """
    resolved_case_config_path = Path(case_config_path) if case_config_path else None
    settings = require_settings(settings, case_config_path=resolved_case_config_path)
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
        # Step 1: Load assignment configuration into temp dir
        assignment = load_assignment(
            job_config_path=config_path,
            data_config_path=resolved_data_config_path,
            neuralls_settings=settings,
            case_config_path=resolved_case_config_path,
            output_root=tmp_path,
            assignment_id=assignment_id,
            assignment_display_name=assignment_display_name,
            dataset_registry_id=dataset_registry_id,
            dataset_display_name=dataset_display_name,
            job_registry_id=job_registry_id,
            job_display_name=job_display_name,
        )
        workflow_settings = assignment.settings
        workspace = assignment.workspace
        contract = default_training_dataset_contract()
        dataset_id = workspace.dataset_id
        resolved_assignment_display_name = assignment.spec.assignment_display_name
        resolved_dataset_display_name = assignment.spec.dataset_display_name or dataset_id

        # Step 2: Resolve training dataset artifacts
        _, features, targets = _load_and_prepare_data(workflow_settings, workspace, contract)

        # Step 3: Build execute()-time MLflow naming and tags
        run_config = _build_training_run_config(
            assignment_id=assignment.spec.assignment_id,
            assignment_display_name=resolved_assignment_display_name,
            dataset_registry_id=assignment.spec.dataset_id,
            job_registry_id=assignment.spec.job_id,
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
        )

        # Step 5: Execute training via DLKit
        if max_epochs is not None:
            workflow_settings = patch_model(
                workflow_settings,
                {"training": {"trainer": {"max_epochs": max_epochs}}},
            )
        # Read backend and uri from tracking.toml so both come from config, not code.
        _shared_tracking = load_tracking_config()
        _tracking_backend = _shared_tracking.backend if _shared_tracking else "mlflow"
        workflow_settings = patch_model(
            workflow_settings,
            {
                "tracking": {
                    "backend": _tracking_backend,
                    "uri": runtime_mlflow_env["MLFLOW_TRACKING_URI"],
                }
            },
        )
        with scoped_mlflow_environment(runtime_mlflow_env):
            # neuralls owns all model registration; dlkit's own registration surface is removed.
            execution_result = execute(
                workflow_settings,
                overrides=ExecutionOverrides(
                    experiment_name=run_config.experiment_name,
                    run_name=run_config.run_name,
                    tags=dict(run_config.tags),
                ),
                hooks=build_parent_link_hooks(parent_run_id),
            )
            training_result = _unwrap_execution_result(execution_result)

            # Step 6: Resolve MLflow run metadata and retrieve the checkpoint
            tracking_uri, _ = resolve_runtime_tracking_config()
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
                tracking_uri, _, run_id = mlflow_coords
                _log_training_context(
                    tracking_uri=tracking_uri,
                    run_id=run_id,
                    assignment_id=assignment.spec.assignment_id,
                    assignment_display_name=resolved_assignment_display_name,
                    resolved_dataset_id=dataset_id,
                    dataset_display_name=resolved_dataset_display_name,
                    dataset_registry_id=assignment.spec.dataset_id,
                    job_registry_id=assignment.spec.job_id,
                    job_display_name=assignment.spec.job_display_name or workspace.run_id,
                )
                log_extra_feature_names_tag(
                    tracking_uri,
                    run_id,
                    _extra_feature_names_from_settings(workflow_settings, contract),
                )
                # Step 8: Make the checkpoint durable in MLflow — this is the only
                # copy that survives the tempdir being deleted below. Idempotent:
                # no-ops if the artifact is already present under this run.
                ensure_checkpoint_artifact(
                    tracking_uri=tracking_uri,
                    run_id=run_id,
                    checkpoint_path=local_checkpoint,
                )
                normalized_numpy = _get_normalized_training_numpy_payload(
                    training_result,
                    contract,
                )
                _stage_training_artifacts(
                    workspace=workspace,
                    training_result=training_result,
                    numpy_payload=normalized_numpy,
                    job_config_path=config_path,
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
                    "Training completed without MLflow run metadata; creating fallback run."
                )
                try:
                    fallback_exp_id, fallback_run_id = create_fallback_training_run(
                        tracking_uri=tracking_uri or "",
                        experiment_name=run_config.experiment_name,
                        run_name=run_config.run_name,
                        tags=dict(run_config.tags),
                        checkpoint_path=local_checkpoint,
                    )
                    mlflow_coords = (tracking_uri or "", fallback_exp_id, fallback_run_id)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Could not create fallback MLflow run: {}", exc)

        # temp dir deleted here
        if mlflow_coords is None:
            raise RuntimeError(
                f"Training completed but no MLflow run could be established for "
                f"assignment '{assignment_id}' — nothing to attach the checkpoint to."
            )
        resolved_final_tracking_uri, _, resolved_final_run_id = mlflow_coords
        return resolved_final_run_id, resolved_final_tracking_uri
