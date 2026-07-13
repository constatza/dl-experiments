"""Assignment orchestration for training neural network models.

This module provides the main orchestration functions for running assignments
(one job run on one dataset):
- `run_assignment()`: Run a single assignment (data generation + training)
- `run_assignment_matrix()`: Run every assignment from one case config

Architecture:
    1. Data generation (with caching) - process_config() from generation module
    2. Model training (with checkpoint detection) - train_model() from training module
    3. Solver comparison (separate) - compare_preconditioners() from compare module

Note:
    This orchestrator does NOT handle solver comparisons. For CG solver benchmarking,
    use the comparison workflows after training completes.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from neuralls.platform.config.models.dataset_identity import resolve_dataset_identity
from neuralls.composition.assignments.assembler import (
    load_assignment_batch,
    load_validated_case_config,
)
from neuralls.composition.tracking.run_specs import build_session_run_spec
from neuralls.platform.config.loaders import load_data_config
from neuralls.platform.config.settings import NeurallsSettings, require_settings
from neuralls.platform.storage.base import load_matrix
from neuralls.application.models import AssignmentResult
from neuralls.platform.caching import compute_directory_hash
from neuralls.composition.generation.processing import process_config
from neuralls.composition.assignments.training import train_model
from neuralls.platform.storage.dataset_readers import resolve_dataset_artifacts
from neuralls.platform.tracking.environment import scoped_mlflow_environment
from neuralls.platform.tracking.mlflow import build_workflow_environment
from neuralls.platform.tracking.mlflow_client import find_successful_run
from neuralls.composition.tracking.session import session_parent_run


def run_assignment(
    *,
    settings: NeurallsSettings,
    job_config_path: Path,
    data_config_path: Path,
    output_root: Path,
    force: bool,
    src_hash: str,
    max_epochs: int | None = None,
    assignment_id: str,
    assignment_display_name: str,
    dataset_registry_id: str | None = None,
    dataset_display_name: str | None = None,
    job_registry_id: str | None = None,
    job_display_name: str | None = None,
    parent_run_id: str | None = None,
    mlflow_experiment_name: str | None = None,
    tracking_uri: str | None = None,
) -> AssignmentResult:
    """Run data generation and model training for a single assignment.

    This function orchestrates a complete assignment workflow:
    1. Load data configuration from TOML file
    2. Generate/cache dataset artifacts (manifest + .npy + sparse pack)
    3. Check MLflow for an already-completed run of this assignment (skip if found and force=False)
    4. Train model if needed (creates checkpoint)
    5. Return success/failure result

    Args:
        job_config_path: Path to a job configuration TOML (e.g., /path/to/job.toml)
        data_config_path: Path to a dataset configuration TOML (e.g., /path/to/dataset.toml)
        output_root: Root directory for all assignment outputs
        force: If True, retrain even if a completed run already exists. If False, reuse it.
        src_hash: Hash of source code directory for cache invalidation
        tracking_uri: MLflow tracking URI used to check for a prior completed run.

    Returns:
        AssignmentResult with status='Success' or 'Failed' and optional error message

    Note:
        For solver comparison, use compare_preconditioners() after training completes.
        This function only generates data and trains models.

    Example:
        >>> result = run_assignment(
        ...     job_config_path=Path("/tmp/job.toml"),
        ...     data_config_path=Path("/tmp/dataset.toml"),
        ...     output_root=Path("output"),
        ...     force=False,
        ...     src_hash="abc123",
        ... )
        >>> print(result.status)
        'Success'
    """
    try:
        # Step 1: Load data configuration and generate/cache dataset.
        # Fails fast if the dataset config has no resolvable identity — the
        # resolved name itself isn't needed here now that the checkpoint-dir
        # computation that used it has moved to MLflow (Step 2 below).
        data_cfg = load_data_config(data_config_path, settings)
        resolve_dataset_identity(data_cfg=data_cfg, config_path=data_config_path)
        if data_cfg.source.matrix_path is None:
            raise ValueError("Missing 'source.matrix_path' in data config")
        if data_cfg.output.data_dir is None:
            data_cfg = data_cfg.model_copy(
                update={"output": data_cfg.output.with_data_dir(settings.processed_dir)}
            )
        matrix_path = data_cfg.source.matrix_path
        if matrix_path is None:
            raise ValueError("Missing 'source.matrix_path' in data config")
        matrix = load_matrix(Path(matrix_path))
        data_dir = process_config(data_cfg, matrix)
        artifacts = resolve_dataset_artifacts(data_dir)
        missing = [
            str(p)
            for p in (artifacts.rhs.path, artifacts.solutions.path, artifacts.matrix.path)
            if not p.exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"Required data files not found in {data_dir}:\n  - " + "\n  - ".join(missing)
            )

        # Step 2: Check MLflow for a completed run of this exact assignment. A FINISHED
        # run tagged with this assignment_id is trusted as equivalent to training again —
        # nothing is reused locally, whatever needs the checkpoint later (comparison,
        # inference) resolves it independently through MLflow.
        already_done = (
            not force
            and mlflow_experiment_name is not None
            and tracking_uri is not None
            and find_successful_run(
                tracking_uri=tracking_uri,
                mlflow_experiment_name=mlflow_experiment_name,
                assignment_id=assignment_id,
            )
            is not None
        )

        # Step 3: Train model unless a completed run already exists
        if already_done:
            logger.info(f"Using existing MLflow run for assignment '{assignment_id}'")
        else:
            run_id, _ = train_model(
                config_path=job_config_path,
                job_registry_id=job_registry_id,
                job_display_name=job_display_name,
                settings=settings,
                data_config_path=data_config_path,
                output_root=output_root,
                max_epochs=max_epochs,
                assignment_id=assignment_id,
                assignment_display_name=assignment_display_name,
                dataset_registry_id=dataset_registry_id,
                dataset_display_name=dataset_display_name,
                parent_run_id=parent_run_id,
                mlflow_experiment_name=mlflow_experiment_name,
            )
            logger.info(f"Training complete: run {run_id}")

        # Step 4: Return success result
        return AssignmentResult(
            assignment_id=assignment_id,
            assignment_display_name=assignment_display_name,
            status="Success",
        )
    except Exception as exc:  # noqa: BLE001
        # Broad by design: one assignment's failure (including dlkit-internal
        # errors, e.g. a leaked MLflow run from a prior search job) must never
        # abort the rest of the batch.
        logger.error(f"Assignment {assignment_id} failed: {exc}")
        return AssignmentResult(
            assignment_id=assignment_id,
            assignment_display_name=assignment_display_name,
            status="Failed",
            error=str(exc),
        )


def run_assignment_matrix(
    case_config_path: Path,
    settings: NeurallsSettings | None = None,
    *,
    force: bool = False,
    project_root: Path | None = None,
    max_epochs: int | None = None,
) -> list[AssignmentResult]:
    """Run training for every assignment defined in one case config.

    This is the main orchestrator for running multiple assignments in sequence.
    Each assignment consists of:
    1. Dataset generation (with caching based on data config + src hash)
    2. Model training (with MLflow-backed reuse unless force=True)

    The function processes assignments sequentially to avoid resource contention.
    Failed assignments don't stop the batch - each returns a result with status.

    Args:
        case_config_path: Path to a case config defining all assignments
        force: If True, retrain all models even if a completed run already exists
        project_root: Root directory for resolving src/ (auto-detected if None)

    Returns:
        List of AssignmentResult objects, one per assignment (success or failure)

    Note:
        For solver comparison after training, use comparison workflows:
        >>> from neuralls.composition.assignments.comparison_batch import run_comparison_batch
        >>> run_batch_comparison(case_config_path, comparison_config_path, params)

    Example:
        >>> results = run_assignment_matrix(
        ...     Path("/tmp/case.toml"),
        ...     force=False,
        ... )
        >>> success_count = sum(1 for r in results if r.status == "Success")
        >>> print(f"{success_count}/{len(results)} assignments succeeded")
    """
    settings = require_settings(settings, case_config_path=case_config_path)
    # Auto-detect project root if not provided (for src hash calculation)
    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]

    # Load all assignment definitions from the case config
    batch = load_assignment_batch(case_config_path, settings)
    assignments = batch.assignments

    # Compute source code hash for cache invalidation
    # When src/ changes, datasets are regenerated to ensure consistency
    src_dir = project_root / "src"
    if not src_dir.exists():
        src_dir = Path(__file__).resolve().parent.parent
    src_hash = compute_directory_hash(src_dir)

    logger.info(f"Training {len(assignments)} assignments from {case_config_path}")

    # Open one case-level MLflow parent run so every training/search run below
    # (including dlkit's own nested Optuna trial runs) nests under it, and so
    # the MLflow experiment gets its artifact_location pinned before dlkit auto-creates it.
    cfg, _ = load_validated_case_config(case_config_path, settings)
    training_mlflow_env = build_workflow_environment(
        tracking_uri=cfg.mlflow.tracking_uri,
        artifact_location=cfg.mlflow.artifacts_destination,
        default_output_root=batch.output_root,
    )
    mlflow_experiment_name = cfg.names.training

    results: list[AssignmentResult] = []
    with scoped_mlflow_environment(training_mlflow_env.env):
        session_run_name, session_tags = build_session_run_spec(
            case_config_path=case_config_path.resolve(),
            experiment_name=mlflow_experiment_name,
            phase="session_training",
        )
        with session_parent_run(
            tracking_uri=training_mlflow_env.tracking_uri,
            artifact_uri=training_mlflow_env.artifact_uri,
            run_name=session_run_name,
            tags=session_tags.as_mlflow_tags(),
            experiment_name=mlflow_experiment_name,
        ) as handle:
            # Run each assignment sequentially
            # Sequential execution avoids GPU/memory contention and makes logs clearer
            for assignment in assignments:
                # Log assignment details for progress tracking
                logger.info(f"\n{'=' * 60}")
                logger.info(f"Assignment: {assignment.spec.assignment_display_name}")
                logger.info(
                    f"  Job: {assignment.spec.job_display_name or assignment.spec.job_config_path.stem}"
                )
                logger.info(
                    f"  Dataset: {assignment.spec.dataset_display_name or assignment.workspace.dataset_id}"
                )
                logger.info(f"{'=' * 60}")

                # Run single assignment (catches exceptions internally)
                result = run_assignment(
                    settings=settings,
                    job_config_path=assignment.spec.job_config_path,
                    data_config_path=assignment.spec.data_config_path,
                    output_root=batch.output_root,
                    force=force,
                    src_hash=src_hash,
                    max_epochs=max_epochs,
                    assignment_id=assignment.spec.assignment_id,
                    assignment_display_name=assignment.spec.assignment_display_name,
                    dataset_registry_id=assignment.spec.dataset_id,
                    dataset_display_name=assignment.spec.dataset_display_name,
                    job_registry_id=assignment.spec.job_id,
                    job_display_name=assignment.spec.job_display_name,
                    parent_run_id=handle.parent_run_id,
                    mlflow_experiment_name=mlflow_experiment_name,
                    tracking_uri=training_mlflow_env.tracking_uri,
                )
                results.append(result)
                if not result.is_success:
                    handle.mark_failed()

    return results
