"""Batch training workflow: prepare/reuse-check every assignment, then train them as one dlkit multirun sweep.

This module provides the main orchestration functions for running assignments
(one job run on one dataset):
- `run_assignment()`: Generate/cache one assignment's dataset, run the MLflow
  reuse check, and either report a cache-hit success or prepare it for the sweep.
- `run_assignment_matrix()`: Run every assignment from one case config as a
  single dlkit multirun sweep.

Architecture:
    1. Data generation (with caching) - process_config() from generation module
    2. Model training (with checkpoint detection) - dlkit's run_multirun_spec()
    3. Solver comparison (separate) - compare_preconditioners() from compare module

Note:
    This orchestrator does NOT handle solver comparisons. For CG solver benchmarking,
    use the comparison workflows after training completes.
"""

from __future__ import annotations

from pathlib import Path

from dlkit.common import ChildFailure, ChildSuccess
from dlkit.engine.workflows.multi_run import MultiRunSpec, RunSpec
from dlkit.interfaces.api import run_multirun_spec
from loguru import logger

from neuralls.application.models import AssignmentResult
from neuralls.composition.assignments.assembler import (
    load_assignment_batch,
    load_validated_case_config,
)
from neuralls.composition.assignments.training import (
    PreparedTraining,
    cleanup_prepared_training,
    finalize_prepared_training,
    prepare_training_settings,
    to_run_spec,
)
from neuralls.composition.generation.processing import process_config
from neuralls.composition.tracking.run_specs import build_session_run_spec
from neuralls.platform.caching import compute_directory_hash
from neuralls.platform.config.loaders import load_data_config
from neuralls.platform.config.models.dataset_identity import resolve_dataset_identity
from neuralls.platform.config.settings import NeurallsSettings, require_settings
from neuralls.platform.storage.base import load_matrix
from neuralls.platform.storage.dataset_readers import resolve_dataset_artifacts
from neuralls.platform.tracking.environment import scoped_mlflow_environment
from neuralls.platform.tracking.mlflow import (
    build_workflow_environment,
    finalize_session_parent_run,
)
from neuralls.platform.tracking.mlflow_client import find_successful_run


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
    mlflow_experiment_name: str | None = None,
    tracking_uri: str | None = None,
) -> AssignmentResult | PreparedTraining:
    """Generate/cache one assignment's dataset and prepare it for the training sweep.

    This function performs everything a dlkit multirun sweep child can't do for
    itself, ahead of `run_assignment_matrix()` building that sweep:
    1. Load data configuration from TOML file
    2. Generate/cache dataset artifacts (manifest + .npy + sparse pack)
    3. Check MLflow for an already-completed run of this assignment (skip if found and force=False)
    4. If training is needed, resolve dlkit settings via `prepare_training_settings()`

    Args:
        job_config_path: Path to a job configuration TOML (e.g., /path/to/job.toml)
        data_config_path: Path to a dataset configuration TOML (e.g., /path/to/dataset.toml)
        output_root: Root directory for all assignment outputs
        force: If True, retrain even if a completed run already exists. If False, reuse it.
        src_hash: Hash of source code directory for cache invalidation
        tracking_uri: MLflow tracking URI used to check for a prior completed run.

    Returns:
        An `AssignmentResult` with status='Success' (a completed MLflow run already
        exists and `force=False`) or status='Failed' (dataset generation, the reuse
        check, or dlkit settings preparation raised). Otherwise a `PreparedTraining`
        ready to become one child of the training sweep in `run_assignment_matrix()`.

    Note:
        For solver comparison, use compare_preconditioners() after training completes.
        This function only generates data and prepares models for training.

    Example:
        >>> outcome = run_assignment(
        ...     settings=settings,
        ...     job_config_path=Path("/tmp/job.toml"),
        ...     data_config_path=Path("/tmp/dataset.toml"),
        ...     output_root=Path("output"),
        ...     force=False,
        ...     src_hash="abc123",
        ...     assignment_id="assignment-1",
        ...     assignment_display_name="Assignment 1",
        ... )
        >>> isinstance(outcome, AssignmentResult) and outcome.status
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

        # Step 3: Skip training if a completed run already exists; otherwise resolve
        # this assignment's dlkit settings so the caller can add it to the sweep.
        if already_done:
            logger.info(f"Using existing MLflow run for assignment '{assignment_id}'")
            return AssignmentResult(
                assignment_id=assignment_id,
                assignment_display_name=assignment_display_name,
                status="Success",
            )

        return prepare_training_settings(
            config_path=job_config_path,
            data_config_path=data_config_path,
            settings=settings,
            output_root=output_root,
            max_epochs=max_epochs,
            assignment_id=assignment_id,
            assignment_display_name=assignment_display_name,
            dataset_registry_id=dataset_registry_id,
            dataset_display_name=dataset_display_name,
            job_registry_id=job_registry_id,
            job_display_name=job_display_name,
            mlflow_experiment_name=mlflow_experiment_name,
            batched=True,
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


def _finalize_assignment_child(
    *,
    prepared: PreparedTraining,
    execution_result: object,
) -> AssignmentResult:
    """Finalize one successful multirun child: make its checkpoint durable in MLflow.

    Converts a durability failure into a Failed `AssignmentResult` rather than
    raising — mirrors `run_assignment()`'s original per-assignment failure
    isolation, which used to cover the full train+finalize path via `train_model()`.

    Args:
        prepared: This assignment's prepared training inputs.
        execution_result: The multirun child's dispatch result.

    Returns:
        `AssignmentResult` with status='Success' or 'Failed'.
    """
    spec = prepared.assignment.spec
    try:
        run_id, _ = finalize_prepared_training(prepared, execution_result)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Assignment {spec.assignment_id} failed: {exc}")
        return AssignmentResult(
            assignment_id=spec.assignment_id,
            assignment_display_name=spec.assignment_display_name,
            status="Failed",
            error=str(exc),
        )
    logger.info(f"Training complete: run {run_id}")
    return AssignmentResult(
        assignment_id=spec.assignment_id,
        assignment_display_name=spec.assignment_display_name,
        status="Success",
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

    This is the main orchestrator for running multiple assignments. Each
    assignment consists of:
    1. Dataset generation (with caching based on data config + src hash)
    2. Model training (with MLflow-backed reuse unless force=True), dispatched
       as one dlkit multirun sweep across every assignment that needs it

    Assignments whose MLflow reuse check finds a completed run are resolved
    immediately without joining the sweep. Failed assignments — whether the
    failure happened during dataset generation/preparation or during the
    sweep itself — don't stop the batch; each returns a result with status.

    Args:
        case_config_path: Path to a case config defining all assignments
        force: If True, retrain all models even if a completed run already exists
        project_root: Root directory for resolving src/ (auto-detected if None)

    Returns:
        List of AssignmentResult objects, one per assignment (success or
        failure), in the same order as the case config's `[[assignments]]`.

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

    # Resolve the case-level MLflow topology once. Every assignment needing
    # training joins one dlkit multirun sweep below — dlkit owns that sweep's
    # parent run's lifecycle and per-child failure isolation natively.
    cfg, _ = load_validated_case_config(case_config_path, settings)
    training_mlflow_env = build_workflow_environment(
        tracking_uri=cfg.mlflow.tracking_uri,
        artifact_location=cfg.mlflow.artifacts_destination,
        default_output_root=batch.output_root,
    )
    mlflow_experiment_name = cfg.names.training

    results_by_id: dict[str, AssignmentResult] = {}
    prepared_by_child_id: dict[str, PreparedTraining] = {}
    run_specs: list[RunSpec] = []
    any_failed = False

    with scoped_mlflow_environment(training_mlflow_env.env):
        session_run_name, session_tags = build_session_run_spec(
            case_config_path=case_config_path.resolve(),
            experiment_name=mlflow_experiment_name,
            phase="session_training",
        )

        # Phase 1: Generate/cache each assignment's dataset and run its MLflow
        # reuse check. Cache hits resolve immediately; everything else is
        # prepared as one child of the sweep below.
        for assignment in assignments:
            spec = assignment.spec
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Assignment: {spec.assignment_display_name}")
            logger.info(f"  Job: {spec.job_display_name or spec.job_config_path.stem}")
            logger.info(
                f"  Dataset: {spec.dataset_display_name or assignment.workspace.dataset_id}"
            )
            logger.info(f"{'=' * 60}")

            outcome = run_assignment(
                settings=settings,
                job_config_path=spec.job_config_path,
                data_config_path=spec.data_config_path,
                output_root=batch.output_root,
                force=force,
                src_hash=src_hash,
                max_epochs=max_epochs,
                assignment_id=spec.assignment_id,
                assignment_display_name=spec.assignment_display_name,
                dataset_registry_id=spec.dataset_id,
                dataset_display_name=spec.dataset_display_name,
                job_registry_id=spec.job_id,
                job_display_name=spec.job_display_name,
                mlflow_experiment_name=mlflow_experiment_name,
                tracking_uri=training_mlflow_env.tracking_uri,
            )
            match outcome:
                case PreparedTraining():
                    prepared_by_child_id[spec.assignment_id] = outcome
                    run_specs.append(to_run_spec(outcome))
                case AssignmentResult():
                    results_by_id[spec.assignment_id] = outcome
                    if not outcome.is_success:
                        any_failed = True

        # Phase 2: Dispatch every assignment that needs training as one sweep.
        sweep_result = None
        if run_specs:
            sweep_result = run_multirun_spec(
                MultiRunSpec(
                    experiment_name=mlflow_experiment_name,
                    parent_run_name=session_run_name,
                    parent_tags=dict(session_tags.as_mlflow_tags()),
                    failure_policy="continue",
                    children=tuple(run_specs),
                )
            )
            for child_outcome in sweep_result.children:
                prepared = prepared_by_child_id[child_outcome.child_id]
                try:
                    match child_outcome:
                        case ChildSuccess():
                            result = _finalize_assignment_child(
                                prepared=prepared,
                                execution_result=child_outcome.result,
                            )
                            results_by_id[child_outcome.child_id] = result
                            if not result.is_success:
                                any_failed = True
                        case ChildFailure():
                            any_failed = True
                            logger.error(
                                f"Assignment '{child_outcome.child_id}' failed: {child_outcome.message}"
                            )
                            results_by_id[child_outcome.child_id] = AssignmentResult(
                                assignment_id=child_outcome.child_id,
                                assignment_display_name=prepared.resolved_assignment_display_name,
                                status="Failed",
                                error=child_outcome.message,
                            )
                finally:
                    cleanup_prepared_training(prepared)

        if sweep_result is not None:
            finalize_session_parent_run(
                tracking_uri=sweep_result.tracking_uri or training_mlflow_env.tracking_uri,
                run_id=sweep_result.parent_run_id,
                status="FAILED" if any_failed else "FINISHED",
            )

    return [results_by_id[assignment.spec.assignment_id] for assignment in assignments]
