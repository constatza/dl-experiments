"""Experiment orchestration for training neural network models.

This module provides the main orchestration functions for running experiments:
- `run_experiment()`: Run single experiment (data generation + training)
- `run_experiment_matrix()`: Run multiple experiments from one case config

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
from neuralls.composition.experiments.assembler import load_batch, load_validated_case_config
from neuralls.composition.tracking.run_specs import build_session_run_spec
from neuralls.platform.config.loaders import load_data_config
from neuralls.platform.config.settings import NeurallsSettings, require_settings
from neuralls.platform.storage.base import load_matrix
from neuralls.application.models import ExperimentResult
from neuralls.platform.caching import compute_directory_hash
from neuralls.composition.generation.processing import process_config
from neuralls.platform.storage.filesystem import extract_model_name
from neuralls.composition.experiments.training import train_model
from neuralls.platform.storage.dataset_readers import resolve_dataset_artifacts
from neuralls.platform.storage.checkpoints import get_latest_checkpoint
from neuralls.platform.tracking.environment import scoped_mlflow_environment
from neuralls.platform.tracking.mlflow import (
    build_workflow_environment,
    create_session_parent_run,
    finalize_session_parent_run,
)


def run_experiment(
    *,
    settings: NeurallsSettings,
    job_config_path: Path,
    data_config_path: Path,
    output_root: Path,
    force: bool,
    src_hash: str,
    max_epochs: int | None = None,
    experiment_id: str,
    experiment_display_name: str,
    dataset_registry_id: str | None = None,
    dataset_display_name: str | None = None,
    job_registry_id: str | None = None,
    job_display_name: str | None = None,
    parent_run_id: str | None = None,
    mlflow_experiment_name: str | None = None,
) -> ExperimentResult:
    """Run data generation and model training for a single experiment.

    This function orchestrates a complete experiment workflow:
    1. Load data configuration from TOML file
    2. Generate/cache dataset artifacts (manifest + .npy + sparse pack)
    3. Check for existing checkpoint (skip training if found and force=False)
    4. Train model if needed (creates checkpoint)
    5. Return success/failure result

    Args:
        job_config_path: Path to a job configuration TOML (e.g., /path/to/job.toml)
        data_config_path: Path to a dataset configuration TOML (e.g., /path/to/dataset.toml)
        output_root: Root directory for all experiment outputs
        force: If True, retrain even if checkpoint exists. If False, reuse existing checkpoint.
        src_hash: Hash of source code directory for cache invalidation

    Returns:
        ExperimentResult with status='Success' or 'Failed' and optional error message

    Note:
        For solver comparison, use compare_preconditioners() after training completes.
        This function only generates data and trains models.

    Example:
        >>> result = run_experiment(
        ...     job_config_path=Path("/tmp/job.toml"),
        ...     data_config_path=Path("/tmp/dataset.toml"),
        ...     output_root=Path("output"),
        ...     force=False,
        ...     src_hash="abc123",
        ... )
        >>> print(result.status)
        'Success'
    """
    # Extract model identifier from the config filename stem.
    model_name = extract_model_name(job_config_path)
    try:
        # Step 1: Load data configuration and generate/cache dataset
        data_cfg = load_data_config(data_config_path, settings)
        dataset_id = resolve_dataset_identity(
            data_cfg=data_cfg,
            config_path=data_config_path,
        ).name
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

        # Step 2: Check for existing checkpoint to avoid redundant training
        # Workspace structure: output_root/{dataset_id}/{model_name}/checkpoints/
        checkpoint_dir = output_root / dataset_id / model_name / "checkpoints"
        checkpoint = get_latest_checkpoint(checkpoint_dir)

        # Step 3: Train model if no checkpoint exists or force=True
        if force or checkpoint is None:
            checkpoint = train_model(
                config_path=job_config_path,
                job_registry_id=job_registry_id,
                job_display_name=job_display_name,
                settings=settings,
                data_config_path=data_config_path,
                output_root=output_root,
                max_epochs=max_epochs,
                experiment_id=experiment_id,
                experiment_display_name=experiment_display_name,
                dataset_registry_id=dataset_registry_id,
                dataset_display_name=dataset_display_name,
                parent_run_id=parent_run_id,
                mlflow_experiment_name=mlflow_experiment_name,
            )
            logger.info(f"Training complete: {checkpoint}")
        else:
            logger.info(f"Using existing checkpoint: {checkpoint}")

        # Step 4: Return success result
        return ExperimentResult(
            experiment_id=experiment_id,
            experiment_display_name=experiment_display_name,
            status="Success",
        )
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        # Catch expected exceptions to ensure batch runs continue even if one experiment fails
        logger.error(f"Experiment {experiment_id} failed: {exc}")
        return ExperimentResult(
            experiment_id=experiment_id,
            experiment_display_name=experiment_display_name,
            status="Failed",
            error=str(exc),
        )


def run_experiment_matrix(
    experiments_config_path: Path,
    settings: NeurallsSettings | None = None,
    *,
    force: bool = False,
    project_root: Path | None = None,
    max_epochs: int | None = None,
) -> list[ExperimentResult]:
    """Run training for all experiments defined in one case config.

    This is the main orchestrator for running multiple experiments in sequence.
    Each experiment consists of:
    1. Dataset generation (with caching based on data config + src hash)
    2. Model training (with checkpoint reuse unless force=True)

    The function processes experiments sequentially to avoid resource contention.
    Failed experiments don't stop the batch - each returns a result with status.

    Args:
        experiments_config_path: Path to a case config defining all experiments
        force: If True, retrain all models even if checkpoints exist
        project_root: Root directory for resolving src/ (auto-detected if None)

    Returns:
        List of ExperimentResult objects, one per experiment (success or failure)

    Note:
        For solver comparison after training, use comparison workflows:
        >>> from neuralls.composition.experiments.comparison_batch import run_comparison_batch
        >>> run_batch_comparison(experiments_config_path, comparison_config_path, params)

    Example:
        >>> results = run_experiment_matrix(
        ...     Path("/tmp/case.toml"),
        ...     force=False,
        ... )
        >>> success_count = sum(1 for r in results if r.status == "Success")
        >>> print(f"{success_count}/{len(results)} experiments succeeded")
    """
    settings = require_settings(settings, case_config_path=experiments_config_path)
    # Auto-detect project root if not provided (for src hash calculation)
    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]

    # Load all experiment definitions from the case config
    batch = load_batch(experiments_config_path, settings)
    experiments = batch.experiments

    # Compute source code hash for cache invalidation
    # When src/ changes, datasets are regenerated to ensure consistency
    src_dir = project_root / "src"
    if not src_dir.exists():
        src_dir = Path(__file__).resolve().parent.parent
    src_hash = compute_directory_hash(src_dir)

    logger.info(f"Training {len(experiments)} experiments from {experiments_config_path}")

    # Open one case-level MLflow parent run so every training/search run below
    # (including dlkit's own nested Optuna trial runs) nests under it, and so
    # the experiment gets its artifact_location pinned before dlkit auto-creates it.
    cfg, _ = load_validated_case_config(experiments_config_path, settings)
    training_mlflow_env = build_workflow_environment(
        tracking_uri=cfg.mlflow.tracking_uri,
        artifact_location=cfg.mlflow.artifacts_destination,
        default_output_root=batch.output_root,
    )
    mlflow_experiment_name = cfg.names.training

    results: list[ExperimentResult] = []
    with scoped_mlflow_environment(training_mlflow_env.env):
        session_run_name, session_tags = build_session_run_spec(
            case_config_path=experiments_config_path.resolve(),
            experiment_name=mlflow_experiment_name,
            phase="session_training",
        )
        parent_run_id = create_session_parent_run(
            tracking_uri=training_mlflow_env.tracking_uri,
            artifact_uri=training_mlflow_env.artifact_uri,
            run_name=session_run_name,
            tags=session_tags.as_mlflow_tags(),
            experiment_name=mlflow_experiment_name,
        )
        try:
            # Run each experiment sequentially
            # Sequential execution avoids GPU/memory contention and makes logs clearer
            for exp in experiments:
                # Log experiment details for progress tracking
                logger.info(f"\n{'=' * 60}")
                logger.info(f"Experiment: {exp.spec.experiment_display_name}")
                logger.info(f"  Job: {exp.spec.job_display_name or exp.spec.job_config_path.stem}")
                logger.info(
                    f"  Dataset: {exp.spec.dataset_display_name or exp.workspace.dataset_id}"
                )
                logger.info(f"{'=' * 60}")

                # Run single experiment (catches exceptions internally)
                result = run_experiment(
                    settings=settings,
                    job_config_path=exp.spec.job_config_path,
                    data_config_path=exp.spec.data_config_path,
                    output_root=batch.output_root,
                    force=force,
                    src_hash=src_hash,
                    max_epochs=max_epochs,
                    experiment_id=exp.spec.experiment_id,
                    experiment_display_name=exp.spec.experiment_display_name,
                    dataset_registry_id=exp.spec.dataset_registry_id,
                    dataset_display_name=exp.spec.dataset_display_name,
                    job_registry_id=exp.spec.job_registry_id,
                    job_display_name=exp.spec.job_display_name,
                    parent_run_id=parent_run_id,
                    mlflow_experiment_name=mlflow_experiment_name,
                )
                results.append(result)
        finally:
            session_status = "FAILED" if any(not r.is_success for r in results) else "FINISHED"
            finalize_session_parent_run(
                tracking_uri=training_mlflow_env.tracking_uri,
                run_id=parent_run_id,
                status=session_status,
            )

    return results
