"""Experiment orchestration for training neural network models.

This module provides the main orchestration functions for running experiments:
- `run_experiment()`: Run single experiment (data generation + training)
- `run_experiment_matrix()`: Run multiple experiments from experiments.toml

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

from neuralls.configuration.loader import load_batch
from neuralls.workflows.reporting import ExperimentResult
from neuralls.workflows.utils.hashing import compute_directory_hash
from neuralls.workflows.data import load_data_config
from neuralls.generation import process_config
from neuralls.workflows.utils.paths import extract_model_name
from neuralls.workflows.training import train_model
from neuralls.validation import validate_data_exists
from neuralls.io.checkpoints import get_latest_checkpoint
from neuralls.constants import (
    DATASET_MANIFEST_FILENAME,
    MATRIX_COO_DIRNAME,
    RHS_ARRAY_FILENAME,
    SOLUTIONS_ARRAY_FILENAME,
)


def run_experiment(
    *,
    model_config_path: Path,
    data_config_path: Path,
    output_root: Path,
    force: bool,
    src_hash: str,
    max_epochs: int | None = None,
) -> ExperimentResult:
    """Run data generation and model training for a single experiment.

    This function orchestrates a complete experiment workflow:
    1. Load data configuration from TOML file
    2. Generate/cache dataset artifacts (manifest + .npy + sparse pack)
    3. Check for existing checkpoint (skip training if found and force=False)
    4. Train model if needed (creates checkpoint)
    5. Return success/failure result

    Args:
        model_config_path: Path to model configuration TOML (e.g., configs/linear.toml)
        data_config_path: Path to data configuration TOML (e.g., data-configs/collect-504.toml)
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
        ...     model_config_path=Path("configs/linear.toml"),
        ...     data_config_path=Path("data-configs/collect-504.toml"),
        ...     output_root=Path("output"),
        ...     force=False,
        ...     src_hash="abc123",
        ... )
        >>> print(result.status)
        'Success'
    """
    # Extract model identifier from config file (e.g., "linear" from "linear.toml")
    model_name = extract_model_name(model_config_path)
    try:
        # Step 1: Load data configuration and generate/cache dataset
        data_cfg = load_data_config(data_config_path)
        data_dir = process_config(data_cfg, config_path=data_config_path)
        validate_data_exists(
            data_dir,
            [
                DATASET_MANIFEST_FILENAME,
                RHS_ARRAY_FILENAME,
                SOLUTIONS_ARRAY_FILENAME,
                MATRIX_COO_DIRNAME,
            ],
        )

        # Step 2: Check for existing checkpoint to avoid redundant training
        # Workspace structure: output_root/{dataset_id}/{model_name}/checkpoints/
        checkpoint_dir = (
            output_root / data_config_path.stem / model_name / "checkpoints"
        )
        checkpoint = get_latest_checkpoint(checkpoint_dir)

        # Step 3: Train model if no checkpoint exists or force=True
        if force or checkpoint is None:
            checkpoint = train_model(
                config_path=model_config_path,
                data_config_path=data_config_path,
                output_root=output_root,
                max_epochs=max_epochs,
            )
            logger.info(f"Training complete: {checkpoint}")
        else:
            logger.info(f"Using existing checkpoint: {checkpoint}")

        # Step 4: Return success result
        return ExperimentResult(
            experiment_id=model_name,
            status="Success",
        )
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        # Catch expected exceptions to ensure batch runs continue even if one experiment fails
        logger.error(f"Experiment {model_name} failed: {exc}")
        return ExperimentResult(
            experiment_id=model_name, status="Failed", error=str(exc)
        )


def run_experiment_matrix(
    experiments_config_path: Path,
    *,
    force: bool = False,
    project_root: Path | None = None,
    max_epochs: int | None = None,
) -> list[ExperimentResult]:
    """Run training for all experiments defined in experiments.toml.

    This is the main orchestrator for running multiple experiments in sequence.
    Each experiment consists of:
    1. Dataset generation (with caching based on data config + src hash)
    2. Model training (with checkpoint reuse unless force=True)

    The function processes experiments sequentially to avoid resource contention.
    Failed experiments don't stop the batch - each returns a result with status.

    Args:
        experiments_config_path: Path to experiments.toml defining all experiments
        force: If True, retrain all models even if checkpoints exist
        project_root: Root directory for resolving src/ (auto-detected if None)

    Returns:
        List of ExperimentResult objects, one per experiment (success or failure)

    Note:
        For solver comparison after training, use comparison workflows:
        >>> from neuralls.workflows.comparison import run_batch_comparison
        >>> run_batch_comparison(experiments_config_path, comparison_config_path, params)

    Example:
        >>> results = run_experiment_matrix(
        ...     Path("configs/experiments.toml"),
        ...     force=False,
        ... )
        >>> success_count = sum(1 for r in results if r.status == "Success")
        >>> print(f"{success_count}/{len(results)} experiments succeeded")
    """
    # Auto-detect project root if not provided (for src hash calculation)
    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]

    # Load all experiment definitions from experiments.toml
    batch = load_batch(experiments_config_path)
    experiments = batch.experiments

    # Compute source code hash for cache invalidation
    # When src/ changes, datasets are regenerated to ensure consistency
    src_dir = project_root / "src"
    if not src_dir.exists():
        src_dir = Path(__file__).resolve().parent.parent
    src_hash = compute_directory_hash(src_dir)

    logger.info(f"Training {len(experiments)} experiments from {experiments_config_path}")

    # Run each experiment sequentially
    # Sequential execution avoids GPU/memory contention and makes logs clearer
    results: list[ExperimentResult] = []
    for exp in experiments:
        # Log experiment details for progress tracking
        logger.info(f"\n{'='*60}")
        logger.info(f"Experiment: {exp.spec.id}")
        logger.info(f"  Model: {exp.spec.model_config_path.stem}")
        logger.info(f"  Dataset: {exp.spec.data_config_path.stem}")
        logger.info(f"{'='*60}")

        # Run single experiment (catches exceptions internally)
        result = run_experiment(
            model_config_path=exp.spec.model_config_path,
            data_config_path=exp.spec.data_config_path,
            output_root=batch.output_root,
            force=force,
            src_hash=src_hash,
            max_epochs=max_epochs,
        )
        results.append(result)

    return results
