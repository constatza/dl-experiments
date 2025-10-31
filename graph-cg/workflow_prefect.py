"""Prefect workflow for orchestrating graph-cg experiments.

This workflow provides complete end-to-end experiment execution:
- Data generation (with filesystem-based caching)
- Model training
- Prediction/inference (parity plots)
- Method comparison (preconditioner analysis)

All steps are integrated into the flow with automatic caching and parallel execution.

Setup (Optional but Recommended):
    Enable result persistence globally for automatic task result caching:

        prefect config set PREFECT_RESULTS_PERSIST_BY_DEFAULT=true

    This allows Prefect to automatically cache task results, avoiding
    redundant computation when inputs haven't changed.

Usage:
    # Run all experiments (data generation, training, prediction, comparison)
    uv run python graph-cg/workflow_prefect.py

    # Or import and run programmatically
    from workflow_prefect import run_experiment_matrix_flow
    run_experiment_matrix_flow()

How Caching Works:
    All tasks use persist_result=True and cache_policy=INPUTS for persistent caching:

    - Data generation:
      * Filesystem check: skips if data exists (source of truth)
      * Prefect INPUTS cache: memoizes across flow runs
      * persist_result=True ensures cache persists across Prefect restarts

    - Training:
      * Experiment names come directly from experiments.toml
      * Prefect INPUTS cache: memoizes based on experiment name + inputs
      * Checkpoint paths live under ``output/<experiment-name>/checkpoints``
      * persist_result=True ensures checkpoint paths persist

    - Prediction & Comparison:
      * Prefect INPUTS cache: memoizes based on checkpoint + config
      * persist_result=True ensures results persist across runs
      * Automatic re-execution only when upstream tasks change

    - Benefits:
        * No manual cache management needed
        * No stale checkpoint concerns (directories owned by flow)
        * Automatic cache invalidation on config/data changes
        * Clear traceability (outputs keyed by experiment name)
        * Complete pipeline runs once, then cached

Note:
    Uses Prefect 3.x API where .submit() is available for tasks, not flows.
    Experiments are implemented as tasks to enable parallel execution.
"""

from __future__ import annotations

import hashlib
import os
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("DASK_DISTRIBUTED__LOGGING__DISTRIBUTED", "error")
from pathlib import Path
from typing import Any
import tomllib

from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact, create_table_artifact
from prefect.cache_policies import INPUTS
from prefect.futures import PrefectFuture
from prefect_dask import DaskTaskRunner

from src.cli.data import load_data_config
from src.cli.training import train_model
from src.cli.prediction import run_inference
from src.cli.comparison import compare_preconditioners
from src.common import get_latest_checkpoint
from src.experiment_manifest import load_manifest, update_manifest
from src.config_utils import resolve_data_dir, resolve_training_paths
from src.prefect_utils import (
    compute_directory_hash,
    compute_data_files_hash,
    compute_experiment_output_hash,
)
from src.validation import validate_data_exists
from src.unified_data_processing import process_config


def build_signature(**parts: str | None) -> str:
    """Create stable signature string from keyword parts."""
    tokens = []
    for key in sorted(parts):
        value = parts[key] or "none"
        tokens.append(f"{key}:{value}")
    return "|".join(tokens)


def to_relative_path(path: Path, base: Path) -> str:
    """Return path relative to base when possible, otherwise absolute."""
    try:
        return str(Path(path).resolve().relative_to(base.resolve()))
    except Exception:
        return str(path)


def compute_file_signature(path: Path | str) -> str:
    """Compute a stable SHA-1 signature for a file's contents."""

    file_path = Path(path)
    if not file_path.exists():
        return "missing"

    hasher = hashlib.sha1()
    with open(file_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def extract_model_name(model_config_path: Path | str) -> str:
    """Extract model name from config SESSION.name.

    Args:
        model_config_path: Path to model config file

    Returns:
        Model name from SESSION.name, or config filename if not set
    """
    model_config_path = Path(model_config_path)

    with open(model_config_path, "rb") as f:
        config = tomllib.load(f)

    session = config.get("SESSION") or {}
    name = session.get("name")

    # If SESSION.name not set, derive from config filename
    if not isinstance(name, str) or not name:
        name = model_config_path.stem

    return name


def resolve_output_root(paths_cfg: dict[str, str] | None) -> Path:
    """Resolve base output root from experiments config or environment."""

    if paths_cfg:
        configured = paths_cfg.get("output_root") or paths_cfg.get("output_dir")
        if configured:
            return Path(configured).expanduser().resolve()

    env_override = os.getenv("GRAPH_CG_OUTPUT_DIR")
    if env_override:
        return Path(env_override).expanduser().resolve()

    return Path(__file__).resolve().parent / "output"


def prepare_experiment_outputs(
    model_name: str, data_config_name: str, output_root: Path
) -> dict[str, Path]:
    """Ensure experiment-specific output directories exist.

    Structure:
        output/
          {data_config_name}/            # data-config filename stem (experiment level)
            {model_name}/                # SESSION.name from model config (run level)
              checkpoints/
                checkpoint.ckpt
              figures/
              metrics/
              predictions/

    Args:
        model_name: Model identifier from SESSION.name
        data_config_name: Data config filename stem (e.g., "generate-90")
        output_root: Base output directory from experiments.toml

    Returns:
        Dictionary with paths to output directories
    """
    output_root.mkdir(parents=True, exist_ok=True)

    # Create data/model hierarchy
    experiment_dir = output_root / data_config_name / model_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Create output type subdirectories
    checkpoints_dir = experiment_dir / "checkpoints"
    figures_dir = experiment_dir / "figures"
    metrics_dir = experiment_dir / "metrics"
    predictions_dir = experiment_dir / "predictions"

    for directory in (checkpoints_dir, figures_dir, metrics_dir, predictions_dir):
        directory.mkdir(parents=True, exist_ok=True)

    return {
        "experiment_dir": experiment_dir,
        "checkpoints_dir": checkpoints_dir,
        "figures_dir": figures_dir,
        "metrics_dir": metrics_dir,
        "predictions_dir": predictions_dir,
    }


@task(
    persist_result=True,
    cache_policy=INPUTS,
    task_run_name="get-or-generate-{data_config_path}",
    retries=1,
    retry_delay_seconds=5,
)
def get_or_generate_data_task(
    data_config_path: str,
    data_config_version: str = "",
    src_hash: str = "",
    force: bool = False,
) -> dict[str, Any]:
    """Get existing data or generate if missing (idempotent).

    This task implements a two-layer idempotence strategy:
    1. Filesystem check: Skip generation if data exists on disk (source of truth)
    2. Prefect INPUTS cache: Memoize result across runs (in-memory/DB)

    The filesystem check is the primary source of truth, ensuring correctness
    even when Prefect cache is cleared (e.g., server restart, cache expiry).

    This ensures:
    - No redundant generation if data exists
    - Multiple experiments share the same data config
    - Only missing data is regenerated (not all data)
    - Automatic retry on transient failures (network, disk I/O)

    Args:
        data_config_path: Path to data generation config (e.g.,
            "data-configs/generate-90.toml").
        data_config_version: Hash of config file content for cache invalidation
        src_hash: Hash of source code for cache invalidation
        force: Force regeneration even if data exists

    Returns:
        dict: Dictionary with keys:
            - data_dir: Path to data directory
            - data_hash: Content hash of data files for downstream cache invalidation

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If data generation fails after retries.

    Example:
        >>> result = get_or_generate_data_task("data-configs/generate-90.toml")
        >>> print(result["data_dir"])
        /data/projects/graph-cg/data/processed/generate-90-norm
        >>> print(result["data_hash"][:12])
        a3f5c8d9e2b1
    """
    config_path = Path(data_config_path)
    _ = data_config_version  # Ensures Prefect cache keys include file signature
    _ = src_hash  # Ensures cache keys include source code changes

    # Resolve expected output directory (pure function - no I/O)
    expected_output = resolve_data_dir(config_path)

    # Load config and process using unified interface
    # Note: process_config handles its own filesystem checks for idempotence
    cfg = load_data_config(config_path)
    output_path = process_config(cfg, config_path=config_path)

    print(f"\nData ready at: {output_path}")

    # Compute content-based hash of data files
    data_hash = compute_data_files_hash(output_path)
    print(f"Data content hash: {data_hash[:12]}...")

    # Create artifact for data status
    create_markdown_artifact(
        key=f"data-status-{config_path.stem}".lower(),
        markdown=(
            f"# Data Ready\n\n"
            f"**Config**: `{config_path.name}`\n\n"
            f"**Location**: `{output_path}`\n\n"
            f"**Data Hash**: `{data_hash[:12]}...`\n\n"
            f"**Status**: ✓ Data ready"
        ),
        description=f"Data status for {config_path.stem}",
    )

    return {"data_dir": output_path, "data_hash": data_hash}


@task(
    persist_result=True,
    cache_policy=INPUTS,
)
def train_model_task(
    model_template_path: str,
    data_dir: Path,
    data_config_path: str | Path,
    data_hash: str,
    output_root: Path | None = None,
    checkpoint_state: str = "",
    model_config_version: str = "",
    data_config_version: str = "",
    src_hash: str = "",
    force: bool = False,
) -> Path:
    """Train model with resolved config paths.

    Experiment outputs are organized as: {output_root}/{data_config_name}/{model_name}/
    where data_config_name is from the data config filename (experiment level)
    and model_name is from SESSION.name or config filename (run level).

    Args:
        model_template_path: Path to model config template (e.g., "configs/ffnn.toml")
        data_dir: Directory containing the training data
        data_config_path: Path to the data config (used to extract data_config_name)
        data_hash: Content hash of data files for cache invalidation
        output_root: Optional override for output root
        checkpoint_state: Filesystem state token for checkpoint directory
        model_config_version: Hash of model config file content
        data_config_version: Hash of data config file content
        src_hash: Hash of source code
        force: Force re-training even if checkpoint exists

    Returns:
        Path: Path to saved model checkpoint

    Raises:
        FileNotFoundError: If data files don't exist
        RuntimeError: If training fails after retries

    Example:
        >>> checkpoint = train_model_task(
        ...     "configs/ffnn.toml",
        ...     Path("/data/projects/graph-cg/data/processed/generate-90-norm"),
        ...     "data-configs/generate-90.toml",
        ...     "a3f5c8d9e2b1...",
        ... )
        >>> print(checkpoint)
        /data/projects/graph-cg/data/output/FFNN-NormScaled-504/generate-90/checkpoints/checkpoint.ckpt
    """
    # Extract identifiers from configs
    logger = get_run_logger()
    model_name = extract_model_name(model_template_path)
    data_config_name = Path(data_config_path).stem

    logger.info("Training %s on %s", model_name, data_config_name)
    logger.info("Model template: %s", model_template_path)
    logger.info("Data directory: %s", data_dir)
    if model_config_version:
        logger.debug("Model config signature: %s", model_config_version)
    if data_config_version:
        logger.debug("Data config signature: %s", data_config_version)
    if src_hash:
        logger.debug("Source code signature: %s", src_hash)
    if data_hash:
        logger.debug("Data content hash: %s", data_hash[:12])
    if checkpoint_state:
        logger.debug("Checkpoint state token: %s", checkpoint_state)
    _ = src_hash  # Ensures cache keys include source code changes
    _ = data_hash  # Ensures cache keys include data content hash
    _ = checkpoint_state  # Part of cache key via Prefect INPUTS policy
    stage_signature = build_signature(
        data=data_hash,
        data_config=data_config_version,
        model=model_config_version,
        src=src_hash,
    )
    manifest_metadata = {
        "signature": stage_signature,
        "model_config_version": model_config_version,
        "data_config_version": data_config_version,
        "data_hash": data_hash,
        "src_hash": src_hash,
    }

    # Prepare experiment-specific output directories
    root_dir = output_root or resolve_output_root(None)
    output_dirs = prepare_experiment_outputs(model_name, data_config_name, root_dir)
    logger.info("Experiment output root: %s", output_dirs["experiment_dir"])
    manifest = load_manifest(output_dirs["experiment_dir"])
    if not force:
        training_entry = manifest.get("training") or {}
        stored_signature = training_entry.get("signature")
        if stored_signature == stage_signature:
            checkpoint_rel = training_entry.get("checkpoint_path")
            cached_checkpoint = None
            if checkpoint_rel:
                candidate = output_dirs["experiment_dir"] / checkpoint_rel
                if candidate.exists():
                    cached_checkpoint = candidate
            if cached_checkpoint is None:
                cached_checkpoint = get_latest_checkpoint(output_dirs["checkpoints_dir"])
            if cached_checkpoint and cached_checkpoint.exists():
                logger.info(
                    "Checkpoint already up to date for %s/%s — skipping training",
                    data_config_name,
                    model_name,
                )
                return cached_checkpoint

    # Validate data exists
    validate_data_exists(data_dir, ["rhs-samples.npy", "sol-samples.npy"])

    # Resolve training paths
    paths = resolve_training_paths(data_dir)
    logger.info("Features file: %s", paths["features_path"])
    logger.info("Targets file: %s", paths["targets_path"])

    # Train with path overrides
    # output_dir points to {output_root}/{data_config_name}/{model_name}/
    # Lightning will create checkpoints at {output_dir}/checkpoints/
    logger.info("Starting training...")
    print(f"\n{'=' * 60}")
    print(f"Training: {data_config_name}/{model_name}")
    print(f"{'=' * 60}")

    checkpoint_path = train_model(
        config_path=model_template_path,
        data_config_path=data_config_path,
        features_path=paths["features_path"],
        targets_path=paths["targets_path"],
        output_dir=output_dirs["experiment_dir"],
        session_name=model_name,
        manifest_metadata=manifest_metadata,
    )

    if output_dirs["checkpoints_dir"] not in checkpoint_path.parents:
        raise RuntimeError(
            "Checkpoint written outside checkpoints directory: "
            f"{checkpoint_path} not in {output_dirs['checkpoints_dir']}"
        )

    logger.info("Checkpoint saved to %s", checkpoint_path)

    # Create artifact for training completion
    experiment_id = f"{data_config_name}/{model_name}"
    create_markdown_artifact(
        key=f"training-{data_config_name}-{model_name}".lower(),
        markdown=(
            f"# Training Complete\n\n"
            f"**Experiment**: `{experiment_id}`\n\n"
            f"**Model**: `{model_name}`\n\n"
            f"**Data Config**: `{data_config_name}`\n\n"
            f"**Checkpoint**: `{checkpoint_path}`\n\n"
            f"**Status**: ✓ Training complete"
        ),
        description=f"Training results for {experiment_id}",
    )

    return checkpoint_path


@task(
    persist_result=True,
    cache_policy=INPUTS,
)
def predict_task(
    model_template_path: str,
    data_dir: Path,
    data_config_path: str | Path,
    checkpoint_path: Path,
    data_hash: str,
    output_root: Path | None = None,
    prediction_state: str = "",
    model_config_version: str = "",
    data_config_version: str = "",
    src_hash: str = "",
    force: bool = False,
) -> dict[str, Any]:
    """Run prediction/inference on trained model.

    Args:
        model_template_path: Path to model config template
        data_dir: Directory containing the data
        data_config_path: Path to the data config
        checkpoint_path: Path to trained model checkpoint
        data_hash: Content hash of data files for cache invalidation
        output_root: Optional override for output root
        prediction_state: Filesystem state token for prediction artifacts
        model_config_version: Hash of model config file content
        data_config_version: Hash of data config file content
        src_hash: Hash of source code
        force: Force re-prediction even if outputs exist

    Returns:
        dict: Prediction results including plot paths

    Example:
        >>> results = predict_task(
        ...     "configs/ffnn.toml",
        ...     Path("/data/projects/graph-cg/data/processed/generate-90-norm"),
        ...     "data-configs/generate-90.toml",
        ...     Path("/data/projects/graph-cg/output/FFNN/generate-90/checkpoints/checkpoint.ckpt"),
        ...     "a3f5c8d9e2b1...",
        ... )
    """
    logger = get_run_logger()
    model_name = extract_model_name(model_template_path)
    data_config_name = Path(data_config_path).stem

    stage_signature = build_signature(
        checkpoint=str(checkpoint_path),
        data=data_hash,
        data_config=data_config_version,
        model=model_config_version,
        prediction_state=prediction_state,
        src=src_hash,
    )
    manifest_metadata = {
        "signature": stage_signature,
        "model_config_version": model_config_version,
        "data_config_version": data_config_version,
        "data_hash": data_hash,
        "src_hash": src_hash,
    }

    output_root_path = (
        Path(output_root)
        if output_root is not None
        else resolve_output_root(None)
    )
    output_dirs = prepare_experiment_outputs(
        model_name, data_config_name, output_root_path
    )
    manifest = load_manifest(output_dirs["experiment_dir"])
    if not force:
        inference_entry = manifest.get("inference") or {}
        stored_signature = inference_entry.get("signature")
        if stored_signature == stage_signature:
            plot_rel = inference_entry.get("plot_path")
            plot_path = None
            if plot_rel:
                candidate = output_dirs["experiment_dir"] / plot_rel
                if candidate.exists():
                    plot_path = candidate
            logger.info(
                "Prediction already up to date for %s/%s — skipping inference",
                data_config_name,
                model_name,
            )
            return {
                "predictions": None,
                "y_true": None,
                "y_pred": None,
                "duration_seconds": 0.0,
                "plot_path": plot_path,
                "skipped": True,
            }

    logger.info(
        "Running prediction for %s on %s using checkpoint %s",
        model_name,
        data_config_name,
        checkpoint_path,
    )
    if model_config_version:
        logger.debug("Model config signature: %s", model_config_version)
    if data_config_version:
        logger.debug("Data config signature: %s", data_config_version)
    if src_hash:
        logger.debug("Source code signature: %s", src_hash)
    if data_hash:
        logger.debug("Data content hash: %s", data_hash[:12])
    if prediction_state:
        logger.debug("Prediction state token: %s", prediction_state)
    _ = src_hash  # Ensures cache keys include source code changes
    _ = data_hash  # Ensures cache keys include data content hash
    _ = prediction_state  # Part of cache key via Prefect INPUTS policy

    # Validate checkpoint exists
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "This may indicate the training task returned a cached path "
            "to a deleted checkpoint."
        )

    # Resolve paths for prediction
    features_path = data_dir / "rhs-samples.npy"
    targets_path = data_dir / "sol-samples.npy"

    # Run inference
    logger.info("Running prediction...")
    print(f"\n{'=' * 60}")
    print(f"Predicting: {data_config_name}/{model_name}")
    print(f"{'=' * 60}")
    results = run_inference(
        config_path=model_template_path,
        data_config_path=data_config_path,
        checkpoint_path=checkpoint_path,
        features_path=features_path,
        targets_path=targets_path,
        save_plots=True,
        figures_dir=output_dirs["figures_dir"],
    )

    plot_path = results.get("plot_path")
    if plot_path:
        logger.info("Prediction plots saved to %s", plot_path)
    else:
        logger.warning("Prediction returned no plot path")

    checkpoint_rel = to_relative_path(checkpoint_path, output_dirs["experiment_dir"])
    manifest_payload = {
        "checkpoint_path": checkpoint_rel,
        **manifest_metadata,
    }
    if plot_path:
        manifest_payload["plot_path"] = to_relative_path(
            Path(plot_path), output_dirs["experiment_dir"]
        )
    update_manifest(output_dirs["experiment_dir"], "inference", manifest_payload)

    # Create artifact for prediction completion
    experiment_id = f"{data_config_name}/{model_name}"
    create_markdown_artifact(
        key=f"prediction-{data_config_name}-{model_name}".lower(),
        markdown=(
            f"# Prediction Complete\n\n"
            f"**Experiment**: `{experiment_id}`\n\n"
            f"**Checkpoint**: `{checkpoint_path}`\n\n"
            f"**Plot**: `{results.get('plot_path', 'N/A')}`\n\n"
            f"**Duration**: {results.get('duration_seconds', 0):.2f}s\n\n"
            f"**Status**: ✓ Prediction complete"
        ),
        description=f"Prediction results for {experiment_id}",
    )

    logger.info("Prediction completed for %s/%s", model_name, data_config_name)

    return results


@task(
    persist_result=True,
    cache_policy=INPUTS,
)
def compare_methods_task(
    model_template_path: str,
    data_dir: Path,
    data_config_path: str | Path,
    checkpoint_path: Path,
    data_hash: str,
    output_root: Path | None = None,
    comparison_state: str = "",
    model_config_version: str = "",
    data_config_version: str = "",
    src_hash: str = "",
    force: bool = False,
) -> dict[str, Any]:
    """Compare different preconditioner methods.

    Args:
        model_template_path: Path to model config template
        data_dir: Directory containing the data
        data_config_path: Path to the data config
        checkpoint_path: Path to trained model checkpoint
        data_hash: Content hash of data files for cache invalidation
        output_root: Optional override for output root
        comparison_state: Filesystem state token for comparison artifacts
        model_config_version: Hash of model config file content
        data_config_version: Hash of data config file content
        src_hash: Hash of source code
        force: Force re-comparison even if outputs exist

    Returns:
        dict: Comparison results including plot paths

    Example:
        >>> results = compare_methods_task(
        ...     "configs/ffnn.toml",
        ...     Path("/data/projects/graph-cg/data/processed/generate-90-norm"),
        ...     "data-configs/generate-90.toml",
        ...     Path("/data/projects/graph-cg/output/FFNN/generate-90/checkpoints/checkpoint.ckpt"),
        ...     "a3f5c8d9e2b1...",
        ... )
    """
    logger = get_run_logger()
    model_name = extract_model_name(model_template_path)
    data_config_name = Path(data_config_path).stem

    stage_signature = build_signature(
        checkpoint=str(checkpoint_path),
        comparison_state=comparison_state,
        data=data_hash,
        data_config=data_config_version,
        model=model_config_version,
        src=src_hash,
    )
    manifest_metadata = {
        "signature": stage_signature,
        "model_config_version": model_config_version,
        "data_config_version": data_config_version,
        "data_hash": data_hash,
        "src_hash": src_hash,
    }

    output_root_path = (
        Path(output_root)
        if output_root is not None
        else resolve_output_root(None)
    )
    output_dirs = prepare_experiment_outputs(
        model_name, data_config_name, output_root_path
    )
    manifest = load_manifest(output_dirs["experiment_dir"])
    if not force:
        comparison_entry = manifest.get("comparison") or {}
        stored_signature = comparison_entry.get("signature")
        if stored_signature == stage_signature:
            cached_plots: dict[str, Path] = {}
            for key, rel_path in (comparison_entry.get("plot_paths") or {}).items():
                candidate = output_dirs["experiment_dir"] / rel_path
                if candidate.exists():
                    cached_plots[key] = candidate
            logger.info(
                "Comparison already up to date for %s/%s — skipping comparison",
                data_config_name,
                model_name,
            )
            return {
                "results": None,
                "summary": comparison_entry.get("summary"),
                "plot_paths": cached_plots,
                "preconditioners": comparison_entry.get("preconditioners", []),
                "warm_starts": comparison_entry.get("warm_starts", []),
                "solver_params": comparison_entry.get("solver_params", {}),
                "skipped": True,
            }

    logger.info(
        "Comparing preconditioners for %s on %s",
        model_name,
        data_config_name,
    )
    logger.info("Using checkpoint %s", checkpoint_path)
    if model_config_version:
        logger.debug("Model config signature: %s", model_config_version)
    if data_config_version:
        logger.debug("Data config signature: %s", data_config_version)
    if src_hash:
        logger.debug("Source code signature: %s", src_hash)
    if data_hash:
        logger.debug("Data content hash: %s", data_hash[:12])
    if comparison_state:
        logger.debug("Comparison state token: %s", comparison_state)
    _ = src_hash  # Ensures cache keys include source code changes
    _ = data_hash  # Ensures cache keys include data content hash
    _ = comparison_state  # Part of cache key via Prefect INPUTS policy

    # Validate checkpoint exists
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "This may indicate the training task returned a cached path "
            "to a deleted checkpoint."
        )

    # Resolve paths for comparison
    # Use single matrix and extract test RHS sample
    import numpy as np
    from src.constants import DEFAULT_TEST_SAMPLE_INDEX

    matrix_path = data_dir / "matrix.npy"
    rhs_samples_path = data_dir / "rhs-samples.npy"

    if not matrix_path.exists():
        raise FileNotFoundError(f"Matrix file not found: {matrix_path}")
    if not rhs_samples_path.exists():
        raise FileNotFoundError(f"RHS samples file not found: {rhs_samples_path}")

    # Load matrix and extract test RHS sample
    matrix = np.load(matrix_path)
    rhs_samples = np.load(rhs_samples_path)

    test_sample_idx = DEFAULT_TEST_SAMPLE_INDEX
    if test_sample_idx >= len(rhs_samples):
        raise ValueError(
            f"Test sample index {test_sample_idx} exceeds dataset size {len(rhs_samples)}"
        )

    # Save test RHS sample for comparison
    rhs_path = data_dir / "rhs.npy"
    np.save(rhs_path, rhs_samples[test_sample_idx])

    logger.info(
        "Using sample %d for comparison: matrix shape=%s, rhs shape=%s",
        test_sample_idx,
        matrix.shape,
        rhs_samples[test_sample_idx].shape,
    )

    # Run comparison
    logger.info("Running comparison...")
    print(f"\n{'=' * 60}")
    print(f"Comparing: {data_config_name}/{model_name}")
    print(f"{'=' * 60}")
    results = compare_preconditioners(
        config_path=model_template_path,
        data_config_path=data_config_path,
        matrix_path=matrix_path,
        rhs_path=rhs_path,
        checkpoint_path=checkpoint_path,
        save_plots=True,
        output_dir=output_dirs["experiment_dir"],
        figures_dir=output_dirs["figures_dir"],
    )

    comparison_results = results or {}

    plot_paths = comparison_results.get("plot_paths", {})
    if plot_paths:
        logger.info("Comparison plots saved: %s", plot_paths)
    else:
        logger.warning("Comparison returned no plot paths")

    checkpoint_rel = to_relative_path(checkpoint_path, output_dirs["experiment_dir"])
    manifest_payload = {
        "checkpoint_path": checkpoint_rel,
        **manifest_metadata,
        "plot_paths": {
            key: to_relative_path(Path(path), output_dirs["experiment_dir"])
            for key, path in plot_paths.items()
        },
        "preconditioners": comparison_results.get("preconditioners", []),
        "warm_starts": comparison_results.get("warm_starts", []),
        "solver_params": comparison_results.get("solver_params", {}),
        "summary": comparison_results.get("summary"),
    }
    update_manifest(output_dirs["experiment_dir"], "comparison", manifest_payload)

    # Create artifact for comparison completion
    experiment_id = f"{data_config_name}/{model_name}"
    create_markdown_artifact(
        key=f"comparison-{data_config_name}-{model_name}".lower(),
        markdown=(
            f"# Method Comparison Complete\n\n"
            f"**Experiment**: `{experiment_id}`\n\n"
            f"**Preconditioners**: {', '.join(results.get('preconditioners', []))}\n\n"
            f"**Warm starts**: {', '.join(results.get('warm_starts', []))}\n\n"
            f"**Plots**: {', '.join(str(p) for p in results.get('plot_paths', {}).values())}\n\n"
            f"**Status**: ✓ Comparison complete"
        ),
        description=f"Comparison results for {experiment_id}",
    )

    logger.info(
        "Comparison completed for %s/%s",
        model_name,
        data_config_name,
    )

    return results


@task(
    task_run_name="run-experiment",
)
def run_experiment_task(
    model_template_path: str,
    data_config_path: str,
    output_root: Path,
    src_hash: str = "",
    force: bool = False,
) -> dict[str, Any]:
    """Run a complete experiment: data generation + training + prediction + comparison.

    Experiment identity is derived from:
    - Model name: SESSION.name from model config
    - Data name: filename stem from data config

    Args:
        model_template_path: Path to model config template
        data_config_path: Path to data generation config
        output_root: Base output directory
        src_hash: Hash of source code for cache invalidation
        force: Force re-run of all tasks

    Returns:
        dict: Dictionary containing:
            - checkpoint_path: Path to saved model checkpoint
            - prediction_results: Results from inference
            - comparison_results: Results from method comparison

    Example:
        >>> results = run_experiment_task(
        ...     "configs/ffnn.toml",
        ...     "data-configs/generate-90.toml",
        ...     Path("/data/projects/graph-cg/data/output"),
        ... )
        >>> print(results['checkpoint_path'])
        /data/projects/graph-cg/data/output/FFNN-NormScaled-504/generate-90/checkpoints/checkpoint.ckpt
    """
    model_config_version = compute_file_signature(model_template_path)
    data_config_version = compute_file_signature(data_config_path)

    # Stage-specific cache tokens derived from current filesystem state
    output_root_path = Path(output_root)
    model_name = extract_model_name(model_template_path)
    data_config_name = Path(data_config_path).stem
    experiment_dir = output_root_path / data_config_name / model_name
    checkpoints_dir = experiment_dir / "checkpoints"
    figures_dir = experiment_dir / "figures"
    metrics_dir = experiment_dir / "metrics"
    predictions_dir = experiment_dir / "predictions"

    checkpoint_state = compute_experiment_output_hash(checkpoints_dir)

    # Step 1: Get or generate data (cached based on config content + src hash)
    data_result = get_or_generate_data_task(
        data_config_path,
        data_config_version=data_config_version,
        src_hash=src_hash,
        force=force,
    )

    # Extract data directory and content hash
    data_dir = data_result["data_dir"]
    data_hash = data_result["data_hash"]

    # Step 2: Train model on generated data
    checkpoint_path = train_model_task(
        model_template_path,
        data_dir,
        data_config_path,
        data_hash,
        output_root_path,
        checkpoint_state=checkpoint_state,
        model_config_version=model_config_version,
        data_config_version=data_config_version,
        src_hash=src_hash,
        force=force,
    )

    # Step 3: Run prediction/inference
    prediction_state = "|".join(
        [
            f"figures:{compute_experiment_output_hash(figures_dir)}",
            f"predictions:{compute_experiment_output_hash(predictions_dir)}",
        ]
    )
    prediction_results = predict_task(
        model_template_path,
        data_dir,
        data_config_path,
        checkpoint_path,
        data_hash,
        output_root_path,
        prediction_state=prediction_state,
        model_config_version=model_config_version,
        data_config_version=data_config_version,
        src_hash=src_hash,
        force=force,
    )

    # Step 4: Compare preconditioner methods
    comparison_state = "|".join(
        [
            f"figures:{compute_experiment_output_hash(figures_dir)}",
            f"metrics:{compute_experiment_output_hash(metrics_dir)}",
            f"predictions:{compute_experiment_output_hash(predictions_dir)}",
        ]
    )
    comparison_results = compare_methods_task(
        model_template_path,
        data_dir,
        data_config_path,
        checkpoint_path,
        data_hash,
        output_root_path,
        comparison_state=comparison_state,
        model_config_version=model_config_version,
        data_config_version=data_config_version,
        src_hash=src_hash,
        force=force,
    )

    return {
        "checkpoint_path": checkpoint_path,
        "prediction_results": prediction_results,
        "comparison_results": comparison_results,
    }


@flow(
    name="run_experiment_matrix",
    task_runner=DaskTaskRunner(
        cluster_kwargs={
            "n_workers": 4,
            "threads_per_worker": 2,
            "silence_logs": "ERROR",
        }
    ),
)
def run_experiment_matrix_flow(
    experiments_config_path: str | None = None,
    force: bool = False,
) -> dict[str, dict[str, Any]]:
    """Run all experiments defined in experiments.toml in parallel.

    This flow orchestrates the entire experiment matrix:
    1. Reads experiment definitions from experiments.toml
    2. Submits all experiments in parallel (non-blocking with .submit())
    3. Generates all unique datasets in parallel (with caching)
    4. Trains all models in parallel (limited by DaskTaskRunner workers)
    5. Runs predictions for each trained model
    6. Compares preconditioner methods for each experiment

    Prefect handles:
    - Automatic caching (data generation runs once per unique config)
    - True parallel execution via .submit() and futures
    - Controlled concurrency (max 4 experiments in parallel to prevent resource exhaustion)
    - Incremental computation (only reruns changed configs)

    Args:
        experiments_config_path: Path to experiments definition file.
            If None, uses GRAPH_CG_EXPERIMENTS_CONFIG env var or
            defaults to "graph-cg/experiments.toml".

    Returns:
        dict: Mapping of experiment names to experiment results containing:
            - checkpoint_path: Path to trained model checkpoint
            - prediction_results: Results from inference
            - comparison_results: Results from method comparison

    Example:
        >>> results = run_experiment_matrix_flow()
        >>> for name, exp_results in results.items():
        ...     print(f"{name}:")
        ...     print(f"  Checkpoint: {exp_results['checkpoint_path']}")
        ...     print(f"  Prediction: {exp_results['prediction_results']['plot_path']}")

    Note:
        Experiments are submitted as tasks (not subflows) to enable
        parallel execution via .submit() in Prefect 3.x.
    """
    # Resolve config path (env var or default)
    if experiments_config_path is None:
        experiments_config_path = os.getenv(
            "GRAPH_CG_EXPERIMENTS_CONFIG",
            "graph-cg/experiments.toml",
        )
    experiments_config_path = Path(experiments_config_path)

    # Validate config exists
    if not experiments_config_path.exists():
        raise FileNotFoundError(
            f"Experiments config not found: {experiments_config_path}\n"
            f"Set GRAPH_CG_EXPERIMENTS_CONFIG env var or pass explicit path."
        )

    print(f"\n{'=' * 80}")
    print(f"Loading experiment matrix: {experiments_config_path}")
    print(f"{'=' * 80}")

    # Load experiments config
    with open(experiments_config_path, "rb") as f:
        config = tomllib.load(f)

    path_settings = config.get("paths") or {}
    output_root = resolve_output_root(path_settings)

    experiments = config.get("experiments", [])
    if not experiments:
        raise ValueError(f"No experiments found in {experiments_config_path}")

    print(f"\nFound {len(experiments)} experiments:")
    for idx, exp in enumerate(experiments, 1):
        model_template = exp.get("model_template", "?")
        data_config = exp.get("data_config", "?")
        print(f"  {idx}. {Path(model_template).stem} + {Path(data_config).stem}")
    print()

    # Resolve paths relative to config directory
    config_dir = experiments_config_path.parent

    # Compute source code hash for cache invalidation
    src_dir = config_dir / "src"
    src_hash = compute_directory_hash(src_dir)
    print(f"Source code hash: {src_hash[:12]}...")

    # Submit all experiments in parallel (non-blocking)
    print("Submitting experiments in parallel...")
    print(f"Output root: {output_root}")
    if force:
        print("Force mode: Ignoring filesystem checks, re-running all tasks")
    futures: dict[str, PrefectFuture] = {}

    for exp in experiments:
        model_template = str(config_dir / exp["model_template"])
        data_config = str(config_dir / exp["data_config"])

        # Build unique ID for tracking: model_name/data_config_name
        experiment_id = f"{Path(model_template).stem}/{Path(data_config).stem}"

        # .submit() returns immediately (non-blocking)
        future = run_experiment_task.submit(
            model_template_path=model_template,
            data_config_path=data_config,
            output_root=output_root,
            src_hash=src_hash,
            force=force,
        )
        futures[experiment_id] = future
        print(f"  ✓ Submitted: {experiment_id}")

    print(f"\nWaiting for {len(futures)} experiments to complete...")
    print("(Limited to 4 parallel workers to prevent resource exhaustion)\n")

    # Wait for all experiments and collect results
    results: dict[str, dict[str, Any]] = {}
    failed_experiments: list[tuple[str, Exception]] = []

    for experiment_id, future in futures.items():
        try:
            experiment_results = future.result()  # Blocks until this experiment completes
            results[experiment_id] = experiment_results
            print(f"  ✓ Completed: {experiment_id}")
        except Exception as e:
            print(f"  ✗ Failed: {experiment_id} - {e}")
            failed_experiments.append((experiment_id, e))
            # Continue with other experiments (partial failure tolerance)

    print(f"\n{'=' * 80}")
    print(
        f"Experiment matrix completed! "
        f"{len(results)}/{len(experiments)} succeeded, "
        f"{len(failed_experiments)} failed"
    )
    print(f"{'=' * 80}")

    if results:
        print("\nSuccessful experiments:")
        for experiment_id, experiment_results in results.items():
            checkpoint = experiment_results.get("checkpoint_path")
            print(f"  ✓ {experiment_id}: {checkpoint}")

    if failed_experiments:
        print("\nFailed experiments:")
        for experiment_id, error in failed_experiments:
            print(f"  ✗ {experiment_id}: {error}")

    # Create summary artifact for Prefect UI
    experiment_data = []
    for exp in experiments:
        model_template = exp["model_template"]
        data_config = exp["data_config"]
        experiment_id = f"{Path(model_template).stem}/{Path(data_config).stem}"

        if experiment_id in results:
            experiment_results = results[experiment_id]
            checkpoint_path = experiment_results.get("checkpoint_path")
            prediction_results = experiment_results.get("prediction_results", {})
            comparison_results = experiment_results.get("comparison_results", {})

            # Format additional info
            info_parts = []
            if prediction_results.get("plot_path"):
                info_parts.append("✓ Prediction")
            if comparison_results.get("plot_paths"):
                info_parts.append("✓ Comparison")
            additional_info = ", ".join(info_parts) if info_parts else "Training only"

            experiment_data.append(
                {
                    "Experiment": experiment_id,
                    "Model": Path(model_template).stem,
                    "Data": Path(data_config).stem,
                    "Status": "✓ Success",
                    "Checkpoint": str(checkpoint_path),
                    "Additional": additional_info,
                }
            )
        else:
            # Find the error for this experiment
            error_msg = next(
                (
                    str(err)
                    for failed_id, err in failed_experiments
                    if failed_id == experiment_id
                ),
                "Unknown error",
            )
            experiment_data.append(
                {
                    "Experiment": experiment_id,
                    "Model": Path(model_template).stem,
                    "Data": Path(data_config).stem,
                    "Status": "✗ Failed",
                    "Checkpoint": error_msg,
                    "Additional": "N/A",
                }
            )

    create_table_artifact(
        key="experiment-results-summary",
        table=experiment_data,
        description=f"Experiment matrix results: {len(results)}/{len(experiments)} succeeded",
    )

    return results


if __name__ == "__main__":
    # Run all experiments
    run_experiment_matrix_flow()
