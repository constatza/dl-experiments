"""Prefect workflow for orchestrating graph-cg experiments.

This workflow provides complete end-to-end experiment execution:
- Data generation (with filesystem-based caching)
- Model training
- Prediction/inference (parity plots)
- Method comparison (preconditioner analysis)

All steps are integrated into the flow with automatic caching and sequential execution
to prevent memory exhaustion.

Setup (Optional but Recommended):
    Enable result persistence globally for automatic task result caching:

        prefect config set PREFECT_RESULTS_PERSIST_BY_DEFAULT=true

    This allows Prefect to automatically cache task results, avoiding
    redundant computation when inputs haven't changed.

Usage:
    # Run all experiments (data generation, training, prediction, comparison)
    uv run python graph-cg/scripts/run_experiments.py

    # Or import and run programmatically
    from src.workflows.workflow_prefect import run_experiment_matrix_flow
    run_experiment_matrix_flow()

Debugging:
    For detailed logging during data generation or other tasks:

        # Enable debug logging in source modules (e.g., data_collection.py)
        from loguru import logger
        logger.enable("src.data_collection")  # or other module name

    Prefect task logs can be viewed at the server UI (typically http://127.0.0.1:4200)

How Caching Works:
    Data generation and training use persist_result=True and cache_policy=INPUTS for persistent caching; prediction and comparison are intentionally uncached:

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
      * Currently uncached to always refresh plots/CSV diagnostics

    - Benefits:
        * No manual cache management needed
        * No stale checkpoint concerns (directories owned by flow)
        * Automatic cache invalidation on config/data changes
        * Clear traceability (outputs keyed by experiment name)
        * Complete pipeline runs once, then cached

Note:
    Experiments run sequentially to prevent memory exhaustion from large datasets.
    Each experiment can internally parallelize data generation steps if needed.
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
from prefect.task_runners import ConcurrentTaskRunner

from dlkit.interfaces.servers.mlflow_adapter import MLflowServerContext
from dlkit.tools.config.mlflow_settings import MLflowServerSettings

from src.cli.data import load_data_config
from src.cli.training import train_model
from src.cli.prediction import run_inference
from src.cli.comparison import compare_preconditioners
from src.system_loading import get_latest_checkpoint
from src.constants import (
    DEFAULT_EXPERIMENTS_CONFIG,
    DEFAULT_MLARTIFACTS_DIR,
    DEFAULT_MLRUNS_DIR,
    DEFAULT_OUTPUT_DIR,
)
from src.prefect_utils import (
    compute_directory_hash,
    compute_data_files_hash,
    compute_experiment_output_hash,
)
from src.paths.core import DataPaths, FlowPaths, ProjectRoots, parse_flow_keys
from src.validation import validate_data_exists
from src.generation import process_config
from src.diagnostics.prediction_diagnostics import save_prediction_samples_to_csv


def build_signature(**parts: str | None) -> str:
    """Create stable signature string from keyword parts."""
    tokens = []
    for key in sorted(parts):
        value = parts[key] or "none"
        tokens.append(f"{key}:{value}")
    return "|".join(tokens)


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


def resolve_expected_data_dir(data_config_path: Path | str) -> Path:
    """Predict the dataset directory without executing generation."""
    config_path = Path(data_config_path)
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    flow_id, dataset_id = parse_flow_keys(config, config_path=config_path)
    output_cfg = config.get("output") or {}
    roots = ProjectRoots.from_overrides(
        project_root=output_cfg.get("project_root"),
        processed_root=output_cfg.get("processed_dir"),
        output_root=output_cfg.get("output_root"),
        figures_root=output_cfg.get("figures_root"),
    )
    flow = FlowPaths(flow_id=flow_id, roots=roots)
    return DataPaths(flow=flow, dataset_id=dataset_id).base_dir


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

    return DEFAULT_OUTPUT_DIR


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
    data_state: str = "",
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
        data_state: Hash/state token for existing data directory (cache key)
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
    _ = data_state  # Ensures Prefect cache keys include filesystem state

    # Load config and process using unified interface
    # Note: process_config handles its own filesystem checks for idempotence
    cfg = load_data_config(config_path)
    output_path = process_config(cfg, config_path=config_path)

    print(f"\nData ready at: {output_path}")

    # Verify critical files were actually persisted to disk
    required_files = ["normalized.npz"]
    missing_files = []
    for filename in required_files:
        filepath = output_path / filename
        if not filepath.exists():
            missing_files.append(str(filepath))
        elif filepath.stat().st_size == 0:
            missing_files.append(f"{filepath} (empty)")

    if missing_files:
        files_str = "\n  - ".join(missing_files)
        raise RuntimeError(
            f"Data generation completed but required files are missing or empty:\n  - {files_str}\n"
            f"This may indicates a Dask worker file system sync issue."
        )

    # Compute content-based hash of data files
    data_hash = compute_data_files_hash(output_path)
    print(f"Data content hash: {data_hash[:12]}...")
    print("✓ Verified data files persisted successfully")

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
    experiment_dir: Path | None = None,
    checkpoint_state: str = "",
    model_config_version: str = "",
    data_config_version: str = "",
    src_hash: str = "",
    force: bool = False,
) -> Path:
    """Train model with resolved config paths.

    Experiment outputs are organized as:
    - Single-pair: {output_root}/{data_config_name}/{model_name}/
    - Multi-pair: {experiment_dir}/ (if experiment_dir is provided)

    Args:
        model_template_path: Path to model config template (e.g., "configs/ffnn.toml")
        data_dir: Directory containing the training data
        data_config_path: Path to the data config (used to extract data_config_name)
        data_hash: Content hash of data files for cache invalidation
        output_root: Optional override for output root (single-pair experiments)
        experiment_dir: Optional experiment directory (multi-pair experiments)
                        If provided, overrides automatic directory creation
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
    if experiment_dir is not None:
        # Multi-pair experiment: use provided experiment directory
        output_dirs = {
            "experiment_dir": experiment_dir,
            "checkpoints_dir": experiment_dir / "checkpoints",
            "figures_dir": experiment_dir / "figures",
            "metrics_dir": experiment_dir / "metrics",
            "predictions_dir": experiment_dir / "predictions",
        }
        # Ensure directories exist
        for dir_path in output_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Experiment output dir (multi-pair): %s", output_dirs["experiment_dir"]
        )
    else:
        # Single-pair experiment: use traditional directory structure
        root_dir = output_root or resolve_output_root(None)
        output_dirs = prepare_experiment_outputs(model_name, data_config_name, root_dir)
        logger.info(
            "Experiment output root (single-pair): %s", output_dirs["experiment_dir"]
        )
    # Check for existing checkpoint (filesystem-based caching)
    if not force:
        cached_checkpoint = get_latest_checkpoint(
            output_dirs["checkpoints_dir"]
        )
        if cached_checkpoint and cached_checkpoint.exists():
            logger.info(
                "Checkpoint already exists for %s/%s — skipping training",
                data_config_name,
                model_name,
            )
            return cached_checkpoint

    # Validate data exists
    validate_data_exists(data_dir, ["normalized.npz"])

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


@task(persist_result=False, cache_policy=None, cache_result_in_memory=False)
def predict_task(
    model_template_path: str,
    data_dir: Path,
    data_config_path: str | Path,
    checkpoint_path: Path,
    data_hash: str,
    output_root: Path | None = None,
    experiment_dir: Path | None = None,
    prediction_state: str = "",
    model_config_version: str = "",
    data_config_version: str = "",
    src_hash: str = "",
    force: bool = False,
) -> dict[str, Any]:
    """Run prediction/inference on trained model.

    Experiment outputs are organized as:
    - Single-pair: {output_root}/{data_config_name}/{model_name}/
    - Multi-pair: {experiment_dir}/ (if experiment_dir is provided)

    Args:
        model_template_path: Path to model config template
        data_dir: Directory containing the data
        data_config_path: Path to the data config
        checkpoint_path: Path to trained model checkpoint
        data_hash: Content hash of data files for cache invalidation
        output_root: Optional override for output root (single-pair experiments)
        experiment_dir: Optional experiment directory (multi-pair experiments)
                        If provided, overrides automatic directory creation
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

    # Prepare experiment-specific output directories
    if experiment_dir is not None:
        # Multi-pair experiment: use provided experiment directory
        output_dirs = {
            "experiment_dir": experiment_dir,
            "checkpoints_dir": experiment_dir / "checkpoints",
            "figures_dir": experiment_dir / "figures",
            "metrics_dir": experiment_dir / "metrics",
            "predictions_dir": experiment_dir / "predictions",
        }
        # Ensure directories exist
        for dir_path in output_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Experiment output dir (multi-pair): %s", output_dirs["experiment_dir"]
        )
    else:
        # Single-pair experiment: use traditional directory structure
        output_root_path = (
            Path(output_root) if output_root is not None else resolve_output_root(None)
        )
        output_dirs = prepare_experiment_outputs(
            model_name, data_config_name, output_root_path
        )
        logger.info(
            "Experiment output dir (single-pair): %s", output_dirs["experiment_dir"]
        )
    # Run inference
    # Note: run_inference will automatically resolve features_path and targets_path
    # from the data config, handling both normalized.npz and separate .npy formats
    logger.info("Running prediction...")
    print(f"\n{'=' * 60}")
    print(f"Predicting: {data_config_name}/{model_name}")
    print(f"{'=' * 60}")
    results = run_inference(
        config_path=model_template_path,
        data_config_path=data_config_path,
        checkpoint_path=checkpoint_path,
        features_path=None,
        targets_path=None,
        save_plots=True,
        figures_dir=output_dirs["figures_dir"],
    )

    plot_path = results.get("plot_path")
    if plot_path:
        logger.info("Prediction plots saved to %s", plot_path)
    else:
        logger.warning("Prediction returned no plot path")

    samples_csv_path = save_prediction_samples_to_csv(
        results.get("y_true"),
        results.get("y_pred"),
        output_dirs["predictions_dir"],
        f"{data_config_name}_{model_name}",
        num_samples=1,
    )
    if samples_csv_path:
        logger.info("Saved prediction sample CSVs: %s", samples_csv_path)
        results["samples_csv_paths"] = samples_csv_path
    else:
        logger.warning("Prediction sample CSV was not saved (missing targets/predictions or shape mismatch)")

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


@task(persist_result=False, cache_policy=None, cache_result_in_memory=False)
def compare_methods_task(
    model_template_path: str,
    data_dir: Path,
    data_config_path: str | Path,
    data_hash: str,
    output_root: Path | None = None,
    checkpoint_path: Path | None = None,
    comparison_state: str = "",
    model_config_version: str = "",
    data_config_version: str = "",
    src_hash: str = "",
    solver_config_path: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Compare different preconditioner methods.

    Args:
        model_template_path: Path to model config template
        data_dir: Directory containing the data
        data_config_path: Path to the data config
        data_hash: Content hash of data files for cache invalidation
        output_root: Optional override for output root
        checkpoint_path: Path to model checkpoint (optional)
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
    solver_config_version = (
        compute_file_signature(solver_config_path) if solver_config_path else ""
    )

    stage_signature = build_signature(
        checkpoint=str(checkpoint_path),
        comparison_state=comparison_state,
        data=data_hash,
        data_config=data_config_version,
        model=model_config_version,
        src=src_hash,
        solver_config=solver_config_version,
    )
    manifest_metadata = {
        "signature": stage_signature,
        "model_config_version": model_config_version,
        "data_config_version": data_config_version,
        "data_hash": data_hash,
        "src_hash": src_hash,
        "solver_config_version": solver_config_version,
    }

    output_root_path = (
        Path(output_root) if output_root is not None else resolve_output_root(None)
    )
    output_dirs = prepare_experiment_outputs(
        model_name, data_config_name, output_root_path
    )
    logger.info(
        "Comparing preconditioners for %s on %s",
        model_name,
        data_config_name,
    )

    if checkpoint_path:
        logger.info("Using checkpoint: %s", checkpoint_path)
    else:
        logger.info("No neural checkpoints provided (using classical methods only)")

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
    if checkpoint_path is not None and not checkpoint_path.exists():
        raise FileNotFoundError(
            f"checkpoint_path not found: {checkpoint_path}\n"
            "This may indicate the training task returned a cached path "
            "to a deleted checkpoint."
        )

    # Load data directly from normalized.npz
    from src.constants import DEFAULT_TEST_SAMPLE_INDEX

    logger.info(f"Loading comparison data from: {data_dir}")

    # compare_preconditioners loads data via load_system_data which supports
    # directory paths and will load from normalized.npz automatically

    # Run comparison
    logger.info("Running comparison...")
    print(f"\n{'=' * 60}")
    print(f"Comparing: {data_config_name}/{model_name}")
    print(f"{'=' * 60}")
    results = compare_preconditioners(
        config_path=model_template_path,
        data_config_path=data_config_path,
        matrix_path=data_dir / "normalized.npz",
        checkpoint_path=checkpoint_path,
        solver_config_path=solver_config_path,
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
    experiment_name: str,
    pairs: list[dict[str, str]],
    output_root: Path,
    solver_config_path: str | None = None,
    src_hash: str = "",
    force: bool = False,
) -> dict[str, Any]:
    """Run a complete experiment with multiple (data, model) pairs.

    Supports both single-pair and multi-pair experiments:
    - Single-pair: Traditional workflow with one dataset and one model
    - Multi-pair: Parallel data generation and training for multiple pairs

    For multi-pair experiments, pairs are processed in parallel:
    1. All data generation tasks run in parallel (Barrier 1)
    2. All training tasks run in parallel after data is ready (Barrier 2)
    3. Comparison runs with all trained checkpoints

    Args:
        experiment_name: Unique experiment identifier
        pairs: List of (name, model_template, data_config) dicts
               Each pair's "name" specifies its role (warm_start, preconditioner, both, etc.)
        output_root: Base output directory
        src_hash: Hash of source code for cache invalidation
        force: Force re-run of all tasks

    Returns:
        dict: Dictionary containing:
            - checkpoints: Mapping of pair names to checkpoint paths
            - prediction_results: Results from inference (single-pair only)
            - comparison_results: Results from method comparison

    Example (single-pair):
        >>> results = run_experiment_task(
        ...     "ffnn/generate-90",
        ...     [{"name": "both", "model_template": "configs/ffnn.toml", "data_config": "data-configs/generate-90.toml"}],
        ...     Path("/data/projects/graph-cg/data/output"),
        ... )

    Example (multi-pair):
        >>> results = run_experiment_task(
        ...     "warmstart-precond",
        ...     [
        ...         {"name": "warm_start", "model_template": "configs/ffnn.toml", "data_config": "data-configs/warmstart.toml"},
        ...         {"name": "preconditioner", "model_template": "configs/linear.toml", "data_config": "data-configs/precond.toml"},
        ...     ],
        ...     Path("/data/projects/graph-cg/data/output"),
        ... )
    """
    logger = get_run_logger()
    output_root_path = Path(output_root)

    # Determine if this is single-pair or multi-pair experiment
    is_single_pair = len(pairs) == 1 and pairs[0]["name"] in (
        "both",
        "warm_start",
        "preconditioner",
    )

    if is_single_pair:
        # Single-pair experiment: use traditional workflow with role-based checkpoint mapping
        pair = pairs[0]
        model_template_path = pair["model_template"]
        data_config_path = pair["data_config"]
        role = pair["name"]

        model_config_version = compute_file_signature(model_template_path)
        data_config_version = compute_file_signature(data_config_path)

        # Stage-specific cache tokens derived from current filesystem state
        model_name = extract_model_name(model_template_path)
        data_config_name = Path(data_config_path).stem
        experiment_dir = output_root_path / data_config_name / model_name
        checkpoints_dir = experiment_dir / "checkpoints"
        figures_dir = experiment_dir / "figures"
        metrics_dir = experiment_dir / "metrics"
        predictions_dir = experiment_dir / "predictions"

        checkpoint_state = compute_experiment_output_hash(checkpoints_dir)
        expected_data_dir = resolve_expected_data_dir(data_config_path)
        data_state = compute_data_files_hash(expected_data_dir)

        # Step 1: Get or generate data
        data_result = get_or_generate_data_task(
            data_config_path,
            data_config_version=data_config_version,
            src_hash=src_hash,
            data_state=data_state,
            force=force,
        )

        data_dir = data_result["data_dir"]
        data_hash = data_result["data_hash"]

        # Step 2: Train model
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
        # Map checkpoint based on role
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
            data_hash,
            output_root_path,
        comparison_state=comparison_state,
        model_config_version=model_config_version,
        data_config_version=data_config_version,
        src_hash=src_hash,
        force=force,
        checkpoint_path=checkpoint_path,
        solver_config_path=solver_config_path,
    )

        return {
            "checkpoints": {role: checkpoint_path},
            "prediction_results": prediction_results,
            "comparison_results": comparison_results,
        }

    else:
        # Multi-pair experiment: parallel data generation + training with barriers
        logger.info(
            f"Running multi-pair experiment: {experiment_name} with {len(pairs)} pairs"
        )

        # Phase 1: Submit all data generation tasks in parallel
        logger.info("Phase 1: Parallel data generation for all pairs")
        data_futures: dict[str, PrefectFuture] = {}
        for pair in pairs:
            pair_name = pair["name"]
            data_config_path = pair["data_config"]
            data_config_version = compute_file_signature(data_config_path)
            expected_data_dir = resolve_expected_data_dir(data_config_path)
            data_state = compute_data_files_hash(expected_data_dir)

            future = get_or_generate_data_task.submit(
                data_config_path,
                data_config_version=data_config_version,
                src_hash=src_hash,
                data_state=data_state,
                force=force,
            )
            data_futures[pair_name] = future
            logger.info(f"  Submitted data generation for pair '{pair_name}'")

        # Barrier 1: Wait for all data generation to complete
        logger.info("Barrier 1: Waiting for all data generation tasks...")
        data_results: dict[str, dict[str, Any]] = {}
        for pair_name, future in data_futures.items():
            data_results[pair_name] = future.result()
            logger.info(f"  ✓ Data ready for pair '{pair_name}'")

        # Phase 2: Submit all training tasks in parallel
        logger.info("Phase 2: Parallel training for all pairs")
        training_futures: dict[str, PrefectFuture] = {}
        experiment_dirs: dict[str, dict[str, Path]] = {}

        for pair in pairs:
            pair_name = pair["name"]
            model_template_path = pair["model_template"]
            data_config_path = pair["data_config"]

            model_config_version = compute_file_signature(model_template_path)
            data_config_version = compute_file_signature(data_config_path)

            data_result = data_results[pair_name]
            data_dir = data_result["data_dir"]
            data_hash = data_result["data_hash"]

            # Create experiment-specific directories: {output_root}/{experiment_name}/{pair_name}/
            model_name = extract_model_name(model_template_path)
            pair_experiment_dir = output_root_path / experiment_name / pair_name
            pair_checkpoints_dir = pair_experiment_dir / "checkpoints"
            pair_figures_dir = pair_experiment_dir / "figures"
            pair_metrics_dir = pair_experiment_dir / "metrics"
            pair_predictions_dir = pair_experiment_dir / "predictions"

            experiment_dirs[pair_name] = {
                "experiment_dir": pair_experiment_dir,
                "checkpoints_dir": pair_checkpoints_dir,
                "figures_dir": pair_figures_dir,
                "metrics_dir": pair_metrics_dir,
                "predictions_dir": pair_predictions_dir,
            }

            checkpoint_state = compute_experiment_output_hash(pair_checkpoints_dir)

            future = train_model_task.submit(
                model_template_path,
                data_dir,
                data_config_path,
                data_hash,
                output_root=output_root_path,
                experiment_dir=pair_experiment_dir,
                checkpoint_state=checkpoint_state,
                model_config_version=model_config_version,
                data_config_version=data_config_version,
                src_hash=src_hash,
                force=force,
            )
            training_futures[pair_name] = future
            logger.info(f"  Submitted training for pair '{pair_name}'")

        # Barrier 2: Wait for all training to complete
        logger.info("Barrier 2: Waiting for all training tasks...")
        checkpoints: dict[str, Path] = {}
        for pair_name, future in training_futures.items():
            checkpoints[pair_name] = future.result()
            logger.info(
                f"  ✓ Training complete for pair '{pair_name}': {checkpoints[pair_name]}"
            )

        # Phase 2.5: Submit all prediction tasks in parallel
        logger.info("Phase 2.5: Parallel predictions for all pairs")
        prediction_futures: dict[str, PrefectFuture] = {}

        for pair in pairs:
            pair_name = pair["name"]
            model_template_path = pair["model_template"]
            data_config_path = pair["data_config"]

            data_result = data_results[pair_name]
            data_dir = data_result["data_dir"]
            data_hash = data_result["data_hash"]

            checkpoint_path = checkpoints[pair_name]
            pair_dirs = experiment_dirs[pair_name]

            model_config_version = compute_file_signature(model_template_path)
            data_config_version = compute_file_signature(data_config_path)

            # Compute prediction state for cache invalidation
            prediction_state = "|".join(
                [
                    f"figures:{compute_experiment_output_hash(pair_dirs['figures_dir'])}",
                    f"predictions:{compute_experiment_output_hash(pair_dirs['predictions_dir'])}",
                ]
            )

            future = predict_task.submit(
                model_template_path,
                data_dir,
                data_config_path,
                checkpoint_path,
                data_hash,
                output_root=output_root_path,
                experiment_dir=pair_dirs["experiment_dir"],
                prediction_state=prediction_state,
                model_config_version=model_config_version,
                data_config_version=data_config_version,
                src_hash=src_hash,
                force=force,
            )
            prediction_futures[pair_name] = future
            logger.info(f"  Submitted prediction for pair '{pair_name}'")

        # Barrier 2.5: Wait for all predictions
        logger.info("Barrier 2.5: Waiting for all prediction tasks...")
        prediction_results: dict[str, dict[str, Any]] = {}
        for pair_name, future in prediction_futures.items():
            prediction_results[pair_name] = future.result()
            logger.info(f"  ✓ Predictions complete for pair '{pair_name}'")

        # Phase 3: Run comparison with all checkpoints
        # Use first pair's config for comparison (they should share system data)
        first_pair = pairs[0]
        model_template_path = first_pair["model_template"]
        data_config_path = first_pair["data_config"]
        data_result = data_results[first_pair["name"]]
        data_dir = data_result["data_dir"]
        data_hash = data_result["data_hash"]

        model_config_version = compute_file_signature(model_template_path)
        data_config_version = compute_file_signature(data_config_path)

        # Create comparison directory at experiment level
        comparison_dir = output_root_path / experiment_name / "comparison"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        figures_dir = comparison_dir / "figures"
        metrics_dir = comparison_dir / "metrics"
        figures_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)

        comparison_state = "|".join(
            [
                f"figures:{compute_experiment_output_hash(figures_dir)}",
                f"metrics:{compute_experiment_output_hash(metrics_dir)}",
            ]
        )

        # Select checkpoint for comparison (prefer explicitly named preconditioner, else first)
        checkpoint_path = checkpoints.get("preconditioner", next(iter(checkpoints.values())))

        logger.info("Phase 3: Running comparison with all checkpoints")
        comparison_results = compare_methods_task(
            model_template_path,
            data_dir,
            data_config_path,
            data_hash,
            output_root_path,
            comparison_state=comparison_state,
            model_config_version=model_config_version,
            data_config_version=data_config_version,
            src_hash=src_hash,
            force=force,
            checkpoint_path=checkpoint_path,
            solver_config_path=solver_config_path,
        )

        return {
            "checkpoints": checkpoints,
            "prediction_results": prediction_results,
            "comparison_results": comparison_results,
        }


@flow(
    name="run_experiment_matrix",
    task_runner=ConcurrentTaskRunner(max_workers=1),
)
def run_experiment_matrix_flow(
    experiments_config_path: str | None = None,
    force: bool = False,
) -> dict[str, dict[str, Any]]:
    """Run all experiments defined in configs/experiments.toml sequentially.

    This flow orchestrates the entire experiment matrix:
    1. Reads experiment definitions from configs/experiments.toml
    2. Submits all experiments (sequential execution to avoid memory issues)
    3. Generates all unique datasets (with caching)
    4. Trains all models sequentially
    5. Runs predictions for each trained model
    6. Compares preconditioner methods for each experiment

    Prefect handles:
    - Automatic caching (data generation runs once per unique config)
    - Sequential execution to prevent memory exhaustion
    - Incremental computation (only reruns changed configs)

    Args:
        experiments_config_path: Path to experiments definition file.
            If None, uses GRAPH_CG_EXPERIMENTS_CONFIG env var or
            defaults to the repository configs/experiments.toml.

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
        Experiments are submitted as tasks to enable async submission.
        Sequential execution prevents memory exhaustion from parallel workers.
    """
    # Resolve config path (env var or default)
    if experiments_config_path is None:
        experiments_config_path = os.getenv("GRAPH_CG_EXPERIMENTS_CONFIG")

    if experiments_config_path is None:
        repo_root = Path(__file__).resolve().parent.parent
        experiments_config_path = repo_root / DEFAULT_EXPERIMENTS_CONFIG
    else:
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

    # Convert experiments to normalized format (list of pairs)
    normalized_experiments = []
    for idx, exp in enumerate(experiments):
        # Support unified network format
        model_template = exp.get("model_template")
        data_config = exp.get("data_config")

        # Support legacy multi-role format for backward compatibility
        precond_cfg = exp.get("preconditioner")
        warmstart_cfg = exp.get("warm_start")
        same = exp.get("same", False)

        exp_solver = exp.get("solver_config", "solver-configs/default.toml")

        if model_template and data_config:
            # Standard unified network format
            pairs = [
                {
                    "name": "both",
                    "model_template": model_template,
                    "data_config": data_config,
                }
            ]
            exp_name = f"{Path(model_template).stem}/{Path(data_config).stem}"
        elif precond_cfg or warmstart_cfg:
            # Legacy multi-role format for backward compatibility
            pairs = []

            if same:
                # Unified network for both roles
                cfg = precond_cfg or warmstart_cfg
                if not cfg:
                    raise ValueError(
                        f"Experiment {idx} with same=true must have preconditioner or warm_start config"
                    )

                model_template = cfg.get("model")
                data_config = cfg.get("data")
                if not model_template or not data_config:
                    raise ValueError(
                        f"Experiment {idx} config missing 'model' or 'data' field"
                    )

                pairs.append(
                    {
                        "name": "both",
                        "model_template": model_template,
                        "data_config": data_config,
                    }
                )
                exp_name = f"{Path(model_template).stem}/{Path(data_config).stem}"
            else:
                # Multiple checkpoints for different roles
                exp_parts = []

                if precond_cfg:
                    model_template = precond_cfg.get("model")
                    data_config = precond_cfg.get("data")
                    if not model_template or not data_config:
                        raise ValueError(
                            f"Experiment {idx} preconditioner config missing 'model' or 'data' field"
                        )

                    pairs.append(
                        {
                            "name": "preconditioner",
                            "model_template": model_template,
                            "data_config": data_config,
                        }
                    )
                    exp_parts.append(f"precond-{Path(model_template).stem}")

                if warmstart_cfg:
                    model_template = warmstart_cfg.get("model")
                    data_config = warmstart_cfg.get("data")
                    if not model_template or not data_config:
                        raise ValueError(
                            f"Experiment {idx} warm_start config missing 'model' or 'data' field"
                        )

                    pairs.append(
                        {
                            "name": "warm_start",
                            "model_template": model_template,
                            "data_config": data_config,
                        }
                    )
                    exp_parts.append(f"warmstart-{Path(model_template).stem}")

                exp_name = "-".join(exp_parts)
        else:
            raise ValueError(
                f"Experiment {idx} must have either 'model_template' and 'data_config' fields "
                f"(unified network) or 'preconditioner'/'warm_start' fields (legacy multi-role)"
            )

        normalized_experiments.append(
            {
                "name": exp_name,
                "pairs": pairs,
                "solver_config": exp_solver,
            }
        )

    print(f"\nFound {len(normalized_experiments)} experiments:")
    for idx, exp in enumerate(normalized_experiments, 1):
        exp_name = exp["name"]
        pairs = exp["pairs"]

        if len(pairs) == 1:
            pair = pairs[0]
            role = pair["name"]
            model = Path(pair["model_template"]).stem
            data = Path(pair["data_config"]).stem
            if role == "both":
                print(f"  {idx}. {model}/{data} (unified network)")
            else:
                print(f"  {idx}. {model}/{data} (role: {role})")
        else:
            print(f"  {idx}. {exp_name}")
            for pair in pairs:
                model = Path(pair["model_template"]).stem
                data = Path(pair["data_config"]).stem
                print(f"      - {pair['name']}: {model} + {data}")
    print()

    # Resolve paths relative to config directory
    config_dir = experiments_config_path.parent

    # Compute source code hash for cache invalidation
    src_dir = config_dir / "src"
    src_hash = compute_directory_hash(src_dir)
    print(f"Source code hash: {src_hash[:12]}...")

    # Configure MLflow paths from config or defaults
    mlruns_dir = Path(path_settings.get("mlruns_dir", str(DEFAULT_MLRUNS_DIR)))
    mlartifacts_dir = Path(
        path_settings.get("mlartifacts_dir", str(DEFAULT_MLARTIFACTS_DIR))
    )

    # Ensure MLflow directories exist
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    mlartifacts_dir.mkdir(parents=True, exist_ok=True)
    print(f"MLflow tracking: {mlruns_dir}")
    print(f"MLflow artifacts: {mlartifacts_dir}")

    # Start MLflow server once for entire workflow
    # Use absolute resolved paths with forward slashes for SQLite URI
    mlruns_db_path = mlruns_dir.resolve() / "mlflow.db"
    mlflow_server_config = MLflowServerSettings(
        host="127.0.0.1",
        port=5000,
        backend_store_uri=f"sqlite:///{mlruns_db_path.as_posix()}",
        artifacts_destination=str(mlartifacts_dir.resolve()),
    )

    with MLflowServerContext(server_config=mlflow_server_config) as server_info:
        print(f"\nMLflow server started at: {server_info.url}")

        # Submit all experiments in parallel (non-blocking)
        print("Submitting experiments in parallel...")
        print(f"Output root: {output_root}")
        if force:
            print("Force mode: Ignoring filesystem checks, re-running all tasks")
        futures: dict[str, PrefectFuture] = {}

        for exp in normalized_experiments:
            experiment_name = exp["name"]
            pairs = exp["pairs"]

            # Resolve all pair paths relative to config directory
            resolved_pairs = []
            for pair in pairs:
                resolved_pairs.append(
                    {
                        "name": pair["name"],
                        "model_template": str(config_dir / pair["model_template"]),
                        "data_config": str(config_dir / pair["data_config"]),
                    }
                )
            solver_config_path = str((config_dir / exp.get("solver_config", "solver-configs/default.toml")).resolve())

            # .submit() returns immediately (non-blocking)
            future = run_experiment_task.submit(
                experiment_name=experiment_name,
                pairs=resolved_pairs,
                output_root=output_root,
                solver_config_path=solver_config_path,
                src_hash=src_hash,
                force=force,
            )
            futures[experiment_name] = future
            print(f"  ✓ Submitted: {experiment_name}")

        print(f"\nWaiting for {len(futures)} experiments to complete...")
        print("(Running sequentially to prevent memory exhaustion)\n")

        # Wait for all experiments and collect results
        results: dict[str, dict[str, Any]] = {}
        failed_experiments: list[tuple[str, Exception]] = []

        for experiment_id, future in futures.items():
            try:
                experiment_results = (
                    future.result()
                )  # Blocks until this experiment completes
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
                checkpoints = experiment_results.get("checkpoints", {})
                if checkpoints:
                    checkpoint_info = ", ".join(
                        f"{role}={path}" for role, path in checkpoints.items()
                    )
                    print(f"  ✓ {experiment_id}: {checkpoint_info}")
                else:
                    print(f"  ✓ {experiment_id}: No checkpoints")

        if failed_experiments:
            print("\nFailed experiments:")
            for experiment_id, error in failed_experiments:
                print(f"  ✗ {experiment_id}: {error}")

        # Create summary artifact for Prefect UI
        experiment_data = []
        for exp in normalized_experiments:
            experiment_id = exp["name"]
            pairs = exp["pairs"]

            if experiment_id in results:
                experiment_results = results[experiment_id]
                checkpoints = experiment_results.get("checkpoints", {})
                prediction_results = experiment_results.get("prediction_results", {})
                comparison_results = experiment_results.get("comparison_results", {})

                # Format checkpoint info
                if checkpoints:
                    checkpoint_str = ", ".join(
                        f"{role}: {path.name}" for role, path in checkpoints.items()
                    )
                else:
                    checkpoint_str = "None"

                # Format additional info
                info_parts = []
                if prediction_results and prediction_results.get("plot_path"):
                    info_parts.append("✓ Prediction")
                if comparison_results and comparison_results.get("plot_paths"):
                    info_parts.append("✓ Comparison")
                additional_info = (
                    ", ".join(info_parts) if info_parts else "Comparison only"
                )

                # Format pair info
                if len(pairs) == 1:
                    pair_info = f"{Path(pairs[0]['model_template']).stem} + {Path(pairs[0]['data_config']).stem}"
                else:
                    pair_info = f"{len(pairs)} pairs"

                experiment_data.append(
                    {
                        "Experiment": experiment_id,
                        "Pairs": pair_info,
                        "Status": "✓ Success",
                        "Checkpoints": checkpoint_str,
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

                # Format pair info
                if len(pairs) == 1:
                    pair_info = f"{Path(pairs[0]['model_template']).stem} + {Path(pairs[0]['data_config']).stem}"
                else:
                    pair_info = f"{len(pairs)} pairs"

                experiment_data.append(
                    {
                        "Experiment": experiment_id,
                        "Pairs": pair_info,
                        "Status": "✗ Failed",
                        "Checkpoints": error_msg,
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
