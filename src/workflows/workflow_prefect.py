"""Prefect workflow for orchestrating graph-cg experiments.

This workflow provides complete end-to-end experiment execution:
- Data generation (with filesystem-based caching)
- Model training
- Prediction/inference (parity plots)
- Method comparison (preconditioner analysis)

All steps are integrated into the flow with automatic caching and sequential execution
to prevent memory exhaustion.
"""

from __future__ import annotations

import glob
import hashlib
import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("DASK_DISTRIBUTED__LOGGING__DISTRIBUTED", "error")
from pathlib import Path
from typing import Any

from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact, create_table_artifact
from prefect.cache_policies import INPUTS
from prefect.futures import PrefectFuture
from prefect.task_runners import ConcurrentTaskRunner

from dlkit import GeneralSettings
from dlkit.interfaces.servers.mlflow_adapter import MLflowServerContext
from dlkit.tools.config.mlflow_settings import MLflowServerSettings

from src.cli.data import load_data_config
from src.cli.training import train_model
from src.cli.prediction import run_inference
from src.cli.comparison import compare_preconditioners
from src.system_loading import get_latest_checkpoint
from src.constants import (
    DEFAULT_MLARTIFACTS_DIR,
    DEFAULT_MLRUNS_DIR,
    DEFAULT_OUTPUT_DIR,
)
from src.prefect_utils import (
    compute_directory_hash,
    compute_data_files_hash,
    compute_experiment_output_hash,
)
from src.paths.core import FlowContext
from src.validation import validate_data_exists
from src.generation import process_config
from src.configuration.loader import load_experiments


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
    """Extract model name from config SESSION.name."""
    # This function now has to load the toml file itself, as it only gets a path.
    import tomllib
    model_config_path = Path(model_config_path)
    with open(model_config_path, "rb") as f:
        config = tomllib.load(f)
    session = config.get("SESSION") or {}
    name = session.get("name")
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
    """Ensure experiment-specific output directories exist."""
    output_root.mkdir(parents=True, exist_ok=True)
    experiment_dir = output_root / data_config_name / model_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
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
    """Get existing data or generate if missing (idempotent)."""
    config_path = Path(data_config_path)
    _ = data_config_version
    _ = src_hash
    _ = data_state

    cfg = load_data_config(config_path)
    output_path = process_config(cfg, config_path=config_path)
    print(f"\nData ready at: {output_path}")

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
            f"Data generation completed but required files are missing or empty:\n  - {files_str}"
        )

    data_hash = compute_data_files_hash(output_path)
    print(f"Data content hash: {data_hash[:12]}...")
    create_markdown_artifact(
        key=f"data-status-{config_path.stem}".lower(),
        markdown=(
            f"# Data Ready\n\n"
            f"**Config**: `{config_path.name}`\n\n"
            f"**Location**: `{output_path}`\n\n"
            f"**Data Hash**: `{data_hash[:12]}...`"
        ),
        description=f"Data status for {config_path.stem}",
    )
    return {"data_dir": output_path, "data_hash": data_hash}


@task(persist_result=True, cache_policy=INPUTS)
def train_model_task(
    model_config_path: str,
    data_config_path: str,
    data_dir: Path,
    data_hash: str,
    output_root: Path,
    checkpoint_state: str = "",
    model_config_version: str = "",
    data_config_version: str = "",
    src_hash: str = "",
    force: bool = False,
) -> Path:
    """Train model with resolved config paths."""
    logger = get_run_logger()
    model_name = extract_model_name(model_config_path)
    data_config_name = Path(data_config_path).stem
    logger.info("Training %s on %s", model_name, data_config_name)

    _ = src_hash
    _ = data_hash
    _ = checkpoint_state

    output_dirs = prepare_experiment_outputs(model_name, data_config_name, output_root)

    if not force:
        cached_checkpoint = get_latest_checkpoint(output_dirs["checkpoints_dir"])
        if cached_checkpoint and cached_checkpoint.exists():
            logger.info("Checkpoint already exists, skipping training.")
            return cached_checkpoint

    validate_data_exists(data_dir, ["normalized.npz"])

    logger.info("Starting training...")
    checkpoint_path = train_model(
        config_path=model_config_path,
        data_config_path=data_config_path,
        output_dir=output_dirs["experiment_dir"],
        session_name=model_name,
    )

    if output_dirs["checkpoints_dir"] not in checkpoint_path.parents:
        raise RuntimeError("Checkpoint written outside checkpoints directory.")

    logger.info("Checkpoint saved to %s", checkpoint_path)
    return checkpoint_path


@task(persist_result=True, cache_policy=INPUTS)
def predict_task(
    model_config_path: str,
    data_config_path: str,
    data_dir: Path,
    checkpoint_path: Path,
    data_hash: str,
    output_root: Path,
    prediction_state: str = "",
    model_config_version: str = "",
    data_config_version: str = "",
    src_hash: str = "",
    force: bool = False,
) -> dict[str, Any]:
    """Run prediction/inference on trained model."""
    logger = get_run_logger()
    model_name = extract_model_name(model_config_path)
    data_config_name = Path(data_config_path).stem

    _ = src_hash
    _ = data_hash
    _ = prediction_state

    output_dirs = prepare_experiment_outputs(model_name, data_config_name, output_root)

    if not force:
        plot_pattern = output_dirs["figures_dir"] / "parity_*.png"
        if glob.glob(str(plot_pattern)):
            logger.info("Prediction already exists, skipping inference.")
            return {"skipped": True}

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info("Running prediction...")
    results = run_inference(
        config_path=model_config_path,
        data_config_path=data_config_path,
        checkpoint_path=checkpoint_path,
        save_plots=True,
        figures_dir=output_dirs["figures_dir"],
    )
    logger.info("Prediction completed for %s/%s", model_name, data_config_name)
    return results


@task(persist_result=True, cache_policy=INPUTS)
def compare_methods_task(
    model_config_path: str,
    data_config_path: str,
    solver_config_path: str,
    data_dir: Path,
    data_hash: str,
    output_root: Path,
    checkpoint_path: Path | None = None,
    comparison_state: str = "",
    model_config_version: str = "",
    data_config_version: str = "",
    src_hash: str = "",
    force: bool = False,
) -> dict[str, Any]:
    """Compare different preconditioner methods."""
    logger = get_run_logger()
    model_name = extract_model_name(model_config_path)
    data_config_name = Path(data_config_path).stem

    _ = src_hash
    _ = data_hash
    _ = comparison_state

    output_dirs = prepare_experiment_outputs(model_name, data_config_name, output_root)

    if not force:
        plot_pattern = output_dirs["figures_dir"] / "convergence_*.png"
        if glob.glob(str(plot_pattern)):
            logger.info("Comparison already exists, skipping.")
            return {"skipped": True}

    if checkpoint_path and not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info("Running comparison...")
    results = compare_preconditioners(
        config_path=model_config_path,
        data_config_path=data_config_path,
        matrix_path=data_dir / "normalized.npz",
        checkpoint_path=checkpoint_path,
        solver_config_path=solver_config_path,
        save_plots=True,
        output_dir=output_dirs["experiment_dir"],
        figures_dir=output_dirs["figures_dir"],
    )
    logger.info("Comparison completed for %s/%s", model_name, data_config_name)
    return results


@task(task_run_name="run-experiment-{experiment_name}")
def run_experiment_task(
    experiment_name: str,
    model_config_path: Path,
    data_gen_config_path: Path,
    solver_config_path: Path,
    output_root: Path,
    src_hash: str = "",
    force: bool = False,
) -> dict[str, Any]:
    """Run a complete, self-contained experiment."""
    logger = get_run_logger()
    logger.info(f"Starting experiment: {experiment_name}")

    model_config_version = compute_file_signature(model_config_path)
    data_config_version = compute_file_signature(data_gen_config_path)
    
    data_state = compute_data_files_hash(data_gen_config_path.parent)

    data_result = get_or_generate_data_task(
        str(data_gen_config_path),
        data_config_version=data_config_version,
        src_hash=src_hash,
        data_state=data_state,
        force=force,
    )
    data_dir = data_result["data_dir"]
    data_hash = data_result["data_hash"]

    checkpoint_state = compute_experiment_output_hash(
        output_root / Path(data_gen_config_path).stem / extract_model_name(model_config_path) / "checkpoints"
    )
    
    checkpoint_path = train_model_task(
        str(model_config_path),
        str(data_gen_config_path),
        data_dir,
        data_hash,
        output_root,
        checkpoint_state=checkpoint_state,
        model_config_version=model_config_version,
        data_config_version=data_config_version,
        src_hash=src_hash,
        force=force,
    )

    prediction_state = compute_experiment_output_hash(
         output_root / Path(data_gen_config_path).stem / extract_model_name(model_config_path) / "predictions"
    )
    prediction_results = predict_task(
        str(model_config_path),
        str(data_gen_config_path),
        data_dir,
        checkpoint_path,
        data_hash,
        output_root,
        prediction_state=prediction_state,
        model_config_version=model_config_version,
        data_config_version=data_config_version,
        src_hash=src_hash,
        force=force,
    )
    
    comparison_state = compute_experiment_output_hash(
        output_root / Path(data_gen_config_path).stem / extract_model_name(model_config_path) / "figures" # Also uses figures
    )
    comparison_results = compare_methods_task(
        str(model_config_path),
        str(data_gen_config_path),
        str(solver_config_path),
        data_dir,
        data_hash,
        output_root,
        checkpoint_path=checkpoint_path,
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


@flow(name="run_experiment_matrix", task_runner=ConcurrentTaskRunner(max_workers=1))
def run_experiment_matrix_flow(
    experiments_config_path: str | Path | None = None,
    force: bool = False,
) -> dict[str, dict[str, Any]]:
    """Run all experiments defined in the configuration system."""
    experiments = load_experiments(experiments_config_path)

    print(f"\nFound {len(experiments)} experiments to run:")
    for idx, (exp_name, _, _, model_path, data_gen_path, solver_path) in enumerate(experiments, 1):
        print(f"  {idx}. {exp_name}")
    print()

    project_root = Path.cwd()
    output_root = resolve_output_root(None)

    src_dir = project_root / "src"
    src_hash = compute_directory_hash(src_dir)
    print(f"Source code hash: {src_hash[:12]}...")

    mlruns_dir = project_root / "data" / "mlruns"
    mlartifacts_dir = project_root / "data" / "mlartifacts"
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    mlartifacts_dir.mkdir(parents=True, exist_ok=True)
    print(f"MLflow tracking: {mlruns_dir}")
    print(f"MLflow artifacts: {mlartifacts_dir}")

    mlruns_db_path = mlruns_dir.resolve() / "mlflow.db"
    mlflow_server_config = MLflowServerSettings(
        host="127.0.0.1",
        port=5000,
        backend_store_uri=f"sqlite:///{mlruns_db_path.as_posix()}",
        artifacts_destination=str(mlartifacts_dir.resolve()),
    )

    with MLflowServerContext(server_config=mlflow_server_config) as server_info:
        print(f"\nMLflow server started at: {server_info.url}")
        print("Submitting experiments sequentially...")
        print(f"Output root: {output_root}")
        if force:
            print("Force mode: Ignoring filesystem checks, re-running all tasks")

        futures: dict[str, PrefectFuture] = {}
        for exp_name, _, _, model_path, data_gen_path, solver_path in experiments:
            future = run_experiment_task.submit(
                experiment_name=exp_name,
                model_config_path=model_path,
                data_gen_config_path=data_gen_path,
                solver_config_path=solver_path,
                output_root=output_root,
                src_hash=src_hash,
                force=force,
            )
            futures[exp_name] = future
            print(f"  ✓ Submitted: {exp_name}")

        print(f"\nWaiting for {len(futures)} experiments to complete...")
        print("(Running sequentially to prevent memory exhaustion)\n")

        results: dict[str, dict[str, Any]] = {}
        failed_experiments: list[tuple[str, Exception]] = []

        for experiment_id, future in futures.items():
            try:
                experiment_results = future.result()
                results[experiment_id] = experiment_results
                print(f"  ✓ Completed: {experiment_id}")
            except Exception as e:
                print(f"  ✗ Failed: {experiment_id} - {e}")
                failed_experiments.append((experiment_id, e))

        print(f"\n{'=' * 80}")
        print(
            f"Experiment matrix completed! "
            f"{len(results)}/{len(experiments)} succeeded, "
            f"{len(failed_experiments)} failed"
        )
        print(f"{ '=' * 80}")

        if failed_experiments:
            print("\nFailed experiments:")
            for experiment_id, error in failed_experiments:
                print(f"  ✗ {experiment_id}: {error}")

        # Create summary artifact for Prefect UI
        experiment_data = []
        for exp_name, _, _, model_path, data_gen_path, solver_path in experiments:
            status = "✓ Success" if exp_name in results else "✗ Failed"
            details = "OK"
            if exp_name not in results:
                error_msg = next(
                    (str(err) for failed_id, err in failed_experiments if failed_id == exp_name),
                    "Unknown error",
                )
                details = error_msg

            experiment_data.append(
                {
                    "Experiment": exp_name,
                    "Status": status,
                    "Model": model_path.name,
                    "Data Gen": data_gen_path.name,
                    "Solver": solver_path.name,
                    "Details": details,
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
