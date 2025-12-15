"""Method comparison task."""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

from prefect import task, get_run_logger
from prefect.cache_policies import INPUTS

from src.cli.comparison import compare_preconditioners
from src.workflows.utils.paths import extract_model_name, prepare_experiment_outputs


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
    """Compare different preconditioner methods.
    
    Args:
        model_config_path: Path to model config.
        data_config_path: Path to data config.
        solver_config_path: Path to solver config.
        data_dir: Directory containing matrix data.
        data_hash: Hash of data content.
        output_root: Root output directory.
        checkpoint_path: Optional path to neural checkpoint.
        comparison_state: Hash of output state (figures/metrics).
        model_config_version: Hash of model config.
        data_config_version: Hash of data config.
        src_hash: Hash of source code.
        force: Force re-run.
        
    Returns:
        Dictionary with comparison results.
    """
    logger = get_run_logger()
    model_name = extract_model_name(model_config_path)
    data_config_name = Path(data_config_path).stem

    # Unused args for cache key
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
