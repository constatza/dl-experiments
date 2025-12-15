"""Inference / Prediction task."""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

from prefect import task, get_run_logger
from prefect.cache_policies import INPUTS

from src.cli.prediction import run_inference
from src.workflows.utils.paths import extract_model_name, prepare_experiment_outputs


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
    """Run prediction/inference on trained model.
    
    Args:
        model_config_path: Path to model config.
        data_config_path: Path to data config.
        data_dir: Directory containing input data.
        checkpoint_path: Path to trained checkpoint.
        data_hash: Hash of data content.
        output_root: Root output directory.
        prediction_state: Hash of output prediction dir (for cache invalidation).
        model_config_version: Hash of model config.
        data_config_version: Hash of data config.
        src_hash: Hash of source code.
        force: Force re-run.
        
    Returns:
        Dictionary with prediction results.
    """
    logger = get_run_logger()
    model_name = extract_model_name(model_config_path)
    data_config_name = Path(data_config_path).stem

    # Unused args for cache key
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
