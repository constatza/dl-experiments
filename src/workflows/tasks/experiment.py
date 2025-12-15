"""Experiment orchestration task."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from prefect import task, get_run_logger

from src.workflows.tasks.data import get_or_generate_data_task
from src.workflows.tasks.training import train_model_task
from src.workflows.tasks.inference import predict_task
from src.workflows.tasks.comparison import compare_methods_task
from src.workflows.utils.hashing import (
    compute_data_files_hash,
    compute_experiment_output_hash,
    compute_file_signature,
)
from src.workflows.utils.paths import extract_model_name


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
    """Run a complete, self-contained experiment.
    
    Orchestrates: Data -> Training -> Prediction -> Comparison.
    """
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
        output_root / Path(data_gen_config_path).stem / extract_model_name(model_config_path) / "figures"
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
