"""Lightweight orchestration without Prefect."""

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
from neuralls.system_loading import get_latest_checkpoint


def run_experiment(
    *,
    model_config_path: Path,
    data_config_path: Path,
    output_root: Path,
    force: bool,
    src_hash: str,
) -> ExperimentResult:
    """Run data generation and model training for a single experiment.

    This function handles:
    1. Data generation (with caching)
    2. Model training (with checkpoint detection)

    For solver comparison, use compare_preconditioners.py after training.
    """
    model_name = extract_model_name(model_config_path)
    try:
        data_cfg = load_data_config(data_config_path)
        data_dir = process_config(data_cfg, config_path=data_config_path)
        validate_data_exists(data_dir, ["normalized.npz"])

        checkpoint_dir = (
            output_root / data_config_path.stem / model_name / "checkpoints"
        )
        checkpoint = get_latest_checkpoint(checkpoint_dir)
        if force or checkpoint is None:
            checkpoint = train_model(
                config_path=model_config_path,
                data_config_path=data_config_path,
                output_root=output_root,
            )
            logger.info(f"Training complete: {checkpoint}")
        else:
            logger.info(f"Using existing checkpoint: {checkpoint}")

        return ExperimentResult(
            experiment_id=model_name,
            status="Success",
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Experiment {model_name} failed: {exc}")
        return ExperimentResult(
            experiment_id=model_name, status="Failed", error=str(exc)
        )


def run_experiment_matrix(
    experiments_config_path: Path,
    *,
    force: bool = False,
    project_root: Path | None = None,
) -> list[ExperimentResult]:
    """Run training for all experiments defined in experiments.toml.

    This function:
    1. Generates all unique datasets in parallel (with caching)
    2. Trains all models

    For solver comparison after training, use compare_preconditioners.py with:
    - --experiments <path to experiments.toml>
    - --solver-config <path to solver config>
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]

    batch = load_batch(experiments_config_path)
    experiments = batch.experiments

    src_dir = project_root / "src"
    if not src_dir.exists():
        src_dir = Path(__file__).resolve().parent.parent
    src_hash = compute_directory_hash(src_dir)

    logger.info(f"Training {len(experiments)} experiments from {experiments_config_path}")

    results: list[ExperimentResult] = []
    for exp in experiments:
        logger.info(f"\n{'='*60}")
        logger.info(f"Experiment: {exp.spec.id}")
        logger.info(f"  Model: {exp.spec.model_config_path.stem}")
        logger.info(f"  Dataset: {exp.spec.data_config_path.stem}")
        logger.info(f"{'='*60}")

        result = run_experiment(
            model_config_path=exp.spec.model_config_path,
            data_config_path=exp.spec.data_config_path,
            output_root=batch.global_output_dir,
            force=force,
            src_hash=src_hash,
        )
        results.append(result)
    return results
