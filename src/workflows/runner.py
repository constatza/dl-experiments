"""Lightweight orchestration without Prefect."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.configuration.loader import load_batch
from src.workflows.reporting import ExperimentResult
from src.workflows.utils.hashing import compute_directory_hash
from src.cli.data import load_data_config
from src.generation import process_config
from src.workflows.utils.paths import extract_model_name
from src.cli.training import train_model
from src.cli.prediction import run_inference
from src.cli.comparison import compare_preconditioners
from src.validation import validate_data_exists
from src.system_loading import get_latest_checkpoint


def run_training_stage(
    *, model_config_path: Path, data_config_path: Path, output_dir: Path, session_name: str
) -> Path:
    return train_model(
        config_path=model_config_path,
        data_config_path=data_config_path,
        output_dir=output_dir,
        session_name=session_name,
    )


def run_prediction_stage(
    *, model_config_path: Path, data_config_path: Path, checkpoint_path: Path, figures_dir: Path
) -> None:
    run_inference(
        config_path=model_config_path,
        data_config_path=data_config_path,
        checkpoint_path=checkpoint_path,
        figures_dir=figures_dir,
    )


def run_comparison_stage(
    *,
    model_config_path: Path,
    data_config_path: Path,
    solver_config_path: Path,
    matrix_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    figures_dir: Path,
) -> None:
    compare_preconditioners(
        config_path=model_config_path,
        data_config_path=data_config_path,
        solver_config_path=solver_config_path,
        matrix_path=matrix_path,
        checkpoint_path=checkpoint_path,
        save_plots=True,
        output_dir=output_dir,
        figures_dir=figures_dir,
    )


def run_experiment(
    *,
    model_config_path: Path,
    data_config_path: Path,
    solver_config_path: Path,
    output_root: Path,
    force: bool,
    src_hash: str,
) -> ExperimentResult:
    """Run data -> training -> prediction -> comparison sequentially."""
    model_name = extract_model_name(model_config_path)
    try:
        data_cfg = load_data_config(data_config_path)
        data_dir = process_config(data_cfg, config_path=data_config_path)
        validate_data_exists(data_dir, ["normalized.npz"])

        checkpoint_dir = output_root / data_config_path.stem / model_name / "checkpoints"
        checkpoint = get_latest_checkpoint(checkpoint_dir)
        if force or checkpoint is None:
            checkpoint = run_training_stage(
                model_config_path=model_config_path,
                data_config_path=data_config_path,
                output_dir=output_root,
                session_name=model_name,
            )

        run_prediction_stage(
            model_config_path=model_config_path,
            data_config_path=data_config_path,
            checkpoint_path=checkpoint,
            figures_dir=output_root / data_config_path.stem / model_name / "figures",
        )

        run_comparison_stage(
            model_config_path=model_config_path,
            data_config_path=data_config_path,
            solver_config_path=solver_config_path,
            matrix_path=data_dir / "normalized.npz",
            checkpoint_path=checkpoint,
            output_dir=output_root / data_config_path.stem / model_name,
            figures_dir=output_root / data_config_path.stem / model_name / "figures",
        )

        return ExperimentResult(experiment_id=model_name, status="Success")
    except Exception as exc:  # noqa: BLE001
        return ExperimentResult(experiment_id=model_name, status="Failed", error=str(exc))


def run_experiment_matrix(
    experiments_config_path: Path,
    *,
    force: bool = False,
    project_root: Path | None = None,
) -> list[ExperimentResult]:
    """Run all experiments defined in experiments.toml without Prefect."""
    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]

    batch = load_batch(experiments_config_path)
    experiments = batch.experiments

    src_dir = project_root / "src"
    if not src_dir.exists():
        src_dir = Path(__file__).resolve().parent.parent
    src_hash = compute_directory_hash(src_dir)

    results: list[ExperimentResult] = []
    for exp in experiments:
        result = run_experiment(
            model_config_path=exp.spec.model_config_path,
            data_config_path=exp.spec.data_config_path,
            solver_config_path=exp.spec.solver_config_path,
            output_root=batch.global_output_dir,
            force=force,
            src_hash=src_hash,
        )
        results.append(result)
    return results
