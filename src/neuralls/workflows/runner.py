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
from neuralls.workflows.prediction import run_inference
from neuralls.validation import validate_data_exists
from neuralls.system_loading import get_latest_checkpoint
from neuralls.workflows.compare import compare_preconditioners
from neuralls.io.comparison import load_solver_config
from neuralls.preconditioner_factory import build_preconditioner_configs


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

        run_inference(
            config_path=model_config_path,
            data_config_path=data_config_path,
            checkpoint_path=checkpoint,
            figures_dir=output_root / data_config_path.stem / model_name / "figures",
            output_root=output_root,
            synthetic_benchmark=True,
            solver_config_path=solver_config_path,
        )

        solver_cfg = load_solver_config(solver_config_path)
        normalized_path = data_dir / "normalized.npz"
        # Use Pydantic's model_copy to update paths
        updated_general_params = solver_cfg.general.model_copy(
            update={
                "matrix_path": str(normalized_path),
                "rhs_path": str(normalized_path),
            }
        )
        precond_configs = build_preconditioner_configs(
            [{"name": spec.name, "type": spec.type, **spec.model_dump(exclude={'name', 'type'})}
             for spec in solver_cfg.solvers]
        )
        compare_preconditioners(
            general_params=updated_general_params,
            preconditioner_configs=precond_configs,
            output_root=output_root / data_config_path.stem / model_name,
            figures_root=output_root / data_config_path.stem / model_name / "figures",
        )

        return ExperimentResult(experiment_id=model_name, status="Success")
    except Exception as exc:  # noqa: BLE001
        return ExperimentResult(
            experiment_id=model_name, status="Failed", error=str(exc)
        )


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
