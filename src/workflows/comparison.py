"""Backend helpers for comparison workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Any

from loguru import logger

from src.configuration.loader import load_batch
from src.configuration.services import WorkspaceFactory
from src.configuration.domain import ExperimentWorkspace
from src.constants import DEFAULT_OUTPUT_DIR, DEFAULT_PROCESSED_DATA_DIR
from src.workflows.checkpoints import resolve_checkpoint
from src.workflows.specs import (
    ComparisonSpec,
    ComparisonParams,
    ComparisonOutcome,
    PreconditionerLimits,
)
from src.workflows.utils.paths import extract_model_name
from src.cli.comparison import compare_preconditioners
from src.workflows.utils.paths import resolve_output_root


def _make_workspace(
    output_root: Path, processed_root: Path, data_id: str, model_name: str
) -> ExperimentWorkspace:
    factory = WorkspaceFactory(output_root, processed_root)
    workspace = factory.create(data_id, model_name)
    workspace.root_dir.mkdir(parents=True, exist_ok=True)
    workspace.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return workspace


def _build_batch_spec(
    exp: Any,
    checkpoint_path: Path | None,
    checkpoint_config: Path | None,
    matrix: Path | None,
    rhs: Path | None,
) -> ComparisonSpec | None:
    checkpoint = resolve_checkpoint(
        explicit=checkpoint_path,
        config_file=checkpoint_config,
        solver_config=exp.spec.solver_config_path,
        checkpoint_dir=exp.workspace.checkpoint_dir,
    )
    if checkpoint is None:
        raise FileNotFoundError(
            f"No checkpoint found for experiment '{exp.spec.id}'. "
            f"Checked solver config and {exp.workspace.checkpoint_dir}."
        )
    default_matrix = matrix or exp.workspace.data_dir / "normalized.npz"
    return ComparisonSpec(
        name=exp.spec.id,
        model_config=exp.spec.model_config_path,
        data_config=exp.spec.data_config_path,
        solver_config=exp.spec.solver_config_path,
        workspace=exp.workspace,
        checkpoint=checkpoint,
        matrix_override=default_matrix,
        rhs_override=rhs,
        figures_dir=exp.workspace.figures_dir,
        output_dir=exp.workspace.root_dir,
    )


def _build_direct_spec(
    name: str,
    model_config: Path,
    data_config: Path,
    solver_config: Path,
    workspace: ExperimentWorkspace,
    checkpoint: Path,
    matrix: Path | None,
    rhs: Path | None,
) -> ComparisonSpec:
    default_matrix = matrix or workspace.data_dir / "normalized.npz"
    return ComparisonSpec(
        name=name,
        model_config=model_config,
        data_config=data_config,
        solver_config=solver_config,
        workspace=workspace,
        checkpoint=checkpoint,
        matrix_override=default_matrix,
        rhs_override=rhs,
        figures_dir=workspace.figures_dir,
        output_dir=workspace.root_dir,
    )


def build_batch_comparisons(
    experiments_config: Path,
    *,
    checkpoint_path: Path | None = None,
    checkpoint_config: Path | None = None,
    matrix: Path | None = None,
    rhs: Path | None = None,
) -> list[ComparisonSpec]:
    batch = load_batch(experiments_config)
    specs: list[ComparisonSpec] = []
    for exp in batch.experiments:
        spec = _build_batch_spec(
            exp,
            checkpoint_path,
            checkpoint_config,
            matrix,
            rhs,
        )
        if spec:
            specs.append(spec)
    return specs


def build_direct_comparisons(
    *,
    model_config: Path,
    data_config: Path,
    solver_config: Path,
    output_root: Path | None,
    processed_root: Path | None = None,
    checkpoint_path: Path | None,
    checkpoint_config: Path | None,
    matrix: Path | None,
    rhs: Path | None,
) -> list[ComparisonSpec]:
    resolved_output = resolve_output_root(None) if output_root is None else output_root
    resolved_processed = processed_root or DEFAULT_PROCESSED_DATA_DIR
    model_name = extract_model_name(model_config)
    data_id = data_config.stem
    workspace = _make_workspace(resolved_output, resolved_processed, data_id, model_name)
    checkpoint = resolve_checkpoint(
        explicit=checkpoint_path,
        config_file=checkpoint_config,
        solver_config=solver_config,
        checkpoint_dir=workspace.checkpoint_dir,
    )
    if checkpoint is None:
        return []
    return [_build_direct_spec(model_name, model_config, data_config, solver_config, workspace, checkpoint, matrix, rhs)]


def run_comparisons(
    specs: Iterable[ComparisonSpec], params: ComparisonParams
) -> list[ComparisonOutcome]:
    outcomes: list[ComparisonOutcome] = []
    for spec in specs:
        merged_limits = PreconditionerLimits(
            apply_every=params.limits.apply_every,
            first_n=params.limits.first_n,
            neural_iters=params.limits.neural_iters,
            fallback_preconditioner=params.limits.fallback_preconditioner,
        )
        merged_params = ComparisonParams(
            matrix=params.matrix or spec.matrix_override,
            rhs=params.rhs or spec.rhs_override,
            output_dir=params.output_dir or spec.output_dir or DEFAULT_OUTPUT_DIR,
            figures_dir=params.figures_dir or spec.figures_dir,
            save_plots=params.save_plots,
            breakdown_tol=params.breakdown_tol,
            limits=merged_limits,
            reorthogonalize=params.reorthogonalize,
            reorthog_window=params.reorthog_window,
            reorthog_threshold=params.reorthog_threshold,
        )
        try:
            result = compare_preconditioners(
                config_path=spec.model_config,
                data_config_path=spec.data_config,
                solver_config_path=spec.solver_config,
                checkpoint_path=spec.checkpoint,
                params=merged_params,
                custom_combinations=None,
            )
            outcomes.append(
                ComparisonOutcome(name=spec.name, success=True, payload=result)
            )
        except Exception as exc:  # noqa: BLE001
            outcomes.append(
                ComparisonOutcome(name=spec.name, success=False, error=str(exc))
            )
    return outcomes
