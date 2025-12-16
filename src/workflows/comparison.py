"""Backend helpers for comparison workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Any
from types import SimpleNamespace
import tomllib

from loguru import logger

from src.configuration.loader import load_batch
from src.configuration.services import WorkspaceFactory
from src.configuration.domain import ExperimentWorkspace
from src.constants import DEFAULT_OUTPUT_DIR, DEFAULT_PROCESSED_DATA_DIR, DEFAULT_PROJECT_ROOT
from src.workflows.checkpoints import resolve_checkpoint
from src.workflows.specs import (
    ComparisonSpec,
    ComparisonParams,
    ComparisonOutcome,
    PreconditionerLimits,
)
from src.workflows.utils.paths import extract_model_name
from src.workflows.compare import compare_preconditioners
from src.io.comparison import load_solver_config
from src.preconditioner_factory import build_preconditioner_configs
from src.workflows.utils.paths import resolve_output_root
from src.mlflow_utils import build_run_config, finalize_run, open_run


COMPARISON_ARTIFACTS: tuple[str, ...] = ("figures", "reports", "metrics")


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
        settings=exp.settings,
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


def _coerce_mlflow_settings(model_config: Path) -> Any | None:
    try:
        with model_config.open("rb") as handle:
            config = tomllib.load(handle)
        mlflow_cfg = config.get("MLFLOW")
        if not isinstance(mlflow_cfg, dict):
            return None
        client_cfg = mlflow_cfg.get("client") if isinstance(mlflow_cfg.get("client"), dict) else {}
        server_cfg = mlflow_cfg.get("server") if isinstance(mlflow_cfg.get("server"), dict) else {}
        return SimpleNamespace(
            MLFLOW=SimpleNamespace(
                enabled=bool(mlflow_cfg.get("enabled")),
                client=SimpleNamespace(**client_cfg),
                server=SimpleNamespace(**server_cfg),
            ),
            PATHS=SimpleNamespace(project_root=str(DEFAULT_PROJECT_ROOT)),
        )
    except Exception:
        return None


def _start_comparison_run(spec: ComparisonSpec, enable_mlflow: bool):
    settings = spec.settings or _coerce_mlflow_settings(spec.model_config)
    config = build_run_config(
        settings=settings,
        workspace_root=spec.workspace.root_dir,
        dataset_id=spec.data_config.stem,
        model_name=f"{spec.name}-compare",
        enabled=enable_mlflow,
    )
    if config is None:
        return None
    try:
        return open_run(config)
    except ModuleNotFoundError:
        logger.info("MLflow not installed; skipping MLflow logging.")
        return None


def _comparison_metrics(result: dict[str, Any] | None) -> dict[str, float]:
    if not result:
        return {}
    recs = result.get("recommendations") if isinstance(result, dict) else None
    best = recs.get("best_overall") if isinstance(recs, dict) else None
    if not isinstance(best, dict):
        return {}
    metrics: dict[str, float] = {}
    for key, target in (("residual", "best_residual"), ("iterations", "best_iterations")):
        value = best.get(key)
        try:
            metrics[target] = float(value)
        except (TypeError, ValueError):
            continue
    return metrics


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
    specs: Iterable[ComparisonSpec],
    params: ComparisonParams,
    *,
    enable_mlflow: bool = False,
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
        mlflow_state = _start_comparison_run(spec, enable_mlflow)
        metrics: dict[str, float] | None = None
        error: Exception | None = None
        result: dict[str, Any] | None = None
        try:
            general_params, solver_entries = load_solver_config(spec.solver_config)
            precond_configs = build_preconditioner_configs(
                [
                    {"name": solver.name, "type": solver.type, **solver.args}
                    for solver in solver_entries
                ]
            )
            result = compare_preconditioners(
                general_params=general_params,
                preconditioner_configs=precond_configs,
                output_root=spec.output_dir or spec.workspace.root_dir,
                figures_root=spec.figures_dir or spec.workspace.figures_dir,
            )
        except Exception as exc:  # noqa: BLE001
            error = exc
        finally:
            metrics = _comparison_metrics(result)
            finalize_run(
                mlflow_state,
                metrics=metrics,
                workspace_root=spec.workspace.root_dir,
                allowlist=COMPARISON_ARTIFACTS,
                failed=error is not None,
            )
        if error:
            outcomes.append(
                ComparisonOutcome(name=spec.name, success=False, error=str(error))
            )
        else:
            outcomes.append(
                ComparisonOutcome(name=spec.name, success=True, payload=result)
            )
    return outcomes
