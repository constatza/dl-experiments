"""Backend helpers for comparison workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Any

from loguru import logger

from neuralls.configuration.loader import load_batch
from neuralls.configuration.services import WorkspaceFactory
from neuralls.configuration.domain import ExperimentWorkspace
from neuralls.constants import DEFAULT_OUTPUT_DIR, DEFAULT_PROCESSED_DATA_DIR, DEFAULT_PROJECT_ROOT
from neuralls.workflows.checkpoints import resolve_checkpoint
from neuralls.workflows.specs import (
    ComparisonSpec,
    ComparisonParams,
    ComparisonOutcome,
)
from neuralls.workflows.utils.paths import extract_model_name
from neuralls.workflows.compare import compare_preconditioners
from neuralls.io.comparison import load_solver_config
from neuralls.preconditioner_factory import build_preconditioner_configs_from_specs
from neuralls.workflows.utils.paths import resolve_output_root
from neuralls.mlflow_utils import build_run_config, finalize_run, open_run


COMPARISON_ARTIFACTS: tuple[str, ...] = ("figures", "reports", "metrics")


def _make_workspace(
    output_root: Path, processed_root: Path, data_id: str, model_name: str
) -> ExperimentWorkspace:
    factory = WorkspaceFactory(output_root, processed_root)
    workspace = factory.create(data_id, model_name)
    workspace.root_dir.mkdir(parents=True, exist_ok=True)
    workspace.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return workspace


def _build_batch_spec(exp: Any) -> ComparisonSpec | None:
    checkpoint = resolve_checkpoint(
        explicit=None,
        config_file=None,
        solver_config=exp.spec.solver_config_path,
        checkpoint_dir=exp.workspace.checkpoint_dir,
    )
    if checkpoint is None:
        raise FileNotFoundError(
            f"No checkpoint found for experiment '{exp.spec.id}'. "
            f"Checked solver config and {exp.workspace.checkpoint_dir}."
        )
    return ComparisonSpec(
        name=exp.spec.id,
        model_config=exp.spec.model_config_path,
        data_config=exp.spec.data_config_path,
        solver_config=exp.spec.solver_config_path,
        workspace=exp.workspace,
        checkpoint=checkpoint,
        matrix_override=None,
        rhs_override=None,
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
) -> ComparisonSpec:
    return ComparisonSpec(
        name=name,
        model_config=model_config,
        data_config=data_config,
        solver_config=solver_config,
        workspace=workspace,
        checkpoint=checkpoint,
        matrix_override=None,
        rhs_override=None,
        figures_dir=workspace.figures_dir,
        output_dir=workspace.root_dir,
    )


def _get_mlflow_config(model_config: Path):
    """Load validated model config for MLflow settings.

    Returns ModelConfigFile or None if loading fails.
    """
    try:
        from neuralls.configuration.loaders import load_model_config
        return load_model_config(model_config)
    except Exception:
        return None


def _get_mlflow_enabled(spec: ComparisonSpec) -> bool:
    """Read MLflow enabled setting from model config."""
    # First try settings if available
    if spec.settings and hasattr(spec.settings, "MLFLOW"):
        mlflow_cfg = getattr(spec.settings, "MLFLOW", None)
        if mlflow_cfg:
            return getattr(mlflow_cfg, "enabled", False)

    # Fallback: load config directly using Pydantic model
    model_cfg = _get_mlflow_config(spec.model_config)
    return model_cfg.MLFLOW.enabled if model_cfg else False


def _start_comparison_run(spec: ComparisonSpec, enable_mlflow: bool):
    settings = spec.settings
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
) -> list[ComparisonSpec]:
    batch = load_batch(experiments_config)
    specs: list[ComparisonSpec] = []
    for exp in batch.experiments:
        spec = _build_batch_spec(exp)
        if spec:
            specs.append(spec)
    return specs


def build_direct_comparisons(
    *,
    model_config: Path,
    data_config: Path,
    solver_config: Path,
) -> list[ComparisonSpec]:
    resolved_output = resolve_output_root(None)
    resolved_processed = DEFAULT_PROCESSED_DATA_DIR
    model_name = extract_model_name(model_config)
    data_id = data_config.stem
    workspace = _make_workspace(resolved_output, resolved_processed, data_id, model_name)
    checkpoint = resolve_checkpoint(
        explicit=None,
        config_file=None,
        solver_config=solver_config,
        checkpoint_dir=workspace.checkpoint_dir,
    )
    if checkpoint is None:
        return []
    return [_build_direct_spec(model_name, model_config, data_config, solver_config, workspace, checkpoint)]


def run_comparisons(
    specs: Iterable[ComparisonSpec],
    params: ComparisonParams,
) -> list[ComparisonOutcome]:
    outcomes: list[ComparisonOutcome] = []
    for spec in specs:
        # Read MLflow setting from config
        enable_mlflow = _get_mlflow_enabled(spec)
        mlflow_state = _start_comparison_run(spec, enable_mlflow)
        metrics: dict[str, float] | None = None
        error: Exception | None = None
        result: dict[str, Any] | None = None
        try:
            solver_cfg = load_solver_config(spec.solver_config)
            # Use Pydantic SolverConfigFile directly
            precond_configs = build_preconditioner_configs_from_specs(solver_cfg.solvers)
            result = compare_preconditioners(
                general_params=solver_cfg.general,
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
