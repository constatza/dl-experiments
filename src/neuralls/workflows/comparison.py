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


def _resolve_neural_preconditioners(
    solver_specs: list,
    experiments_map: dict[str, Any],
) -> list:
    """Resolve checkpoints for neural preconditioners from experiment references.

    For each neural solver with 'experiment' field:
    1. Look up experiment by ID in experiments_map
    2. Get checkpoint_path from that experiment
    3. Return updated solver spec with resolved checkpoint

    Args:
        solver_specs: List of SolverSpecConfig from solver.toml
        experiments_map: Dict mapping experiment IDs to RunnableExperiment objects

    Returns:
        List of solver specs with resolved checkpoints

    Raises:
        ValueError: If experiment reference is invalid
        FileNotFoundError: If checkpoint not found for referenced experiment
    """
    resolved_specs = []

    for spec in solver_specs:
        if spec.type != "neural":
            # Non-neural solvers don't need checkpoint resolution
            resolved_specs.append(spec)
            continue

        # Neural solver must have either checkpoint_path OR experiment reference
        if spec.checkpoint_path:
            # Explicit checkpoint path (for cross-experiment reuse or external checkpoints)
            resolved_specs.append(spec)
        elif spec.experiment:
            # Reference to experiment ID
            exp_id = spec.experiment
            if exp_id not in experiments_map:
                raise ValueError(
                    f"Neural solver '{spec.name}' references unknown experiment '{exp_id}'. "
                    f"Available experiments: {list(experiments_map.keys())}"
                )

            experiment = experiments_map[exp_id]
            checkpoint = experiment.spec.checkpoint_path

            if not checkpoint or not checkpoint.exists():
                raise FileNotFoundError(
                    f"No checkpoint found for experiment '{exp_id}' "
                    f"(referenced by solver '{spec.name}'). "
                    f"Expected: {checkpoint}"
                )

            # Create updated spec with resolved checkpoint
            resolved_spec = spec.model_copy(update={"checkpoint_path": checkpoint})
            resolved_specs.append(resolved_spec)
        else:
            raise ValueError(
                f"Neural solver '{spec.name}' must specify either 'checkpoint_path' or 'experiment'"
            )

    return resolved_specs


def run_comparisons(
    specs: Iterable[ComparisonSpec],
    params: ComparisonParams,
) -> list[ComparisonOutcome]:
    """Run comparisons with checkpoint resolution from experiment references.

    Args:
        specs: Comparison specifications
        params: Comparison parameters

    Returns:
        List of comparison outcomes
    """
    # Load experiments map for checkpoint resolution
    # This allows solver configs to reference experiments by ID
    try:
        from neuralls.configuration.loader import load_batch
        from neuralls.constants import DEFAULT_PROJECT_ROOT
        experiments_toml = DEFAULT_PROJECT_ROOT / "configs" / "experiments.toml"
        batch = load_batch(experiments_toml)
        experiments_map = {exp.spec.id: exp for exp in batch.experiments}
    except Exception as e:
        logger.warning(f"Could not load experiments map for checkpoint resolution: {e}")
        experiments_map = {}

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

            # Resolve checkpoints for neural preconditioners that reference experiments
            if experiments_map:
                resolved_specs = _resolve_neural_preconditioners(
                    solver_cfg.solvers,
                    experiments_map,
                )
            else:
                resolved_specs = solver_cfg.solvers

            # Use Pydantic SolverConfigFile directly
            precond_configs = build_preconditioner_configs_from_specs(resolved_specs)
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
