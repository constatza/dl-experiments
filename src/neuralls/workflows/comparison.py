"""Backend helpers for comparison workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from collections.abc import Iterable

from loguru import logger

from neuralls.constants import DEFAULT_PROCESSED_DATA_DIR
from neuralls.configuration.loader import load_batch
from neuralls.configuration.services import WorkspaceFactory
from neuralls.configuration.domain import ExperimentWorkspace
from neuralls.workflows.checkpoints import resolve_checkpoint
from neuralls.workflows.specs import (
    ComparisonSpec,
    ComparisonParams,
    ComparisonOutcome,
)
from neuralls.workflows.utils.paths import resolve_output_root, extract_model_name
from neuralls.workflows.compare import compare_preconditioners
from neuralls.io.toml_loader import load_solver_config
from neuralls.configuration.preconditioner import PreconditionerConfig


def _make_workspace(
    output_root: Path, processed_root: Path, data_id: str, model_name: str
) -> ExperimentWorkspace:
    factory = WorkspaceFactory(output_root, processed_root)
    workspace = factory.create(data_id, model_name)
    workspace.root_dir.mkdir(parents=True, exist_ok=True)
    workspace.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return workspace


def _build_batch_spec(exp: Any, solver_config: Path) -> ComparisonSpec | None:
    checkpoint = resolve_checkpoint(
        explicit=None,
        config_file=None,
        solver_config=solver_config,
        checkpoint_dir=exp.workspace.checkpoint_dir,
    )
    if checkpoint is None:
        logger.error(
            f"No checkpoint found for experiment '{exp.spec.id}'. "
            f"Checked solver config and {exp.workspace.checkpoint_dir}."
        )
        return None
    return ComparisonSpec(
        name=exp.spec.id,
        model_config=exp.spec.model_config_path,
        data_config=exp.spec.data_config_path,
        solver_config=solver_config,
        workspace=exp.workspace,
        checkpoint=checkpoint,
        matrix_override=None,
        rhs_override=None,
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


def _comparison_output_root(general_params: Any, solver_config_path: Path) -> Path:
    configured_root = getattr(general_params, "output_root", None)
    if configured_root is None:
        raise ValueError("general.output_root is required for comparisons.")
    return Path(configured_root).expanduser().resolve() / solver_config_path.stem


def build_batch_comparisons(
    experiments_config: Path,
    solver_config: Path,
) -> list[ComparisonSpec]:
    batch = load_batch(experiments_config)
    specs: list[ComparisonSpec] = []
    for exp in batch.experiments:
        spec = _build_batch_spec(exp, solver_config)
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
    workspace = _make_workspace(
        resolved_output, resolved_processed, data_id, model_name
    )
    checkpoint = resolve_checkpoint(
        explicit=None,
        config_file=None,
        solver_config=solver_config,
        checkpoint_dir=workspace.checkpoint_dir,
    )
    if checkpoint is None:
        return []
    return [
        _build_direct_spec(
            model_name, model_config, data_config, solver_config, workspace, checkpoint
        )
    ]


def _resolve_neural_preconditioners(
    solver_specs: list,
    experiments_map: dict[str, Any],
) -> list[PreconditionerConfig]:
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

        if not spec.experiment:
            raise ValueError(
                f"Neural solver '{spec.name}' must specify either 'checkpoint_path' or 'experiment'"
            )

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

    return resolved_specs


def run_comparisons(
    specs: Iterable[ComparisonSpec],
    params: ComparisonParams,
    experiments_map: dict[str, Any] | None = None,
) -> list[ComparisonOutcome]:
    """Run comparisons with checkpoint resolution from experiment references.

    A comparison run executes each solver listed in the solver TOML exactly once and
    produces shared diagnostics (plots, summaries) for that solver set. Experiments
    are only used to resolve neural checkpoints; comparisons do not repeat per
    experiment.
    """
    if experiments_map is None:
        try:
            from neuralls.configuration.loader import load_batch
            from neuralls.constants import DEFAULT_PROJECT_ROOT

            experiments_toml = DEFAULT_PROJECT_ROOT / "configs" / "experiments.toml"
            batch = load_batch(experiments_toml)
            experiments_map = {exp.spec.id: exp for exp in batch.experiments}
        except Exception as e:
            logger.warning(
                f"Could not load experiments map for checkpoint resolution: {e}"
            )
            experiments_map = {}

    outcomes: list[ComparisonOutcome] = []
    for spec in specs:
        error: Exception | None = None
        try:
            solver_cfg = load_solver_config(spec.solver_config)
            comparison_root = _comparison_output_root(
                solver_cfg.general, spec.solver_config
            )

            # Resolve checkpoints for neural preconditioners that reference experiments
            if experiments_map:
                resolved_specs = _resolve_neural_preconditioners(
                    solver_cfg.solvers,
                    experiments_map,
                )
            else:
                resolved_specs = solver_cfg.solvers

            # Use Pydantic SolverConfigFile directly
            result = compare_preconditioners(
                general_params=solver_cfg.general,
                preconditioner_configs=resolved_specs,
                output_root=comparison_root,
                save_plots=params.save_plots,
            )
        except Exception as exc:  # noqa: BLE001
            error = exc
        if error:
            outcomes.append(
                ComparisonOutcome(name=spec.name, success=False, error=str(error))
            )
        else:
            outcomes.append(
                ComparisonOutcome(name=spec.name, success=True, payload=result)
            )
    return outcomes


def run_batch_comparison(
    experiments_config: Path, solver_config: Path, params: ComparisonParams
) -> list[ComparisonOutcome]:
    """Run a single aggregated comparison for a solver config.

    Experiments are read only to resolve neural checkpoints; the comparison itself
    runs once for the solver config and writes shared diagnostics under
    general.output_root/<solver-stem>/.
    """
    batch = load_batch(experiments_config)
    if not batch.experiments:
        raise ValueError(
            "No experiments found to resolve checkpoints for neural solvers."
        )

    specs = [_build_batch_spec(exp, solver_config) for exp in batch.experiments]
    specs = [s for s in specs if s is not None]
    if not specs:
        raise ValueError(
            "No checkpoints found for any experiments; cannot run comparison."
        )

    experiments_map = {exp.spec.id: exp for exp in batch.experiments}
    return run_comparisons([specs[0]], params, experiments_map=experiments_map)
