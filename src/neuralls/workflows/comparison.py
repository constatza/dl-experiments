"""Backend helpers for comparison workflows.

This module provides orchestration for running CG solver comparisons across experiments.
It handles:
- Building comparison specifications from experiments.toml
- Resolving neural preconditioner checkpoints from experiment references
- Running solver comparisons with resolved configurations

Architecture:
    Comparison workflows work in two modes:
    1. Batch mode: Load experiments from experiments.toml, resolve neural checkpoints
    2. Direct mode: Use explicit model/data/solver configs

    Neural preconditioners can reference experiments by ID instead of explicit checkpoint paths.
    This module resolves those references by looking up checkpoints from completed experiments.

Key Functions:
    - `run_batch_comparison()`: Run comparison for experiments.toml + solver config
    - `run_comparisons()`: Run comparisons for list of specs (with checkpoint resolution)
    - `build_batch_comparisons()`: Build specs from experiments.toml
    - `_resolve_neural_preconditioners()`: Resolve checkpoint paths from experiment refs

Example:
    >>> from neuralls.workflows.comparison import run_batch_comparison, ComparisonParams
    >>> outcomes = run_batch_comparison(
    ...     Path("configs/experiments.toml"),
    ...     Path("solver-configs/default.toml"),
    ...     ComparisonParams(save_plots=True),
    ... )
    >>> print(f"{sum(o.success for o in outcomes)}/{len(outcomes)} comparisons succeeded")
"""

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


def _validate_neural_preconditioner(spec: Any) -> None:
    """Validate neural preconditioner has required configuration.

    Neural preconditioners need either:
    1. Explicit checkpoint_path - direct path to .ckpt file
    2. Experiment reference - ID of experiment to resolve checkpoint from

    This is a pure validation function with no side effects.

    Args:
        spec: Solver specification (SolverSpecConfig from solver.toml)

    Raises:
        ValueError: If neural solver lacks both checkpoint_path and experiment reference

    Example:
        Valid configs:
        >>> spec1 = SolverSpec(name="neural", type="neural",
        ...                    checkpoint_path="checkpoints/model.ckpt")
        >>> _validate_neural_preconditioner(spec1)  # OK

        >>> spec2 = SolverSpec(name="neural", type="neural",
        ...                    experiment="linear-baseline")
        >>> _validate_neural_preconditioner(spec2)  # OK

        Invalid config:
        >>> spec3 = SolverSpec(name="neural", type="neural")
        >>> _validate_neural_preconditioner(spec3)  # Raises ValueError
    """
    if spec.type != "neural":
        return

    if not spec.checkpoint_path and not spec.experiment:
        raise ValueError(
            f"Neural solver '{spec.name}' must specify "
            "either 'checkpoint_path' or 'experiment'"
        )


def _validate_experiment_reference(
    exp_id: str,
    experiments_map: dict[str, Any],
    solver_name: str,
) -> None:
    """Validate experiment reference exists in experiments map.

    When a neural solver references an experiment by ID, we need to verify
    that experiment exists in the loaded experiments. This provides clear
    error messages with available options if the reference is invalid.

    This is a pure validation function with no side effects.

    Args:
        exp_id: Experiment identifier from solver spec (e.g., "linear-baseline")
        experiments_map: Map of experiment IDs to RunnableExperiment objects
        solver_name: Solver name for error messages (e.g., "neural-precond")

    Raises:
        ValueError: If experiment reference not found, with list of valid experiments

    Example:
        >>> experiments_map = {"linear-baseline": exp1, "gcn-model": exp2}
        >>> _validate_experiment_reference("linear-baseline", experiments_map, "neural")  # OK
        >>> _validate_experiment_reference("invalid", experiments_map, "neural")  # Raises ValueError
    """
    if exp_id not in experiments_map:
        raise ValueError(
            f"Neural solver '{solver_name}' references unknown experiment '{exp_id}'. "
            f"Available experiments: {list(experiments_map.keys())}"
        )


def _resolve_checkpoint_from_experiment(
    exp_id: str,
    experiments_map: dict[str, Any],
    solver_name: str,
) -> Path:
    """Resolve checkpoint path from experiment reference.

    Looks up an experiment by ID and retrieves its checkpoint path.
    The checkpoint must exist on disk or this raises an error.

    This is a pure function that only reads data structures and filesystem.

    Args:
        exp_id: Experiment identifier (must exist in experiments_map)
        experiments_map: Map of experiment IDs to RunnableExperiment objects
        solver_name: Solver name for error messages (e.g., "neural-precond")

    Returns:
        Path to existing checkpoint file (.ckpt)

    Raises:
        FileNotFoundError: If checkpoint doesn't exist or wasn't set

    Example:
        >>> experiments_map = {"linear": RunnableExperiment(...)}
        >>> checkpoint = _resolve_checkpoint_from_experiment(
        ...     "linear", experiments_map, "neural"
        ... )
        >>> print(checkpoint)
        Path('output/collect-504/linear/.../checkpoints/epoch=9.ckpt')
    """
    experiment = experiments_map[exp_id]
    checkpoint = experiment.spec.checkpoint_path

    if not checkpoint or not checkpoint.exists():
        raise FileNotFoundError(
            f"No checkpoint found for experiment '{exp_id}' "
            f"(referenced by solver '{solver_name}'). "
            f"Expected: {checkpoint}"
        )

    return checkpoint


def _resolve_preconditioner(
    spec: Any,
    experiments_map: dict[str, Any],
) -> Any:
    """Resolve single preconditioner configuration.

    This is a pure transformation function that handles three cases:
    1. Non-neural solvers (jacobi, ilu, etc.) - return unchanged
    2. Neural with explicit checkpoint_path - return unchanged
    3. Neural with experiment reference - resolve checkpoint and inject path

    The function uses single-responsibility helpers for validation and resolution,
    making it easy to test and understand each step.

    Args:
        spec: Solver specification (SolverSpecConfig from solver.toml)
        experiments_map: Map of experiment IDs to RunnableExperiment objects

    Returns:
        Resolved solver specification (unchanged or with checkpoint_path injected)

    Raises:
        ValueError: If neural solver configuration is invalid
        FileNotFoundError: If checkpoint not found for referenced experiment

    Example:
        >>> # Case 1: Non-neural passes through
        >>> jacobi_spec = SolverSpec(name="jacobi", type="jacobi")
        >>> resolved = _resolve_preconditioner(jacobi_spec, {})
        >>> assert resolved == jacobi_spec

        >>> # Case 2: Neural with checkpoint passes through
        >>> neural_spec = SolverSpec(name="neural", type="neural",
        ...                          checkpoint_path="model.ckpt")
        >>> resolved = _resolve_preconditioner(neural_spec, {})
        >>> assert resolved == neural_spec

        >>> # Case 3: Neural with experiment reference gets resolved
        >>> neural_ref = SolverSpec(name="neural", type="neural",
        ...                         experiment="linear-baseline")
        >>> experiments = {"linear-baseline": RunnableExperiment(...)}
        >>> resolved = _resolve_preconditioner(neural_ref, experiments)
        >>> assert resolved.checkpoint_path is not None
    """
    # Case 1: Non-neural solvers pass through unchanged
    if spec.type != "neural":
        return spec

    # Case 2: Neural with explicit checkpoint passes through unchanged
    if spec.checkpoint_path:
        return spec

    # Case 3: Neural with experiment reference needs resolution
    _validate_neural_preconditioner(spec)

    exp_id = spec.experiment
    _validate_experiment_reference(exp_id, experiments_map, spec.name)

    checkpoint = _resolve_checkpoint_from_experiment(
        exp_id, experiments_map, spec.name
    )

    return spec.model_copy(update={"checkpoint_path": checkpoint})


def _resolve_neural_preconditioners(
    solver_specs: list,
    experiments_map: dict[str, Any],
) -> list[PreconditionerConfig]:
    """Resolve checkpoints for neural preconditioners from experiment references.

    This is a pure mapping function that applies _resolve_preconditioner to each
    spec in the list. It uses functional iteration (list comprehension) for clarity.

    The function handles mixed solver types:
    - Non-neural solvers (jacobi, ilu) pass through unchanged
    - Neural with explicit checkpoints pass through unchanged
    - Neural with experiment refs get checkpoints resolved and injected

    Args:
        solver_specs: List of SolverSpecConfig from solver.toml [solvers] section
        experiments_map: Dict mapping experiment IDs to RunnableExperiment objects

    Returns:
        List of solver specs with all neural checkpoints resolved

    Raises:
        ValueError: If any experiment reference is invalid
        FileNotFoundError: If any checkpoint not found for referenced experiment

    Example:
        >>> solver_specs = [
        ...     SolverSpec(name="jacobi", type="jacobi"),  # Unchanged
        ...     SolverSpec(name="neural", type="neural",
        ...                experiment="linear-baseline"),  # Gets resolved
        ... ]
        >>> experiments = {"linear-baseline": RunnableExperiment(...)}
        >>> resolved = _resolve_neural_preconditioners(solver_specs, experiments)
        >>> assert resolved[0] == solver_specs[0]  # jacobi unchanged
        >>> assert resolved[1].checkpoint_path is not None  # neural resolved
    """
    return [
        _resolve_preconditioner(spec, experiments_map)
        for spec in solver_specs
    ]


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
        except (FileNotFoundError, ValueError, OSError) as e:
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
        except (ValueError, RuntimeError, OSError, FileNotFoundError) as exc:
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
