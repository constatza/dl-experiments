"""Simplified preconditioner comparison workflow driven by comparison config.

This module provides the core comparison workflow for benchmarking CG solver
preconditioners. It follows clean architecture principles:
- Domain models for data structures (ComparisonPaths, LinearSystem)
- Service pattern for preconditioner creation (PreconditionerService)
- Single-responsibility helper functions
- Pure orchestration in main function

Architecture:
    The comparison workflow consists of 9 clear steps:
    1. Validate inputs
    2. Resolve paths (matrix, rhs, output, figures)
    3. Load and validate linear system
    4. Create preconditioners via service (uses factory function)
    5. Compute condition numbers for diagnostics
    6. Wrap preconditioners with scheduling (iteration limits, fallbacks)
    7. Run CG comparison
    8. Generate plots
    9. Return result

Key Components:
    - `compare_preconditioners()`: Main entry point (orchestration only)
    - `PreconditionerService`: Centralized preconditioner creation
    - `ComparisonPaths`: Type-safe path resolution
    - `LinearSystem`: Validated system matrices

Example:
    >>> from neuralls.composition.comparison.single_run import compare_preconditioners
    >>> from neuralls.platform.config.models.comparison import (
    ...     ComparisonGeneral,
    ...     SolverParams,
    ...     ComparisonData,
    ... )
    >>> from neuralls.platform.config.models.preconditioner import StandardPreconditionerConfig
    >>>
    >>> general = ComparisonGeneral(
    ...     params=SolverParams(
    ...         rtol=1e-6,
    ...         atol=1e-14,
    ...         max_iterations=100,
    ...         stopping_criterion="residual_norm",
    ...         m_max=20,
    ...         breakdown_tol=None,
    ...     ),
    ...     data=ComparisonData(matrix_path="data/matrix.txt", rhs_path="data/rhs.txt"),
    ...     tracking=None,
    ...     model_store=None,
    ... )
    >>> configs = [
    ...     StandardPreconditionerConfig(name="jacobi", type="jacobi"),
    ...     StandardPreconditionerConfig(name="ilu", type="ilu"),
    ... ]
    >>> result = compare_preconditioners(
    ...     general_params=general,
    ...     preconditioner_configs=configs,
    ... )
    >>> print(result.summary)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast
from collections.abc import Sequence

import numpy as np
from loguru import logger

from neuralls.platform.config.models.preconditioner import PreconditionerConfig
from neuralls.platform.config.models.comparison import ComparisonGeneral
from neuralls.domain.analysis.spectra import (
    PreconditionerCallable,
    compute_condition_numbers,
    plot_condition_numbers,
)
from neuralls.platform.config.resolution import resolve_user_path
from neuralls.platform.storage.filesystem import ensure_dir
import zarr

from neuralls.platform.storage.comparison import load_system_arrays
from neuralls.platform.storage.training_artifacts import load_training_arrays
from neuralls.domain.solver.preconditioners.base import BindableInputs, Preconditioner
from neuralls.domain.linalg import compute_condition_number
from neuralls.domain.normalization import IScale, create_scale_from_config
from neuralls.platform.reporting.plots import plot_convergence_comparison, plot_metric_comparison
from neuralls.composition.preconditioners.factory import (
    create_preconditioner,
    create_scheduled_preconditioner,
    PreconditionerScheduleConfig,
)
from neuralls.domain.solver.comparison import (
    format_results_summary,
    run_cg_comparison,
)
from neuralls.domain.solver.utils.validation import validate_matrix, validate_rhs_vector
from neuralls.domain.solver.models.result import CGComparisonResult
from neuralls.domain.solver.models.result import (
    ComparisonRecommendations,
    ComparisonResult,
    PlotPaths,
)


# Arrays always available from LinearSystem — never need loading from disk.
_SYSTEM_ARRAYS_ALWAYS_AVAILABLE: frozenset[str] = frozenset({"matrix"})

StoppingCriterion = Literal["tolerance", "fixed_iterations"]


@dataclass(frozen=True)
class ComparisonPaths:
    """Resolved paths for comparison workflow.

    Immutable container for all paths needed during comparison.
    Using a dataclass instead of passing 4 Path arguments improves:
    - Type safety (can't mix up arguments)
    - Extensibility (can add fields without changing signatures)
    - Clarity (self-documenting structure)

    Attributes:
        matrix: Path to system matrix file (.txt or .npy)
        rhs: Path to right-hand side vector file (.txt or .npy)
        output: Root output directory for comparison results
        figures: Directory for diagnostic plots
    """

    matrix: Path
    rhs: Path
    output: Path
    figures: Path


@dataclass(frozen=True)
class LinearSystem:
    """Loaded and validated linear system.

    Immutable container for validated system arrays.
    Matrices and vectors have been validated for:
    - Correct shapes (n x n matrix, n-length vector)
    - Compatibility (rhs matches matrix dimensions)
    - Numeric properties (no NaN/inf values)

    Attributes:
        matrix: System matrix A in Ax=b (shape: n x n)
        rhs: Right-hand side vector b in Ax=b (shape: n,)
    """

    matrix: np.ndarray
    rhs: np.ndarray


class PreconditionerService:
    """Service for creating and managing preconditioners.

    This service centralizes preconditioner creation using the factory function.

    The service pattern provides:
    - Single source of truth for preconditioner creation
    - Clear dependency injection point (can inject custom adapter for testing)
    - Consistent preconditioner creation

    Example:
        >>> service = PreconditionerService()
        >>> jacobi_config = StandardPreconditionerConfig(name="jacobi", type="jacobi")
        >>> matrix = np.eye(100)
        >>> jacobi_precond = service.create_preconditioner(matrix, jacobi_config)
        >>> precond_set = service.create_preconditioner_set(matrix, [jacobi_config, ...])
    """

    def __init__(self, adapter=None):
        """Initialize service.

        Args:
            adapter: Optional adapter for neural preconditioner (DI for testing)
        """
        self._adapter = adapter

    def create_preconditioner(
        self,
        matrix: np.ndarray,
        config: PreconditionerConfig,
    ) -> Preconditioner:
        """Create a single preconditioner.

        Args:
            matrix: System matrix to precondition (shape: n x n)
            config: Preconditioner configuration (type, parameters, etc.)

        Returns:
            Preconditioner instance

        Example:
            >>> service = PreconditionerService()
            >>> A = np.array([[4, 1], [1, 3]])
            >>> config = StandardPreconditionerConfig(name="jacobi", type="jacobi")
            >>> M = service.create_preconditioner(A, config)
            >>> r = np.array([1.0, 2.0])
            >>> z = M.apply(r)  # Apply preconditioner
        """
        return create_preconditioner(matrix, config, adapter=self._adapter)

    def create_preconditioner_set(
        self,
        matrix: np.ndarray,
        configs: Sequence[PreconditionerConfig],
    ) -> dict[str, Preconditioner]:
        """Create multiple preconditioners for comparison.

        Args:
            matrix: System matrix to precondition (same for all)
            configs: Sequence of preconditioner configurations

        Returns:
            Dictionary mapping preconditioner names to Preconditioner instances

        Example:
            >>> service = PreconditionerService()
            >>> A = np.eye(100)
            >>> configs = [
            ...     StandardPreconditionerConfig(name="jacobi", type="jacobi"),
            ...     StandardPreconditionerConfig(name="ilu", type="ilu"),
            ... ]
            >>> preconditioners = service.create_preconditioner_set(A, configs)
            >>> print(list(preconditioners.keys()))
            ['jacobi', 'ilu']
        """
        return {cfg.name: self.create_preconditioner(matrix, cfg) for cfg in configs}


# Registry for stopping criterion name mappings
# Easy to extend: just add new mappings to this dict
_STOPPING_CRITERION_REGISTRY: dict[str, StoppingCriterion] = {
    "tolerance": "tolerance",
    "residual_norm": "tolerance",  # Config name -> internal name
    "fixed": "fixed_iterations",
    "fixed_iterations": "fixed_iterations",
}


def register_stopping_criterion(name: str, criterion: StoppingCriterion) -> None:
    """Register a stopping criterion name mapping.

    Args:
        name: The criterion name (e.g., "my_criterion")
        criterion: The typed literal ("tolerance" or "fixed_iterations")
    """
    _STOPPING_CRITERION_REGISTRY[name.lower()] = criterion


def _map_stopping_criterion(name: str) -> StoppingCriterion:
    """Map string stopping criterion to typed literal.

    Args:
        name: Stopping criterion name from config (e.g., "tolerance", "fixed")

    Returns:
        Typed literal "tolerance" or "fixed_iterations"

    Raises:
        ValueError: If the criterion name is not registered
    """
    normalized = name.lower()
    criterion = _STOPPING_CRITERION_REGISTRY.get(normalized)
    if criterion is None:
        valid_names = ", ".join(sorted(_STOPPING_CRITERION_REGISTRY.keys()))
        raise ValueError(f"Unknown stopping criterion: '{name}'. Valid options: {valid_names}")
    return criterion


def _resolve_comparison_paths(
    *,
    general_params: ComparisonGeneral,
    output_root: Path | None,
    figures_root: Path | None,
) -> ComparisonPaths:
    """Resolve all paths for comparison workflow.

    This is a pure function that validates and resolves paths from config.
    It ensures all required paths are set and handles optional overrides.

    Args:
        general_params: Comparison general configuration with params+data context
        output_root: Optional override for output root directory
        figures_root: Optional override for figures directory

    Returns:
        ComparisonPaths with all resolved and validated paths

    Raises:
        ValueError: If matrix_path/rhs_path are missing or invalid

    Example:
        >>> general = ComparisonGeneral(
        ...     params=SolverParams(
        ...         rtol=1e-6,
        ...         atol=1e-14,
        ...         max_iterations=100,
        ...         stopping_criterion="residual_norm",
        ...         m_max=20,
        ...         breakdown_tol=None,
        ...     ),
        ...     data=ComparisonData(matrix_path="data/matrix.txt", rhs_path="data/rhs.txt"),
        ...     tracking=None,
        ...     model_store=None,
        ... )
        >>> paths = _resolve_comparison_paths(
        ...     general_params=general,
        ...     output_root=None,
        ...     figures_root=None,
        ... )
        >>> print(paths.figures)
        Path('output/figures')
    """
    matrix_file = Path(general_params.data.matrix_path)
    rhs_file = Path(general_params.data.rhs_path)

    if output_root is not None:
        # Caller supplies exact directory — use as-is (no timestamp suffix)
        output_base = resolve_user_path(output_root)
    else:
        output_base = (Path.cwd() / "comparison" / matrix_file.stem).resolve()

    figs_base = Path(figures_root) if figures_root else output_base / "figures"

    return ComparisonPaths(
        matrix=matrix_file,
        rhs=rhs_file,
        output=output_base,
        figures=figs_base,
    )


def _ensure_comparison_directories(paths: ComparisonPaths) -> None:
    """Ensure the figures directory exists.

    Only the figures subdirectory needs explicit creation; the output root
    is either caller-supplied or an MLflow-managed artifact directory.
    ``mkdir(parents=True)`` inside ``ensure_dir`` creates any intermediate
    directories (including ``paths.output``) as a side effect.

    Args:
        paths: Comparison paths with output and figures directories
    """
    ensure_dir(paths.figures)


def _normalize_linear_system(
    matrix: np.ndarray,
    rhs: np.ndarray,
    normalize_system: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply comparison-time normalization to matrix and RHS."""
    if normalize_system == "none":
        return matrix, rhs
    if normalize_system == "rhs":
        rhs_norm = float(np.linalg.norm(rhs))
        if rhs_norm == 0.0:
            return matrix, rhs
        return matrix, rhs / rhs_norm
    if normalize_system == "both":
        matrix, rhs = _normalize_linear_system(matrix, rhs, "matrix")
        return _normalize_linear_system(matrix, rhs, "rhs")

    strategy = normalize_system
    if strategy not in {"matrix", "diagonal", "spectral"}:
        raise ValueError(
            f"Unsupported normalize_system value: {normalize_system!r}. "
            "Expected one of none, matrix, rhs, both, diagonal, spectral."
        )

    rhs_samples = rhs.reshape(1, -1) if strategy == "spectral" else None
    scale_strategy = cast(Literal["none", "matrix", "spectral", "diagonal"], strategy)
    scale = create_scale_from_config(scale_strategy, matrix, rhs_samples=rhs_samples)
    if scale is None:
        return matrix, rhs
    if isinstance(scale, list):
        if not scale:
            return matrix, rhs
        scale = scale[0]
    if not isinstance(scale, IScale):
        raise TypeError(f"Expected IScale, got {type(scale).__name__}")
    return scale.scale_matrix(matrix), scale.scale_rhs(rhs)


def _load_linear_system(
    paths: ComparisonPaths,
    *,
    rhs_index: int,
    matrix_index: int,
    normalize_system: str,
) -> LinearSystem:
    """Load and validate linear system.

    This function:
    1. Loads matrix and rhs from files (txt or npy)
    2. Validates matrix properties (square, no NaN/inf, etc.)
    3. Validates rhs compatibility with matrix dimensions
    4. Returns validated LinearSystem

    Args:
        paths: Comparison paths with matrix and rhs file locations.
        rhs_index: Row index to select from a multi-RHS file; -1 selects row 0.
        matrix_index: Sample index to select from a multi-matrix dataset directory.
        normalize_system: Scaling mode — one of ``"none"``, ``"matrix"``,
            ``"rhs"``, ``"both"``, ``"diagonal"``, ``"spectral"``.

    Returns:
        LinearSystem: Validated (A, b) pair, scaled and shaped.

    Raises:
        ValueError: If validation fails (wrong shape, NaN values, incompatible dimensions).
        FileNotFoundError: If matrix or rhs files don't exist.
    """
    A, b = load_system_arrays(
        paths.matrix, paths.rhs, rhs_index=rhs_index, matrix_index=matrix_index
    )
    A, b = _normalize_linear_system(A, b, normalize_system)
    validate_matrix(A)
    validate_rhs_vector(b, A)
    return LinearSystem(matrix=A, rhs=b)


def _log_matrix_condition_number(
    matrix: np.ndarray,
    *,
    matrix_path: Path,
    display_name: str | None,
) -> None:
    """Log the matrix condition number once per comparison."""
    label = display_name or matrix_path.stem or matrix_path.name
    try:
        value = float(compute_condition_number(matrix))
    except np.linalg.LinAlgError as exc:
        logger.warning(
            f"Matrix condition number unavailable: comparison={label} "
            f"matrix={matrix_path.name} error={exc}"
        )
        return
    logger.info(
        f"Matrix condition number: comparison={label} matrix={matrix_path.name} value={value:.4e}"
    )


def _create_scheduled_preconditioners(
    preconditioner_configs: Sequence[PreconditionerConfig],
    matrix: np.ndarray,
    base_preconditioners: dict[str, Any],
) -> dict[str, Preconditioner]:
    """Wrap preconditioners with scheduling if configured.

    Args:
        preconditioner_configs: Config objects with scheduling parameters
        matrix: System matrix (needed for fallback creation)
        base_preconditioners: Already-created preconditioner instances

    Returns:
        Dict mapping names to scheduled preconditioner instances
    """
    scheduled: dict[str, Preconditioner] = {}

    for cfg in preconditioner_configs:
        primary = base_preconditioners[cfg.name]

        # Create schedule config from preconditioner config
        schedule = PreconditionerScheduleConfig(
            limit_iters=cfg.limit_iters,
            fallback=cfg.fallback,
        )

        # Wrap with scheduling if limit_iters is specified
        scheduled[cfg.name] = create_scheduled_preconditioner(
            primary=primary,
            schedule=schedule,
        )

    return scheduled


def _generate_comparison_plots(
    results: dict[str, CGComparisonResult],
    cond_numbers: dict[str, float],
    paths: ComparisonPaths,
    display_name: str | None = None,
    rtol: float | None = None,
    atol: float | None = None,
    max_iterations: int | None = None,
) -> PlotPaths:
    """Generate diagnostic plots for comparison.

    Args:
        results: CG comparison results
        cond_numbers: Condition numbers
        paths: Comparison paths
        display_name: Optional display name for plots
        rtol: Optional relative tolerance parameter to display
        atol: Optional absolute tolerance parameter to display
        max_iterations: Optional max iterations parameter to display

    Returns:
        Typed plot paths.
    """
    suffix = paths.matrix.stem or "comparison"
    cond_path = plot_condition_numbers(
        cond_numbers,
        save_dir=paths.figures,
        suffix=suffix,
        title=display_name,
        rtol=rtol,
        atol=atol,
    )

    convergence_path = paths.figures / f"preconditioner_convergence_{suffix}.png"
    plot_convergence_comparison(
        results,
        metadata=None,
        save_path=convergence_path,
        title=display_name,
        rtol=rtol,
        atol=atol,
        max_iterations=max_iterations,
    )

    # Generate horizontal iterations barplot
    iter_path = paths.figures / f"preconditioner_iterations_{suffix}.png"
    plot_metric_comparison(
        list(results.keys()),
        [r.iterations for r in results.values()],
        metric_name="CG Iterations",
        title=display_name,
        horizontal=True,
        save_path=iter_path,
    )

    return PlotPaths(
        convergence=convergence_path,
        condition_numbers=cond_path,
        iterations_barplot=iter_path,
    )


def _bind_system_inputs(
    preconditioners: dict[str, Preconditioner],
    system_data: dict[str, np.ndarray],
) -> None:
    """Bind dataset-sourced extra inputs to preconditioners that declare them.

    Reads extra_input_names from each preconditioner. For each name present in
    system_data, calls bind_inputs() so the preconditioner can forward them to
    the model without changing the CG-facing apply(residual) interface.

    Args:
        preconditioners: Map of name to preconditioner.
        system_data: Available named arrays (always includes "matrix").
    """
    for precond in preconditioners.values():
        if not isinstance(precond, BindableInputs):
            continue
        needed = {k: v for k, v in system_data.items() if k in precond.extra_input_names}
        if needed:
            precond.bind_inputs(**needed)


def compare_preconditioners(
    *,
    general_params: ComparisonGeneral,
    preconditioner_configs: Sequence[PreconditionerConfig],
    output_root: Path | None = None,
    figures_root: Path | None = None,
    display_name: str | None = None,
) -> ComparisonResult:
    """Run CG comparisons and generate diagnostics.

    This is the main entry point for preconditioner comparison. It orchestrates
    an 11-step workflow using single-responsibility helpers.

    The function is pure orchestration - no business logic here, only composition.
    Each step delegates to a focused helper function.

    Architecture:
        1. Validation - Ensure preconditioner configs provided
        2. Path resolution - Resolve matrix, rhs, output, figures paths
        3. System loading - Load and validate linear system
        4. Preconditioner creation - Create all preconditioners via service
        5. Diagnostics - Compute condition numbers
        6. Solver options - Build iteration limits and fallbacks
        7. Fallback creation - Create identity preconditioner fallback
        8. Parameter configuration - Map stopping criterion, reorthogonalization
        9. Comparison execution - Run CG solver comparison
        10. Plotting - Generate convergence and condition number plots
        11. Result packaging - Return comprehensive result object

    Args:
        general_params: Comparison general configuration
        preconditioner_configs: Sequence of preconditioner configurations
        output_root: Optional override for output root directory
        figures_root: Optional override for figures directory
        display_name: Optional display name for plots
    Returns:
        ComparisonResult containing:
            - results: Raw CG comparison results (iterations, residuals, etc.)
            - summary: Formatted text summary of results
            - plot_paths: Dict mapping plot types to file paths
            - preconditioners: List of preconditioner names tested
            - solver_params: Solver parameters used
            - recommendations: Reserved placeholder for future ranking metadata

    Raises:
        ValueError: If no preconditioner configs provided or required paths missing
        FileNotFoundError: If matrix/rhs files don't exist
        ValidationError: If linear system validation fails

    Example:
        >>> from neuralls.composition.comparison.single_run import compare_preconditioners
        >>> from neuralls.platform.config.models.comparison import (
        ...     ComparisonGeneral,
        ...     SolverParams,
        ...     ComparisonData,
        ... )
        >>> from neuralls.platform.config.models.preconditioner import StandardPreconditionerConfig
        >>>
        >>> general = ComparisonGeneral(
        ...     params=SolverParams(
        ...         rtol=1e-6,
        ...         atol=0.0,
        ...         max_iterations=1000,
        ...         stopping_criterion="residual_norm",
        ...         m_max=20,
        ...         breakdown_tol=None,
        ...     ),
        ...     data=ComparisonData(matrix_path="data/matrix.txt", rhs_path="data/rhs.txt"),
        ...     tracking=None,
        ...     model_store=None,
        ... )
        >>> configs = [
        ...     StandardPreconditionerConfig(name="jacobi", type="jacobi"),
        ...     StandardPreconditionerConfig(name="ilu", type="ilu"),
        ... ]
        >>> result = compare_preconditioners(
        ...     general_params=general,
        ...     preconditioner_configs=configs,
        ... )
        >>> print(result.summary)
    """
    # Step 1: Validate inputs
    if not preconditioner_configs:
        raise ValueError("At least one preconditioner config must be provided.")

    # Step 2: Resolve paths
    paths = _resolve_comparison_paths(
        general_params=general_params,
        output_root=output_root,
        figures_root=figures_root,
    )
    _ensure_comparison_directories(paths)

    # Step 3: Load and validate linear system
    system = _load_linear_system(
        paths,
        rhs_index=general_params.data.rhs_index,
        matrix_index=general_params.data.matrix_index,
        normalize_system=general_params.data.normalize_system,
    )
    _log_matrix_condition_number(
        system.matrix,
        matrix_path=paths.matrix,
        display_name=display_name,
    )

    # Step 4: Create preconditioners via service (uses factory function)
    service = PreconditionerService()
    preconditioners = service.create_preconditioner_set(system.matrix, preconditioner_configs)

    # Step 5: Wrap preconditioners with scheduling if configured
    scheduled_preconditioners = _create_scheduled_preconditioners(
        preconditioner_configs=preconditioner_configs,
        matrix=system.matrix,
        base_preconditioners=preconditioners,
    )

    # Step 5.5: Bind extra inputs to scheduled preconditioners that declare them.
    # Binding on the scheduled wrapper ensures ScheduledPreconditioner.bind_inputs()
    # propagates to the inner preconditioner — no hidden ordering invariant.
    needed_extra_names = frozenset(
        name
        for p in scheduled_preconditioners.values()
        if isinstance(p, BindableInputs)
        for name in p.extra_input_names
        if name not in _SYSTEM_ARRAYS_ALWAYS_AVAILABLE
    )
    extra_data: dict[str, np.ndarray] = {}
    if needed_extra_names and paths.matrix.is_dir():
        _arrays = load_training_arrays(paths.matrix)
        _extra_name_list = sorted(needed_extra_names)
        extra_data = {
            name: np.asarray(
                zarr.open_array(str(_arrays.parameters_zarr[i]), mode="r")[
                    general_params.data.matrix_index
                ],
                dtype=np.float64,
            )
            for i, name in enumerate(_extra_name_list)
            if i < len(_arrays.parameters_zarr)
        }
    _bind_system_inputs(
        scheduled_preconditioners,
        {"matrix": system.matrix, **extra_data},
    )

    # Step 6: Compute condition numbers on fully-initialised preconditioners.
    # Must run after binding so neural preconditioners forward their bound extra
    # inputs internally when __call__(column) is invoked.
    # Wrap as single-input callables: Preconditioner.__call__(r) already
    # dispatches through apply(r) which includes bound _extra_inputs.
    cond_callables: dict[str, PreconditionerCallable] = {
        name: p for name, p in scheduled_preconditioners.items()
    }
    cond_numbers = compute_condition_numbers(system.matrix, cond_callables)

    # Step 7: Run CG comparison with scheduled preconditioners
    results = run_cg_comparison(
        system.matrix,
        system.rhs,
        preconditioners=scheduled_preconditioners,
        rtol=general_params.params.rtol,
        atol=general_params.params.atol,
        maxiter=general_params.params.max_iterations,
        m_max=general_params.params.m_max,
        breakdown_tol=general_params.params.breakdown_tol,
    )
    recommendations = ComparisonRecommendations()

    # Step 8: Generate diagnostic plots.
    plot_paths = _generate_comparison_plots(
        results,
        cond_numbers,
        paths,
        display_name=display_name,
        rtol=general_params.params.rtol,
        atol=general_params.params.atol,
        max_iterations=general_params.params.max_iterations,
    )

    # Step 9: Package and return result
    return ComparisonResult(
        results=results,
        summary=format_results_summary(results),
        plot_paths=plot_paths,
        preconditioners=tuple(preconditioners.keys()),
        condition_numbers=cond_numbers,
        solver_params=general_params,
        recommendations=recommendations,
        output_dir=paths.output,
    )
