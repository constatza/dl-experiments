"""Simplified preconditioner comparison workflow driven by solver config.

This module provides the core comparison workflow for benchmarking CG solver
preconditioners. It follows clean architecture principles:
- Domain models for data structures (ComparisonPaths, LinearSystem)
- Service pattern for preconditioner creation (PreconditionerService)
- Single-responsibility helper functions
- Pure orchestration in main function

Architecture:
    The comparison workflow consists of 12 clear steps:
    1. Validate inputs
    2. Resolve paths (matrix, rhs, output, figures)
    3. Load and validate linear system
    4. Create preconditioners via service (eliminates registry duplication)
    5. Compute condition numbers for diagnostics
    6. Build solver options (iteration limits, fallbacks)
    7. Create fallback preconditioner
    8. Configure solver parameters
    9. Run CG comparison
    10. Generate recommendations
    11. Generate plots (if enabled)
    12. Return result

Key Components:
    - `compare_preconditioners()`: Main entry point (orchestration only)
    - `PreconditionerService`: Centralized preconditioner creation
    - `ComparisonPaths`: Type-safe path resolution
    - `LinearSystem`: Validated system matrices

Example:
    >>> from neuralls.workflows.compare import compare_preconditioners
    >>> from neuralls.configuration.comparison import GeneralSolverConfig
    >>> from neuralls.configuration.preconditioner import StandardPreconditionerConfig
    >>>
    >>> general = GeneralSolverConfig(
    ...     matrix_path="data/matrix.txt",
    ...     rhs_path="data/rhs.txt",
    ...     output_root="output/comparison",
    ... )
    >>> configs = [
    ...     StandardPreconditionerConfig(name="jacobi", type="jacobi"),
    ...     StandardPreconditionerConfig(name="ilu", type="ilu"),
    ... ]
    >>> result = compare_preconditioners(
    ...     general_params=general,
    ...     preconditioner_configs=configs,
    ...     save_plots=True,
    ... )
    >>> print(result.summary)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from collections.abc import Callable, Sequence

import numpy as np

from neuralls.configuration.preconditioner import PreconditionerConfig
from neuralls.configuration.comparison import GeneralSolverConfig
from ..diagnostics import compute_condition_numbers, plot_condition_numbers
from neuralls.io.filesystem import ensure_dir
from ..io.comparison import load_system_arrays
from ..plotting import plot_convergence_comparison
from ..preconditioner import create_default_registry
from ..solver import (
    format_results_summary,
    run_cg_comparison,
    summarize_best_combinations,
)
from ..validation import validate_matrix, validate_rhs_vector
from .results import ComparisonResult


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

    This service centralizes preconditioner creation to eliminate registry duplication.
    Before this service, the registry was created 3 times in different functions.

    The service pattern provides:
    - Single source of truth for registry
    - Clear dependency injection point
    - Easy to test (can inject mock registry)
    - Consistent preconditioner creation

    Example:
        >>> service = PreconditionerService()
        >>> jacobi_config = StandardPreconditionerConfig(name="jacobi", type="jacobi")
        >>> matrix = np.eye(100)
        >>> jacobi_precond = service.create_preconditioner(matrix, jacobi_config)
        >>> precond_set = service.create_preconditioner_set(matrix, [jacobi_config, ...])
    """

    def __init__(self):
        """Initialize service with default registry.

        Creates the preconditioner registry once and reuses it for all
        preconditioner creation requests.
        """
        self.registry = create_default_registry()

    def create_preconditioner(
        self,
        matrix: np.ndarray,
        config: PreconditionerConfig,
    ) -> Callable:
        """Create a single preconditioner.

        Args:
            matrix: System matrix to precondition (shape: n x n)
            config: Preconditioner configuration (type, parameters, etc.)

        Returns:
            Preconditioner function: ndarray -> ndarray

        Example:
            >>> service = PreconditionerService()
            >>> A = np.array([[4, 1], [1, 3]])
            >>> config = StandardPreconditionerConfig(name="jacobi", type="jacobi")
            >>> M = service.create_preconditioner(A, config)
            >>> r = np.array([1.0, 2.0])
            >>> z = M(r)  # Apply preconditioner
        """
        return self.registry.create(matrix, config)

    def create_preconditioner_set(
        self,
        matrix: np.ndarray,
        configs: Sequence[PreconditionerConfig],
    ) -> dict[str, Callable]:
        """Create multiple preconditioners for comparison.

        Args:
            matrix: System matrix to precondition (same for all)
            configs: Sequence of preconditioner configurations

        Returns:
            Dictionary mapping preconditioner names to functions

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
        return {
            cfg.name: self.create_preconditioner(matrix, cfg)
            for cfg in configs
        }


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
        raise ValueError(
            f"Unknown stopping criterion: '{name}'. "
            f"Valid options: {valid_names}"
        )
    return criterion


def _resolve_comparison_paths(
    *,
    general_params: GeneralSolverConfig,
    output_root: Path | None,
    figures_root: Path | None,
) -> ComparisonPaths:
    """Resolve all paths for comparison workflow.

    This is a pure function that validates and resolves paths from config.
    It ensures all required paths are set and handles optional overrides.

    Args:
        general_params: General solver configuration with matrix_path, rhs_path, output_root
        output_root: Optional override for output root directory
        figures_root: Optional override for figures directory

    Returns:
        ComparisonPaths with all resolved and validated paths

    Raises:
        ValueError: If matrix_path, rhs_path, or output_root not configured

    Example:
        >>> general = GeneralSolverConfig(
        ...     matrix_path="data/matrix.txt",
        ...     rhs_path="data/rhs.txt",
        ...     output_root="output",
        ... )
        >>> paths = _resolve_comparison_paths(
        ...     general_params=general,
        ...     output_root=None,
        ...     figures_root=None,
        ... )
        >>> print(paths.figures)
        Path('output/figures')
    """
    if general_params.matrix_path is None or general_params.rhs_path is None:
        raise ValueError("Matrix and RHS must be provided in solver config.")

    matrix_file = Path(general_params.matrix_path)
    rhs_file = Path(general_params.rhs_path)

    base_root = output_root or getattr(general_params, "output_root", None)
    if base_root is None:
        raise ValueError("output_root must be set in solver general config.")

    output_base = Path(base_root).expanduser().resolve()
    figs_base = Path(figures_root) if figures_root else output_base / "figures"

    return ComparisonPaths(
        matrix=matrix_file,
        rhs=rhs_file,
        output=output_base,
        figures=figs_base,
    )


def _ensure_comparison_directories(paths: ComparisonPaths) -> None:
    """Ensure output directories exist.

    Creates directories if they don't exist. Safe to call multiple times.

    Args:
        paths: Comparison paths with output and figures directories
    """
    ensure_dir(paths.output)
    ensure_dir(paths.figures)


def _load_linear_system(paths: ComparisonPaths) -> LinearSystem:
    """Load and validate linear system.

    This function:
    1. Loads matrix and rhs from files (txt or npy)
    2. Validates matrix properties (square, no NaN/inf, etc.)
    3. Validates rhs compatibility with matrix dimensions
    4. Returns validated LinearSystem

    Args:
        paths: Comparison paths with matrix and rhs file locations

    Returns:
        LinearSystem with loaded and validated arrays

    Raises:
        ValueError: If validation fails (wrong shape, NaN values, incompatible dimensions)
        FileNotFoundError: If matrix or rhs files don't exist
    """
    A, b = load_system_arrays(paths.matrix, paths.rhs)
    validate_matrix(A)
    validate_rhs_vector(b, A)
    return LinearSystem(matrix=A, rhs=b)


def _resolve_fallback_callable(
    name: str, A: np.ndarray, preconditioners: dict[str, Any]
) -> Callable[[np.ndarray], np.ndarray]:
    """Resolve fallback preconditioner by name.

    This uses the registry pattern to create preconditioners on-demand,
    eliminating code duplication from the old factory approach.
    """
    # Check if already created
    if name in preconditioners:
        return preconditioners[name]

    # Create on-demand using registry
    from neuralls.configuration.preconditioner import StandardPreconditionerConfig

    registry = create_default_registry()
    config = StandardPreconditionerConfig(name=name, type=name)
    return registry.create(A, config)


def _build_solver_options(
    preconditioner_configs: Sequence[PreconditionerConfig],
    matrix: np.ndarray,
    preconditioners: dict[str, Callable],
) -> dict[str, dict[str, Any]]:
    """Build solver options for each preconditioner.

    Args:
        preconditioner_configs: Preconditioner configurations
        matrix: System matrix
        preconditioners: Created preconditioner functions

    Returns:
        Dictionary mapping solver names to options
    """
    solver_options: dict[str, dict[str, Any]] = {}
    for cfg in preconditioner_configs:
        limit = cfg.limit_iters if cfg.limit_iters >= 0 else None
        solver_options[cfg.name] = {
            "limit_iters": limit,
            "fallback": _resolve_fallback_callable(cfg.fallback, matrix, preconditioners),
        }
    return solver_options


def _generate_comparison_plots(
    results: dict[str, Any],
    cond_numbers: dict[str, float],
    paths: ComparisonPaths,
) -> dict[str, Path]:
    """Generate diagnostic plots for comparison.

    Args:
        results: CG comparison results
        cond_numbers: Condition numbers
        paths: Comparison paths

    Returns:
        Dictionary mapping plot types to paths
    """
    suffix = paths.matrix.stem or "comparison"
    plot_condition_numbers(cond_numbers, save_dir=paths.figures, suffix=suffix)

    convergence_path = paths.figures / f"preconditioner_convergence_{suffix}.png"
    plot_convergence_comparison(results, metadata=None, save_path=convergence_path)

    return {"convergence": convergence_path}


def compare_preconditioners(
    *,
    general_params: GeneralSolverConfig,
    preconditioner_configs: Sequence[PreconditionerConfig],
    output_root: Path | None = None,
    figures_root: Path | None = None,
    save_plots: bool = True,
) -> ComparisonResult:
    """Run CG comparisons and generate diagnostics.

    This is the main entry point for preconditioner comparison. It orchestrates
    a 12-step workflow using single-responsibility helpers.

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
        10. Recommendations - Summarize best solver combinations
        11. Plotting - Generate convergence and condition number plots
        12. Result packaging - Return comprehensive result object

    Args:
        general_params: General solver configuration (rtol, atol, max_iter, paths)
        preconditioner_configs: Sequence of preconditioner configurations
        output_root: Optional override for output root directory
        figures_root: Optional override for figures directory
        save_plots: Whether to generate diagnostic plots

    Returns:
        ComparisonResult containing:
            - results: Raw CG comparison results (iterations, residuals, etc.)
            - summary: Formatted text summary of results
            - plot_paths: Dict mapping plot types to file paths
            - preconditioners: List of preconditioner names tested
            - solver_params: Solver parameters used
            - recommendations: Best solver combinations

    Raises:
        ValueError: If no preconditioner configs provided or required paths missing
        FileNotFoundError: If matrix/rhs files don't exist
        ValidationError: If linear system validation fails

    Example:
        >>> from neuralls.workflows.compare import compare_preconditioners
        >>> from neuralls.configuration.comparison import GeneralSolverConfig
        >>> from neuralls.configuration.preconditioner import StandardPreconditionerConfig
        >>>
        >>> general = GeneralSolverConfig(
        ...     matrix_path="data/matrix.txt",
        ...     rhs_path="data/rhs.txt",
        ...     output_root="output/comparison",
        ...     rtol=1e-6,
        ...     atol=0.0,
        ...     max_iterations=1000,
        ... )
        >>> configs = [
        ...     StandardPreconditionerConfig(name="jacobi", type="jacobi"),
        ...     StandardPreconditionerConfig(name="ilu", type="ilu"),
        ... ]
        >>> result = compare_preconditioners(
        ...     general_params=general,
        ...     preconditioner_configs=configs,
        ...     save_plots=True,
        ... )
        >>> print(result.summary)
        >>> print(f"Best: {result.recommendations}")
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
    system = _load_linear_system(paths)

    # Step 4: Create preconditioners via service (eliminates registry duplication)
    service = PreconditionerService()
    preconditioners = service.create_preconditioner_set(
        system.matrix, preconditioner_configs
    )
    solver_types = {cfg.name: cfg.type for cfg in preconditioner_configs}

    # Step 5: Compute condition numbers for diagnostics
    cond_numbers = compute_condition_numbers(system.matrix, preconditioners)

    # Step 6: Build solver options (iteration limits, fallbacks)
    solver_options = _build_solver_options(
        preconditioner_configs, system.matrix, preconditioners
    )

    # Step 7: Create fallback preconditioner (identity)
    from neuralls.configuration.preconditioner import StandardPreconditionerConfig

    fallback_config = StandardPreconditionerConfig(name="identity", type="identity")
    fallback_precond = service.create_preconditioner(system.matrix, fallback_config)

    # Step 8: Configure solver parameters
    stopping_criterion = _map_stopping_criterion(general_params.stopping_criterion)

    # Step 9: Run CG comparison
    results = run_cg_comparison(
        system.matrix,
        system.rhs,
        preconditioners=preconditioners,
        rtol=general_params.rtol,
        atol=general_params.atol,
        maxiter=general_params.max_iterations,
        stopping_criterion=stopping_criterion,
        breakdown_tol=0.0,
        m_max=general_params.m_max,
        precond_iters=None,
        fallback_preconditioner=fallback_precond,
        precond_every=1,
        precond_first_n=None,
        combination_plan=None,
        limited_preconditioner=None,
        solver_types=solver_types,
        solver_options=solver_options,
    )

    # Step 10: Generate recommendations (best solver combinations)
    recommendations = summarize_best_combinations(results)

    # Step 11: Generate diagnostic plots (if enabled)
    plot_paths = (
        _generate_comparison_plots(results, cond_numbers, paths)
        if save_plots
        else {}
    )

    # Step 12: Package and return result
    return ComparisonResult(
        results=results,
        summary=format_results_summary(results),
        plot_paths=plot_paths,
        preconditioners=list(preconditioners.keys()),
        solver_params=general_params,
        recommendations=recommendations,
    )
