"""Modular utilities for exporting solver results and configurations.

This package provides a three-level API for exporting solver data:

Level 1: Convenience API (quick_export)
    Quick one-liner for common case - export everything to one directory.

Level 2: Full control API (save_solver_data)
    Backward-compatible function with all parameters for complete control.

Level 3: Granular API (save_system_arrays, save_iteration_history, etc.)
    Individual functions for specific export tasks, maximum flexibility.

Format Support:
    - NPZ: Fast binary format, 10x smaller than TXT (default for new code)
    - TXT: Human-readable text format (backward compatible)
    - BOTH: Export both formats for debugging

Example:
    >>> from pathlib import Path
    >>> import numpy as np
    >>> from neuralls.domain.solver.utils.export import quick_export, ArrayFormat
    >>> from neuralls.domain.solver.models.config import SolverConfig
    >>> from neuralls.domain.solver.models.result import SolverResult
    >>> from torchalg.monitoring import TraceMode
    >>>
    >>> # Level 1: Quick export (new code)
    >>> config = SolverConfig(
    ...     algorithm="cg",
    ...     rtol=1e-6,
    ...     atol=1e-10,
    ...     maxiter=100,
    ...     trace_mode=TraceMode.MINIMAL,
    ... )
    >>> result = SolverResult(
    ...     converged=True,
    ...     iterations=10,
    ...     residual=1e-8,
    ...     residual_abs=1e-8,
    ...     rhs_norm=1.0,
    ...     breakdown=False,
    ... )
    >>> A = np.eye(3)
    >>> b = np.ones(3)
    >>> x_sol = np.ones(3)
    >>> quick_export(result, config, A, b, x_sol, Path("/tmp/output"))
    >>>
    >>> # Level 2: Full control (backward compatible)
    >>> from neuralls.domain.solver.utils.export import save_solver_data
    >>> save_solver_data(
    ...     output_dir=Path("/tmp/output"),
    ...     name_stem="solver_run",
    ...     A=A,
    ...     b=b,
    ...     x0=None,
    ...     x_sol=x_sol,
    ...     result=result,
    ...     solver_config=config,
    ...     format=ArrayFormat.NPZ,  # Use NPZ for smaller files
    ... )
    >>>
    >>> # Level 3: Granular control (advanced)
    >>> from neuralls.domain.solver.utils.export import (
    ...     save_system_arrays,
    ...     save_iteration_history,
    ...     save_full_config,
    ...     save_solver_result,
    ... )
    >>> save_system_arrays(Path("/tmp/output"), A, b, x_sol, format=ArrayFormat.NPZ)
    >>> save_full_config(Path("/tmp/output/config.toml"), config)
    >>> save_solver_result(Path("/tmp/output/results.toml"), result)
"""

from pathlib import Path
from typing import Any

import numpy as np

from neuralls.domain.solver.models.config import SolverConfig
from neuralls.domain.solver.models.result import SolverResult

# Re-export submodule functions (Level 3: Granular API)
from .arrays import save_iteration_history, save_system_arrays
from .configs import save_full_config, save_solver_config
from .formats import ArrayFormat
from .results import save_solver_result

__all__ = [
    "ArrayFormat",
    # Level 1: Convenience
    "quick_export",
    "save_full_config",
    "save_iteration_history",
    "save_solver_config",
    # Level 2: Full control
    "save_solver_data",
    "save_solver_result",
    # Level 3: Granular
    "save_system_arrays",
]


def quick_export(
    result: SolverResult,
    config: SolverConfig,
    A: np.ndarray,
    b: np.ndarray,
    x_sol: np.ndarray,
    output_dir: Path,
    name_stem: str = "solver_output",
    x0: np.ndarray | None = None,
    format: ArrayFormat = ArrayFormat.NPZ,
) -> None:
    """Quick export for common case - everything to one directory.

    Convenience function that exports all solver data in one call:
    - System arrays (A, b, x0, x_sol)
    - Iteration history (if available)
    - Configuration (solver config)
    - Results (outcomes)

    Args:
        result: Solver execution result.
        config: Solver configuration.
        A: System matrix.
        b: Right-hand side vector.
        x_sol: Solution vector.
        output_dir: Base output directory.
        name_stem: Subdirectory name (default: "solver_output").
        x0: Initial guess (optional).
        format: Export format (NPZ, TXT, or BOTH).

    Example:
        >>> from pathlib import Path
        >>> import numpy as np
        >>> from neuralls.domain.solver.utils.export import quick_export, ArrayFormat
        >>> from neuralls.domain.solver.models.config import SolverConfig
        >>> from neuralls.domain.solver.models.result import SolverResult
        >>> from torchalg.monitoring import TraceMode
        >>> config = SolverConfig(
        ...     algorithm="cg",
        ...     rtol=1e-6,
        ...     atol=1e-10,
        ...     maxiter=100,
        ...     trace_mode=TraceMode.MINIMAL,
        ... )
        >>> result = SolverResult(
        ...     converged=True,
        ...     iterations=10,
        ...     residual=1e-8,
        ...     residual_abs=1e-8,
        ...     rhs_norm=1.0,
        ...     breakdown=False,
        ... )
        >>> A = np.eye(3)
        >>> b = np.ones(3)
        >>> x_sol = np.ones(3)
        >>> quick_export(result, config, A, b, x_sol, Path("/tmp/output"))
    """
    target_dir = Path(output_dir) / name_stem
    target_dir.mkdir(parents=True, exist_ok=True)

    # Export arrays
    save_system_arrays(target_dir, A, b, x_sol, x0=x0, format=format)

    # Export history if available
    if result.iteration_history is not None:
        save_iteration_history(target_dir, result.iteration_history, format=format)

    # Export configuration
    save_solver_config(target_dir / "config.toml", config)

    # Export results
    save_solver_result(target_dir / "results.toml", result)


def save_solver_data(
    output_dir: Path,
    name_stem: str,
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray | None,
    x_sol: np.ndarray,
    result: SolverResult,
    solver_config: SolverConfig,
    data_config: dict[str, Any] | None = None,
    experiment_config: dict[str, Any] | None = None,
    format: ArrayFormat = ArrayFormat.TXT,  # Default to TXT for backward compatibility
) -> None:
    """Save complete solver data (backward compatible API).

    Saves all solver artifacts and configurations to a directory:
    - System arrays: A, b, x0, x_sol
    - Iteration history: residual_norms, (optionally) residuals, solutions
    - Configuration: config.toml (solver, data, experiment tables)
    - Results: results.toml (outcomes table)

    Format options:
    - NPZ: Fast binary format, 10x smaller (recommended for new code)
    - TXT: Human-readable text format (backward compatible, default)
    - BOTH: Export both formats for debugging

    Args:
        output_dir: Root directory for results.
        name_stem: Name of the subdirectory for this run.
        A: System matrix.
        b: Right-hand side vector.
        x0: Initial guess (optional).
        x_sol: Computed solution.
        result: Solver result container.
        solver_config: Structured solver configuration.
        data_config: Optional metadata about the system/matrix.
        experiment_config: Optional metadata about the experiment context.
        format: Export format (TXT, NPZ, or BOTH).

    Example:
        >>> from pathlib import Path
        >>> import numpy as np
        >>> from neuralls.domain.solver.utils.export import save_solver_data, ArrayFormat
        >>> from neuralls.domain.solver.models.config import SolverConfig
        >>> from neuralls.domain.solver.models.result import SolverResult
        >>> from torchalg.monitoring import TraceMode
        >>> config = SolverConfig(
        ...     algorithm="cg",
        ...     rtol=1e-6,
        ...     atol=1e-10,
        ...     maxiter=100,
        ...     trace_mode=TraceMode.MINIMAL,
        ... )
        >>> result = SolverResult(
        ...     converged=True,
        ...     iterations=10,
        ...     residual=1e-8,
        ...     residual_abs=1e-8,
        ...     rhs_norm=1.0,
        ...     breakdown=False,
        ... )
        >>> A = np.eye(3)
        >>> b = np.ones(3)
        >>> x_sol = np.ones(3)
        >>> save_solver_data(
        ...     output_dir=Path("/tmp/results"),
        ...     name_stem="test_run",
        ...     A=A,
        ...     b=b,
        ...     x0=None,
        ...     x_sol=x_sol,
        ...     result=result,
        ...     solver_config=config,
        ...     format=ArrayFormat.NPZ,  # Use NPZ for smaller files
        ... )
    """
    target_dir = Path(output_dir) / name_stem
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save numerical data (arrays)
    save_system_arrays(target_dir, A, b, x_sol, x0=x0, format=format)

    # 2. Save iteration history
    if result.iteration_history is not None:
        save_iteration_history(target_dir, result.iteration_history, format=format)

    # 3. Save configuration (TOML)
    save_full_config(
        target_dir / "config.toml",
        solver_config,
        data_config=data_config,
        experiment_config=experiment_config,
    )

    # 4. Save results (TOML)
    save_solver_result(target_dir / "results.toml", result)
