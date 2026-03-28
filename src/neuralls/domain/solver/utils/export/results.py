"""Results export utilities for solver data.

This module provides functions for saving solver execution results and
outcomes to TOML files for analysis and comparison.

All functions are I/O actions with clear contracts.
"""

from pathlib import Path

import tomli_w

from neuralls.domain.solver.models.result import SolverResult

from .serialization import sanitize_for_toml


def save_solver_result(
    output_path: Path,
    result: SolverResult,
) -> None:
    """Save solver results/outcomes to TOML file.

    Creates TOML file with [outcomes] table containing:
    - converged: Convergence status (bool)
    - iterations: Iteration count (int)
    - final_residual_norm: Absolute residual norm (float)
    - final_relative_residual: Relative residual norm (float)
    - rtol: Relative tolerance used (float, optional)
    - atol: Absolute tolerance used (float, optional)
    - stopping_criterion: Why solver stopped (str, optional)

    Args:
        output_path: Path to output TOML file.
        result: SolverResult dataclass instance.

    Example:
        >>> from pathlib import Path
        >>> from neuralls.domain.solver.models.result import SolverResult
        >>> result = SolverResult(
        ...     converged=True,
        ...     iterations=10,
        ...     residual=1e-8,
        ...     residual_abs=1e-8,
        ...     rhs_norm=1.0,
        ...     breakdown=False,
        ...     tol=1e-6,
        ...     atol=1e-10,
        ...     stopping_criterion="converged",
        ... )
        >>> save_solver_result(Path("/tmp/results.toml"), result)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_dict = {
        "outcomes": {
            "converged": bool(result.converged),
            "iterations": int(result.iterations),
            "final_residual_norm": float(result.residual_abs),
            "final_relative_residual": float(result.residual),
            "rtol": float(result.tol) if result.tol is not None else None,
            "atol": float(result.atol) if result.atol is not None else None,
            "stopping_criterion": result.stopping_criterion,
        }
    }

    with output_path.open("wb") as f:
        tomli_w.dump(sanitize_for_toml(results_dict), f)
