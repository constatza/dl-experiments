"""CG solver comparison runner.

This module provides orchestration for running CG comparisons with multiple
preconditioners. It handles solver routing (PCG vs Flexible CG), result
formatting, and analysis.

Moved from solver/comparison.py to workflows/ as these are workflow orchestration
concerns, not core solver algorithms.
"""

from __future__ import annotations

from collections.abc import Mapping
import numpy as np
from scipy.linalg import norm

from ..constants import DEFAULT_ATOL, DEFAULT_M_MAX, DEFAULT_RTOL
from ..solver.factories import flexible_cg, pcg
from ..solver.models.result import CGComparisonResult
from ..solver.preconditioners.base import Preconditioner
from .results import ComparisonRecommendations, RankedRecommendation


def run_cg_comparison(
    A: np.ndarray,
    b: np.ndarray,
    *,
    preconditioners: Mapping[str, Preconditioner],
    x0: np.ndarray | None = None,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    maxiter: int = 100,
    m_max: int = DEFAULT_M_MAX,
    breakdown_tol: float | None = None,
) -> dict[str, CGComparisonResult]:
    """Run CG with multiple preconditioners for comparison.

    Uses type-based routing:
    - Contextual/iteration-dependent preconditioners → flexible_cg
    - Non-linear preconditioners (for example neural models) → flexible_cg
    - Static linear preconditioners → pcg

    Args:
        A: System matrix
        b: Right-hand side vector
        preconditioners: Dict mapping names to Preconditioner instances
        x0: Initial guess (defaults to zero)
        rtol: Relative tolerance
        atol: Absolute tolerance
        maxiter: Maximum iterations
        m_max: FCG orthogonalization restart parameter
        breakdown_tol: Breakdown detection tolerance.

    Returns:
        Dict mapping preconditioner names to CGComparisonResult

    Example:
        >>> from neuralls.solver.preconditioners import Identity, JacobiPreconditioner
        >>> preconditioners = {
        ...     "none": Identity(),
        ...     "jacobi": JacobiPreconditioner(A),
        ... }
        >>> results = run_cg_comparison(A, b, preconditioners=preconditioners)
    """
    if x0 is None:
        x0 = np.zeros_like(b, dtype=np.float64)
    x0_base = x0.copy()

    # Add identity baseline if not present
    if "none" not in preconditioners:
        from ..solver.preconditioners import Identity

        preconditioners = dict(preconditioners)
        preconditioners["none"] = Identity()

    # Compute exact solution for error analysis
    x_exact = np.linalg.solve(A.astype(np.float64, copy=False), b.astype(np.float64, copy=False))

    results: dict[str, CGComparisonResult] = {}

    for precond_name, precond in preconditioners.items():
        try:
            if _requires_flexible_cg(precond):
                # Use flexible CG for contextual preconditioners
                x_sol, info = flexible_cg(
                    A,
                    b,
                    x0,
                    rtol=rtol,
                    atol=atol,
                    maxiter=maxiter,
                    preconditioner=precond,
                    m_max=m_max,
                    breakdown_tol=breakdown_tol,
                )
            else:
                # Use standard PCG for constant preconditioners
                x_sol, info = pcg(
                    A,
                    b,
                    x0,
                    rtol=rtol,
                    atol=atol,
                    maxiter=maxiter,
                    preconditioner=precond,
                    breakdown_tol=breakdown_tol,
                )
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as solver_exc:
            result = CGComparisonResult(
                x=x0_base.copy(),
                converged=False,
                iterations=0,
                residual=float("inf"),
                residual_abs=float("inf"),
                residual_history=[],
                residual_history_abs=[],
                preconditioner=precond_name,
                initial_guess=x0_base.copy(),
                exact_error=None,
                rhs_norm=norm(b),
                breakdown=False,
                error=f"CG solver failed: {solver_exc}",
            )
        else:
            exact_norm = norm(x_exact)
            exact_error = (
                norm(x_sol - x_exact) / exact_norm if exact_norm != 0 else norm(x_sol - x_exact)
            )

            # Extract residual history from iteration history if available
            if info.iteration_history is not None:
                residual_history: list[float] = info.iteration_history.residual_norms.to_list()
            else:
                residual_history = [info.residual]

            result = CGComparisonResult(
                x=x_sol,
                converged=info.converged,
                iterations=info.iterations,
                residual=info.residual,
                residual_abs=info.residual_abs,
                residual_history=residual_history,
                residual_history_abs=residual_history,  # Same as relative for now
                preconditioner=precond_name,
                initial_guess=x0_base.copy(),
                exact_error=exact_error,
                rhs_norm=info.rhs_norm,
                breakdown=info.breakdown,
            )

        results[precond_name] = result

    return results


def _requires_flexible_cg(preconditioner: Preconditioner) -> bool:
    """Check if preconditioner requires flexible CG.

    Delegates to the preconditioner's own knowledge of solver compatibility,
    following OCP: adding a new preconditioner type never requires updating
    this function.
    """
    return preconditioner.requires_flexible_cg


def format_results_summary(results: dict[str, CGComparisonResult]) -> str:
    """Format CG results into a readable summary.

    Status codes follow SciPy convention:
    - "ok": converged (info=0)
    - "fail": did not converge (info>0), either max iterations or breakdown

    Args:
        results: CG comparison results

    Returns:
        Formatted summary string
    """
    lines = ["Flexible CG results:"]
    for name, result in results.items():
        status = "ok" if result.converged else "fail"
        iters = result.iterations
        res = result.residual
        exact_err = result.exact_error
        res_abs = result.residual_abs

        if exact_err is not None:
            line = f"- {name:<18} status={status:<4} iters={iters:>3}  rel_res={res:.3e}"
            if res_abs is not None:
                line += f" (abs={res_abs:.3e})"
            line += f"  exact_err={exact_err:.3e}"
        else:
            line = f"- {name:<18} status={status:<4} iters={iters:>3}  rel_res={res:.3e}"
            if res_abs is not None:
                line += f" (abs={res_abs:.3e})"

        if result.error:
            line += f"  note={result.error}"

        # Distinguish breakdown types for better diagnostics
        if result.breakdown and not result.converged:
            line += "  note=breakdown"
        elif result.breakdown and result.converged:
            # Breakdown occurred after achieving convergence tolerance
            line += "  note=breakdown_post_convergence"

        lines.append(line)

    return "\n".join(lines)


def summarize_best_combinations(
    results: dict[str, CGComparisonResult],
) -> ComparisonRecommendations:
    """Summarize best-performing CG combinations by preconditioner and overall.

    Args:
        results: CG comparison results

    Returns:
        Typed ranked recommendations with an overall best entry.
    """
    ranked: list[RankedRecommendation] = []
    for label, info in results.items():
        if not info.converged:
            continue
        ranked.append(
            RankedRecommendation(
                label=label,
                iterations=info.iterations,
                residual=info.residual,
                residual_abs=info.residual_abs,
                breakdown=info.breakdown,
            )
        )

    ranked = sorted(ranked, key=lambda entry: entry.residual)
    overall_best = ranked[0] if ranked else None

    return ComparisonRecommendations(
        ranked=tuple(ranked),
        overall_best=overall_best,
    )
