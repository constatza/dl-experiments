"""Comparison and helper wrappers for the solver API."""

from __future__ import annotations

from typing import Any, Literal
from collections.abc import Callable, Sequence

import numpy as np
from scipy.linalg import norm

from ..constants import DEFAULT_ATOL, DEFAULT_BREAKDOWN_TOL, DEFAULT_RTOL
from .factories import flexible_cg, preconditioned_cg as _preconditioned_cg
from .models.result import CGComparisonResult, IterationContext, SolverResult
from .strategies.reorthogonalization import ReorthogonalizationStrategy


# Registry of constant (non-adaptive) preconditioner types
# These use the standard preconditioned_cg path (SciPy-based)
# Easy to extend: just add new type to this set
_CONSTANT_PRECONDITIONER_TYPES: set[str] = {
    "none",
    "jacobi",
    "ilu",
    "identity",
}


def register_constant_preconditioner_type(type_name: str) -> None:
    """Register a new constant preconditioner type.

    Args:
        type_name: The type identifier for the constant preconditioner
    """
    _CONSTANT_PRECONDITIONER_TYPES.add(type_name)


def is_constant_preconditioner(type_name: str) -> bool:
    """Check if a preconditioner type is constant (non-adaptive).

    Args:
        type_name: The preconditioner type identifier

    Returns:
        True if the type is a constant preconditioner, False otherwise
    """
    return type_name in _CONSTANT_PRECONDITIONER_TYPES


# Signature detection removed: all preconditioners now have uniform signature (residual) -> result


def _scheduled_preconditioner(
    preconditioner: Callable[[np.ndarray], np.ndarray] | None,
    *,
    fallback: Callable[[np.ndarray], np.ndarray] | None,
    limit_iters: int | None,
    apply_every: int,
    first_n: int | None,
) -> Callable[[IterationContext], np.ndarray] | None:
    if preconditioner is None and fallback is None:
        return None

    active_precond = preconditioner or (lambda residual: residual)
    fallback_precond = fallback or (lambda residual: residual)

    def wrapped(context: IterationContext) -> np.ndarray:
        """Scheduled preconditioner taking IterationContext.

        All preconditioners now have uniform signature: (residual) -> result.
        No signature detection needed.
        """
        iteration = context.iteration
        residual = context.residual

        # Determine which preconditioner to use based on scheduling
        use_main = True
        if first_n is not None and iteration >= first_n:
            use_main = False
        if limit_iters is not None and iteration >= limit_iters:
            use_main = False
        if apply_every > 1 and iteration % apply_every != 0:
            use_main = False

        # All preconditioners have signature: (residual) -> result
        if use_main:
            return active_precond(residual)
        return fallback_precond(residual)

    return wrapped


def preconditioned_cg(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray | None = None,
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    max_iter: int = 100,
    preconditioner: Callable[[np.ndarray], np.ndarray] | None = None,
    stopping_criterion: Literal["tolerance", "fixed_iterations"] = "tolerance",
    breakdown_tol: float = 0.0,
) -> tuple[np.ndarray, SolverResult]:
    """Preconditioned Conjugate Gradient solver wrapper."""
    if x0 is None:
        x0 = np.zeros_like(b, dtype=np.float64)
    # Note: stopping_criterion and breakdown_tol are legacy parameters, not passed to factory
    return _preconditioned_cg(
        A,
        b,
        x0,
        rtol=rtol,
        atol=atol,
        max_iter=max_iter,
        preconditioner=preconditioner,
    )


def run_cg_comparison(
    A: np.ndarray,
    b: np.ndarray,
    *,
    preconditioners: dict[str, Callable[[np.ndarray], np.ndarray]],
    x0: np.ndarray | None = None,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    max_iter: int = 100,
    stopping_criterion: Literal["tolerance", "fixed_iterations"] = "tolerance",
    breakdown_tol: float = DEFAULT_BREAKDOWN_TOL,
    precond_iters: int | None = None,
    fallback_preconditioner: Callable[[np.ndarray], np.ndarray] | None = None,
    precond_every: int = 1,
    precond_first_n: int | None = None,
    reorthogonalize: ReorthogonalizationStrategy | None = None,
    combination_plan: Sequence[tuple[str, str, str]] | None = None,
    limited_preconditioner: str | None = None,
    solver_types: dict[str, str] | None = None,
    solver_options: dict[str, dict[str, Any]] | None = None,
) -> dict[str, CGComparisonResult]:
    """Run CG with multiple preconditioners for comparison."""
    if x0 is None:
        x0 = np.zeros_like(b, dtype=np.float64)
    x0_base = x0.copy()

    preconditioners = dict(preconditioners)
    if "none" not in preconditioners:
        preconditioners["none"] = lambda residual: residual

    x_exact = np.linalg.solve(
        A.astype(np.float64, copy=False), b.astype(np.float64, copy=False)
    )

    results: dict[str, CGComparisonResult] = {}

    if combination_plan is None:
        combos = [
            ("none", precond_name, "none") for precond_name in preconditioners.keys()
        ]
    else:
        combos = []
        for warm_name, precond_name, helper_name in combination_plan:
            if warm_name != "none" or helper_name != "none":
                continue
            if precond_name not in preconditioners:
                raise ValueError(f"Unknown preconditioner '{precond_name}' requested.")
            combos.append((warm_name, precond_name, helper_name))

    for _, precond_name, _ in combos:
        effective_type = (solver_types or {}).get(precond_name, precond_name)
        precond = preconditioners[precond_name]
        precond_fn = precond if effective_type != "none" else None

        should_limit = (
            limited_preconditioner is None or precond_name == limited_preconditioner
        )
        opts = (solver_options or {}).get(precond_name, {})
        limit_iters = opts.get("limit_iters", precond_iters if should_limit else None)
        limit_fallback = opts.get(
            "fallback", fallback_preconditioner if limit_iters is not None else None
        )
        if isinstance(limit_fallback, str) and limit_fallback in preconditioners:
            limit_fallback = preconditioners[limit_fallback]
        apply_every = opts.get("apply_every", precond_every)
        first_n = opts.get("first_n", precond_first_n)
        scheduled_precond = _scheduled_preconditioner(
            precond_fn,
            fallback=limit_fallback,
            limit_iters=limit_iters,
            apply_every=apply_every,
            first_n=first_n,
        )

        try:
            if is_constant_preconditioner(effective_type):
                # SciPy CG/PCG path for static preconditioners (None goes through the same wrapper)
                x_sol, info = preconditioned_cg(
                    A,
                    b,
                    x0,
                    rtol=rtol,
                    atol=atol,
                    max_iter=max_iter,
                    preconditioner=precond_fn,  # None for baseline CG
                    stopping_criterion=stopping_criterion,
                    breakdown_tol=breakdown_tol,
                )
            else:
                # Flexible CG path (neural/nonlinear/time-varying preconditioners)
                x_sol, info = flexible_cg(
                    A,
                    b,
                    x0,
                    rtol=rtol,
                    atol=atol,
                    max_iter=max_iter,
                    preconditioner=scheduled_precond,
                    reorthogonalize=reorthogonalize,
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
                warm_start="none",
                preconditioner=precond_name,
                step_helper="none",
                initial_guess=x0_base.copy(),
                exact_error=None,
                rhs_norm=norm(b),
                breakdown=False,
                helper_iterations=[],
                helper_norms=[],
                error=f"CG solver failed: {solver_exc}",
            )
        else:
            exact_norm = norm(x_exact)
            exact_error = (
                norm(x_sol - x_exact) / exact_norm
                if exact_norm != 0
                else norm(x_sol - x_exact)
            )

            # Extract residual history from event log if available
            if info.event_log is not None:
                res_hist_raw = info.event_log.get_history("residual_norm")
                # Convert to list[float] if needed
                residual_history: list[float] = [float(x) for x in res_hist_raw]
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
                warm_start="none",
                preconditioner=precond_name,
                step_helper="none",
                initial_guess=x0_base.copy(),
                exact_error=exact_error,
                rhs_norm=info.rhs_norm,
                breakdown=info.breakdown,
                helper_iterations=[],  # Not used in new architecture
                helper_norms=[],  # Not used in new architecture
            )

        results[precond_name] = result

    return results


def format_results_summary(results: dict[str, CGComparisonResult]) -> str:
    """Format CG results into a readable summary.

    Status codes follow SciPy convention:
    - "ok": converged (info=0)
    - "fail": did not converge (info>0), either max iterations or breakdown
    """
    lines = ["Flexible CG results:"]
    for name, result in results.items():
        status = "ok" if result.converged else "fail"
        iters = result.iterations
        res = result.residual
        exact_err = result.exact_error
        res_abs = result.residual_abs

        if exact_err is not None:
            line = (
                f"- {name:<18} status={status:<4} iters={iters:>3}  rel_res={res:.3e}"
            )
            if res_abs is not None:
                line += f" (abs={res_abs:.3e})"
            line += f"  exact_err={exact_err:.3e}"
        else:
            line = (
                f"- {name:<18} status={status:<4} iters={iters:>3}  rel_res={res:.3e}"
            )
            if res_abs is not None:
                line += f" (abs={res_abs:.3e})"

        if result.helper_iterations:
            line += f"  helper_calls={len(result.helper_iterations)}"

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
) -> dict[str, Any]:
    """Summarize best-performing CG combinations by preconditioner and overall."""
    ranked: list[dict[str, Any]] = []
    for label, info in results.items():
        if not info.converged:
            continue
        ranked.append(
            {
                "label": label,
                "iterations": info.iterations,
                "residual": info.residual,
                "residual_abs": info.residual_abs,
                "breakdown": info.breakdown,
            }
        )

    ranked = sorted(ranked, key=lambda entry: entry["residual"])
    overall_best = ranked[0] if ranked else None

    return {
        "ranked": ranked,
        "overall_best": overall_best,
    }
