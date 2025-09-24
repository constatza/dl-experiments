"""Conjugate Gradient solver implementations."""

from __future__ import annotations
from collections.abc import Callable
from typing import Literal
import numpy as np
from scipy.linalg import norm


def preconditioned_cg(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    *,
    tol: float = 1e-8,
    max_iter: int = 100,
    preconditioner: Callable[[np.ndarray], np.ndarray] | None = None,
    stopping_criterion: Literal["tolerance", "fixed_iterations"] = "tolerance",
) -> tuple[np.ndarray, dict]:
    """Preconditioned Conjugate Gradient solver.

    Args:
        A: System matrix
        b: Right-hand side vector
        x0: Initial guess
        tol: Convergence tolerance
        max_iter: Maximum iterations
        preconditioner: Preconditioner function (None for identity)
        stopping_criterion: When to stop ("tolerance" or "fixed_iterations")

    Returns:
        Tuple of (solution, info_dict)
    """
    if preconditioner is None:
        preconditioner = lambda x: x

    x = x0.copy()
    r = b - A.dot(x)
    z = preconditioner(r)
    p = z.copy()
    rz_old = np.dot(r, z)

    residual_norm = norm(r)
    residuals = [residual_norm]

    converged = False
    for k in range(max_iter):
        if stopping_criterion == "tolerance" and residual_norm < tol:
            converged = True
            break

        Ap = A.dot(p)
        pAp = np.dot(p, Ap)

        if abs(pAp) < 1e-15:
            break

        alpha = rz_old / pAp
        x += alpha * p
        r -= alpha * Ap

        residual_norm = norm(r)
        residuals.append(residual_norm)

        if stopping_criterion == "tolerance" and residual_norm < tol:
            converged = True
            break

        z = preconditioner(r)
        rz_new = np.dot(r, z)

        if abs(rz_old) < 1e-15:
            break

        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new

    # Final convergence check for fixed_iterations mode
    if stopping_criterion == "fixed_iterations":
        converged = residual_norm < tol

    info = {
        "converged": converged,
        "iterations": len(residuals) - 1,
        "residual": residual_norm,
        "residual_history": residuals,
    }

    return x, info


def run_cg_comparison(
    A: np.ndarray,
    b: np.ndarray,
    *,
    preconditioners: dict[str, Callable[[np.ndarray], np.ndarray]],
    warm_starts: dict[str, Callable[[np.ndarray], np.ndarray | None]] | None = None,
    x0: np.ndarray | None = None,
    tol: float = 1e-8,
    max_iter: int = 100,
    stopping_criterion: Literal["tolerance", "fixed_iterations"] = "tolerance",
) -> dict[str, dict]:
    """Run CG with multiple preconditioners and warm starts for comparison.

    Args:
        A: System matrix
        b: Right-hand side vector
        preconditioners: Dict mapping names to preconditioner functions
        warm_starts: Optional dict mapping names to initial-guess providers
        x0: Initial guess (zeros if None)
        tol: Convergence tolerance
        max_iter: Maximum iterations
        stopping_criterion: When to stop

    Returns:
        Dictionary mapping combo name -> results
    """
    if x0 is None:
        x0 = np.zeros_like(b)

    if warm_starts is None:
        warm_starts = {"none": lambda _: None}

    warm_starts = dict(warm_starts)
    if "none" not in warm_starts:
        warm_starts["none"] = lambda _: None

    # Compute exact solution for error analysis
    x_exact = np.linalg.solve(A, b)

    results = {}
    def combo_label(warm_name: str, precond_name: str) -> str:
        if warm_name == "none" and precond_name == "none":
            return "none"
        if warm_name == "none":
            return precond_name
        if precond_name == "none":
            return warm_name
        return f"{warm_name}+{precond_name}"

    for warm_name, warm_fn in warm_starts.items():
        try:
            warm_guess = warm_fn(b) if warm_fn is not None else None
            if warm_guess is None:
                warm_x0 = x0
            else:
                warm_x0 = np.asarray(warm_guess, dtype=b.dtype)
                if warm_x0.shape != b.shape:
                    warm_x0 = warm_x0.reshape(b.shape)
        except Exception as exc:  # noqa: BLE001
            label = combo_label(warm_name, "none")
            results[label] = {
                "converged": False,
                "iterations": 0,
                "residual": float("inf"),
                "residual_history": [],
                "x": x0.copy(),
                "exact_error": None,
                "warm_start": warm_name,
                "preconditioner": "none",
                "error": f"Warm start '{warm_name}' failed: {exc}",
            }
            continue

        for precond_name, precond in preconditioners.items():
            if warm_name != "none" and precond_name != "none":
                continue

            label = combo_label(warm_name, precond_name)
            precond_fn = precond if precond_name != "none" else None

            x_sol, info = preconditioned_cg(
                A,
                b,
                warm_x0,
                tol=tol,
                max_iter=max_iter,
                preconditioner=precond_fn,
                stopping_criterion=stopping_criterion,
            )

            info["x"] = x_sol
            info["exact_error"] = norm(x_sol - x_exact) / norm(x_exact)
            info["warm_start"] = warm_name
            info["preconditioner"] = precond_name

            results[label] = info

    return results


def format_results_summary(results: dict[str, dict]) -> str:
    """Format CG results into a readable summary.

    Args:
        results: Results dictionary from run_cg_comparison

    Returns:
        Formatted summary string
    """
    lines = ["Preconditioned CG results:"]
    for name, info in results.items():
        status = "ok" if info.get("converged") else "fail"
        iters = info.get("iterations", 0)
        res = info.get("residual", float('inf'))
        exact_err = info.get("exact_error")

        if exact_err is not None:
            line = (f"- {name:<18} status={status:<4} iters={iters:>3}  "
                    f"resid={res:.3e}  exact_err={exact_err:.3e}")
        else:
            line = f"- {name:<18} status={status:<4} iters={iters:>3}  resid={res:.3e}"

        warm_name = info.get("warm_start")
        precond_name = info.get("preconditioner")
        if warm_name is not None or precond_name is not None:
            line += f"  warm={warm_name}  precond={precond_name}"

        if info.get("error"):
            line += f"  note={info['error']}"

        lines.append(line)

    return "\n".join(lines)
