"""Conjugate Gradient solver implementations."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np
from scipy.linalg import norm


@dataclass(frozen=True)
class IterationContext:
    """Context passed to flexible preconditioners and helpers."""

    iteration: int
    residual: np.ndarray
    solution: np.ndarray
    matrix: np.ndarray
    rhs: np.ndarray


def _ensure_vector(vector: np.ndarray | None, template: np.ndarray, *, name: str) -> np.ndarray:
    """Ensure helper/preconditioner outputs are float64 arrays with correct shape."""

    if vector is None:
        raise ValueError(f"{name} returned None; expected ndarray")

    arr = np.asarray(vector, dtype=np.float64)
    if arr.shape != template.shape:
        arr = arr.reshape(template.shape)
    return arr


def _wrap_preconditioner(
    preconditioner: Callable[..., np.ndarray] | None,
) -> Callable[[IterationContext], np.ndarray]:
    """Adapt legacy preconditioner signatures to flexible iteration context."""

    if preconditioner is None:
        return lambda ctx: ctx.residual.copy()

    try:
        signature = inspect.signature(preconditioner)
    except (TypeError, ValueError):  # Built-in or C-extension callable without signature
        return lambda ctx: _ensure_vector(preconditioner(ctx.residual), ctx.residual, name="preconditioner")

    if len(signature.parameters) == 1:
        return lambda ctx: _ensure_vector(preconditioner(ctx.residual), ctx.residual, name="preconditioner")

    return lambda ctx: _ensure_vector(preconditioner(ctx), ctx.residual, name="preconditioner")


def _wrap_helper(
    helper: Callable[..., np.ndarray | None] | None,
) -> Callable[[IterationContext], np.ndarray | None] | None:
    """Adapt helper callback to the iteration context signature."""

    if helper is None:
        return None

    try:
        signature = inspect.signature(helper)
    except (TypeError, ValueError):
        return lambda ctx: helper(ctx.residual)

    if len(signature.parameters) == 1:
        return lambda ctx: helper(ctx.residual)

    return lambda ctx: helper(ctx)


def flexible_pcg(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    *,
    tol: float = 1e-8,
    max_iter: int = 100,
    preconditioner: Callable[..., np.ndarray] | None = None,
    helper: Callable[..., np.ndarray | None] | None = None,
    stopping_criterion: Literal["tolerance", "fixed_iterations"] = "tolerance",
    breakdown_tol: float = 1e-12,
) -> tuple[np.ndarray, dict]:
    """Flexible Preconditioned Conjugate Gradient solver with step helpers."""

    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    x = np.asarray(x0, dtype=np.float64).copy()

    preconditioner_fn = _wrap_preconditioner(preconditioner)
    helper_fn = _wrap_helper(helper)

    r = b - A.dot(x)
    rhs_norm = norm(b)
    if rhs_norm == 0:
        rhs_norm = 1.0

    context = IterationContext(
        iteration=0,
        residual=r.copy(),
        solution=x.copy(),
        matrix=A,
        rhs=b,
    )

    z = preconditioner_fn(context)
    helper_iterations: list[int] = []
    helper_norms: list[float] = []

    if helper_fn is not None:
        helper_vec = helper_fn(context)
        if helper_vec is not None:
            helper_arr = _ensure_vector(helper_vec, r, name="helper")
            z = z + helper_arr
            helper_iterations.append(0)
            helper_norms.append(float(norm(helper_arr)))

    p = z.copy()
    rz_old = float(np.dot(r, z))

    residual_norm = norm(r)
    residuals_abs = [residual_norm]
    residuals_rel = [residual_norm / rhs_norm]

    converged = residual_norm < tol if stopping_criterion == "tolerance" else False
    breakdown_occurred = False

    iteration = 0
    while iteration < max_iter:
        if stopping_criterion == "tolerance" and residual_norm < tol:
            converged = True
            break

        Ap = A.dot(p)
        pAp = float(np.dot(p, Ap))
        if abs(pAp) < breakdown_tol:
            breakdown_occurred = True
            break

        alpha = rz_old / pAp
        x = x + alpha * p
        r = r - alpha * Ap

        residual_norm = norm(r)
        residuals_abs.append(residual_norm)
        residuals_rel.append(residual_norm / rhs_norm)

        iteration += 1

        if stopping_criterion == "tolerance" and residual_norm < tol:
            converged = True
            break

        context = IterationContext(
            iteration=iteration,
            residual=r.copy(),
            solution=x.copy(),
            matrix=A,
            rhs=b,
        )

        z = preconditioner_fn(context)

        if helper_fn is not None:
            helper_vec = helper_fn(context)
            if helper_vec is not None:
                helper_arr = _ensure_vector(helper_vec, r, name="helper")
                z = z + helper_arr
                helper_iterations.append(iteration)
                helper_norms.append(float(norm(helper_arr)))

        rz_new = float(np.dot(r, z))
        if abs(rz_old) < breakdown_tol:
            breakdown_occurred = True
            break

        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new

    if stopping_criterion == "fixed_iterations":
        converged = residual_norm < tol

    info = {
        "converged": converged,
        "iterations": iteration,
        "residual": residual_norm / rhs_norm,
        "residual_abs": residual_norm,
        "residual_history": residuals_rel,
        "residual_history_abs": residuals_abs,
        "rhs_norm": rhs_norm,
        "breakdown": breakdown_occurred,
        "helper_iterations": helper_iterations,
        "helper_norms": helper_norms,
    }

    return x, info


def preconditioned_cg(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    *,
    tol: float = 1e-8,
    max_iter: int = 100,
    preconditioner: Callable[[np.ndarray], np.ndarray] | None = None,
    stopping_criterion: Literal["tolerance", "fixed_iterations"] = "tolerance",
    breakdown_tol: float = 1e-12,
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
    x, info = flexible_pcg(
        A,
        b,
        x0,
        tol=tol,
        max_iter=max_iter,
        preconditioner=preconditioner,
        helper=None,
        stopping_criterion=stopping_criterion,
        breakdown_tol=breakdown_tol,
    )

    # Remove helper-specific fields for backwards compatibility
    info.pop("helper_iterations", None)
    info.pop("helper_norms", None)
    return x, info


def run_cg_comparison(
    A: np.ndarray,
    b: np.ndarray,
    *,
    preconditioners: dict[str, Callable[[np.ndarray], np.ndarray]],
    warm_starts: dict[str, Callable[[np.ndarray], np.ndarray | None]] | None = None,
    step_helpers: dict[str, Callable[..., np.ndarray | None]] | None = None,
    x0: np.ndarray | None = None,
    tol: float = 1e-8,
    max_iter: int = 100,
    stopping_criterion: Literal["tolerance", "fixed_iterations"] = "tolerance",
    breakdown_tol: float = 1e-12,
    combination_plan: Sequence[tuple[str, str, str]] | None = None,
) -> dict[str, dict]:
    """Run CG with multiple preconditioners and warm starts for comparison.

    Args:
        A: System matrix
        b: Right-hand side vector
        preconditioners: Dict mapping names to preconditioner functions
        warm_starts: Optional dict mapping names to initial-guess providers
        step_helpers: Optional dict of per-iteration helpers ("none" auto-added)
        x0: Initial guess (zeros if None)
        tol: Convergence tolerance
        max_iter: Maximum iterations
        stopping_criterion: When to stop

    Returns:
        Dictionary mapping combo name -> results
    """
    if x0 is None:
        x0 = np.zeros_like(b, dtype=np.float64)

    if warm_starts is None:
        warm_starts = {"none": lambda _: None}

    warm_starts = dict(warm_starts)
    if "none" not in warm_starts:
        warm_starts["none"] = lambda _: None

    preconditioners = dict(preconditioners)
    if "none" not in preconditioners:
        preconditioners["none"] = lambda residual: residual

    if step_helpers is None:
        step_helpers = {"none": lambda _: None}
    else:
        step_helpers = dict(step_helpers)
    if "none" not in step_helpers:
        step_helpers["none"] = lambda _: None

    # Compute exact solution for error analysis (ensure float64)
    x_exact = np.linalg.solve(A.astype(np.float64, copy=False), b.astype(np.float64, copy=False))

    results = {}
    def combo_label(warm_name: str, precond_name: str, helper_name: str) -> str:
        parts: list[str] = []
        if warm_name != "none":
            parts.append(warm_name)
        if precond_name != "none":
            parts.append(precond_name)
        if helper_name != "none":
            parts.append(helper_name)
        return "+".join(parts) if parts else "none"

    if combination_plan is None:
        combos_iterable = [
            (warm_name, precond_name, helper_name)
            for warm_name in warm_starts.keys()
            for precond_name in preconditioners.keys()
            for helper_name in step_helpers.keys()
            if not (
                precond_name != "none" and (warm_name != "none" or helper_name != "none")
            )
        ]
    else:
        combos_iterable = list(combination_plan)

    seen: set[tuple[str, str, str]] = set()
    combos: list[tuple[str, str, str]] = []
    for combo in combos_iterable:
        if combo not in seen:
            combos.append(combo)
            seen.add(combo)

    combos_by_warm: dict[str, list[tuple[str, str, str]]] = {}
    for warm_name, precond_name, helper_name in combos:
        if warm_name not in warm_starts:
            raise ValueError(f"Unknown warm start '{warm_name}' requested.")
        if precond_name not in preconditioners:
            raise ValueError(f"Unknown preconditioner '{precond_name}' requested.")
        if helper_name not in step_helpers:
            raise ValueError(f"Unknown step helper '{helper_name}' requested.")
        combos_by_warm.setdefault(warm_name, []).append((warm_name, precond_name, helper_name))

    for warm_name, warm_combos in combos_by_warm.items():
        warm_fn = warm_starts[warm_name]
        try:
            warm_guess = warm_fn(b) if warm_fn is not None else None
            if warm_guess is None:
                warm_x0 = x0
            else:
                warm_x0 = np.asarray(warm_guess, dtype=np.float64)
                if warm_x0.shape != b.shape:
                    warm_x0 = warm_x0.reshape(b.shape)
        except Exception as exc:  # noqa: BLE001
            for _, precond_name, helper_name in warm_combos:
                label = combo_label(warm_name, precond_name, helper_name)
                info = {
                    "converged": False,
                    "iterations": 0,
                    "residual": float("inf"),
                    "residual_abs": float("inf"),
                    "residual_history": [],
                    "residual_history_abs": [],
                    "x": x0.copy(),
                    "exact_error": None,
                    "warm_start": warm_name,
                    "preconditioner": precond_name,
                    "step_helper": helper_name,
                    "error": f"Warm start '{warm_name}' failed: {exc}",
                    "breakdown": False,
                }
                info["helper_iterations"] = []
                info["helper_norms"] = []
                results[label] = info
            continue

        for _, precond_name, helper_name in warm_combos:
            precond = preconditioners[precond_name]
            helper_fn = step_helpers[helper_name]
            precond_fn = precond if precond_name != "none" else None
            helper_callable = helper_fn if helper_name != "none" else None
            label = combo_label(warm_name, precond_name, helper_name)

            try:
                x_sol, info = flexible_pcg(
                    A,
                    b,
                    warm_x0,
                    tol=tol,
                    max_iter=max_iter,
                    preconditioner=precond_fn,
                    helper=helper_callable,
                    stopping_criterion=stopping_criterion,
                    breakdown_tol=breakdown_tol,
                )
            except Exception as solver_exc:  # noqa: BLE001
                info = {
                    "converged": False,
                    "iterations": 0,
                    "residual": float("inf"),
                    "residual_abs": float("inf"),
                    "residual_history": [],
                    "residual_history_abs": [],
                    "x": warm_x0.copy(),
                    "exact_error": None,
                    "breakdown": False,
                    "error": f"CG solver failed: {solver_exc}",
                }
                info["helper_iterations"] = []
                info["helper_norms"] = []
                x_sol = warm_x0.copy()
            else:
                info["x"] = x_sol
                exact_norm = norm(x_exact)
                if exact_norm == 0:
                    info["exact_error"] = norm(x_sol - x_exact)
                else:
                    info["exact_error"] = norm(x_sol - x_exact) / exact_norm

            info["warm_start"] = warm_name
            info["preconditioner"] = precond_name
            info["step_helper"] = helper_name
            info["initial_guess"] = warm_x0.copy()

            results[label] = info

    return results


def format_results_summary(results: dict[str, dict]) -> str:
    """Format CG results into a readable summary.

    Args:
        results: Results dictionary from run_cg_comparison

    Returns:
        Formatted summary string
    """
    lines = ["Flexible CG results:"]
    for name, info in results.items():
        status = "ok" if info.get("converged") else "fail"
        iters = info.get("iterations", 0)
        res = info.get("residual", float('inf'))
        exact_err = info.get("exact_error")

        res_abs = info.get("residual_abs")

        if exact_err is not None:
            line = (
                f"- {name:<18} status={status:<4} iters={iters:>3}  "
                f"rel_res={res:.3e}"
            )
            if res_abs is not None:
                line += f" (abs={res_abs:.3e})"
            line += f"  exact_err={exact_err:.3e}"
        else:
            line = f"- {name:<18} status={status:<4} iters={iters:>3}  rel_res={res:.3e}"
            if res_abs is not None:
                line += f" (abs={res_abs:.3e})"

        warm_name = info.get("warm_start")
        precond_name = info.get("preconditioner")
        helper_name = info.get("step_helper")
        warm_label = warm_name if warm_name is not None else "none"
        precond_label = precond_name if precond_name is not None else "none"
        helper_label = helper_name if helper_name not in (None, "none") else None

        if warm_label != "none" or precond_label != "none" or helper_label is not None:
            line += f"  warm={warm_label}  precond={precond_label}"
            if helper_label is not None:
                line += f"  helper={helper_label}"

        helper_iters = info.get("helper_iterations")
        if helper_iters:
            line += f"  helper_calls={len(helper_iters)}"

        if info.get("error"):
            line += f"  note={info['error']}"

        if info.get("breakdown"):
            line += "  note=breakdown"

        lines.append(line)

    return "\n".join(lines)


def summarize_best_combinations(results: dict[str, dict]) -> dict[str, Any]:
    """Summarize best-performing CG combinations by warm start and overall."""

    ranked: list[dict[str, Any]] = []
    for label, info in results.items():
        if not info.get("converged"):
            continue
        iterations = int(info.get("iterations", 0))
        residual = float(info.get("residual", float("inf")))
        residual_abs = float(info.get("residual_abs", float("inf")))
        exact_error = info.get("exact_error")
        ranked.append(
            {
                "label": label,
                "iterations": iterations,
                "residual": residual,
                "residual_abs": residual_abs,
                "exact_error": None if exact_error is None else float(exact_error),
                "warm_start": info.get("warm_start", "none"),
                "preconditioner": info.get("preconditioner", "none"),
                "step_helper": info.get("step_helper", "none"),
            }
        )

    ranked.sort(key=lambda entry: (entry["iterations"], entry["residual"]))

    best_overall = ranked[0] if ranked else None

    best_by_warm: dict[str, dict[str, Any]] = {}
    for entry in ranked:
        warm_name = entry.get("warm_start") or "none"
        existing = best_by_warm.get(warm_name)
        if existing is None or (entry["iterations"], entry["residual"]) < (
            existing["iterations"],
            existing["residual"],
        ):
            best_by_warm[warm_name] = entry

    best_by_precond: dict[str, dict[str, Any]] = {}
    for entry in ranked:
        precond_name = entry.get("preconditioner") or "none"
        existing = best_by_precond.get(precond_name)
        if existing is None or (entry["iterations"], entry["residual"]) < (
            existing["iterations"],
            existing["residual"],
        ):
            best_by_precond[precond_name] = entry

    return {
        "best_overall": best_overall,
        "best_by_warm_start": best_by_warm,
        "best_by_preconditioner": best_by_precond,
        "ranked": ranked,
    }
