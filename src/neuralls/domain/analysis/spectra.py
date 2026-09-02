"""Condition-number diagnostics for matrices and preconditioners."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger

PreconditionerCallable = Callable[[torch.Tensor], torch.Tensor]


def _power_iterate(
    apply: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    *,
    maxiter: int,
    tol: float,
) -> float:
    """Estimate the dominant eigenvalue magnitude of ``apply`` via power iteration.

    Forward-only: only ever calls ``apply``, no inverse or transpose needed.
    Valid for operators with a real, positive spectrum (e.g. an
    SPD-preconditioned SPD system), where ``‖apply(v)‖ -> lambda`` as ``v``
    converges to the dominant eigenvector.
    """
    v = x0 / x0.norm()
    eigenvalue = 0.0
    for _ in range(maxiter):
        w = apply(v)
        eigenvalue_new = float(w.norm())
        if eigenvalue_new == 0.0:
            return 0.0
        v = w / eigenvalue_new
        converged = abs(eigenvalue_new - eigenvalue) <= tol * max(eigenvalue_new, 1.0)
        eigenvalue = eigenvalue_new
        if converged:
            break
    return eigenvalue


def _power_iteration_extremes(
    apply: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    *,
    maxiter: int = 1000,
    tol: float = 1e-6,
) -> tuple[float, float]:
    """Estimate the largest and smallest eigenvalues of a linear operator.

    Assumes ``apply`` has a real, positive spectrum — true for any
    SPD-preconditioned SPD system, which is similar to the symmetric
    ``M^{-1/2} A M^{-1/2}``. The smallest eigenvalue comes from a second,
    shifted power iteration (``shift * I - apply``, folded-spectrum trick) —
    still forward-only, no inverse or transpose required.

    Args:
        apply: Linear operator, shape ``(n,) -> (n,)``.
        x0: Starting vector, shape ``(n,)``.
        maxiter: Maximum power-iteration steps per extreme.
        tol: Relative convergence tolerance.

    Returns:
        ``(lambda_max, lambda_min)`` eigenvalue estimates.
    """
    lambda_max = _power_iterate(apply, x0, maxiter=maxiter, tol=tol)
    shift = lambda_max * 1.01 + 1e-12
    lambda_min = shift - _power_iterate(
        lambda x: shift * x - apply(x), x0, maxiter=maxiter, tol=tol
    )
    return lambda_max, lambda_min


def compute_condition_numbers(
    matrix: np.ndarray,
    preconditioners: dict[str, PreconditionerCallable],
) -> dict[str, float]:
    """Estimate the 2-norm condition number of each preconditioned system.

    Matrix-free: never materializes the dense preconditioned matrix and
    never computes an inverse, via (shifted) power iteration on
    ``x -> preconditioner(matrix @ x)``.

    Args:
        matrix: System matrix A, shape ``(n, n)``.
        preconditioners: Preconditioner callables keyed by name.

    Returns:
        Condition number estimate per preconditioner name (NaN on failure).
    """
    matrix_tensor = torch.as_tensor(matrix, dtype=torch.float64)
    x0 = torch.ones(matrix_tensor.shape[1], dtype=torch.float64, device=matrix_tensor.device)
    cond_numbers: dict[str, float] = {}
    for name, preconditioner in preconditioners.items():
        try:
            lambda_max, lambda_min = _power_iteration_extremes(
                lambda x, p=preconditioner: p(matrix_tensor @ x), x0
            )
            if lambda_min <= 0:
                raise ValueError(f"non-positive smallest eigenvalue estimate: {lambda_min}")
            cond_numbers[name] = lambda_max / lambda_min
        except (ValueError, RuntimeError) as exc:
            cond_numbers[name] = float("nan")
            logger.warning("Could not compute condition number for '{}': {}", name, exc)
    return cond_numbers


def format_scientific(value: float, sig_figs: int = 4) -> str:
    return f"{value:.{sig_figs}e}"


def plot_condition_numbers(
    cond_numbers: dict[str, float],
    *,
    save_dir: Path | None = None,
    suffix: str = "conditions",
    title: str | None = None,
    rtol: float | None = None,
    atol: float | None = None,
) -> Path | None:
    """Plot condition numbers for preconditioners as horizontal bar chart.

    Args:
        cond_numbers: Dictionary mapping preconditioner names to condition numbers
        save_dir: Directory to save the plot
        suffix: Suffix for the output filename
        title: Optional title for the plot
        rtol: Optional relative tolerance parameter to display
        atol: Optional absolute tolerance parameter to display

    Returns:
        Path to saved plot, or None if no condition numbers provided
    """
    if not cond_numbers:
        return None
    labels = list(cond_numbers.keys())
    values = [cond_numbers[name] for name in labels]

    # Dynamic figsize based on number of labels
    figsize = (9, max(3, len(labels) * 0.6))
    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.barh(labels, values)
    ax.set_xscale("log")
    ax.set_xlabel("Condition number (λ_max/λ_min)")

    # Build subtitle from non-None parameters
    subtitle_parts = []
    if rtol is not None:
        subtitle_parts.append(f"rtol={rtol:.0e}")
    if atol is not None:
        subtitle_parts.append(f"atol={atol:.0e}")

    if subtitle_parts:
        ax.set_title(", ".join(subtitle_parts), fontsize=9)

    fig.suptitle(title or "Condition Numbers by Preconditioner", fontsize=13, fontweight="bold")

    # Annotations on the right side of bars
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2.0,
            format_scientific(value, sig_figs=4),
            ha="left",
            va="center",
            fontsize=8,
        )

    fig.tight_layout()
    cond_path = None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        cond_path = Path(save_dir) / f"preconditioner_condition_numbers_{suffix}.png"
        fig.savefig(cond_path, dpi=150)
        logger.info(f"Saved condition number plot to: {cond_path}")
    else:
        plt.show()
    plt.close(fig)
    return cond_path
