"""Condition-number diagnostics for matrices and preconditioners."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as linalg
import torch
from loguru import logger

PreconditionerCallable = Callable[[torch.Tensor], torch.Tensor]


def precondition_matrix(
    preconditioner: PreconditionerCallable,
    matrix: np.ndarray,
) -> np.ndarray:
    """Precondition a matrix by applying preconditioner to each column.

    Args:
        preconditioner: Callable that applies preconditioning (function or Preconditioner)
        matrix: Matrix to precondition

    Returns:
        Preconditioned matrix

    Note:
        All Preconditioner objects are callable via __call__, so this function
        works with both bare functions and Preconditioner objects uniformly.
    """
    matrix_tensor = torch.as_tensor(matrix, dtype=torch.float64)
    columns = [preconditioner(matrix_tensor[:, idx]) for idx in range(matrix.shape[1])]
    return torch.stack(columns, dim=1).detach().cpu().numpy().astype(np.float64, copy=False)


def compute_condition_numbers(
    matrix: np.ndarray,
    preconditioners: dict[str, PreconditionerCallable],
) -> dict[str, float]:
    cond_numbers: dict[str, float] = {}
    for name, preconditioner in preconditioners.items():
        try:
            precond_matrix = precondition_matrix(preconditioner, matrix)
            cond_numbers[name] = float(linalg.cond(precond_matrix))
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
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
    ax.set_xlabel("Condition number (2-norm)")

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
