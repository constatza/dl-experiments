"""Condition-number diagnostics for matrices and preconditioners."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as linalg
from loguru import logger


def precondition_matrix(
    preconditioner: Callable[[np.ndarray], np.ndarray],
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
    columns = [
        np.asarray(preconditioner(matrix[:, idx]), dtype=np.float64, copy=False)
        for idx in range(matrix.shape[1])
    ]
    return np.column_stack(columns)


def compute_condition_numbers(
    matrix: np.ndarray,
    preconditioners: dict[str, Callable[[np.ndarray], np.ndarray]],
) -> dict[str, float]:
    cond_numbers: dict[str, float] = {}
    for name, preconditioner in preconditioners.items():
        try:
            precond_matrix = precondition_matrix(preconditioner, matrix)
            cond_numbers[name] = float(linalg.cond(precond_matrix))
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            cond_numbers[name] = float("nan")
            print(f"Warning: could not compute condition number for '{name}': {exc}")
    return cond_numbers


def format_scientific(value: float, sig_figs: int = 4) -> str:
    return f"{value:.{sig_figs}e}"


def plot_condition_numbers(
    cond_numbers: dict[str, float],
    *,
    save_dir: Path | None = None,
    suffix: str = "conditions",
) -> Path | None:
    if not cond_numbers:
        return None
    labels = list(cond_numbers.keys())
    values = [cond_numbers[name] for name in labels]
    fig, ax = plt.subplots()
    bars = ax.bar(labels, values)
    ax.set_title("Condition number by preconditioner")
    ax.set_ylabel("Condition number (2-norm)")
    ax.set_yscale("log")
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            format_scientific(value, sig_figs=4),
            ha="center",
            va="bottom",
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
