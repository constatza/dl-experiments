"""Plotting utilities for graph-cg project."""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


def plot_parity_and_residuals(y_true: np.ndarray, y_pred: np.ndarray,
                             sample: int = 0, save_path: str | Path | None = None,
                             show: bool = False) -> None:
    """Create parity and residuals plots.

    Args:
        y_true: True values
        y_pred: Predicted values
        sample: Sample number for title
        save_path: Path to save plot
        show: Whether to show plot
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # Calculate metrics
    residuals = y_pred - y_true
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals**2)))
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2)) if y_true.size > 0 else 0.0
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Parity plot
    ax = axes[0]
    y_min = float(np.min([y_true.min(), y_pred.min()]))
    y_max = float(np.max([y_true.max(), y_pred.max()]))
    pad = 0.02 * (y_max - y_min) if y_max > y_min else 1.0

    ax.scatter(y_true, y_pred, s=10, alpha=0.7, color=plt.cm.Set1(0.0))
    ax.plot([y_min - pad, y_max + pad], [y_min - pad, y_max + pad],
            linestyle="dashed", color=plt.cm.Set1(0.1), label="y = x")
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Parity — sample {sample}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.text(0.02, 0.98, f"R²={r2:.3f}\nRMSE={rmse:.3e}\nMAE={mae:.3e}",
            transform=ax.transAxes, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    # Residuals plot
    ax = axes[1]
    ax.scatter(y_pred, residuals, s=10, alpha=0.7, color=plt.cm.Set1(0.0))
    ax.axhline(0.0, color=plt.cm.Set1(0.1), linestyle='dashed', label='residual = 0')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (Pred - True)")
    ax.set_title("Residuals vs Predicted")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        print(f"Saved parity+residuals plot to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_residual_history(results: Dict[str, Dict[str, Any]],
                         save_path: str | Path | None = None,
                         show: bool = False) -> None:
    """Plot residual history for different methods.

    Args:
        results: Dictionary of method results with 'residuals' key
        save_path: Path to save plot
        show: Whether to show plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for method_name, result in results.items():
        residuals = result.get('residual_history') or result.get('residuals')
        if residuals:
            iterations = range(len(residuals))
            ax.semilogy(iterations, residuals, 'o-', label=method_name, markersize=4)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Relative Residual $\\|r\\| / \\|b\\|$')
    ax.set_title('Convergence History (Relative)')
    ax.grid(True, alpha=0.3)
    ax.legend()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        print(f"Saved residual history plot to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_convergence_comparison(results: Dict[str, Dict[str, Any]],
                              save_path: str | Path | None = None,
                              show: bool = False) -> None:
    """Plot convergence comparison between methods.

    Args:
        results: Dictionary of method results
        save_path: Path to save plot
        show: Whether to show plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for method_name, result in results.items():
        residuals = result.get('residual_history') or result.get('residuals')
        if residuals:
            iterations = range(len(residuals))
            ax.semilogy(iterations, residuals, 'o-', label=method_name, markersize=4)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Relative Residual $\\|r\\| / \\|b\\|$')
    ax.set_title('Convergence Comparison (Relative Residual)')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        print(f"Saved convergence comparison plot to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_data_norms(
    data_dir: str | Path,
    save_path: str | Path | None = None,
    show: bool = False
) -> None:
    """Plot RHS and solution norm distributions for a dataset.

    Args:
        data_dir: Path to dataset directory containing .npy files
        save_path: Path to save figure (optional)
        show: Whether to display the plot

    Returns:
        None
    """
    data_dir = Path(data_dir)

    # Load data
    rhs_samples = np.load(data_dir / "rhs-samples.npy")
    sol_samples = np.load(data_dir / "sol-samples.npy")

    # Calculate norms
    rhs_norms = np.linalg.norm(rhs_samples, axis=1)
    sol_norms = np.linalg.norm(sol_samples, axis=1)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot RHS norms histogram
    ax1.hist(rhs_norms, bins=50, alpha=0.7, edgecolor="black")
    ax1.axvline(np.mean(rhs_norms), color="r", linestyle="--", linewidth=2,
                label=f"Mean: {np.mean(rhs_norms):.3e}")
    ax1.axvline(np.median(rhs_norms), color="g", linestyle="--", linewidth=2,
                label=f"Median: {np.median(rhs_norms):.3e}")
    ax1.set_xlabel("||b|| (RHS Norm)", fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    ax1.set_title("RHS Norm Distribution", fontsize=14, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot solution norms histogram
    ax2.hist(sol_norms, bins=50, alpha=0.7, edgecolor="black", color="orange")
    ax2.axvline(np.mean(sol_norms), color="r", linestyle="--", linewidth=2,
                label=f"Mean: {np.mean(sol_norms):.3e}")
    ax2.axvline(np.median(sol_norms), color="g", linestyle="--", linewidth=2,
                label=f"Median: {np.median(sol_norms):.3e}")
    ax2.set_xlabel("||x|| (Solution Norm)", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Solution Norm Distribution", fontsize=14, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add dataset info as suptitle
    dataset_name = data_dir.name
    fig.suptitle(f"Data Norms: {dataset_name}", fontsize=16, fontweight="bold", y=1.02)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved data norms plot to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

