"""Plotting utilities for graph-cg project."""

from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List


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
    ax.scatter(y_true, residuals, s=10, alpha=0.7, color=plt.cm.Set1(0.0))
    ax.axhline(0.0, color=plt.cm.Set1(0.1), linestyle='dashed', label='residual = 0')
    ax.set_xlabel("True")
    ax.set_ylabel("Residual (Pred - True)")
    ax.set_title("Residuals vs True")
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
    ax.set_ylabel('Residual Norm')
    ax.set_title('Convergence History')
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    methods: List[str] = []
    iterations: List[int] = []
    final_residuals: List[float] = []
    converged: List[bool] = []
    skipped: List[str] = []

    for method_name, result in results.items():
        res_val = result.get('residual', float('inf'))
        if not np.isfinite(res_val):
            skipped.append(method_name)
            continue

        methods.append(method_name)
        iterations.append(result.get('iterations', 0))
        final_residuals.append(res_val)
        converged.append(result.get('converged', False))

    # Iterations comparison
    colors = ['green' if c else 'red' for c in converged]
    bars1 = ax1.bar(methods, iterations, color=colors, alpha=0.7)
    ax1.set_ylabel('Iterations to Convergence')
    ax1.set_title('Iterations Required')
    ax1.tick_params(axis='x', rotation=45)

    # Add value labels on bars
    for bar, iters, conv in zip(bars1, iterations, converged):
        height = bar.get_height()
        label = str(iters) if conv else f"{iters}*"
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                label, ha='center', va='bottom')

    # Final residuals comparison
    if final_residuals:
        bars2 = ax2.bar(methods, final_residuals, color=colors, alpha=0.7)
        ax2.set_ylabel('Final Residual')
        ax2.set_title('Final Residual Norms')
        ax2.set_yscale('log')
        ax2.tick_params(axis='x', rotation=45)

        for bar, res in zip(bars2, final_residuals):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                     f"{res:.2e}", ha='center', va='bottom')
    else:
        ax2.axis('off')
        ax2.text(0.5, 0.5, 'No finite residuals to plot', ha='center', va='center')

    if skipped:
        note = ", ".join(skipped)
        ax2.text(0.5, 0.05, f"Skipped: {note}", ha='center', va='bottom', transform=ax2.transAxes)

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
