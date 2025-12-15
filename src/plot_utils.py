"""Plotting utilities for visualization of results."""

from __future__ import annotations
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


def plot_residual_history(
    results: dict[str, dict],
    save_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """Plot residual history for all methods.

    Args:
        results: Dictionary mapping method name -> info dict with residual_history
        save_path: Optional path to save the plot
    """
    fig = plt.figure(figsize=(12, 8))

    for name, info in results.items():
        # Handle both dict and dataclass results
        if hasattr(info, "residual_history"):
            residual_history = info.residual_history
        elif isinstance(info, dict):
            residual_history = info.get("residual_history", [])
        else:
            residual_history = []

        if residual_history:
            plt.semilogy(residual_history, label=name, marker="o", markersize=4)

    plt.xlabel("CG Iteration")
    plt.ylabel("Relative Residual $\\|r\\| / \\|b\\|$")
    plt.title("Convergence History Comparison (Relative)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_convergence_comparison(
    results: dict[str, dict],
    save_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """Plot convergence comparison between different methods.

    Args:
        results: Dictionary mapping method name -> info dict from solver
        save_path: Optional path to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    for name, info in results.items():
        # Handle both dict and dataclass results
        if hasattr(info, "residual_history"):
            residual_history = info.residual_history
        elif isinstance(info, dict):
            residual_history = info.get("residual_history") or []
        else:
            residual_history = []

        if residual_history:
            iterations = range(len(residual_history))
            ax.semilogy(iterations, residual_history, "o-", label=name, markersize=4)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Relative Residual $\\|r\\| / \\|b\\|$")
    ax.set_title("Convergence History Comparison (Relative)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_parity_residuals(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    save_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """Create a parity (y_true vs y_pred) and residuals plot.

    Args:
        y_pred: Predicted values
        y_true: True values
        save_path: Optional path to save the plot
    """
    # Convert to numpy arrays
    y_pred = np.asarray(y_pred).flatten()
    y_true = np.asarray(y_true).flatten()

    # Basic stats/metrics
    residuals = y_pred - y_true
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals**2)))
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2)) if y_true.size > 0 else 0.0
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Parity plot
    axes[0].scatter(y_true, y_pred, alpha=0.6, s=20)
    lims = [
        np.min([axes[0].get_xlim(), axes[0].get_ylim()]),
        np.max([axes[0].get_xlim(), axes[0].get_ylim()]),
    ]
    axes[0].plot(lims, lims, "k--", alpha=0.8, zorder=0)
    axes[0].set_xlabel("True")
    axes[0].set_ylabel("Predicted")
    axes[0].set_title("Parity Plot")
    axes[0].text(
        0.05,
        0.95,
        f"R² = {r2:.3f}\nRMSE = {rmse:.3e}\nMAE = {mae:.3e}",
        transform=axes[0].transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Residuals plot
    axes[1].scatter(y_pred, residuals, alpha=0.6, s=20)
    axes[1].axhline(y=0, color="k", linestyle="--", alpha=0.8)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Residuals (Pred - True)")
    axes[1].set_title("Residuals Plot")

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_noise_robustness(
    noise_results: dict[str, dict[str, dict]],
    save_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """Plot noise robustness analysis results.

    Args:
        noise_results: Nested dict: noise_level -> method -> results
        save_path: Optional path to save the plot
    """
    noise_levels = sorted(noise_results.keys(), key=float)
    methods = set()
    for level_results in noise_results.values():
        methods.update(level_results.keys())
    methods = sorted(methods)

    # Use matplotlib Set1 colormap for consistent colors
    colors = cm.Set1(np.linspace(0, 1, max(len(methods), 3)))
    method_colors = {
        method: colors[i % len(colors)] for i, method in enumerate(methods)
    }

    fig = plt.figure(figsize=(12, 8))

    for method in methods:
        iterations = []
        levels_numeric = []

        for level in noise_levels:
            if method in noise_results[level]:
                result = noise_results[level][method]
                # Handle both dict and dataclass results
                if hasattr(result, "iterations"):
                    iters = result.iterations
                elif isinstance(result, dict):
                    iters = result.get("iterations", 0)
                else:
                    iters = 0
                iterations.append(iters)
                levels_numeric.append(float(level))

        if iterations:
            plt.plot(
                levels_numeric,
                iterations,
                "o-",
                label=method,
                color=method_colors[method],
                linewidth=2,
                markersize=6,
            )

    plt.xlabel("Noise Level (%)")
    plt.ylabel("CG Iterations to Convergence")
    plt.title("Noise Robustness Analysis")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)
