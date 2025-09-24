"""Plotting utilities for visualization of results."""

from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def plot_residual_history(results: dict[str, dict], save_path: str | Path | None = None) -> None:
    """Plot residual history for all methods.

    Args:
        results: Dictionary mapping method name -> info dict with residual_history
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 8))

    for name, info in results.items():
        residual_history = info.get("residual_history", [])
        if residual_history:
            plt.semilogy(residual_history, label=name, marker='o', markersize=4)

    plt.xlabel('CG Iteration')
    plt.ylabel('Residual Norm')
    plt.title('Convergence History Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')

    plt.show()


def plot_convergence_comparison(results: dict[str, dict], save_path: str | Path | None = None) -> None:
    """Plot convergence comparison between different methods.

    Args:
        results: Dictionary mapping method name -> info dict from solver
        save_path: Optional path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Iterations needed for convergence
    names = list(results.keys())
    iterations = [results[name].get("iterations", 0) for name in names]
    converged = [results[name].get("converged", False) for name in names]

    colors = ['green' if conv else 'red' for conv in converged]
    bars = ax1.bar(names, iterations, color=colors, alpha=0.7)
    ax1.set_ylabel('Iterations to Convergence')
    ax1.set_title('Method Performance (Lower is Better)')
    ax1.tick_params(axis='x', rotation=45)

    # Add value labels on bars
    for bar, iters, conv in zip(bars, iterations, converged):
        height = bar.get_height()
        status = "✓" if conv else "✗"
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{iters} {status}', ha='center', va='bottom', fontsize=10)

    # Plot 2: Final residual norms
    residuals = [results[name].get("residual", float('inf')) for name in names]
    bars2 = ax2.bar(names, residuals, color=colors, alpha=0.7)
    ax2.set_ylabel('Final Residual Norm')
    ax2.set_title('Final Residual Comparison (Log Scale)')
    ax2.set_yscale('log')
    ax2.tick_params(axis='x', rotation=45)

    # Add value labels on bars
    for bar, res in zip(bars2, residuals):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                f'{res:.1e}', ha='center', va='bottom', fontsize=10, rotation=45)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')

    plt.show()


def plot_parity_residuals(y_pred: np.ndarray, y_true: np.ndarray, save_path: str | Path | None = None) -> None:
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
    axes[0].plot(lims, lims, 'k--', alpha=0.8, zorder=0)
    axes[0].set_xlabel('True')
    axes[0].set_ylabel('Predicted')
    axes[0].set_title('Parity Plot')
    axes[0].text(0.05, 0.95, f'R² = {r2:.3f}\nRMSE = {rmse:.3e}\nMAE = {mae:.3e}',
                transform=axes[0].transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Residuals plot
    axes[1].scatter(y_true, residuals, alpha=0.6, s=20)
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.8)
    axes[1].set_xlabel('True')
    axes[1].set_ylabel('Residuals (Pred - True)')
    axes[1].set_title('Residuals Plot')

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')

    plt.show()


def plot_noise_robustness(noise_results: dict[str, dict[str, dict]], save_path: str | Path | None = None) -> None:
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
    method_colors = {method: colors[i % len(colors)] for i, method in enumerate(methods)}

    plt.figure(figsize=(12, 8))

    for method in methods:
        iterations = []
        levels_numeric = []

        for level in noise_levels:
            if method in noise_results[level]:
                iters = noise_results[level][method].get("iterations", 0)
                iterations.append(iters)
                levels_numeric.append(float(level))

        if iterations:
            plt.plot(levels_numeric, iterations, 'o-',
                    label=method, color=method_colors[method], linewidth=2, markersize=6)

    plt.xlabel('Noise Level (%)')
    plt.ylabel('CG Iterations to Convergence')
    plt.title('Noise Robustness Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')

    plt.show()