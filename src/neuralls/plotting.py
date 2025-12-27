"""Plotting utilities for graph-cg project."""

from __future__ import annotations
from pathlib import Path
from typing import Any
from collections.abc import Mapping

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from loguru import logger

from .solver.models.result import CGComparisonResult


def plot_parity_and_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample: int = 0,
    save_path: str | Path | None = None,
    show: bool = False,
) -> None:
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

    set1 = cm.get_cmap("Set1")
    ax.scatter(y_true, y_pred, s=10, alpha=0.7, color=set1(0.0))
    ax.plot(
        [y_min - pad, y_max + pad],
        [y_min - pad, y_max + pad],
        linestyle="dashed",
        color=set1(0.1),
        label="y = x",
    )
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Parity — sample {sample}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    ax.text(
        0.02,
        0.98,
        f"R²={r2:.3f}\nRMSE={rmse:.3e}\nMAE={mae:.3e}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
    )

    # Residuals plot
    ax = axes[1]
    ax.scatter(y_pred, residuals, s=10, alpha=0.7, color=set1(0.0))
    ax.axhline(0.0, color=set1(0.1), linestyle="dashed", label="residual = 0")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (Pred - True)")
    ax.set_title("Residuals vs Predicted")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

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


def plot_residual_history(
    results: dict[str, dict[str, Any]],
    save_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """Plot residual history for different methods.

    Args:
        results: Dictionary of method results with 'residuals' key
        save_path: Path to save plot
        show: Whether to show plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for method_name, result in results.items():
        # Handle both dict and dataclass results
        if hasattr(result, "residual_history"):
            residuals = result.residual_history
        elif isinstance(result, dict):
            residuals = result.get("residual_history") or result.get("residuals")
        else:
            residuals = None

        if residuals:
            iterations = range(len(residuals))
            ax.semilogy(iterations, residuals, "o-", label=method_name, markersize=4)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Relative Residual $\\|r\\| / \\|b\\|$")
    ax.set_title("Convergence History (Relative)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        print(f"Saved residual history plot to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_convergence_comparison(
    results: Mapping[str, CGComparisonResult | Mapping[str, Any]],
    metadata: Mapping[str, Any] | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """Plot convergence comparison between preconditioners.

    Args:
        results: Dictionary of method results
        metadata: Optional metadata dict mapping method name -> NeuralPreconditionerMetadata
        save_path: Path to save plot
        show: Whether to show plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    metadata = dict(metadata or {})

    for method_name, result in results.items():
        # Handle both dict and dataclass results
        if isinstance(result, CGComparisonResult):
            residuals = result.residual_history
        elif isinstance(result, Mapping):
            residuals = result.get("residual_history") or result.get("residuals")
        else:
            residuals = None

        if residuals and len(residuals) > 0:
            iterations = range(len(residuals))

            # Build label with metadata if available
            label = method_name
            if method_name in metadata and metadata[method_name] is not None:
                meta = metadata[method_name]
                # Access metadata attributes
                if hasattr(meta, "residual_iters") and hasattr(meta, "applied_iters"):
                    residual_iters = getattr(meta, "residual_iters", None)
                    applied_iters = (
                        getattr(meta, "applied_iters", None) or residual_iters
                    )
                    if residual_iters is not None:
                        label = (
                            f"{method_name} (tr={residual_iters}, ap={applied_iters})"
                        )

            ax.semilogy(iterations, residuals, "o-", label=label, markersize=4)
        else:
            # Log warning for methods with no history
            logger.warning(f"Method '{method_name}' has no residual history to plot")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Relative Residual $\\|r\\| / \\|b\\|$")
    ax.set_title("Convergence Comparison (Preconditioners Only)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

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


def plot_prediction_diagnostics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    sample: int = 0,
    save_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """Create comprehensive diagnostic plots for predictions vs targets.

    This function creates separate visualizations for predictions and targets
    to help identify scaling issues, normalization problems, and data quality issues.

    Args:
        y_pred: Predicted values (1D array)
        y_true: True target values (1D array)
        sample: Sample number for title
        save_path: Path to save plot
        show: Whether to show plot
    """
    y_pred = np.asarray(y_pred).ravel()
    y_true = np.asarray(y_true).ravel()

    # Calculate comprehensive statistics
    pred_stats = {
        "min": float(np.min(y_pred)),
        "max": float(np.max(y_pred)),
        "mean": float(np.mean(y_pred)),
        "median": float(np.median(y_pred)),
        "std": float(np.std(y_pred)),
        "norm": float(np.linalg.norm(y_pred)),
    }

    true_stats = {
        "min": float(np.min(y_true)),
        "max": float(np.max(y_true)),
        "mean": float(np.mean(y_true)),
        "median": float(np.median(y_true)),
        "std": float(np.std(y_true)),
        "norm": float(np.linalg.norm(y_true)),
    }

    error = y_pred - y_true
    error_stats = {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "max_abs": float(np.max(np.abs(error))),
    }

    # Compute scale ratio
    scale_ratio = pred_stats["norm"] / max(true_stats["norm"], 1e-15)

    # Create comprehensive diagnostic figure
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    # Row 1: Value distributions
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(y_pred, bins=50, alpha=0.7, color="blue", edgecolor="black")
    ax1.axvline(
        pred_stats["mean"],
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {pred_stats['mean']:.3e}",
    )
    ax1.axvline(
        pred_stats["median"],
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Median: {pred_stats['median']:.3e}",
    )
    ax1.set_xlabel("Value", fontsize=11)
    ax1.set_ylabel("Count", fontsize=11)
    ax1.set_title("Prediction Distribution", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(y_true, bins=50, alpha=0.7, color="orange", edgecolor="black")
    ax2.axvline(
        true_stats["mean"],
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {true_stats['mean']:.3e}",
    )
    ax2.axvline(
        true_stats["median"],
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Median: {true_stats['median']:.3e}",
    )
    ax2.set_xlabel("Value", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("Target Distribution", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9, loc="upper left")
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(error, bins=50, alpha=0.7, color="red", edgecolor="black")
    ax3.axvline(0, color="black", linestyle="-", linewidth=2)
    ax3.axvline(
        error_stats["mae"],
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f"MAE: {error_stats['mae']:.3e}",
    )
    ax3.set_xlabel("Error (Pred - True)", fontsize=11)
    ax3.set_ylabel("Count", fontsize=11)
    ax3.set_title("Error Distribution", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=9, loc="upper left")
    ax3.grid(True, alpha=0.3)

    # Row 2: Time series plots (showing first N values)
    n_show = min(200, len(y_pred))
    indices = np.arange(n_show)

    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(
        indices, y_pred[:n_show], "b-", linewidth=1, alpha=0.7, label="Predictions"
    )
    ax4.set_xlabel("Index", fontsize=11)
    ax4.set_ylabel("Value", fontsize=11)
    ax4.set_title(
        f"Predictions (first {n_show} values)", fontsize=12, fontweight="bold"
    )
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=9, loc="upper left")

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(
        indices, y_true[:n_show], "orange", linewidth=1, alpha=0.7, label="Targets"
    )
    ax5.set_xlabel("Index", fontsize=11)
    ax5.set_ylabel("Value", fontsize=11)
    ax5.set_title(f"Targets (first {n_show} values)", fontsize=12, fontweight="bold")
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=9, loc="upper left")

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(
        indices, y_pred[:n_show], "b-", linewidth=1, alpha=0.6, label="Predictions"
    )
    ax6.plot(
        indices, y_true[:n_show], "orange", linewidth=1, alpha=0.6, label="Targets"
    )
    ax6.set_xlabel("Index", fontsize=11)
    ax6.set_ylabel("Value", fontsize=11)
    ax6.set_title(f"Overlay (first {n_show} values)", fontsize=12, fontweight="bold")
    ax6.grid(True, alpha=0.3)
    ax6.legend(fontsize=9, loc="upper left")

    # Row 3: Statistical comparisons
    ax7 = fig.add_subplot(gs[2, 0])
    stats_names = ["min", "max", "mean", "median", "std", "norm"]
    pred_vals = [pred_stats[k] for k in stats_names]
    true_vals = [true_stats[k] for k in stats_names]
    x_pos = np.arange(len(stats_names))
    width = 0.35

    ax7.bar(
        x_pos - width / 2,
        pred_vals,
        width,
        label="Predictions",
        alpha=0.8,
        color="blue",
    )
    ax7.bar(
        x_pos + width / 2, true_vals, width, label="Targets", alpha=0.8, color="orange"
    )
    ax7.set_xticks(x_pos)
    ax7.set_xticklabels(stats_names, rotation=45, ha="right", fontsize=9)
    ax7.set_ylabel("Value", fontsize=11)
    ax7.set_title("Statistics Comparison", fontsize=12, fontweight="bold")
    ax7.legend(fontsize=9, loc="upper left")
    ax7.grid(True, alpha=0.3, axis="y")
    ax7.set_yscale("symlog")

    ax8 = fig.add_subplot(gs[2, 1])
    # Parity plot
    y_min = float(np.min([y_true.min(), y_pred.min()]))
    y_max = float(np.max([y_true.max(), y_pred.max()]))
    pad = 0.05 * (y_max - y_min) if y_max > y_min else 1.0

    ax8.scatter(y_true, y_pred, s=5, alpha=0.5, color="purple")
    ax8.plot(
        [y_min - pad, y_max + pad],
        [y_min - pad, y_max + pad],
        "k--",
        linewidth=2,
        label="Perfect prediction",
    )
    ax8.set_xlabel("True Values", fontsize=11)
    ax8.set_ylabel("Predicted Values", fontsize=11)
    ax8.set_title("Parity Plot", fontsize=12, fontweight="bold")
    ax8.grid(True, alpha=0.3)
    ax8.legend(fontsize=9, loc="upper left")
    ax8.axis("equal")

    ax9 = fig.add_subplot(gs[2, 2])
    # Text summary
    ax9.axis("off")
    summary_text = f"""
DIAGNOSTIC SUMMARY (Sample {sample})

PREDICTIONS:
  Min:    {pred_stats["min"]:.6e}
  Max:    {pred_stats["max"]:.6e}
  Mean:   {pred_stats["mean"]:.6e}
  Median: {pred_stats["median"]:.6e}
  Std:    {pred_stats["std"]:.6e}
  ||pred||: {pred_stats["norm"]:.6e}

TARGETS:
  Min:    {true_stats["min"]:.6e}
  Max:    {true_stats["max"]:.6e}
  Mean:   {true_stats["mean"]:.6e}
  Median: {true_stats["median"]:.6e}
  Std:    {true_stats["std"]:.6e}
  ||true||: {true_stats["norm"]:.6e}

ERRORS:
  MAE:     {error_stats["mae"]:.6e}
  RMSE:    {error_stats["rmse"]:.6e}
  Max Abs: {error_stats["max_abs"]:.6e}

SCALE ANALYSIS:
  ||pred|| / ||true||: {scale_ratio:.6e}

{"⚠️  WARNING: Scale ratio > 10x!" if scale_ratio > 10 or scale_ratio < 0.1 else "✓ Scale ratio looks reasonable"}
    """
    ax9.text(
        0.05,
        0.95,
        summary_text,
        transform=ax9.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    fig.suptitle(
        "Prediction Diagnostics - Scaling Analysis", fontsize=16, fontweight="bold"
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved diagnostic plot to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_data_norms(
    data_dir: str | Path, save_path: str | Path | None = None, show: bool = False
) -> None:
    """Plot RHS and solution norm distributions for a dataset.

    Args:
        data_dir: Path to dataset directory containing .npy files or normalized.npz
        save_path: Path to save figure (optional)
        show: Whether to display the plot

    Returns:
        None
    """
    data_dir = Path(data_dir)

    # Load data from normalized.npz
    from .io_utils import load_dataset

    data = load_dataset(data_dir, variant="normalized")
    rhs_samples = data["rhs"].astype(np.float64, copy=False)
    sol_samples = data["solutions"].astype(np.float64, copy=False)

    # Calculate norms
    rhs_norms = np.linalg.norm(rhs_samples, axis=1)
    sol_norms = np.linalg.norm(sol_samples, axis=1)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot RHS norms histogram
    ax1.hist(rhs_norms, bins=50, alpha=0.7, edgecolor="black")
    ax1.axvline(
        np.mean(rhs_norms),
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(rhs_norms):.3e}",
    )
    ax1.axvline(
        np.median(rhs_norms),
        color="g",
        linestyle="--",
        linewidth=2,
        label=f"Median: {np.median(rhs_norms):.3e}",
    )
    ax1.set_xlabel("||b|| (RHS Norm)", fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    ax1.set_title("RHS Norm Distribution", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Plot solution norms histogram
    ax2.hist(sol_norms, bins=50, alpha=0.7, edgecolor="black", color="orange")
    ax2.axvline(
        np.mean(sol_norms),
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(sol_norms):.3e}",
    )
    ax2.axvline(
        np.median(sol_norms),
        color="g",
        linestyle="--",
        linewidth=2,
        label=f"Median: {np.median(sol_norms):.3e}",
    )
    ax2.set_xlabel("||x|| (Solution Norm)", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Solution Norm Distribution", fontsize=14, fontweight="bold")
    ax2.legend(loc="upper left")
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
