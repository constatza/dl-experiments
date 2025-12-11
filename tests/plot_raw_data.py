#!/usr/bin/env python3
"""Plot raw training data before normalization from configs/experiments.toml config."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
import tomllib

from paths.core import DataPaths, FlowPaths, ProjectRoots, parse_flow_keys


def _coerce_mapping(value: Any) -> Mapping[str, Any]:
    """Coerce value to mapping, return empty dict if not a mapping."""
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def resolve_data_dir(data_config_path: Path | str) -> Path:
    """Return the processed data directory declared by a data config.

    Extracts the data directory path from a data configuration TOML file
    by parsing flow ID, dataset ID, and project roots.
    """
    config_path = Path(data_config_path)
    with open(config_path, "rb") as handle:
        raw_config = tomllib.load(handle)

    output_cfg = _coerce_mapping(raw_config.get("output", {}))
    flow_id, dataset_id = parse_flow_keys(raw_config, config_path=config_path)

    roots = ProjectRoots.from_overrides(
        project_root=output_cfg.get("project_root"),
        processed_root=output_cfg.get("processed_dir"),
        output_root=output_cfg.get("output_root"),
        figures_root=output_cfg.get("figures_root"),
    )
    flow_paths = FlowPaths(flow_id=flow_id, roots=roots)
    data_paths = DataPaths(flow=flow_paths, dataset_id=dataset_id)

    return data_paths.base_dir


def load_raw_features_and_targets(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load raw rhs (rhs) and targets (solutions) from dataset directory.

    Args:
        dataset_dir: Path to the dataset directory

    Returns:
        Tuple of (rhs, targets) as numpy arrays
    """
    # Load from unified normalized.npz format
    normalized_path = dataset_dir / "normalized.npz"

    if not normalized_path.exists():
        raise FileNotFoundError(f"Dataset not found at {normalized_path}")

    data = np.load(normalized_path)
    rhs = data["rhs"].astype(np.float64, copy=False)
    solutions = data["solutions"].astype(np.float64, copy=False)

    return rhs, solutions


def plot_2d_scatter(rhs: np.ndarray, solutions: np.ndarray, title: str) -> plt.Figure:
    """Create 2D scatter plot of raw data.

    Args:
        rhs: Feature array (N, D)
        solutions: Target array (N, D)
        title: Plot title

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(figsize=(14, 12))

    # Flatten to get all values
    rhs_flat = rhs.flatten()
    solutions_flat = solutions.flatten()

    # Plot 4: Distribution histogram of targets
    ax = axes
    ax.scatter(x=rhs_flat, y=solutions_flat)
    ax.set_xlabel("rhs")
    ax.set_ylabel("solutions")
    ax.set_title("Distribution")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # Add main title and statistics
    fig.suptitle(title, fontsize=14, fontweight="bold")

    stats_text = (
        f"Features: {rhs.shape[0]} samples × {rhs.shape[1]} dims\n"
        f"Targets: {solutions.shape[0]} samples × {solutions.shape[1]} dims\n"
        f"Feature range: [{np.min(rhs):.3e}, {np.max(rhs):.3e}]\n"
        f"Target range: [{np.min(solutions):.3e}, {np.max(solutions):.3e}]\n"
        f"Feature mean/std: {np.mean(rhs):.3e} / {np.std(rhs):.3e}\n"
        f"Target mean/std: {np.mean(solutions):.3e} / {np.std(solutions):.3e}"
    )

    fig.text(
        0.5,
        0.02,
        stats_text,
        ha="center",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout(rect=[0, 0.08, 1, 0.96])

    return fig


def load_matrix(dataset_dir: Path) -> np.ndarray:
    """Load matrix from dataset (prefer unnormalized for spectral analysis)."""
    from src.io_utils import load_dataset

    # Prefer unnormalized matrix.npy for spectral analysis
    matrix_path = dataset_dir / "matrix.npy"
    if matrix_path.exists():
        return np.load(matrix_path).astype(np.float64, copy=False)

    # Fallback to normalized.npz
    data = load_dataset(dataset_dir, variant="normalized")
    return data["matrix"].astype(np.float64, copy=False)


def plot_matrix_hist(matrix: np.ndarray, title: str) -> plt.Figure:
    """Create histogram of matrix values."""
    fig, axes = plt.subplots(figsize=(14, 12))
    axes.plot(np.abs(matrix.flatten()), "o")
    axes.set_title(title)
    axes.set_xlabel("position")
    axes.set_ylabel("value")
    axes.semilogy()
    plt.tight_layout()
    return fig


def main() -> None:
    """Load data from configs/experiments.toml and create scatter plot."""
    graph_cg_root = Path(__file__).resolve().parent.parent

    # Load experiment configuration
    experiments_path = graph_cg_root / "configs" / "experiments.toml"

    with open(experiments_path, "rb") as f:
        config = tomllib.load(f)

    if not config.get("experiments"):
        print("No experiments found in configs/experiments.toml")
        return

    # Use the first active experiment
    exp = config["experiments"][0]
    data_config_name = exp["data_config"]
    output_root = Path(config["paths"]["output_root"])

    print(f"Loading data from: {data_config_name}")

    # Find the dataset directory
    data_config_path = graph_cg_root / data_config_name
    dataset_dir = resolve_data_dir(data_config_path)

    print(f"Dataset directory: {dataset_dir}")

    if not dataset_dir.exists():
        print(f"ERROR: Dataset directory does not exist: {dataset_dir}")
        print("Please run data collection first.")
        return

    try:
        # Load the raw data
        features, targets = load_raw_features_and_targets(dataset_dir)
        matrix = load_matrix(dataset_dir)

        print(f"Loaded features: {features.shape}")
        print(f"Loaded targets: {targets.shape}")
        print(f"Loaded matrix: {matrix.shape}")

        fig_mat = plot_matrix_hist(
            matrix, title=f"Matrix Histogram: {data_config_path.stem}"
        )

        # Create the plot
        fig = plot_2d_scatter(
            features,
            targets,
            title=f"Raw Data Before Normalization: {data_config_path.stem}",
        )

        # Save the plot to configs/experiments.toml output_root/figures
        figures_dir = output_root / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        output_path = figures_dir / "raw_data_scatter.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to: {output_path}")
        fig_mat.savefig(figures_dir / "matrix_hist.png", dpi=150, bbox_inches="tight")
        print(f"\nMatrix histogram saved to: {figures_dir / 'matrix_hist.png'}")

        plt.close(fig)

    except Exception as e:
        print(f"Error loading or plotting data: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
