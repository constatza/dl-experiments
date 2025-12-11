"""PCA training and persistence utilities for preconditioner."""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger

from dlkit.core.training.transforms.pca import PCA


def fit_pca_from_solutions(
    solution_samples_path: str | Path,
    n_components: int,
    *,
    normalize: bool = True
) -> tuple[PCA, Dict[str, Any]]:
    """Fit PCA on solution samples and return fitted model with statistics.

    Args:
        solution_samples_path: Path to solution samples (.npy or .npz file)
        n_components: Number of principal components to compute
        normalize: Whether to normalize solutions before PCA

    Returns:
        Tuple of (fitted PCA model, statistics dict)
    """
    # Load solution samples
    data = np.load(solution_samples_path)
    if isinstance(data, np.lib.npyio.NpzFile):
        # For .npz files, extract the "solutions" array
        solutions = data["solutions"].astype(np.float64, copy=False)
    else:
        # For .npy files, use directly
        solutions = data.astype(np.float64, copy=False)
    logger.info(f"Loaded solution samples: shape={solutions.shape}")

    # Convert to torch tensor
    solutions_tensor = torch.from_numpy(solutions).double()

    # Normalize if requested
    if normalize:
        mean = solutions_tensor.mean(dim=0, keepdim=True)
        std = solutions_tensor.std(dim=0, keepdim=True) + 1e-8
        solutions_tensor = (solutions_tensor - mean) / std
        logger.info(f"Normalized solutions: mean={mean.mean().item():.3e}, std={std.mean().item():.3e}")

    # Create and fit PCA
    pca = PCA(n_components=n_components)
    pca.fit(solutions_tensor)

    # Gather statistics
    stats = {
        'n_samples': solutions.shape[0],
        'n_features': solutions.shape[1],
        'n_components': n_components,
        'explained_variance': pca.explained_variance.numpy() if pca.explained_variance is not None else None,
        'explained_variance_ratio': pca.explained_variance_ratio.numpy() if pca.explained_variance_ratio is not None else None,
        'total_explained_variance': pca.total_explained_variance,
        'normalized': normalize,
    }

    logger.info(f"PCA fitted: {n_components} components")
    logger.info(f"Total explained variance ratio: {stats['total_explained_variance']:.4f}")
    if stats['explained_variance_ratio'] is not None:
        for i, ratio in enumerate(stats['explained_variance_ratio'][:5]):  # Show first 5
            logger.info(f"  Component {i}: {ratio:.4f}")

    return pca, stats


def save_pca_model(
    pca: PCA,
    stats: Dict[str, Any],
    output_path: str | Path
) -> None:
    """Save PCA model and statistics to disk.

    Args:
        pca: Fitted PCA model
        stats: Statistics dictionary
        output_path: Path to save model (.pt extension)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Package for saving
    package = {
        'mean': pca.mean,
        'components': pca.components,
        'explained_variance': pca.explained_variance,
        'explained_variance_ratio': pca.explained_variance_ratio,
        'n_components': pca.n_components,
        'stats': stats,
    }

    torch.save(package, output_path)
    logger.info(f"Saved PCA model to: {output_path}")


def load_pca_model(model_path: str | Path) -> tuple[PCA, Dict[str, Any]]:
    """Load PCA model from disk.

    Args:
        model_path: Path to saved PCA model

    Returns:
        Tuple of (reconstructed PCA model, statistics dict)
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"PCA model not found: {model_path}")

    # Load package
    package = torch.load(model_path, map_location='cpu', weights_only=False)

    # Reconstruct PCA
    n_components = package['n_components']
    n_features = package['components'].shape[1]
    pca = PCA(n_components=n_components, input_shape=(1, n_features))

    # Set fitted attributes
    pca.mean = package['mean']
    pca.components = package['components']
    pca.explained_variance = package['explained_variance']
    pca.explained_variance_ratio = package['explained_variance_ratio']
    pca.fitted = True

    stats = package.get('stats', {})
    logger.info(f"Loaded PCA model from: {model_path}")
    logger.info(f"Components: {n_components}, Features: {n_features}")

    return pca, stats


def get_pca_components_matrix(pca: PCA) -> np.ndarray:
    """Extract eigenvector matrix (V) from PCA model.

    Args:
        pca: Fitted PCA model

    Returns:
        Numpy array of shape (n_features, n_components) - eigenvector matrix V
    """
    if not pca.fitted or pca.components is None:
        raise RuntimeError("PCA model is not fitted")

    # PCA.components has shape (n_components, n_features)
    # We need V with shape (n_features, n_components) for matrix operations
    V = pca.components.T.numpy()
    return V


def plot_variance_ratios(
    stats: Dict[str, Any],
    output_path: str | Path,
    *,
    show_first_n: int = 50
) -> None:
    """Plot explained variance ratios and save to file.

    Args:
        stats: Statistics dictionary from PCA training
        output_path: Path to save plot
        show_first_n: Number of components to show in detail (default: 50)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    variance_ratio = stats['explained_variance_ratio']
    n_components = len(variance_ratio)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Individual explained variance ratios (first N components)
    n_show = min(show_first_n, n_components)
    components = np.arange(1, n_show + 1)
    ax1.bar(components, variance_ratio[:n_show], alpha=0.7, color='steelblue')
    ax1.set_xlabel('Principal Component', fontsize=12)
    ax1.set_ylabel('Explained Variance Ratio', fontsize=12)
    ax1.set_title(f'Explained Variance Ratio (First {n_show} Components)', fontsize=13)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_xlim(0, n_show + 1)

    # Plot 2: Cumulative explained variance
    cumulative_variance = np.cumsum(variance_ratio)
    all_components = np.arange(1, n_components + 1)
    ax2.plot(all_components, cumulative_variance, 'o-', color='darkgreen',
             linewidth=2, markersize=3, alpha=0.7)
    ax2.axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='80% variance')
    ax2.axhline(y=0.9, color='orange', linestyle='--', alpha=0.5, label='90% variance')
    ax2.axhline(y=0.95, color='purple', linestyle='--', alpha=0.5, label='95% variance')
    ax2.set_xlabel('Number of Components', fontsize=12)
    ax2.set_ylabel('Cumulative Explained Variance', fontsize=12)
    ax2.set_title(f'Cumulative Explained Variance (All {n_components} Components)', fontsize=13)
    ax2.legend(loc='lower right')
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, n_components + 1)
    ax2.set_ylim(0, 1.05)

    # Add text annotations
    total_var = stats['total_explained_variance']
    fig.suptitle(
        f"PCA Variance Analysis - {n_components} components capture {total_var:.1%} variance",
        fontsize=14, fontweight='bold'
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved variance ratio plot to: {output_path}")
