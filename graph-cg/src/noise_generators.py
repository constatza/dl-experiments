"""Noise generation strategies for robustness analysis."""

from __future__ import annotations
from typing import Literal
import numpy as np


def global_gaussian_noise(b: np.ndarray, rho: float, seed: int | None = None) -> np.ndarray:
    """Add global Gaussian noise to RHS vector.

    Args:
        b: Original RHS vector
        rho: Noise level as fraction of ||b||_2
        seed: Random seed

    Returns:
        Noisy RHS vector
    """
    rng = np.random.default_rng(seed)
    sigma = rho * np.linalg.norm(b, 2)
    noise = rng.normal(0, sigma, size=b.shape)
    return b + noise


def single_dimension_noise(
    b: np.ndarray,
    dim_idx: int,
    percentage: float,
    seed: int | None = None
) -> np.ndarray:
    """Add noise to a single dimension.

    Args:
        b: Original RHS vector
        dim_idx: Index of dimension to perturb
        percentage: Perturbation as percentage of original value
        seed: Random seed

    Returns:
        Noisy RHS vector
    """
    b_noisy = b.copy()
    perturbation = percentage * b[dim_idx]
    b_noisy[dim_idx] += perturbation
    return b_noisy


def blockwise_noise(
    b: np.ndarray,
    rho: float,
    block_size: int = 4,
    seed: int | None = None
) -> np.ndarray:
    """Add blockwise correlated noise.

    Args:
        b: Original RHS vector
        rho: Noise level as fraction of ||b||_2
        block_size: Size of each block
        seed: Random seed

    Returns:
        Noisy RHS vector
    """
    rng = np.random.default_rng(seed)
    sigma = rho * np.linalg.norm(b, 2)
    n = len(b)
    b_noisy = b.copy()

    # Create blocks and add same noise to each block
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        noise_val = rng.normal(0, sigma)
        b_noisy[start:end] += noise_val

    return b_noisy


def worst_case_noise(
    b: np.ndarray,
    A: np.ndarray,
    alpha: float = 1.0,
    seed: int | None = None
) -> np.ndarray:
    """Add noise in worst-case direction (minimum eigenvalue direction).

    Args:
        b: Original RHS vector
        A: System matrix
        alpha: Scaling factor
        seed: Random seed

    Returns:
        Noisy RHS vector
    """
    rng = np.random.default_rng(seed)

    # Compute minimum eigenvalue and eigenvector
    eigenvals, eigenvecs = np.linalg.eigh(A.T @ A)
    min_idx = np.argmin(eigenvals)
    v_min = eigenvecs[:, min_idx]

    # Compute condition number
    kappa = np.max(eigenvals) / np.max(eigenvals[eigenvals > 1e-12])

    # Add noise in worst direction
    epsilon = rng.choice([-1, 1]) * alpha / kappa
    noise_magnitude = epsilon * np.linalg.norm(b, 2)
    return b + noise_magnitude * v_min


def load_redistribution_noise(
    b: np.ndarray,
    rho: float,
    seed: int | None = None
) -> np.ndarray:
    """Redistribute load between components while preserving total.

    Args:
        b: Original RHS vector
        rho: Redistribution magnitude as fraction of ||b||_2
        seed: Random seed

    Returns:
        Load-redistributed RHS vector
    """
    rng = np.random.default_rng(seed)
    n = len(b)

    # Pick two random indices
    i, j = rng.choice(n, size=2, replace=False)

    # Generate transfer amount
    tau = rng.uniform(-rho * np.linalg.norm(b, 2), rho * np.linalg.norm(b, 2))

    # Redistribute load
    b_noisy = b.copy()
    b_noisy[i] += tau
    b_noisy[j] -= tau

    return b_noisy


def missing_data_noise(
    b: np.ndarray,
    corruption_rate: float,
    seed: int | None = None
) -> np.ndarray:
    """Set random components to zero (missing data).

    Args:
        b: Original RHS vector
        corruption_rate: Fraction of components to set to zero
        seed: Random seed

    Returns:
        RHS vector with missing data
    """
    rng = np.random.default_rng(seed)
    n = len(b)
    n_corrupt = int(corruption_rate * n)

    b_noisy = b.copy()
    corrupt_indices = rng.choice(n, size=n_corrupt, replace=False)
    b_noisy[corrupt_indices] = 0

    return b_noisy


def corrupted_data_noise(
    b: np.ndarray,
    corruption_rate: float,
    seed: int | None = None
) -> np.ndarray:
    """Replace random components with random values.

    Args:
        b: Original RHS vector
        corruption_rate: Fraction of components to corrupt
        seed: Random seed

    Returns:
        RHS vector with corrupted data
    """
    rng = np.random.default_rng(seed)
    n = len(b)
    n_corrupt = int(corruption_rate * n)

    b_noisy = b.copy()
    corrupt_indices = rng.choice(n, size=n_corrupt, replace=False)

    # Replace with random values with larger variance
    corrupt_values = rng.normal(0, 2 * np.linalg.norm(b, 2), size=n_corrupt)
    b_noisy[corrupt_indices] = corrupt_values

    return b_noisy


def extreme_magnitude_noise(
    b: np.ndarray,
    rho: float,
    seed: int | None = None
) -> np.ndarray:
    """Add extreme magnitude noise with bounded infinity norm.

    Args:
        b: Original RHS vector
        rho: Bound as fraction of ||b||_2
        seed: Random seed

    Returns:
        RHS vector with extreme magnitude noise
    """
    rng = np.random.default_rng(seed)
    n = len(b)

    # Generate uniform noise with bounded infinity norm
    bound = rho * np.linalg.norm(b, 2)
    noise = rng.uniform(-bound, bound, size=n)

    return b + noise


def create_noise_strategy(
    strategy: Literal[
        "none", "global", "single_dim", "blockwise", "worst_case",
        "load_redistribution", "missing_data", "corrupted_data", "extreme_magnitude"
    ],
    rho: float = 0.05,
    dim_idx: int | None = None,
    seed: int | None = None,
    A: np.ndarray | None = None
) -> callable:
    """Create a noise strategy function.

    Args:
        strategy: Noise strategy name
        rho: Noise parameter (meaning depends on strategy)
        dim_idx: Dimension index for single_dim strategy
        seed: Random seed
        A: System matrix (required for worst_case strategy)

    Returns:
        Function that takes RHS vector and returns noisy RHS vector
    """
    if strategy == "none":
        return lambda b: b
    elif strategy == "global":
        return lambda b: global_gaussian_noise(b, rho, seed)
    elif strategy == "single_dim":
        if dim_idx is None:
            dim_idx = 0
        return lambda b: single_dimension_noise(b, dim_idx, rho, seed)
    elif strategy == "blockwise":
        return lambda b: blockwise_noise(b, rho, seed=seed)
    elif strategy == "worst_case":
        if A is None:
            raise ValueError("worst_case strategy requires matrix A")
        return lambda b: worst_case_noise(b, A, rho, seed)
    elif strategy == "load_redistribution":
        return lambda b: load_redistribution_noise(b, rho, seed)
    elif strategy == "missing_data":
        return lambda b: missing_data_noise(b, rho, seed)
    elif strategy == "corrupted_data":
        return lambda b: corrupted_data_noise(b, rho, seed)
    elif strategy == "extreme_magnitude":
        return lambda b: extreme_magnitude_noise(b, rho, seed)
    else:
        raise ValueError(f"Unknown noise strategy: {strategy}")