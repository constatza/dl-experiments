"""Data generation strategies for training data creation."""

from __future__ import annotations
from typing import Callable
import numpy as np
from scipy.linalg import norm


def rng_from_seed(seed: int | None) -> np.random.Generator:
    """Create random number generator from seed.

    Args:
        seed: Random seed (None for random)

    Returns:
        Random number generator
    """
    return np.random.default_rng(seed) if seed is not None else np.random.default_rng()


def rounded_counts(total: int, proportions: dict[str, float]) -> dict[str, int]:
    """Convert proportions to integer counts that sum exactly to total.

    Args:
        total: Total number of samples
        proportions: Dictionary of strategy -> proportion

    Returns:
        Dictionary of strategy -> count
    """
    # Compute raw counts
    raw_counts = {k: v * total for k, v in proportions.items()}
    int_counts = {k: int(v) for k, v in raw_counts.items()}
    remainder = {k: raw_counts[k] - int_counts[k] for k in raw_counts}

    # Distribute remainder using largest-remainder method
    total_assigned = sum(int_counts.values())
    remaining = total - total_assigned

    # Sort by remainder descending and assign extra samples
    sorted_keys = sorted(remainder.keys(), key=lambda k: remainder[k], reverse=True)
    for i in range(remaining):
        key = sorted_keys[i % len(sorted_keys)]
        int_counts[key] += 1

    return int_counts


def normal_strategy(
    A: np.ndarray,
    b: np.ndarray,
    count: int,
    rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Generate samples using normal random solutions.

    Args:
        A: System matrix
        b: RHS vector
        count: Number of samples to generate
        rng: Random number generator

    Returns:
        Tuple of (features, targets) where features are RHS and targets are solutions
    """
    n = A.shape[0]
    X = rng.normal(size=(count, n))  # Random solutions
    R = np.array([A @ x for x in X])  # Corresponding RHS vectors
    return R, X


def krylov_strategy(
    A: np.ndarray,
    b: np.ndarray,
    count: int,
    krylov_iters: int,
    rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Generate samples using Krylov subspace method.

    Args:
        A: System matrix
        b: RHS vector
        count: Number of samples to generate
        krylov_iters: Number of CG iterations to run
        rng: Random number generator

    Returns:
        Tuple of (features, targets) where features are RHS and targets are solutions
    """
    n = A.shape[0]
    R = []
    X = []

    for _ in range(count):
        # Random starting point and random RHS
        x0 = rng.normal(size=n)
        b_rand = rng.normal(size=n)

        # Run a few CG iterations
        x = x0.copy()
        r = b_rand - A @ x
        p = r.copy()
        rr_old = np.dot(r, r)

        for _ in range(krylov_iters):
            if norm(r) < 1e-12:
                break

            Ap = A @ p
            pAp = np.dot(p, Ap)

            if abs(pAp) < 1e-15:
                break

            alpha = rr_old / pAp
            x += alpha * p
            r -= alpha * Ap

            rr_new = np.dot(r, r)
            beta = rr_new / rr_old
            p = r + beta * p
            rr_old = rr_new

        R.append(b_rand)
        X.append(x)

    return np.array(R), np.array(X)


# Registry of available strategies
STRATEGY_REGISTRY: dict[str, Callable] = {
    "normal": normal_strategy,
    "krylov": krylov_strategy,
}


def generate_mixture(
    A: np.ndarray,
    b: np.ndarray,
    mix: dict[str, float],
    total: int,
    krylov_iters: int = 15,
    seed: int = 42,
    shuffle: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate mixed training data from multiple strategies.

    Args:
        A: System matrix
        b: RHS vector (not used directly, but kept for interface compatibility)
        mix: Dictionary of strategy_name -> proportion
        total: Total number of samples
        krylov_iters: Number of CG iterations for Krylov strategy
        seed: Random seed
        shuffle: Whether to shuffle final dataset

    Returns:
        Tuple of (features, targets) arrays
    """
    rng = rng_from_seed(seed)

    # Validate mix proportions
    if abs(sum(mix.values()) - 1.0) > 1e-6:
        raise ValueError(f"Mix proportions must sum to 1.0, got {sum(mix.values())}")

    # Convert proportions to counts
    counts = rounded_counts(total, mix)

    all_features = []
    all_targets = []

    for strategy_name, count in counts.items():
        if count == 0:
            continue

        if strategy_name not in STRATEGY_REGISTRY:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        strategy_func = STRATEGY_REGISTRY[strategy_name]

        if strategy_name == "krylov":
            features, targets = strategy_func(A, b, count, krylov_iters, rng)
        else:
            features, targets = strategy_func(A, b, count, rng)

        all_features.append(features)
        all_targets.append(targets)

    # Concatenate all samples
    X = np.vstack(all_features)
    Y = np.vstack(all_targets)

    # Shuffle if requested
    if shuffle:
        indices = rng.permutation(len(X))
        X = X[indices]
        Y = Y[indices]

    return X, Y