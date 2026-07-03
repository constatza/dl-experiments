"""Direct RHS generators for comparison workflows."""

from __future__ import annotations

import numpy as np

from neuralls.platform.config.models.comparison import (
    GaussianRhsSourceModel,
    SparseRhsSourceModel,
)


def generate_gaussian_rhs(
    dim: int,
    cfg: GaussianRhsSourceModel,
    seed: int | None,
) -> np.ndarray:
    """Generate a direct Gaussian RHS vector."""
    rng = np.random.default_rng(seed)
    return rng.normal(loc=cfg.mean, scale=cfg.std, size=dim).astype(np.float64)


def generate_sparse_rhs(dim: int, cfg: SparseRhsSourceModel) -> np.ndarray:
    """Generate a direct sparse RHS vector from explicit index/value pairs."""
    rhs = np.zeros(dim, dtype=np.float64)
    for index, value in zip(cfg.indices, cfg.values, strict=True):
        rhs[index] = value
    return rhs
