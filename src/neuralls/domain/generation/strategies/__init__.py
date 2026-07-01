"""Data generation strategies."""

from . import (
    constant_strategies,
    gaussian_strategies,
    random_normal,
    krylov,
    residuals,
    search_directions,
    eigenvector,
    rhs_archive,
    solution_archive,
    neutral_ones,
    uniform_strategies,
    validated_archive,
    scaled_solutions,
    sparse_rhs,
)

__all__ = [
    "constant_strategies",
    "gaussian_strategies",
    "random_normal",
    "krylov",
    "residuals",
    "search_directions",
    "eigenvector",
    "rhs_archive",
    "solution_archive",
    "neutral_ones",
    "uniform_strategies",
    "validated_archive",
    "scaled_solutions",
    "sparse_rhs",
]
