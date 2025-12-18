"""Data generation strategies."""

from . import (
    random_normal,
    krylov,
    residual_traces,
    residual_error,
    eigenvector,
    rhs_archive,
    solution_archive,
    neutral_ones,
)

__all__ = [
    "random_normal",
    "krylov",
    "residual_traces",
    "residual_error",
    "eigenvector",
    "rhs_archive",
    "solution_archive",
    "neutral_ones",
]
