"""Normalization utilities shared across data strategies.

Centralizes the logic for transforming raw (A, b, x) samples into the
normalized multi-matrix structure that downstream training code expects.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from .math_utils import (
    calculate_spectral_radius_bound,
    get_dimension_scale,
    normalize_by_matrix,
)


@dataclass
class ResidualTraceSamples:
    """Container for residual → solution trace pairs captured during CG runs."""

    residuals: np.ndarray
    solutions: np.ndarray
    sample_indices: np.ndarray
    iteration_indices: np.ndarray


@dataclass
class NormalizationResult:
    """Result of applying a normalization strategy."""

    matrix: np.ndarray
    rhs_samples: np.ndarray
    sol_samples: np.ndarray
    matrix_scale: float = 1.0
    spectral_radius_bound: float | None = None
    spectral_norm: float | None = None
    residual_traces: ResidualTraceSamples | None = None


def _clone_residual_traces(
    traces: ResidualTraceSamples | None,
) -> ResidualTraceSamples | None:
    """Return a defensive copy of residual trace arrays if present."""

    if traces is None:
        return None

    return ResidualTraceSamples(
        residuals=traces.residuals.copy(),
        solutions=traces.solutions.copy(),
        sample_indices=traces.sample_indices.copy(),
        iteration_indices=traces.iteration_indices.copy(),
    )


def _normalize_none(
    A_original: np.ndarray,
    R: np.ndarray,
    X: np.ndarray,
    dataset_dir: Path,
    residual_traces: ResidualTraceSamples | None = None,
) -> NormalizationResult:
    """No-op normalization strategy."""

    return NormalizationResult(
        A_original,
        R,
        X,
        matrix_scale=1.0,
        residual_traces=_clone_residual_traces(residual_traces),
    )


def _normalize_matrix(
    A_original: np.ndarray,
    R: np.ndarray,
    X: np.ndarray,
    dataset_dir: Path,
    residual_traces: ResidualTraceSamples | None = None,
) -> NormalizationResult:
    """Matrix normalization by spectral radius bound."""

    print("Normalizing system by matrix spectral radius bound with dimension scaling...")
    spectral_radius_bound = calculate_spectral_radius_bound(A_original)
    dimension = A_original.shape[0]
    dimension_scale = get_dimension_scale(dimension)
    scale = spectral_radius_bound * dimension_scale

    A_norm, _ = normalize_by_matrix(A_original, R[0], spectral_radius_bound)
    R_norm = R / scale

    residual_norm = None
    if residual_traces is not None:
        residual_norm = ResidualTraceSamples(
            residuals=residual_traces.residuals / scale,
            solutions=residual_traces.solutions.copy(),
            sample_indices=residual_traces.sample_indices.copy(),
            iteration_indices=residual_traces.iteration_indices.copy(),
        )

    print(f"  Spectral radius bound: {spectral_radius_bound:.6e}")
    print(f"  Dimension: {dimension}, sqrt(d): {dimension_scale:.6f}")

    return NormalizationResult(
        A_norm,
        R_norm,
        X,
        matrix_scale=spectral_radius_bound,
        spectral_radius_bound=spectral_radius_bound,
        residual_traces=residual_norm,
    )


def _normalize_spectral(
    A_original: np.ndarray,
    R: np.ndarray,
    X: np.ndarray,
    dataset_dir: Path,
    residual_traces: ResidualTraceSamples | None = None,
) -> NormalizationResult:
    """Spectral normalization (spectral_norm/||rhs|| with sqrt(d) scaling)."""

    print("Applying spectral normalization (spectral_norm/||rhs||, scaled by sqrt(d))...")
    spectral_norm_value = calculate_spectral_radius_bound(A_original)
    dimension = A_original.shape[0]
    dimension_scale = np.sqrt(dimension)
    num_samples = R.shape[0]

    print(f"  Spectral norm bound: {spectral_norm_value:.6e}")
    print(f"  Dimension: {dimension}, sqrt(d): {dimension_scale:.6f}")

    # Matrix normalization is same for all samples (only scaled by spectral norm)
    A_scaled = A_original / spectral_norm_value
    A_norm = A_scaled / dimension_scale

    # Per-sample normalization of RHS and solution
    R_norm = R.copy()
    X_norm = X.copy()
    rhs_norms = np.zeros(num_samples, dtype=np.float64)

    for i in range(num_samples):
        rhs_norm = float(np.linalg.norm(R[i], ord=2))
        if rhs_norm < 1e-15:
            raise ValueError(f"RHS norm too small for normalization: {rhs_norm}")
        rhs_norms[i] = rhs_norm

        # Normalize RHS by its norm and dimension
        R_norm[i] = R[i] / (rhs_norm * dimension_scale)

        # Solution scaled by spectral_norm / rhs_norm
        X_norm[i] = X[i] * spectral_norm_value / rhs_norm

    residual_norm = None
    if residual_traces is not None:
        residuals_scaled = np.empty_like(residual_traces.residuals)
        solutions_scaled = np.empty_like(residual_traces.solutions)
        for idx, sample_idx in enumerate(residual_traces.sample_indices):
            sample_idx_int = int(sample_idx)
            if not 0 <= sample_idx_int < num_samples:
                raise IndexError(
                    f"Residual trace sample index {sample_idx_int} out of bounds (num_samples={num_samples})"
                )
            rhs_norm = rhs_norms[sample_idx_int]
            residuals_scaled[idx] = residual_traces.residuals[idx] / (rhs_norm * dimension_scale)
            solutions_scaled[idx] = (
                residual_traces.solutions[idx] * spectral_norm_value / rhs_norm
            )

        residual_norm = ResidualTraceSamples(
            residuals=residuals_scaled,
            solutions=solutions_scaled,
            sample_indices=residual_traces.sample_indices.copy(),
            iteration_indices=residual_traces.iteration_indices.copy(),
        )

    print(
        "  Normalized matrix spectral norm bound: "
        f"{calculate_spectral_radius_bound(A_norm) * dimension_scale:.6f}"
    )
    print(
        "  RHS norm range: ["
        f"{np.min([np.linalg.norm(R_norm[i]) for i in range(min(10, num_samples))]):.2e}, "
        f"{np.max([np.linalg.norm(R_norm[i]) for i in range(min(10, num_samples))]):.2e}]"
    )
    print(
        "  Solution magnitude range: ["
        f"{np.min([np.linalg.norm(X_norm[i]) for i in range(min(10, num_samples))]):.2e}, "
        f"{np.max([np.linalg.norm(X_norm[i]) for i in range(min(10, num_samples))]):.2e}]"
    )

    return NormalizationResult(
        A_norm,
        R_norm,
        X_norm,
        matrix_scale=spectral_norm_value,
        spectral_norm=spectral_norm_value,
        residual_traces=residual_norm,
    )


_NORMALIZATION_STRATEGIES = {
    "none": _normalize_none,
    "matrix": _normalize_matrix,
    "spectral": _normalize_spectral,
}


def apply_normalization(
    normalize: Literal["none", "matrix", "spectral"],
    A_original: np.ndarray,
    R: np.ndarray,
    X: np.ndarray,
    dataset_dir: Path,
    residual_traces: ResidualTraceSamples | None = None,
) -> NormalizationResult:
    """Apply requested normalization strategy."""

    if normalize not in _NORMALIZATION_STRATEGIES:
        raise ValueError(
            f"Invalid normalize value: {normalize}. "
            f"Must be one of: {list(_NORMALIZATION_STRATEGIES.keys())}"
        )

    strategy = _NORMALIZATION_STRATEGIES[normalize]
    return strategy(A_original, R, X, dataset_dir, residual_traces=residual_traces)


__all__ = [
    "NormalizationResult",
    "ResidualTraceSamples",
    "apply_normalization",
]
