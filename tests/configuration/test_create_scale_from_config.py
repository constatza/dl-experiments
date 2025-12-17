"""Tests for create_scale_from_config() function."""

from __future__ import annotations

import numpy as np
import pytest

from neuralls.normalization import (
    DiagonalScale,
    MatrixScale,
    SpectralScale,
    create_scale_from_config,
)


@pytest.fixture
def sample_matrix() -> np.ndarray:
    """Sample symmetric positive-definite matrix."""
    return np.array(
        [
            [4.0, -1.0, 0.0],
            [-1.0, 4.0, -1.0],
            [0.0, -1.0, 4.0],
        ]
    )


@pytest.fixture
def sample_rhs_samples() -> np.ndarray:
    """Sample RHS vectors."""
    return np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [0.5, 1.0, 1.5],
        ]
    )


@pytest.fixture
def sample_solution_samples() -> np.ndarray:
    """Sample solution vectors."""
    return np.array(
        [
            [0.5, 0.75, 1.0],
            [1.0, 1.5, 2.0],
            [0.25, 0.375, 0.5],
        ]
    )


def test_create_scale_none() -> None:
    """Test 'none' normalization returns None."""
    A = np.eye(3)
    scale = create_scale_from_config("none", A)
    assert scale is None


def test_create_scale_matrix_auto_bound(sample_matrix: np.ndarray) -> None:
    """Test matrix normalization with auto-computed spectral bound."""
    scale = create_scale_from_config("matrix", sample_matrix)
    assert isinstance(scale, MatrixScale)
    assert scale.spectral_radius_bound > 0
    assert scale.dimension_scale == np.sqrt(3)


def test_create_scale_matrix_explicit_bound(sample_matrix: np.ndarray) -> None:
    """Test matrix normalization with explicit spectral bound."""
    explicit_bound = 10.0
    scale = create_scale_from_config(
        "matrix", sample_matrix, spectral_radius_bound=explicit_bound
    )
    assert isinstance(scale, MatrixScale)
    assert scale.spectral_radius_bound == explicit_bound
    assert scale.dimension_scale == np.sqrt(3)


def test_create_scale_diagonal(sample_matrix: np.ndarray) -> None:
    """Test diagonal normalization."""
    scale = create_scale_from_config("diagonal", sample_matrix)
    assert isinstance(scale, DiagonalScale)
    assert len(scale.diagonal_sqrt_inv) == 3
    # diagonal_sqrt_inv = 1/sqrt(diag(A)) = 1/sqrt(4) = 0.5
    np.testing.assert_array_almost_equal(scale.diagonal_sqrt_inv, [0.5, 0.5, 0.5])


def test_create_scale_diagonal_zero_diagonal() -> None:
    """Test diagonal normalization fails with zero diagonal."""
    A = np.array(
        [
            [1.0, 0.5],
            [0.5, 0.0],  # Zero diagonal
        ]
    )
    with pytest.raises(ValueError, match="near-zero diagonal"):
        create_scale_from_config("diagonal", A)


def test_create_scale_spectral_rhs(
    sample_matrix: np.ndarray, sample_rhs_samples: np.ndarray
) -> None:
    """Test spectral normalization with RHS samples."""
    scales = create_scale_from_config(
        "spectral", sample_matrix, rhs_samples=sample_rhs_samples
    )
    assert isinstance(scales, list)
    assert len(scales) == 3
    assert all(isinstance(s, SpectralScale) for s in scales)

    # Type assertion for pyright
    assert isinstance(scales[0], SpectralScale)

    # All scales should share same spectral_norm and dimension_scale
    spectral_norm = scales[0].spectral_norm
    dimension_scale = scales[0].dimension_scale
    for scale in scales:
        assert isinstance(scale, SpectralScale)  # Type assertion
        assert scale.spectral_norm == spectral_norm
        assert scale.dimension_scale == dimension_scale

    # Each scale should have different rhs_norm
    rhs_norms = [scale.rhs_norm for scale in scales if isinstance(scale, SpectralScale)]
    assert len(set(rhs_norms)) == 3  # All unique


def test_create_scale_spectral_solutions(
    sample_matrix: np.ndarray, sample_solution_samples: np.ndarray
) -> None:
    """Test spectral normalization with solution samples."""
    scales = create_scale_from_config(
        "spectral", sample_matrix, solution_samples=sample_solution_samples
    )
    assert isinstance(scales, list)
    assert len(scales) == 3
    assert all(isinstance(s, SpectralScale) for s in scales)

    # Verify RHS is computed correctly: b = A @ x
    for i, scale in enumerate(scales):
        assert isinstance(scale, SpectralScale)  # Type assertion for pyright
        x = sample_solution_samples[i]
        expected_rhs = sample_matrix @ x
        expected_rhs_norm = np.linalg.norm(expected_rhs)
        np.testing.assert_almost_equal(scale.rhs_norm, expected_rhs_norm)


def test_create_scale_spectral_missing_samples(sample_matrix: np.ndarray) -> None:
    """Test spectral normalization fails without samples."""
    with pytest.raises(
        ValueError, match="requires either rhs_samples or solution_samples"
    ):
        create_scale_from_config("spectral", sample_matrix)


def test_create_scale_spectral_both_samples(
    sample_matrix: np.ndarray,
    sample_rhs_samples: np.ndarray,
    sample_solution_samples: np.ndarray,
) -> None:
    """Test spectral normalization fails with both RHS and solution samples."""
    with pytest.raises(
        ValueError, match="Cannot provide both rhs_samples and solution_samples"
    ):
        create_scale_from_config(
            "spectral",
            sample_matrix,
            rhs_samples=sample_rhs_samples,
            solution_samples=sample_solution_samples,
        )


def test_create_scale_invalid_type(sample_matrix: np.ndarray) -> None:
    """Test invalid normalization type raises ValueError."""
    with pytest.raises(ValueError, match="Invalid normalize_type"):
        create_scale_from_config("invalid", sample_matrix)  # type: ignore[arg-type]
