"""Tests for denormalization methods in normalization module.

Verifies that denormalization is the inverse of normalization for all scale types.
"""

import numpy as np
import pytest

from neuralls.domain.normalization import (
    LinearSystemBatch,
    make_diagonal_scale,
    make_matrix_scale,
    make_spectral_scales,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def test_matrix() -> np.ndarray:
    """Small SPD test matrix."""
    return np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)


@pytest.fixture
def test_solution() -> np.ndarray:
    """Test solution vector."""
    return np.array([1.0, 2.0], dtype=np.float64)


@pytest.fixture
def test_rhs(test_matrix: np.ndarray, test_solution: np.ndarray) -> np.ndarray:
    """Test RHS vector (computed from A @ x)."""
    return test_matrix @ test_solution


@pytest.fixture
def test_system_batch(
    test_matrix: np.ndarray,
    test_rhs: np.ndarray,
    test_solution: np.ndarray,
) -> LinearSystemBatch:
    """Linear system batch with two samples where A @ x = b."""
    return LinearSystemBatch(
        _matrix=test_matrix,
        _rhs_samples=np.array([test_rhs, test_rhs * 2.0]),
        _sol_samples=np.array([test_solution, test_solution * 2.0]),
    )


# =============================================================================
# Test MatrixScale Denormalization
# =============================================================================


def test_matrix_scale_denormalize_matrix_is_inverse(
    test_system_batch: LinearSystemBatch,
) -> None:
    """MatrixScale.denormalize_matrix inverts scale_matrix."""
    scale = make_matrix_scale(test_system_batch)
    matrix_normalized = scale.scale_matrix(test_system_batch.matrix)
    matrix_recovered = scale.denormalize_matrix(matrix_normalized)

    np.testing.assert_allclose(matrix_recovered, test_system_batch.matrix, rtol=1e-10)


def test_matrix_scale_denormalize_rhs_is_inverse(
    test_system_batch: LinearSystemBatch,
) -> None:
    """MatrixScale.denormalize_rhs inverts scale_rhs."""
    scale = make_matrix_scale(test_system_batch)
    for rhs in test_system_batch.rhs_samples:
        rhs_normalized = scale.scale_rhs(rhs)
        rhs_recovered = scale.denormalize_rhs(rhs_normalized)

        np.testing.assert_allclose(rhs_recovered, rhs, rtol=1e-10)


def test_matrix_scale_denormalize_solution_is_inverse(
    test_system_batch: LinearSystemBatch,
) -> None:
    """MatrixScale.denormalize_solution inverts scale_solution (identity)."""
    scale = make_matrix_scale(test_system_batch)
    for sol in test_system_batch.sol_samples:
        sol_normalized = scale.scale_solution(sol)
        sol_recovered = scale.denormalize_solution(sol_normalized)

        np.testing.assert_allclose(sol_recovered, sol, rtol=1e-10)


def test_matrix_scale_solution_unchanged(test_system_batch: LinearSystemBatch) -> None:
    """MatrixScale leaves solutions unchanged (for numerical stability)."""
    scale = make_matrix_scale(test_system_batch)
    for sol in test_system_batch.sol_samples:
        sol_normalized = scale.scale_solution(sol)
        sol_denormalized = scale.denormalize_solution(sol_normalized)

        # Both operations are identity
        np.testing.assert_array_equal(sol_normalized, sol)
        np.testing.assert_array_equal(sol_denormalized, sol)


# =============================================================================
# Test DiagonalScale Denormalization
# =============================================================================


def test_diagonal_scale_denormalize_matrix_is_inverse(
    test_system_batch: LinearSystemBatch,
) -> None:
    """DiagonalScale.denormalize_matrix inverts scale_matrix."""
    scale = make_diagonal_scale(test_system_batch)
    matrix_normalized = scale.scale_matrix(test_system_batch.matrix)
    matrix_recovered = scale.denormalize_matrix(matrix_normalized)

    np.testing.assert_allclose(matrix_recovered, test_system_batch.matrix, rtol=1e-10)


def test_diagonal_scale_denormalize_rhs_is_inverse(
    test_system_batch: LinearSystemBatch,
) -> None:
    """DiagonalScale.denormalize_rhs inverts scale_rhs."""
    scale = make_diagonal_scale(test_system_batch)
    for rhs in test_system_batch.rhs_samples:
        rhs_normalized = scale.scale_rhs(rhs)
        rhs_recovered = scale.denormalize_rhs(rhs_normalized)

        np.testing.assert_allclose(rhs_recovered, rhs, rtol=1e-10)


def test_diagonal_scale_denormalize_solution_is_inverse(
    test_system_batch: LinearSystemBatch,
) -> None:
    """DiagonalScale.denormalize_solution inverts scale_solution (identity)."""
    scale = make_diagonal_scale(test_system_batch)
    for sol in test_system_batch.sol_samples:
        sol_normalized = scale.scale_solution(sol)
        sol_recovered = scale.denormalize_solution(sol_normalized)

        np.testing.assert_allclose(sol_recovered, sol, rtol=1e-10)


def test_diagonal_scale_solution_scaling(test_system_batch: LinearSystemBatch) -> None:
    """DiagonalScale scales solutions by D^(1/2) in forward direction."""
    scale = make_diagonal_scale(test_system_batch)
    for sol in test_system_batch.sol_samples:
        sol_normalized = scale.scale_solution(sol)

        # Solution is scaled by D^(1/2) = 1/diagonal_sqrt_inv
        expected_sol_normalized = sol * scale.diagonal_sqrt
        np.testing.assert_allclose(sol_normalized, expected_sol_normalized, rtol=1e-10)

        # Denormalization recovers original
        sol_denormalized = scale.denormalize_solution(sol_normalized)
        np.testing.assert_allclose(sol_denormalized, sol, rtol=1e-10)


def test_diagonal_scale_normalized_diagonal(
    test_system_batch: LinearSystemBatch,
) -> None:
    """DiagonalScale produces identity diagonal after normalization."""
    scale = make_diagonal_scale(test_system_batch)
    matrix_normalized = scale.scale_matrix(test_system_batch.matrix)

    diagonal = np.diag(matrix_normalized)
    np.testing.assert_allclose(diagonal, np.ones(len(diagonal)), rtol=1e-10)


# =============================================================================
# Test SpectralScale Denormalization
# =============================================================================


def test_spectral_scale_denormalize_matrix_is_inverse(
    test_system_batch: LinearSystemBatch,
) -> None:
    """SpectralScale.denormalize_matrix inverts scale_matrix."""
    scales = make_spectral_scales(test_system_batch)
    scale = scales[0]  # Use first scale (matrix scaling is same for all)

    matrix_normalized = scale.scale_matrix(test_system_batch.matrix)
    matrix_recovered = scale.denormalize_matrix(matrix_normalized)

    np.testing.assert_allclose(matrix_recovered, test_system_batch.matrix, rtol=1e-10)


def test_spectral_scale_denormalize_rhs_is_inverse(
    test_system_batch: LinearSystemBatch,
) -> None:
    """SpectralScale.denormalize_rhs inverts scale_rhs."""
    scales = make_spectral_scales(test_system_batch)

    for scale, rhs in zip(scales, test_system_batch.rhs_samples):
        rhs_normalized = scale.scale_rhs(rhs)
        rhs_recovered = scale.denormalize_rhs(rhs_normalized)

        np.testing.assert_allclose(rhs_recovered, rhs, rtol=1e-10)


def test_spectral_scale_denormalize_solution_is_inverse(
    test_system_batch: LinearSystemBatch,
) -> None:
    """SpectralScale.denormalize_solution inverts scale_solution."""
    scales = make_spectral_scales(test_system_batch)

    for scale, sol in zip(scales, test_system_batch.sol_samples):
        sol_normalized = scale.scale_solution(sol)
        sol_recovered = scale.denormalize_solution(sol_normalized)

        np.testing.assert_allclose(sol_recovered, sol, rtol=1e-10)


def test_spectral_scale_solution_transformed(
    test_system_batch: LinearSystemBatch,
) -> None:
    """SpectralScale transforms solutions (unlike MatrixScale/DiagonalScale)."""
    scales = make_spectral_scales(test_system_batch)

    for scale, sol in zip(scales, test_system_batch.sol_samples):
        sol_normalized = scale.scale_solution(sol)

        # Solution is transformed (not identity)
        assert not np.allclose(sol_normalized, sol)

        # But denormalization recovers original
        sol_recovered = scale.denormalize_solution(sol_normalized)
        np.testing.assert_allclose(sol_recovered, sol, rtol=1e-10)


# =============================================================================
# Test Roundtrip Properties
# =============================================================================


def test_matrix_scale_full_roundtrip(test_system_batch: LinearSystemBatch) -> None:
    """MatrixScale: normalize → denormalize recovers original system."""
    scale = make_matrix_scale(test_system_batch)

    # Normalize
    A_norm = scale.scale_matrix(test_system_batch.matrix)
    R_norm = np.array([scale.scale_rhs(r) for r in test_system_batch.rhs_samples])
    X_norm = np.array([scale.scale_solution(x) for x in test_system_batch.sol_samples])

    # Denormalize
    A_recovered = scale.denormalize_matrix(A_norm)
    R_recovered = np.array([scale.denormalize_rhs(r) for r in R_norm])
    X_recovered = np.array([scale.denormalize_solution(x) for x in X_norm])

    # Verify roundtrip
    np.testing.assert_allclose(A_recovered, test_system_batch.matrix, rtol=1e-10)
    np.testing.assert_allclose(R_recovered, test_system_batch.rhs_samples, rtol=1e-10)
    np.testing.assert_allclose(X_recovered, test_system_batch.sol_samples, rtol=1e-10)


def test_diagonal_scale_full_roundtrip(test_system_batch: LinearSystemBatch) -> None:
    """DiagonalScale: normalize → denormalize recovers original system."""
    scale = make_diagonal_scale(test_system_batch)

    # Normalize
    A_norm = scale.scale_matrix(test_system_batch.matrix)
    R_norm = np.array([scale.scale_rhs(r) for r in test_system_batch.rhs_samples])
    X_norm = np.array([scale.scale_solution(x) for x in test_system_batch.sol_samples])

    # Denormalize
    A_recovered = scale.denormalize_matrix(A_norm)
    R_recovered = np.array([scale.denormalize_rhs(r) for r in R_norm])
    X_recovered = np.array([scale.denormalize_solution(x) for x in X_norm])

    # Verify roundtrip
    np.testing.assert_allclose(A_recovered, test_system_batch.matrix, rtol=1e-10)
    np.testing.assert_allclose(R_recovered, test_system_batch.rhs_samples, rtol=1e-10)
    np.testing.assert_allclose(X_recovered, test_system_batch.sol_samples, rtol=1e-10)


def test_spectral_scale_full_roundtrip(test_system_batch: LinearSystemBatch) -> None:
    """SpectralScale: normalize → denormalize recovers original system."""
    scales = make_spectral_scales(test_system_batch)

    # Normalize (per-sample scales for RHS/solution)
    A_norm = scales[0].scale_matrix(test_system_batch.matrix)
    R_norm = np.array([s.scale_rhs(r) for s, r in zip(scales, test_system_batch.rhs_samples)])
    X_norm = np.array([s.scale_solution(x) for s, x in zip(scales, test_system_batch.sol_samples)])

    # Denormalize
    A_recovered = scales[0].denormalize_matrix(A_norm)
    R_recovered = np.array([s.denormalize_rhs(r) for s, r in zip(scales, R_norm)])
    X_recovered = np.array([s.denormalize_solution(x) for s, x in zip(scales, X_norm)])

    # Verify roundtrip
    np.testing.assert_allclose(A_recovered, test_system_batch.matrix, rtol=1e-10)
    np.testing.assert_allclose(R_recovered, test_system_batch.rhs_samples, rtol=1e-10)
    np.testing.assert_allclose(X_recovered, test_system_batch.sol_samples, rtol=1e-10)


# =============================================================================
# Test Normalized System Consistency
# =============================================================================


def test_matrix_scale_normalized_system_validity(
    test_system_batch: LinearSystemBatch,
) -> None:
    """Verify normalized system A' @ x' = b' holds for MatrixScale."""
    scale = make_matrix_scale(test_system_batch)

    A_norm = scale.scale_matrix(test_system_batch.matrix)
    for rhs, sol in zip(test_system_batch.rhs_samples, test_system_batch.sol_samples):
        b_norm = scale.scale_rhs(rhs)
        x_norm = scale.scale_solution(sol)

        # Check A' @ x' = b'
        result = A_norm @ x_norm
        np.testing.assert_allclose(result, b_norm, rtol=1e-10)


def test_diagonal_scale_normalized_system_validity(
    test_system_batch: LinearSystemBatch,
) -> None:
    """Verify normalized system A' @ x' = b' holds for DiagonalScale."""
    scale = make_diagonal_scale(test_system_batch)

    A_norm = scale.scale_matrix(test_system_batch.matrix)
    for rhs, sol in zip(test_system_batch.rhs_samples, test_system_batch.sol_samples):
        b_norm = scale.scale_rhs(rhs)
        x_norm = scale.scale_solution(sol)

        # Check A' @ x' = b'
        result = A_norm @ x_norm
        np.testing.assert_allclose(result, b_norm, rtol=1e-10)


def test_spectral_scale_normalized_system_validity(
    test_system_batch: LinearSystemBatch,
) -> None:
    """Verify normalized system A' @ x' = b' holds for SpectralScale."""
    scales = make_spectral_scales(test_system_batch)

    A_norm = scales[0].scale_matrix(test_system_batch.matrix)
    for scale, rhs, sol in zip(
        scales, test_system_batch.rhs_samples, test_system_batch.sol_samples
    ):
        b_norm = scale.scale_rhs(rhs)
        x_norm = scale.scale_solution(sol)

        # Check A' @ x' = b'
        result = A_norm @ x_norm
        np.testing.assert_allclose(result, b_norm, rtol=1e-10)
