"""Comprehensive tests for the refactored normalization module with ABC design.

Tests cover:
- ABC interfaces (ILinearSystem, ILinearSystemBatch, IScale, ITraceSamples)
- Frozen dataclasses (immutability)
- Pure functions (scale factories, system scaling, trace scaling)
- Strategy functions (normalize_matrix, normalize_diagonal, normalize_spectral, normalize_none)
- Backward compatibility
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import numpy as np
import pytest

from neuralls.shared.normalization import (
    DiagonalScale,
    ErrorTraceSamples,
    LinearSystem,
    LinearSystemBatch,
    MatrixScale,
    NormalizedSystem,
    NormalizationResult,
    ResidualTraceSamples,
    SpectralScale,
    apply_normalization,
    make_diagonal_scale,
    make_matrix_scale,
    make_spectral_scales,
    normalize_diagonal,
    normalize_matrix,
    normalize_none,
    normalize_spectral,
    scale_error_traces,
    scale_residual_traces,
    scale_residual_traces_spectral,
    scale_system,
    scale_system_spectral,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_spd_matrix() -> np.ndarray:
    """Create a small SPD matrix for testing."""
    A = np.array([[4.0, 1.0, 0.5], [1.0, 3.0, 0.3], [0.5, 0.3, 2.0]], dtype=np.float64)
    return A


@pytest.fixture
def simple_system_batch(simple_spd_matrix: np.ndarray) -> LinearSystemBatch:
    """Create a simple system batch for testing."""
    A = simple_spd_matrix
    R = np.array([[1.0, 2.0, 1.0], [0.5, 1.5, 0.5], [2.0, 1.0, 2.0]], dtype=np.float64)
    X = np.array([[0.1, 0.2, 0.1], [0.05, 0.15, 0.05], [0.2, 0.1, 0.2]], dtype=np.float64)
    return LinearSystemBatch(_matrix=A, _rhs_samples=R, _sol_samples=X)


@pytest.fixture
def residual_traces_fixture() -> ResidualTraceSamples:
    """Create residual traces for testing."""
    return ResidualTraceSamples(
        residuals=np.array([[0.1, 0.2, 0.1], [0.05, 0.1, 0.05]], dtype=np.float64),
        solutions=np.array([[0.01, 0.02, 0.01], [0.005, 0.01, 0.005]], dtype=np.float64),
        sample_indices=np.array([0, 1], dtype=np.int64),
        iteration_indices=np.array([0, 0], dtype=np.int64),
    )


@pytest.fixture
def error_traces_fixture() -> ErrorTraceSamples:
    """Create error traces for testing."""
    return ErrorTraceSamples(
        residuals=np.array([[0.2, 0.1, 0.2]], dtype=np.float64),
        solutions_current=np.array([[0.05, 0.05, 0.05]], dtype=np.float64),
        errors=np.array([[0.05, 0.15, 0.05]], dtype=np.float64),
        true_solutions=np.array([[0.1, 0.2, 0.1]], dtype=np.float64),
        sample_indices=np.array([0], dtype=np.int64),
        iteration_indices=np.array([0], dtype=np.int64),
    )


# =============================================================================
# Test ABC Interfaces
# =============================================================================


def test_ilinear_system_interface() -> None:
    """Test ILinearSystem interface is implemented by LinearSystem."""
    A = np.eye(3)
    b = np.ones(3)
    x = np.array([1.0, 1.0, 1.0])

    system = LinearSystem(_matrix=A, _rhs=b, _solution=x)

    # Test interface methods
    assert isinstance(system.matrix, np.ndarray)
    assert isinstance(system.rhs, np.ndarray)
    assert isinstance(system.solution, np.ndarray)
    assert system.dimension == 3


def test_ilinear_system_batch_interface(simple_system_batch: LinearSystemBatch) -> None:
    """Test ILinearSystemBatch interface is implemented by LinearSystemBatch."""
    system = simple_system_batch

    # Test interface methods
    assert isinstance(system.matrix, np.ndarray)
    assert isinstance(system.rhs_samples, np.ndarray)
    assert isinstance(system.sol_samples, np.ndarray)
    assert system.dimension == 3
    assert system.num_samples == 3


def test_iscale_interface_matrix_scale(simple_system_batch: LinearSystemBatch) -> None:
    """Test IScale interface is implemented by MatrixScale."""
    scale = make_matrix_scale(simple_system_batch)

    # Test interface methods
    assert callable(scale.scale_matrix)
    assert callable(scale.scale_rhs)
    assert callable(scale.scale_solution)
    assert callable(scale.scale_residual)

    # Test scaling produces correct shapes
    A_scaled = scale.scale_matrix(simple_system_batch.matrix)
    assert A_scaled.shape == simple_system_batch.matrix.shape

    rhs_scaled = scale.scale_rhs(simple_system_batch.rhs_samples[0])
    assert rhs_scaled.shape == simple_system_batch.rhs_samples[0].shape


def test_iscale_interface_diagonal_scale(
    simple_system_batch: LinearSystemBatch,
) -> None:
    """Test IScale interface is implemented by DiagonalScale."""
    scale = make_diagonal_scale(simple_system_batch)

    # Test interface methods exist
    assert callable(scale.scale_matrix)
    assert callable(scale.scale_rhs)
    assert callable(scale.scale_solution)
    assert callable(scale.scale_residual)


def test_itrace_samples_interface(
    residual_traces_fixture: ResidualTraceSamples,
) -> None:
    """Test ITraceSamples interface is implemented by ResidualTraceSamples."""
    traces = residual_traces_fixture

    # Test interface property
    assert traces.num_traces == 2
    assert len(traces.sample_indices) == 2
    assert len(traces.iteration_indices) == 2


# =============================================================================
# Test Frozen Dataclasses (Immutability)
# =============================================================================


def test_linear_system_batch_frozen(simple_system_batch: LinearSystemBatch) -> None:
    """Test LinearSystemBatch is frozen and immutable."""
    system = simple_system_batch

    # Should not be able to set attributes
    with pytest.raises((AttributeError, TypeError)):
        system._matrix = np.zeros((3, 3))  # type: ignore[misc]


def test_matrix_scale_frozen() -> None:
    """Test MatrixScale is frozen and immutable."""
    scale = MatrixScale(spectral_radius_bound=10.0, dimension_scale=2.0)

    with pytest.raises((AttributeError, TypeError)):
        scale.spectral_radius_bound = 5.0  # type: ignore[misc]


def test_residual_traces_frozen(residual_traces_fixture: ResidualTraceSamples) -> None:
    """Test ResidualTraceSamples is frozen and immutable."""
    traces = residual_traces_fixture

    with pytest.raises((AttributeError, TypeError)):
        traces.residuals = np.zeros((2, 3))  # type: ignore[misc]


# =============================================================================
# Test Pure Functions: Scale Factories
# =============================================================================


def test_make_matrix_scale(simple_system_batch: LinearSystemBatch) -> None:
    """Test make_matrix_scale creates correct MatrixScale."""
    scale = make_matrix_scale(simple_system_batch)

    assert isinstance(scale, MatrixScale)
    assert scale.spectral_radius_bound > 0
    assert scale.dimension_scale == np.sqrt(3)
    assert scale.composite_scale == scale.spectral_radius_bound * scale.dimension_scale


def test_make_diagonal_scale(simple_system_batch: LinearSystemBatch) -> None:
    """Test make_diagonal_scale creates correct DiagonalScale."""
    scale = make_diagonal_scale(simple_system_batch)

    assert isinstance(scale, DiagonalScale)
    assert len(scale.diagonal_sqrt_inv) == 3
    # Verify it's the inverse square root of the diagonal
    diag = np.diag(simple_system_batch.matrix)
    np.testing.assert_allclose(scale.diagonal_sqrt_inv, 1.0 / np.sqrt(diag))


def test_make_spectral_scales(simple_system_batch: LinearSystemBatch) -> None:
    """Test make_spectral_scales creates per-sample SpectralScale list."""
    scales = make_spectral_scales(simple_system_batch)

    assert len(scales) == 3  # One per sample
    assert all(isinstance(s, SpectralScale) for s in scales)
    # All should share spectral_norm and dimension_scale
    assert all(s.spectral_norm == scales[0].spectral_norm for s in scales)
    assert all(s.dimension_scale == scales[0].dimension_scale for s in scales)
    # But different rhs_norms
    assert not all(s.rhs_norm == scales[0].rhs_norm for s in scales)


def test_make_diagonal_scale_rejects_zero_diagonal() -> None:
    """Test make_diagonal_scale rejects matrices with zero diagonal."""
    A = np.array([[0.0, 1.0], [1.0, 2.0]], dtype=np.float64)
    system = LinearSystemBatch(
        _matrix=A,
        _rhs_samples=np.ones((1, 2)),
        _sol_samples=np.ones((1, 2)),
    )

    with pytest.raises(ValueError, match="near-zero diagonal"):
        make_diagonal_scale(system)


# =============================================================================
# Test Pure Functions: System Scaling
# =============================================================================


def test_scale_system_with_matrix_scale(simple_system_batch: LinearSystemBatch) -> None:
    """Test scale_system scales system correctly with MatrixScale."""
    scale = make_matrix_scale(simple_system_batch)
    scaled = scale_system(simple_system_batch, scale)

    assert isinstance(scaled, LinearSystemBatch)
    assert scaled.dimension == 3
    assert scaled.num_samples == 3

    # Verify scaling was applied
    expected_matrix = simple_system_batch.matrix / scale.composite_scale
    np.testing.assert_allclose(scaled.matrix, expected_matrix)


def test_scale_system_with_diagonal_scale(
    simple_system_batch: LinearSystemBatch,
) -> None:
    """Test scale_system scales system correctly with DiagonalScale."""
    scale = make_diagonal_scale(simple_system_batch)
    scaled = scale_system(simple_system_batch, scale)

    assert isinstance(scaled, LinearSystemBatch)

    # Verify symmetric diagonal scaling: D^(-1/2) @ A @ D^(-1/2)
    diag_sqrt_inv = scale.diagonal_sqrt_inv
    expected_matrix = diag_sqrt_inv[:, None] * simple_system_batch.matrix * diag_sqrt_inv[None, :]
    np.testing.assert_allclose(scaled.matrix, expected_matrix)


def test_scale_system_spectral(simple_system_batch: LinearSystemBatch) -> None:
    """Test scale_system_spectral scales with per-sample scales."""
    scales = make_spectral_scales(simple_system_batch)
    scaled = scale_system_spectral(simple_system_batch, scales)

    assert isinstance(scaled, LinearSystemBatch)
    assert scaled.num_samples == 3

    # All samples share the same scaled matrix
    spec_norm = scales[0].spectral_norm
    dim_scale = scales[0].dimension_scale
    expected_matrix = simple_system_batch.matrix / (spec_norm * dim_scale)
    np.testing.assert_allclose(scaled.matrix, expected_matrix)


# =============================================================================
# Test Pure Functions: Trace Scaling
# =============================================================================


def test_scale_residual_traces(residual_traces_fixture: ResidualTraceSamples) -> None:
    """Test scale_residual_traces scales traces correctly."""
    scale = MatrixScale(spectral_radius_bound=10.0, dimension_scale=2.0)
    scaled = scale_residual_traces(residual_traces_fixture, scale)

    assert isinstance(scaled, ResidualTraceSamples)
    assert scaled.num_traces == 2

    # Verify residuals are scaled
    expected_residuals = residual_traces_fixture.residuals / scale.composite_scale
    np.testing.assert_allclose(scaled.residuals, expected_residuals)

    # Verify solutions unchanged (MatrixScale doesn't scale solutions)
    np.testing.assert_allclose(scaled.solutions, residual_traces_fixture.solutions)


def test_scale_error_traces(error_traces_fixture: ErrorTraceSamples) -> None:
    """Test scale_error_traces scales error traces correctly."""
    scale = MatrixScale(spectral_radius_bound=10.0, dimension_scale=2.0)
    scaled = scale_error_traces(error_traces_fixture, scale)

    assert isinstance(scaled, ErrorTraceSamples)

    # Verify residuals and errors are scaled
    expected_residuals = error_traces_fixture.residuals / scale.composite_scale
    expected_errors = error_traces_fixture.errors / scale.composite_scale
    np.testing.assert_allclose(scaled.residuals, expected_residuals)
    np.testing.assert_allclose(scaled.errors, expected_errors)

    # Verify solutions_current and true_solutions unchanged (reference only)
    np.testing.assert_allclose(scaled.solutions_current, error_traces_fixture.solutions_current)
    np.testing.assert_allclose(scaled.true_solutions, error_traces_fixture.true_solutions)


def test_scale_residual_traces_spectral(
    residual_traces_fixture: ResidualTraceSamples,
    simple_system_batch: LinearSystemBatch,
) -> None:
    """Test scale_residual_traces_spectral with per-sample scaling."""
    scales = make_spectral_scales(simple_system_batch)
    scaled = scale_residual_traces_spectral(residual_traces_fixture, scales)

    assert isinstance(scaled, ResidualTraceSamples)
    assert scaled.num_traces == 2

    # Each trace should be scaled by its sample's scale
    for i, sample_idx in enumerate(residual_traces_fixture.sample_indices):
        sample_scale = scales[int(sample_idx)]
        expected_residual = sample_scale.scale_residual(residual_traces_fixture.residuals[i])
        np.testing.assert_allclose(scaled.residuals[i], expected_residual)


# =============================================================================
# Test Strategy Functions
# =============================================================================


def test_normalize_none(
    simple_system_batch: LinearSystemBatch,
    residual_traces_fixture: ResidualTraceSamples,
) -> None:
    """Test normalize_none returns defensive copies."""
    result = normalize_none(simple_system_batch, residual_traces_fixture, None)

    assert isinstance(result, NormalizedSystem)
    assert result.scale is None

    # Verify system was copied
    np.testing.assert_array_equal(result.system.matrix, simple_system_batch.matrix)
    assert result.system.matrix is not simple_system_batch.matrix

    # Verify traces were copied
    assert result.residual_traces is not None
    np.testing.assert_array_equal(
        result.residual_traces.residuals, residual_traces_fixture.residuals
    )
    assert result.residual_traces.residuals is not residual_traces_fixture.residuals


def test_normalize_matrix(
    simple_system_batch: LinearSystemBatch,
    residual_traces_fixture: ResidualTraceSamples,
) -> None:
    """Test normalize_matrix applies correct scaling."""
    result = normalize_matrix(simple_system_batch, residual_traces_fixture, None)

    assert isinstance(result, NormalizedSystem)
    assert isinstance(result.scale, MatrixScale)

    # Verify matrix was scaled
    scale = result.scale
    expected_matrix = simple_system_batch.matrix / scale.composite_scale
    np.testing.assert_allclose(result.system.matrix, expected_matrix)

    # Verify traces were scaled
    assert result.residual_traces is not None
    expected_residuals = residual_traces_fixture.residuals / scale.composite_scale
    np.testing.assert_allclose(result.residual_traces.residuals, expected_residuals)


def test_normalize_diagonal(simple_system_batch: LinearSystemBatch) -> None:
    """Test normalize_diagonal applies Jacobi scaling."""
    result = normalize_diagonal(simple_system_batch, None, None)

    assert isinstance(result, NormalizedSystem)
    assert isinstance(result.scale, DiagonalScale)

    # Verify diagonal is normalized to identity
    diag = np.diag(result.system.matrix)
    np.testing.assert_allclose(diag, np.ones(3), atol=1e-10)


def test_normalize_spectral(simple_system_batch: LinearSystemBatch) -> None:
    """Test normalize_spectral applies per-sample scaling."""
    result = normalize_spectral(simple_system_batch, None, None)

    assert isinstance(result, NormalizedSystem)
    assert isinstance(result.scale, SpectralScale)

    # Verify matrix was normalized
    spec_norm = result.scale.spectral_norm
    dim_scale = result.scale.dimension_scale
    expected_matrix = simple_system_batch.matrix / (spec_norm * dim_scale)
    np.testing.assert_allclose(result.system.matrix, expected_matrix)


# =============================================================================
# Test Backward Compatibility
# =============================================================================


def test_apply_normalization_returns_legacy_format(
    simple_spd_matrix: np.ndarray,
    tmp_path: Path,
) -> None:
    """Test apply_normalization returns NormalizationResult for compatibility."""
    A = simple_spd_matrix
    R = np.array([[1.0, 2.0, 1.0]], dtype=np.float64)
    X = np.array([[0.1, 0.2, 0.1]], dtype=np.float64)

    result = apply_normalization("matrix", A, R, X, tmp_path, verbose=False)

    # Should return NormalizationResult, not NormalizedSystem
    assert isinstance(result, NormalizationResult)

    # Should have legacy attributes
    assert hasattr(result, "matrix")
    assert hasattr(result, "rhs_samples")
    assert hasattr(result, "sol_samples")
    assert hasattr(result, "normalize_type")
    assert hasattr(result, "matrix_scale")
    assert hasattr(result, "spectral_radius_bound")

    # Verify normalize_type is set correctly
    assert result.normalize_type == "matrix"


def test_legacy_result_conversion(simple_system_batch: LinearSystemBatch) -> None:
    """Test NormalizationResult.from_normalized_system converts correctly."""
    # Create a normalized system
    normalized = normalize_matrix(simple_system_batch, None, None)

    # Convert to legacy format
    legacy = NormalizationResult.from_normalized_system(normalized, "matrix")

    assert isinstance(legacy, NormalizationResult)
    np.testing.assert_array_equal(legacy.matrix, normalized.system.matrix)
    np.testing.assert_array_equal(legacy.rhs_samples, normalized.system.rhs_samples)
    np.testing.assert_array_equal(legacy.sol_samples, normalized.system.sol_samples)

    # Check normalize_type attribute
    assert legacy.normalize_type == "matrix"

    # Check scale metadata
    assert isinstance(normalized.scale, MatrixScale)
    assert legacy.matrix_scale == normalized.scale.spectral_radius_bound
    assert legacy.spectral_radius_bound == normalized.scale.spectral_radius_bound


# =============================================================================
# Test Polymorphism (IScale works with any implementation)
# =============================================================================


def test_scale_system_polymorphic() -> None:
    """Test scale_system works polymorphically with any IScale implementation."""
    A = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    R = np.array([[1.0, 1.0]], dtype=np.float64)
    X = np.array([[0.1, 0.1]], dtype=np.float64)
    system = LinearSystemBatch(_matrix=A, _rhs_samples=R, _sol_samples=X)

    # Test with MatrixScale
    matrix_scale = MatrixScale(spectral_radius_bound=5.0, dimension_scale=np.sqrt(2))
    scaled_matrix = scale_system(system, matrix_scale)
    assert isinstance(scaled_matrix, LinearSystemBatch)

    # Test with DiagonalScale
    diagonal_scale = DiagonalScale(diagonal_sqrt_inv=1.0 / np.sqrt(np.diag(A)))
    scaled_diagonal = scale_system(system, diagonal_scale)
    assert isinstance(scaled_diagonal, LinearSystemBatch)

    # Both should work without modification (polymorphism via IScale)


def test_scale_traces_polymorphic(
    residual_traces_fixture: ResidualTraceSamples,
) -> None:
    """Test scale_residual_traces works polymorphically with any IScale."""
    # Test with different IScale implementations
    matrix_scale = MatrixScale(spectral_radius_bound=10.0, dimension_scale=2.0)
    scaled_matrix = scale_residual_traces(residual_traces_fixture, matrix_scale)
    assert isinstance(scaled_matrix, ResidualTraceSamples)

    diag_scale = DiagonalScale(diagonal_sqrt_inv=np.array([0.5, 0.577, 0.707]))
    scaled_diag = scale_residual_traces(residual_traces_fixture, diag_scale)
    assert isinstance(scaled_diag, ResidualTraceSamples)

    # Both work via polymorphism


# =============================================================================
# Integration Tests
# =============================================================================


def test_full_normalization_pipeline_matrix(
    simple_spd_matrix: np.ndarray,
    tmp_path: Path,
) -> None:
    """Test full normalization pipeline with matrix strategy."""
    A = simple_spd_matrix
    R = np.array([[1.0, 2.0, 1.0], [0.5, 1.5, 0.5]], dtype=np.float64)
    X = np.array([[0.1, 0.2, 0.1], [0.05, 0.15, 0.05]], dtype=np.float64)

    residual_traces = ResidualTraceSamples(
        residuals=np.array([[0.1, 0.2, 0.1]], dtype=np.float64),
        solutions=np.array([[0.01, 0.02, 0.01]], dtype=np.float64),
        sample_indices=np.array([0], dtype=np.int64),
        iteration_indices=np.array([0], dtype=np.int64),
    )

    result = apply_normalization(
        "matrix", A, R, X, tmp_path, residual_traces=residual_traces, verbose=False
    )

    # Verify result structure
    assert isinstance(result, NormalizationResult)
    assert result.matrix.shape == A.shape
    assert result.rhs_samples.shape == R.shape
    assert result.residual_traces is not None
    assert result.residual_traces.num_traces == 1


def test_full_normalization_pipeline_spectral(
    simple_spd_matrix: np.ndarray,
    tmp_path: Path,
) -> None:
    """Test full normalization pipeline with spectral strategy."""
    A = simple_spd_matrix
    R = np.array([[1.0, 2.0, 1.0], [0.5, 1.5, 0.5]], dtype=np.float64)
    X = np.array([[0.1, 0.2, 0.1], [0.05, 0.15, 0.05]], dtype=np.float64)

    result = apply_normalization("spectral", A, R, X, tmp_path, verbose=False)

    assert isinstance(result, NormalizationResult)
    assert result.spectral_norm is not None
    assert result.spectral_norm > 0


def test_all_strategies_produce_valid_results(
    simple_spd_matrix: np.ndarray,
    tmp_path: Path,
) -> None:
    """Test all normalization strategies produce valid results."""
    A = simple_spd_matrix
    R = np.array([[1.0, 2.0, 1.0]], dtype=np.float64)
    X = np.array([[0.1, 0.2, 0.1]], dtype=np.float64)

    strategies = ["none", "matrix", "diagonal", "spectral"]

    for strategy in strategies:
        result = apply_normalization(
            cast(Literal["none", "matrix", "spectral", "diagonal"], strategy),
            A,
            R,
            X,
            tmp_path,
            verbose=False,
        )

        assert isinstance(result, NormalizationResult)
        assert result.matrix.shape == A.shape
        assert result.rhs_samples.shape == R.shape
        assert result.sol_samples.shape == X.shape
        # Verify normalize_type attribute is correctly set
        assert result.normalize_type == strategy
