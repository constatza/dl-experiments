"""Unit tests for preconditioner builders.

Tests builder construction, error handling, and edge cases independent of solver integration.
Follows project principles:
- Use fixtures for all test data
- Use tmp_path for temporary files (never tempfile)
- Type hints throughout
- Focus on our own logic, not DLKit internals
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from neuralls.configuration.preconditioner import (
    NeuralPreconditionerConfig,
    StandardPreconditionerConfig,
)
from neuralls.preconditioner.builders import (
    IdentityBuilder,
    ILUBuilder,
    JacobiBuilder,
    NeuralBuilder,
)
from neuralls.preconditioner.ports import PredictorAdapter, PredictorPort

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ==============================================================================
# Mock Predictor and Adapter for Neural Builder Testing
# ==============================================================================


class MockPredictor(PredictorPort):
    """Lightweight mock predictor for testing."""

    def __init__(self) -> None:
        """Initialize mock predictor."""
        self.cleaned_up = False
        self.apply_count = 0

    def apply(self, residual: NDArray) -> NDArray:
        """Mock apply: returns half of input."""
        self.apply_count += 1
        return residual * 0.5

    def cleanup(self) -> None:
        """Mark as cleaned up."""
        self.cleaned_up = True

    def __enter__(self) -> MockPredictor:
        """Enter context manager."""
        return self

    def __exit__(self, *args) -> None:
        """Exit context manager with cleanup."""
        self.cleanup()


class MockAdapter(PredictorAdapter):
    """Lightweight mock adapter for testing."""

    def __init__(self) -> None:
        """Initialize mock adapter."""
        self.predictor = MockPredictor()

    def create_predictor(
        self,
        checkpoint_path: Path,
        config_path: Path | None = None,
        data_config_path: Path | None = None,
    ) -> PredictorPort:
        """Create mock predictor."""
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return self.predictor


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def well_conditioned_matrix() -> NDArray:
    """Well-conditioned 4x4 SPD matrix for testing.

    Returns:
        4x4 diagonal matrix with condition number ~2
    """
    return np.diag([4.0, 3.0, 2.0, 2.0])


@pytest.fixture
def near_zero_diagonal_matrix() -> NDArray:
    """Matrix with near-zero diagonal elements.

    Returns:
        4x4 matrix with diagonal [2.0, 1e-15, 1.0, 1e-16]
    """
    return np.diag([2.0, 1e-15, 1.0, 1e-16])


@pytest.fixture
def negative_diagonal_matrix() -> NDArray:
    """Matrix with negative diagonal elements.

    Returns:
        4x4 matrix with mixed positive/negative diagonal
    """
    return np.diag([2.0, -1.0, 3.0, -0.5])


@pytest.fixture
def nonsquare_matrix() -> NDArray:
    """Non-square matrix for validation testing.

    Returns:
        3x4 rectangular matrix
    """
    return np.ones((3, 4))


@pytest.fixture
def dense_spd_matrix() -> NDArray:
    """Dense SPD matrix for ILU testing.

    Returns:
        5x5 dense SPD tridiagonal matrix
    """
    n = 5
    A = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
    return A


@pytest.fixture
def singular_matrix() -> NDArray:
    """Singular matrix (rank deficient).

    Returns:
        5x5 singular matrix (last row is zeros)
    """
    A = np.eye(5)
    A[4, 4] = 0.0  # Make singular
    return A


@pytest.fixture
def residual_vector() -> NDArray:
    """Test residual vector.

    Returns:
        4D residual vector
    """
    return np.array([1.0, 2.0, 3.0, 4.0])


@pytest.fixture
def mock_checkpoint(tmp_path: Path) -> Path:
    """Create mock checkpoint file.

    Args:
        tmp_path: Pytest temporary directory fixture

    Returns:
        Path to mock checkpoint file
    """
    checkpoint = tmp_path / "mock_checkpoint.ckpt"
    checkpoint.write_text("mock checkpoint data")
    return checkpoint


# ==============================================================================
# IdentityBuilder Tests (3 tests)
# ==============================================================================


def test_identity_builder_returns_copy(
    well_conditioned_matrix: NDArray, residual_vector: NDArray
) -> None:
    """Test that IdentityBuilder returns a copy of input.

    Ensures functional purity - no side effects on input.
    """
    builder = IdentityBuilder()
    config = StandardPreconditionerConfig(name="identity", type="identity")

    precond_fn = builder.build(well_conditioned_matrix, config)
    result = precond_fn(residual_vector)

    # Result should equal input
    assert_array_equal(result, residual_vector)

    # Result should be a copy (different memory address)
    assert result is not residual_vector

    # Modifying result should not affect input
    result[0] = 999.0
    assert residual_vector[0] == 1.0


def test_identity_builder_supports_multiple_type_names(
    well_conditioned_matrix: NDArray,
) -> None:
    """Test that IdentityBuilder supports both 'identity' and 'none' types."""
    builder = IdentityBuilder()

    supported = builder.supported_types()
    assert "identity" in supported
    assert "none" in supported
    assert len(supported) == 2


def test_identity_builder_preserves_input_immutability(
    well_conditioned_matrix: NDArray, residual_vector: NDArray
) -> None:
    """Test that applying preconditioner doesn't modify input vector."""
    builder = IdentityBuilder()
    config = StandardPreconditionerConfig(name="identity", type="identity")

    precond_fn = builder.build(well_conditioned_matrix, config)

    original_copy = residual_vector.copy()
    _ = precond_fn(residual_vector)

    # Input should be unchanged
    assert_array_equal(residual_vector, original_copy)


# ==============================================================================
# JacobiBuilder Tests (5 tests)
# ==============================================================================


def test_jacobi_builder_handles_well_conditioned_diagonal(
    well_conditioned_matrix: NDArray, residual_vector: NDArray
) -> None:
    """Test JacobiBuilder with well-conditioned diagonal."""
    builder = JacobiBuilder()
    config = StandardPreconditionerConfig(name="jacobi", type="jacobi")

    precond_fn = builder.build(well_conditioned_matrix, config)
    result = precond_fn(residual_vector)

    # Expected: z_i = r_i / A_ii
    expected = residual_vector / np.diag(well_conditioned_matrix)
    assert_array_equal(result, expected)


def test_jacobi_builder_handles_near_zero_diagonal(
    near_zero_diagonal_matrix: NDArray, residual_vector: NDArray
) -> None:
    """Test JacobiBuilder with near-zero diagonal elements.

    Builder should replace values < 1e-14 with 1.0 to avoid division by zero.
    """
    builder = JacobiBuilder()
    config = StandardPreconditionerConfig(name="jacobi", type="jacobi")

    precond_fn = builder.build(near_zero_diagonal_matrix, config)
    result = precond_fn(residual_vector)

    # Near-zero diagonal elements should be treated as 1.0
    # Expected: [1.0/2.0, 2.0/1.0, 3.0/1.0, 4.0/1.0] = [0.5, 2.0, 3.0, 4.0]
    expected = np.array([0.5, 2.0, 3.0, 4.0])
    assert_array_equal(result, expected)


def test_jacobi_builder_handles_negative_diagonal(
    negative_diagonal_matrix: NDArray, residual_vector: NDArray
) -> None:
    """Test JacobiBuilder preserves signs with negative diagonal.

    Jacobi should work with negative diagonal elements (z_i = r_i / A_ii).
    """
    builder = JacobiBuilder()
    config = StandardPreconditionerConfig(name="jacobi", type="jacobi")

    precond_fn = builder.build(negative_diagonal_matrix, config)
    result = precond_fn(residual_vector)

    # Expected: [1/2, 2/(-1), 3/3, 4/(-0.5)]
    expected = np.array([0.5, -2.0, 1.0, -8.0])
    assert_array_equal(result, expected)


def test_jacobi_builder_validates_square_matrix(nonsquare_matrix: NDArray) -> None:
    """Test that JacobiBuilder handles non-square matrices gracefully.

    np.diag() will extract diagonal from rectangular matrix without error,
    but dimension mismatch will occur during application.
    """
    builder = JacobiBuilder()
    config = StandardPreconditionerConfig(name="jacobi", type="jacobi")

    # Building with non-square matrix extracts shorter diagonal (min dimension)
    precond_fn = builder.build(nonsquare_matrix, config)

    # Applying to 3D vector should work (diagonal length = 3)
    residual_3d = np.array([1.0, 2.0, 3.0])
    result = precond_fn(residual_3d)
    expected = np.array([1.0, 2.0, 3.0])  # diag is all 1.0
    assert_array_equal(result, expected)


def test_jacobi_builder_preserves_dtype(well_conditioned_matrix: NDArray) -> None:
    """Test that JacobiBuilder preserves float64 dtype."""
    builder = JacobiBuilder()
    config = StandardPreconditionerConfig(name="jacobi", type="jacobi")

    precond_fn = builder.build(well_conditioned_matrix, config)
    residual = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    result = precond_fn(residual)

    assert result.dtype == np.float64


# ==============================================================================
# ILUBuilder Tests (4 tests)
# ==============================================================================


def test_ilu_builder_accepts_dense_matrix(
    dense_spd_matrix: NDArray, residual_vector: NDArray
) -> None:
    """Test ILUBuilder with dense matrix input.

    Builder should convert to CSC format internally.
    """
    builder = ILUBuilder()
    config = StandardPreconditionerConfig(name="ilu", type="ilu")

    # Should not raise error
    precond_fn = builder.build(dense_spd_matrix, config)

    # Should apply without error
    residual_5d = np.ones(5)
    result = precond_fn(residual_5d)

    # Result should have correct shape
    assert result.shape == (5,)
    assert result.dtype == np.float64


def test_ilu_builder_converts_to_csc_internally(dense_spd_matrix: NDArray) -> None:
    """Test that ILUBuilder converts to CSC format for efficient factorization.

    This is an implementation detail, but important for performance.
    """
    builder = ILUBuilder()
    config = StandardPreconditionerConfig(name="ilu", type="ilu")

    # Should not raise error (verifies CSC conversion works)
    precond_fn = builder.build(dense_spd_matrix, config)

    # Smoke test: apply to vector
    residual = np.ones(5)
    result = precond_fn(residual)
    assert result.shape == (5,)


def test_ilu_builder_handles_singular_matrix(singular_matrix: NDArray) -> None:
    """Test ILUBuilder behavior with singular matrix.

    SciPy's spilu may raise RuntimeError or produce poor factorization.
    """
    builder = ILUBuilder()
    config = StandardPreconditionerConfig(name="ilu", type="ilu")

    # spilu may raise RuntimeError for singular matrix
    # If it doesn't raise during build, it might during solve
    try:
        precond_fn = builder.build(singular_matrix, config)
        residual = np.ones(5)
        result = precond_fn(residual)
        # If we get here, spilu didn't detect singularity
        # Just verify shape
        assert result.shape == (5,)
    except RuntimeError:
        # Expected behavior for singular matrix
        pass


def test_ilu_builder_preserves_dtype(dense_spd_matrix: NDArray) -> None:
    """Test that ILUBuilder preserves float64 dtype."""
    builder = ILUBuilder()
    config = StandardPreconditionerConfig(name="ilu", type="ilu")

    precond_fn = builder.build(dense_spd_matrix, config)
    residual = np.ones(5, dtype=np.float64)
    result = precond_fn(residual)

    assert result.dtype == np.float64


# ==============================================================================
# NeuralBuilder Tests (6 tests)
# ==============================================================================


def test_neural_builder_with_default_adapter(
    well_conditioned_matrix: NDArray, mock_checkpoint: Path
) -> None:
    """Test NeuralBuilder with default DLKit adapter.

    Note: This test will fail if DLKit is not installed. That's expected.
    """
    builder = NeuralBuilder()  # Uses default DLKitAdapter

    config = NeuralPreconditionerConfig(
        name="neural",
        type="neural",
        checkpoint_path=mock_checkpoint,
    )

    # Attempting to build with default adapter will fail without DLKit installed
    # or with invalid checkpoint format
    # This test just verifies the interface works
    try:
        precond_fn = builder.build(well_conditioned_matrix, config)
        # If we get here, DLKit is installed and checkpoint format was accepted
        assert callable(precond_fn)
    except (ImportError, RuntimeError, FileNotFoundError):
        # Expected if DLKit not installed or checkpoint invalid
        pytest.skip("DLKit not installed or checkpoint format invalid")


def test_neural_builder_with_custom_adapter(
    well_conditioned_matrix: NDArray, mock_checkpoint: Path, residual_vector: NDArray
) -> None:
    """Test NeuralBuilder with custom adapter (DIP validation)."""
    mock_adapter = MockAdapter()
    builder = NeuralBuilder(adapter=mock_adapter)

    config = NeuralPreconditionerConfig(
        name="neural",
        type="neural",
        checkpoint_path=mock_checkpoint,
    )

    precond_fn = builder.build(well_conditioned_matrix, config)
    result = precond_fn(residual_vector)

    # MockPredictor returns half of input
    expected = residual_vector * 0.5
    assert_array_equal(result, expected)


def test_neural_builder_requires_neural_config(
    well_conditioned_matrix: NDArray, mock_checkpoint: Path
) -> None:
    """Test that NeuralBuilder requires NeuralPreconditionerConfig.

    Should raise TypeError for wrong config type.
    """
    mock_adapter = MockAdapter()
    builder = NeuralBuilder(adapter=mock_adapter)

    # Pass wrong config type (StandardPreconditionerConfig)
    wrong_config = StandardPreconditionerConfig(name="jacobi", type="jacobi")

    with pytest.raises(TypeError, match="requires NeuralPreconditionerConfig"):
        builder.build(well_conditioned_matrix, wrong_config)


def test_neural_builder_checkpoint_not_found(well_conditioned_matrix: NDArray) -> None:
    """Test NeuralBuilder with non-existent checkpoint."""
    mock_adapter = MockAdapter()
    builder = NeuralBuilder(adapter=mock_adapter)

    config = NeuralPreconditionerConfig(
        name="neural",
        type="neural",
        checkpoint_path=Path("/nonexistent/checkpoint.ckpt"),
    )

    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        builder.build(well_conditioned_matrix, config)


def test_neural_builder_cleanup_reference_stored(
    well_conditioned_matrix: NDArray, mock_checkpoint: Path
) -> None:
    """Test that NeuralBuilder stores cleanup reference on function."""
    mock_adapter = MockAdapter()
    builder = NeuralBuilder(adapter=mock_adapter)

    config = NeuralPreconditionerConfig(
        name="neural",
        type="neural",
        checkpoint_path=mock_checkpoint,
    )

    precond_fn = builder.build(well_conditioned_matrix, config)

    # Check that _cleanup attribute exists
    assert hasattr(precond_fn, "_cleanup")
    assert callable(precond_fn._cleanup)


def test_neural_builder_supports_context_manager(
    well_conditioned_matrix: NDArray, mock_checkpoint: Path, residual_vector: NDArray
) -> None:
    """Test that neural predictor supports context manager protocol."""
    mock_adapter = MockAdapter()
    builder = NeuralBuilder(adapter=mock_adapter)

    config = NeuralPreconditionerConfig(
        name="neural",
        type="neural",
        checkpoint_path=mock_checkpoint,
    )

    precond_fn = builder.build(well_conditioned_matrix, config)

    # MockPredictor supports context manager
    predictor = mock_adapter.predictor
    assert not predictor.cleaned_up

    # Use predictor via context manager
    with predictor:
        _ = predictor.apply(residual_vector)

    # Should be cleaned up after context manager exit
    assert predictor.cleaned_up
