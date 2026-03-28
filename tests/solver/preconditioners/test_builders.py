"""Unit tests for preconditioner factory function.

Tests factory creation, error handling, and edge cases independent of solver integration.
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

from neuralls.platform.config.models.preconditioner import (
    NeuralPreconditionerConfig,
    PreconditionerType,
    StandardPreconditionerConfig,
)
from neuralls.composition.preconditioners.factory import (
    create_preconditioner,
    create_scheduled_preconditioner,
    PreconditionerScheduleConfig,
)
from neuralls.domain.solver.preconditioners.ports import PredictorAdapter, PredictorPort
from neuralls.domain.solver.preconditioners import (
    Identity,
    ILUPreconditioner,
    JacobiPreconditioner,
    PreconditionerContext,
    ScheduledPreconditioner,
)
from neuralls.domain.solver.preconditioners.implementations.neural import NeuralPreconditioner

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ==============================================================================
# Mock Predictor and Adapter for Neural Preconditioner Testing
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
def dense_spd_matrix() -> NDArray:
    """Dense SPD matrix for ILU testing.

    Returns:
        5x5 dense SPD tridiagonal matrix
    """
    n = 5
    A = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
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
# Factory Tests - Identity (3 tests)
# ==============================================================================


def test_factory_creates_identity_preconditioner(well_conditioned_matrix: NDArray) -> None:
    """Test that factory creates Identity preconditioner for identity type."""
    config = StandardPreconditionerConfig(name="identity", type=PreconditionerType.IDENTITY)

    precond = create_preconditioner(well_conditioned_matrix, config)

    assert isinstance(precond, Identity)


def test_factory_creates_identity_for_none_type(well_conditioned_matrix: NDArray) -> None:
    """Test that factory creates Identity for 'none' type alias."""
    config = StandardPreconditionerConfig(name="none", type=PreconditionerType.NONE)

    precond = create_preconditioner(well_conditioned_matrix, config)

    assert isinstance(precond, Identity)


def test_identity_preconditioner_returns_copy(
    well_conditioned_matrix: NDArray, residual_vector: NDArray
) -> None:
    """Test that Identity preconditioner returns a copy of input."""
    config = StandardPreconditionerConfig(name="identity", type=PreconditionerType.IDENTITY)

    precond = create_preconditioner(well_conditioned_matrix, config)
    result = precond.apply(residual_vector)

    # Result should equal input
    assert_array_equal(result, residual_vector)

    # Result should be a copy (different memory address)
    assert result is not residual_vector


# ==============================================================================
# Factory Tests - Jacobi (3 tests)
# ==============================================================================


def test_factory_creates_jacobi_preconditioner(well_conditioned_matrix: NDArray) -> None:
    """Test that factory creates JacobiPreconditioner for jacobi type."""
    config = StandardPreconditionerConfig(name="jacobi", type=PreconditionerType.JACOBI)

    precond = create_preconditioner(well_conditioned_matrix, config)

    assert isinstance(precond, JacobiPreconditioner)


def test_jacobi_preconditioner_applies_diagonal_scaling(
    well_conditioned_matrix: NDArray, residual_vector: NDArray
) -> None:
    """Test JacobiPreconditioner with well-conditioned diagonal."""
    config = StandardPreconditionerConfig(name="jacobi", type=PreconditionerType.JACOBI)

    precond = create_preconditioner(well_conditioned_matrix, config)
    result = precond.apply(residual_vector)

    # Expected: z_i = r_i / A_ii
    expected = residual_vector / np.diag(well_conditioned_matrix)
    assert_array_equal(result, expected)


def test_jacobi_preserves_dtype(well_conditioned_matrix: NDArray) -> None:
    """Test that JacobiPreconditioner preserves float64 dtype."""
    config = StandardPreconditionerConfig(name="jacobi", type=PreconditionerType.JACOBI)

    precond = create_preconditioner(well_conditioned_matrix, config)
    residual = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    result = precond.apply(residual)

    assert result.dtype == np.float64


# ==============================================================================
# Factory Tests - ILU (3 tests)
# ==============================================================================


def test_factory_creates_ilu_preconditioner(dense_spd_matrix: NDArray) -> None:
    """Test that factory creates ILUPreconditioner for ilu type."""
    config = StandardPreconditionerConfig(name="ilu", type=PreconditionerType.ILU)

    precond = create_preconditioner(dense_spd_matrix, config)

    assert isinstance(precond, ILUPreconditioner)


def test_ilu_preconditioner_accepts_dense_matrix(
    dense_spd_matrix: NDArray,
) -> None:
    """Test ILUPreconditioner with dense matrix input.

    Preconditioner should convert to CSC format internally.
    """
    config = StandardPreconditionerConfig(name="ilu", type=PreconditionerType.ILU)

    # Should not raise error
    precond = create_preconditioner(dense_spd_matrix, config)

    # Should apply without error
    residual_5d = np.ones(5)
    result = precond.apply(residual_5d)

    # Result should have correct shape
    assert result.shape == (5,)
    assert result.dtype == np.float64


def test_ilu_preserves_dtype(dense_spd_matrix: NDArray) -> None:
    """Test that ILUPreconditioner preserves float64 dtype."""
    config = StandardPreconditionerConfig(name="ilu", type=PreconditionerType.ILU)

    precond = create_preconditioner(dense_spd_matrix, config)
    residual = np.ones(5, dtype=np.float64)
    result = precond.apply(residual)

    assert result.dtype == np.float64


# ==============================================================================
# Factory Tests - Neural (4 tests)
# ==============================================================================


def test_factory_creates_neural_preconditioner(
    well_conditioned_matrix: NDArray, mock_checkpoint: Path
) -> None:
    """Test that factory creates NeuralPreconditioner for neural type."""
    mock_adapter = MockAdapter()

    config = NeuralPreconditionerConfig(
        name="neural",
        type=PreconditionerType.NEURAL,
        checkpoint_path=mock_checkpoint,
    )

    precond = create_preconditioner(well_conditioned_matrix, config, adapter=mock_adapter)

    assert isinstance(precond, NeuralPreconditioner)


def test_neural_preconditioner_with_custom_adapter(
    well_conditioned_matrix: NDArray, mock_checkpoint: Path, residual_vector: NDArray
) -> None:
    """Test NeuralPreconditioner with custom adapter (DIP validation)."""
    mock_adapter = MockAdapter()

    config = NeuralPreconditionerConfig(
        name="neural",
        type=PreconditionerType.NEURAL,
        checkpoint_path=mock_checkpoint,
    )

    precond = create_preconditioner(well_conditioned_matrix, config, adapter=mock_adapter)
    result = precond.apply(residual_vector)

    # MockPredictor returns half of input
    expected = residual_vector * 0.5
    assert_array_equal(result, expected)


def test_neural_preconditioner_checkpoint_not_found(
    well_conditioned_matrix: NDArray,
    tmp_path: Path,
) -> None:
    """Test NeuralPreconditioner with non-existent checkpoint."""
    mock_adapter = MockAdapter()
    missing_checkpoint = tmp_path / "missing" / "checkpoint.ckpt"

    config = NeuralPreconditionerConfig(
        name="neural",
        type=PreconditionerType.NEURAL,
        checkpoint_path=missing_checkpoint,
    )

    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        create_preconditioner(well_conditioned_matrix, config, adapter=mock_adapter)


def test_neural_preconditioner_cleanup_on_delete(
    well_conditioned_matrix: NDArray, mock_checkpoint: Path, residual_vector: NDArray
) -> None:
    """Test that neural preconditioner cleans up on deletion."""
    mock_adapter = MockAdapter()

    config = NeuralPreconditionerConfig(
        name="neural",
        type=PreconditionerType.NEURAL,
        checkpoint_path=mock_checkpoint,
    )

    precond = create_preconditioner(well_conditioned_matrix, config, adapter=mock_adapter)

    # Use the preconditioner
    assert isinstance(precond, NeuralPreconditioner)
    predictor = mock_adapter.predictor
    assert not predictor.cleaned_up
    _ = precond.apply(residual_vector)

    # Manually trigger cleanup
    precond.cleanup()

    # Should be cleaned up
    assert predictor.cleaned_up


# ==============================================================================
# Factory Tests - Error Handling (2 tests)
# ==============================================================================


def test_factory_rejects_unsupported_type(well_conditioned_matrix: NDArray) -> None:
    """Test that factory raises ValueError for unsupported preconditioner type."""
    # Create config with invalid type (bypass enum validation)
    config = StandardPreconditionerConfig(name="invalid", type=PreconditionerType.IDENTITY)
    config = config.model_copy(update={"type": "invalid_type"})  # Force invalid type

    with pytest.raises(ValueError, match="Unsupported preconditioner type"):
        create_preconditioner(well_conditioned_matrix, config)


def test_factory_requires_neural_config_for_neural_type(
    well_conditioned_matrix: NDArray,
) -> None:
    """Test that factory requires NeuralPreconditionerConfig for neural type."""
    # Create wrong config type with neural type
    config = StandardPreconditionerConfig(name="neural", type=PreconditionerType.IDENTITY)
    config = config.model_copy(update={"type": PreconditionerType.NEURAL})  # Force neural type

    with pytest.raises(TypeError, match="Neural type requires NeuralPreconditionerConfig"):
        create_preconditioner(well_conditioned_matrix, config)


# ==============================================================================
# Scheduled Preconditioner Builder Tests (4 tests)
# ==============================================================================


def test_create_scheduled_preconditioner_no_scheduling() -> None:
    """Test builder returns primary unchanged when no scheduling needed."""
    primary = Identity()
    schedule = PreconditionerScheduleConfig(limit_iters=-1)
    result = create_scheduled_preconditioner(primary, schedule)

    # Should return same instance, not wrapped
    assert result is primary


def test_create_scheduled_preconditioner_with_limit() -> None:
    """Test builder wraps preconditioner when limit_iters specified."""
    primary = Identity()
    schedule = PreconditionerScheduleConfig(limit_iters=10)
    result = create_scheduled_preconditioner(primary, schedule)

    # Should wrap in ScheduledPreconditioner
    assert isinstance(result, ScheduledPreconditioner)
    assert result._limit_iters == 10


def test_create_scheduled_preconditioner_default_fallback() -> None:
    """Test Identity used as default fallback."""
    primary = Identity()
    schedule = PreconditionerScheduleConfig(limit_iters=5)
    result = create_scheduled_preconditioner(primary, schedule)

    # Apply after limit and verify fallback is Identity
    ctx = PreconditionerContext(iteration=10, residual_norm=1.0, rhs_norm=1.0)
    r = np.array([1.0, 2.0])
    z = result.apply(r, ctx)

    # Identity behavior: z = r
    assert_array_equal(z, r)


def test_create_scheduled_preconditioner_with_identity_fallback() -> None:
    """Test explicit Identity fallback type."""
    primary = Identity()
    schedule = PreconditionerScheduleConfig(
        limit_iters=5,
        fallback=PreconditionerType.IDENTITY,
    )
    result = create_scheduled_preconditioner(primary, schedule)

    # Apply after limit and verify fallback is Identity
    ctx = PreconditionerContext(iteration=10, residual_norm=1.0, rhs_norm=1.0)
    r = np.array([1.0, 2.0])
    z = result.apply(r, ctx)

    # Identity behavior: z = r
    assert_array_equal(z, r)
