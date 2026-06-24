"""Unit tests for predictor adapters and ports compliance.

Tests adapter contract, resource management, and error boundaries.
Focuses on OUR abstractions - does NOT test DLKit internals (GPU management, etc.).

Follows project principles:
- Use fixtures for all test data
- Use tmp_path for temporary files (never tempfile)
- Type hints throughout
- Mock DLKit to avoid testing infrastructure we don't own
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from neuralls.platform.config.models.preconditioner import PreconditionerType
from neuralls.domain.solver.preconditioners.ports import (
    ExtraInputPredictorPort,
    PredictorAdapter,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ==============================================================================
# Mock Predictor and Adapter
# ==============================================================================


class MockPredictor(ExtraInputPredictorPort):
    """Mock predictor for testing port contract."""

    def __init__(self) -> None:
        """Initialize mock predictor."""
        self.cleaned_up = False
        self.apply_count = 0

    @property
    def required_inputs(self) -> tuple[str, ...]:
        """Return empty tuple — this mock needs no extra inputs.

        Returns:
            Empty tuple of required input names.
        """
        return ()

    def apply(self, residual: NDArray, **extra_inputs: NDArray) -> NDArray:
        """Mock apply that scales residual by 0.5.

        Args:
            residual: Input residual vector.
            **extra_inputs: Ignored extra inputs.

        Returns:
            Residual scaled by 0.5 as float64.
        """
        self.apply_count += 1
        # Preserve shape and return float64
        result = residual * 0.5
        return result.astype(np.float64)

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
    """Mock adapter for testing adapter contract."""

    def __init__(self, predictor: MockPredictor | None = None) -> None:
        """Initialize with optional predictor."""
        self.predictor = predictor or MockPredictor()

    def create_predictor(
        self,
        checkpoint_path: Path,
        config_path: Path | None = None,
        data_config_path: Path | None = None,
    ) -> ExtraInputPredictorPort:
        """Create predictor, validating checkpoint exists.

        Args:
            checkpoint_path: Path to checkpoint file (must exist).
            config_path: Unused.
            data_config_path: Unused.

        Returns:
            The shared MockPredictor instance.

        Raises:
            FileNotFoundError: If checkpoint does not exist.
        """
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return self.predictor


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def residual_vector() -> NDArray:
    """Test residual vector.

    Returns:
        4D float64 residual vector
    """
    return np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)


@pytest.fixture
def mock_checkpoint(tmp_path: Path) -> Path:
    """Create mock checkpoint file.

    Args:
        tmp_path: Pytest temporary directory fixture

    Returns:
        Path to mock checkpoint file
    """
    checkpoint = tmp_path / "test_checkpoint.ckpt"
    checkpoint.write_text("mock checkpoint data")
    return checkpoint


@pytest.fixture
def mock_predictor() -> MockPredictor:
    """Create mock predictor instance.

    Returns:
        Fresh MockPredictor instance
    """
    return MockPredictor()


@pytest.fixture
def mock_adapter(mock_predictor: MockPredictor) -> MockAdapter:
    """Create mock adapter with predictor.

    Args:
        mock_predictor: MockPredictor fixture

    Returns:
        MockAdapter instance
    """
    return MockAdapter(predictor=mock_predictor)


# ==============================================================================
# Port Contract Tests (3 tests)
# ==============================================================================


def test_predictor_context_manager_calls_cleanup(mock_predictor: MockPredictor) -> None:
    """Test that context manager protocol calls cleanup on exit."""
    assert not mock_predictor.cleaned_up

    with mock_predictor:
        assert not mock_predictor.cleaned_up  # Not cleaned yet

    # Should be cleaned after exit
    assert mock_predictor.cleaned_up


def test_predictor_apply_returns_ndarray_with_correct_shape(
    mock_predictor: MockPredictor, residual_vector: NDArray
) -> None:
    """Test that apply returns NDArray with correct shape and dtype."""
    result = mock_predictor.apply(residual_vector)

    # Check type
    assert isinstance(result, np.ndarray)

    # Check shape preserved
    assert result.shape == residual_vector.shape

    # Check dtype is float64
    assert result.dtype == np.float64

    # Check result is correct (MockPredictor scales by 0.5)
    expected = residual_vector * 0.5
    assert_array_equal(result, expected)


def test_predictor_supports_multiple_apply_calls(
    mock_predictor: MockPredictor, residual_vector: NDArray
) -> None:
    """Test that predictor can be used multiple times before cleanup."""
    # Apply multiple times
    result1 = mock_predictor.apply(residual_vector)
    result2 = mock_predictor.apply(residual_vector * 2)
    result3 = mock_predictor.apply(residual_vector * 3)

    # Check apply was called 3 times
    assert mock_predictor.apply_count == 3

    # Check results are correct
    expected1 = residual_vector * 0.5
    expected2 = residual_vector * 2 * 0.5
    expected3 = residual_vector * 3 * 0.5

    assert_array_equal(result1, expected1)
    assert_array_equal(result2, expected2)
    assert_array_equal(result3, expected3)


# ==============================================================================
# Adapter Integration Tests (3 tests)
# ==============================================================================


def test_adapter_validates_checkpoint_exists(
    mock_adapter: MockAdapter, mock_checkpoint: Path, residual_vector: NDArray
) -> None:
    """Test that adapter validates checkpoint file exists."""
    # Valid checkpoint should work
    predictor = mock_adapter.create_predictor(mock_checkpoint)
    result = predictor.apply(residual_vector)

    # Check result is correct
    expected = residual_vector * 0.5
    assert_array_equal(result, expected)


def test_adapter_handles_missing_checkpoint_file(
    mock_adapter: MockAdapter,
    tmp_path: Path,
) -> None:
    """Test that adapter raises FileNotFoundError for missing checkpoint."""
    nonexistent_path = tmp_path / "missing" / "checkpoint.ckpt"

    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        mock_adapter.create_predictor(nonexistent_path)


def test_adapter_can_be_injected_into_factory(
    mock_adapter: MockAdapter, mock_checkpoint: Path, residual_vector: NDArray
) -> None:
    """Test DIP: adapter can be injected into factory function.

    Validates Dependency Inversion Principle - factory depends on
    PredictorAdapter abstraction, not concrete implementation.
    """
    from neuralls.platform.config.models.preconditioner import NeuralPreconditionerConfig
    from neuralls.composition.preconditioners.factory import create_preconditioner

    config = NeuralPreconditionerConfig(
        name="neural",
        type=PreconditionerType.NEURAL,
        checkpoint_path=mock_checkpoint,
    )

    # Create preconditioner with injected adapter
    matrix = np.eye(4)  # Unused by neural preconditioner
    precond = create_preconditioner(matrix, config, adapter=mock_adapter)

    # Apply preconditioner
    result = precond.apply(residual_vector)

    # Verify mock was used (scales by 0.5)
    expected = residual_vector * 0.5
    assert_array_equal(result, expected)

    # Verify predictor was called
    assert mock_adapter.predictor.apply_count == 1
