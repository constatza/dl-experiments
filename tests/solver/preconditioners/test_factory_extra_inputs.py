"""Unit tests for factory threading of extra_input_names to NeuralPreconditioner.

Tests that the composition factory correctly passes extra_input_names from
NeuralPreconditionerConfig to the NeuralPreconditioner instance.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from neuralls.platform.config.models.preconditioner import (
    NeuralPreconditionerConfig,
)
from neuralls.composition.preconditioners.factory import (
    create_preconditioner,
)
from neuralls.domain.solver.preconditioners.ports import PredictorAdapter, PredictorPort
from neuralls.domain.solver.preconditioners.implementations.neural import NeuralPreconditioner

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ==============================================================================
# Mock Predictor and Adapter
# ==============================================================================


class MockPredictor(PredictorPort):
    """Lightweight mock predictor for testing."""

    def apply(self, residual: NDArray, **extra_inputs: NDArray) -> NDArray:
        """Mock apply: returns input unchanged."""
        return residual.copy()

    def cleanup(self) -> None:
        """No-op cleanup."""
        pass


class MockAdapter(PredictorAdapter):
    """Lightweight mock adapter for testing."""

    def create_predictor(
        self,
        checkpoint_path: Path,
        config_path: Path | None = None,
        data_config_path: Path | None = None,
    ) -> PredictorPort:
        """Create mock predictor."""
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return MockPredictor()


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def well_conditioned_matrix() -> NDArray:
    """Well-conditioned 4x4 SPD matrix for testing.

    Returns:
        4x4 diagonal matrix
    """
    return np.diag([4.0, 3.0, 2.0, 2.0])


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


@pytest.fixture
def mock_adapter() -> MockAdapter:
    """Create mock adapter.

    Returns:
        MockAdapter instance
    """
    return MockAdapter()


# ==============================================================================
# Tests for extra_input_names Threading
# ==============================================================================


def test_factory_passes_single_extra_input_name(
    well_conditioned_matrix: NDArray,
    mock_checkpoint: Path,
    mock_adapter: MockAdapter,
) -> None:
    """Test that factory passes single extra_input_name to NeuralPreconditioner."""
    config = NeuralPreconditionerConfig(
        name="neural",
        checkpoint_path=mock_checkpoint,
        extra_input_names=("matrix",),
    )

    precond = create_preconditioner(well_conditioned_matrix, config, adapter=mock_adapter)

    assert isinstance(precond, NeuralPreconditioner)
    assert precond.extra_input_names == ("matrix",)


def test_factory_passes_multiple_extra_input_names(
    well_conditioned_matrix: NDArray,
    mock_checkpoint: Path,
    mock_adapter: MockAdapter,
) -> None:
    """Test that factory passes multiple extra_input_names to NeuralPreconditioner."""
    config = NeuralPreconditionerConfig(
        name="neural",
        checkpoint_path=mock_checkpoint,
        extra_input_names=("matrix", "coordinates", "bc_mask"),
    )

    precond = create_preconditioner(well_conditioned_matrix, config, adapter=mock_adapter)

    assert isinstance(precond, NeuralPreconditioner)
    assert precond.extra_input_names == ("matrix", "coordinates", "bc_mask")


def test_factory_empty_extra_input_names_by_default(
    well_conditioned_matrix: NDArray,
    mock_checkpoint: Path,
    mock_adapter: MockAdapter,
) -> None:
    """Test that factory uses empty tuple as default for extra_input_names."""
    config = NeuralPreconditionerConfig(
        name="neural",
        checkpoint_path=mock_checkpoint,
        # extra_input_names not specified - should default to ()
    )

    precond = create_preconditioner(well_conditioned_matrix, config, adapter=mock_adapter)

    assert isinstance(precond, NeuralPreconditioner)
    assert precond.extra_input_names == ()


def test_factory_preserves_extra_input_names_order(
    well_conditioned_matrix: NDArray,
    mock_checkpoint: Path,
    mock_adapter: MockAdapter,
) -> None:
    """Test that factory preserves order of extra_input_names."""
    names = ("first", "second", "third")
    config = NeuralPreconditionerConfig(
        name="neural",
        checkpoint_path=mock_checkpoint,
        extra_input_names=names,
    )

    precond = create_preconditioner(well_conditioned_matrix, config, adapter=mock_adapter)

    assert isinstance(precond, NeuralPreconditioner)
    assert precond.extra_input_names == names
    # Also verify it's a tuple (not mutated to list or other type)
    assert isinstance(precond.extra_input_names, tuple)
