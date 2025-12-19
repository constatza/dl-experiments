"""Tests for neutral_ones strategy."""

import numpy as np
import pytest

from neuralls.generation.strategies.neutral_ones import NeutralOnesStrategy, NeutralOnesConfig


@pytest.fixture
def sample_matrix():
    """Create a simple 3x3 test matrix."""
    return np.array([
        [4.0, 1.0, 0.0],
        [1.0, 4.0, 1.0],
        [0.0, 1.0, 4.0]
    ], dtype=np.float64)


def test_neutral_ones_strategy_basic(sample_matrix):
    """Test neutral_ones strategy generates x=ones and b=A@x."""
    strategy = NeutralOnesStrategy()
    config = {"samples": 1, "seed": 42}

    result = strategy.generate(
        matrix=sample_matrix,
        rhs=None,
        cfg=config,
        archive=None,
    )

    # Check shapes
    assert result.rhs.shape == (1, 3)
    assert result.solutions.shape == (1, 3)

    # Check solution is all ones
    np.testing.assert_array_equal(result.solutions, np.ones((1, 3)))

    # Check b = A @ x
    expected_rhs = sample_matrix @ np.ones(3)
    np.testing.assert_array_almost_equal(result.rhs[0], expected_rhs)


def test_neutral_ones_multiple_samples(sample_matrix):
    """Test neutral_ones strategy with multiple samples."""
    strategy = NeutralOnesStrategy()
    config = {"samples": 3, "seed": 42}

    result = strategy.generate(
        matrix=sample_matrix,
        rhs=None,
        cfg=config,
        archive=None,
    )

    # Check shapes
    assert result.rhs.shape == (3, 3)
    assert result.solutions.shape == (3, 3)

    # All solutions should be ones
    np.testing.assert_array_equal(result.solutions, np.ones((3, 3)))

    # All RHS should be identical (A @ ones)
    expected_rhs = sample_matrix @ np.ones(3)
    for i in range(3):
        np.testing.assert_array_almost_equal(result.rhs[i], expected_rhs)


def test_neutral_ones_requires_rhs():
    """Test that neutral_ones doesn't require mother RHS."""
    strategy = NeutralOnesStrategy()
    assert strategy.requires_rhs() is False


def test_neutral_ones_deterministic(sample_matrix):
    """Test that neutral_ones produces deterministic results."""
    strategy = NeutralOnesStrategy()
    config = {"samples": 2, "seed": 42}

    result1 = strategy.generate(matrix=sample_matrix, rhs=None, cfg=config, archive=None)
    result2 = strategy.generate(matrix=sample_matrix, rhs=None, cfg=config, archive=None)

    np.testing.assert_array_equal(result1.rhs, result2.rhs)
    np.testing.assert_array_equal(result1.solutions, result2.solutions)


def test_neutral_ones_config_validation():
    """Test NeutralOnesConfig validation."""
    # Valid config
    config = NeutralOnesConfig(samples=1, seed=42)
    assert config.samples == 1
    assert config.seed == 42

    # Negative samples should raise validation error
    with pytest.raises(Exception):  # Pydantic ValidationError
        NeutralOnesConfig(samples=-2, seed=42)
