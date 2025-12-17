from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scipy.linalg import eigh

from neuralls.generation import generate_mixture
from neuralls.generation.base import _generate_eigenvector_combinations
from neuralls.normalization import (
    ErrorTraceSamples,
    ResidualTraceSamples,
    apply_normalization,
)





def test_residual_strategy_with_archive() -> None:
    """Test residual strategy accepts pre-existing solutions from archive."""
    A = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    b = np.array([1.0, 0.0], dtype=np.float64)

    # Create archive solutions
    archive_sols = np.array(
        [[0.5, 0.3], [0.2, 0.8], [0.1, 0.4]],
        dtype=np.float64,
    )
    archive_rhs = np.array([A @ x for x in archive_sols], dtype=np.float64)

    # Generate with archive
    rhs, solutions, residuals, error_traces = generate_mixture(
        A=A,
        b=b,
        mix={"cg_residual": 1.0},
        total=2,
        strategy_overrides={"cg_residual": {"residual_iters": 3}},
        seed=7,
        shuffle=False,
        archive_solutions=archive_sols,
        archive_rhs=archive_rhs,
    )

    # Verify we got the archive solutions (first 2)
    assert np.allclose(solutions, archive_sols[:2])
    assert np.allclose(rhs, archive_rhs[:2])

    # Verify residuals were collected
    assert residuals is not None
    assert residuals.residuals.shape[1] == A.shape[0]


def test_residual_strategy_archive_validation() -> None:
    """Test residual strategy validates archive has enough samples."""
    A = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    b = np.array([1.0, 0.0], dtype=np.float64)

    # Archive with only 1 solution, but we need 2
    archive_sols = np.array([[0.5, 0.3]], dtype=np.float64)

    try:
        rhs, solutions, residuals, error_traces = generate_mixture(
            A=A,
            b=b,
            mix={"cg_residual": 1.0},
            total=2,  # Request more than archive has
            strategy_overrides={"cg_residual": {"residual_iters": 3}},
            seed=7,
            shuffle=False,
            archive_solutions=archive_sols,
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Not enough archive solutions" in str(e)


def test_diagonal_normalization_scales_rows(tmp_path: Path) -> None:
    """Diagonal normalization should apply symmetric diagonal scaling: D^(-1/2) @ A @ D^(-1/2)."""
    A = np.array([[4.0, 1.0], [2.0, 6.0]], dtype=np.float64)
    R = np.array([[2.0, 3.0], [1.0, -1.0]], dtype=np.float64)
    X = np.array([[0.5, 0.25], [0.1, 0.2]], dtype=np.float64)

    residual_traces = ResidualTraceSamples(
        residuals=np.array([[0.4, 0.2], [0.3, 0.1]], dtype=np.float64),
        solutions=np.array([[0.05, 0.02], [0.04, 0.01]], dtype=np.float64),
        sample_indices=np.array([0, 1], dtype=np.int64),
        iteration_indices=np.array([0, 1], dtype=np.int64),
        search_directions=np.array([[0.5, 0.4], [0.2, 0.3]], dtype=np.float64),
        search_direction_products=np.array([[0.8, 0.6], [0.3, 0.2]], dtype=np.float64),
    )
    error_traces = ErrorTraceSamples(
        residuals=np.array([[0.2, -0.1]], dtype=np.float64),
        solutions_current=np.array([[0.0, 0.0]], dtype=np.float64),
        errors=np.array([[0.05, -0.02]], dtype=np.float64),
        true_solutions=np.array([[0.5, 0.4]], dtype=np.float64),
        sample_indices=np.array([0], dtype=np.int64),
        iteration_indices=np.array([0], dtype=np.int64),
    )

    normalized = apply_normalization(
        normalize="diagonal",
        A_original=A,
        R=R,
        X=X,
        dataset_dir=tmp_path,
        residual_traces=residual_traces,
        error_traces=error_traces,
    )

    # Compute expected symmetric diagonal scaling
    diag_sqrt_inv = 1.0 / np.sqrt(np.diag(A))
    diag_sqrt = 1.0 / diag_sqrt_inv
    expected_matrix = diag_sqrt_inv[:, None] * A * diag_sqrt_inv[None, :]
    expected_rhs = R * diag_sqrt_inv
    expected_solutions = X * diag_sqrt

    np.testing.assert_allclose(normalized.matrix, expected_matrix)
    np.testing.assert_allclose(normalized.rhs_samples, expected_rhs)
    np.testing.assert_allclose(normalized.sol_samples, expected_solutions)
    assert normalized.residual_traces is not None
    np.testing.assert_allclose(
        normalized.residual_traces.residuals,
        residual_traces.residuals * diag_sqrt_inv,
    )
    np.testing.assert_allclose(
        normalized.residual_traces.solutions,
        residual_traces.solutions * diag_sqrt,
    )
    assert normalized.residual_traces.search_directions is not None
    assert normalized.residual_traces.search_direction_products is not None
    np.testing.assert_allclose(
        normalized.residual_traces.search_directions,
        residual_traces.search_directions * diag_sqrt,
    )
    np.testing.assert_allclose(
        normalized.residual_traces.search_direction_products,
        residual_traces.search_direction_products * diag_sqrt_inv,
    )
    assert normalized.error_traces is not None
    np.testing.assert_allclose(
        normalized.error_traces.residuals,
        error_traces.residuals * diag_sqrt_inv,
    )
    np.testing.assert_allclose(
        normalized.error_traces.errors,
        error_traces.errors * diag_sqrt_inv,
    )


def test_diagonal_normalization_rejects_zero_diagonal(tmp_path: Path) -> None:
    """Diagonal normalization should guard against zero diagonal entries."""
    A = np.array([[0.0, 1.0], [1.0, 2.0]], dtype=np.float64)
    R = np.ones((1, 2), dtype=np.float64)
    X = np.ones((1, 2), dtype=np.float64)

    with pytest.raises(ValueError, match="near-zero diagonal"):
        apply_normalization(
            normalize="diagonal",
            A_original=A,
            R=R,
            X=X,
            dataset_dir=tmp_path,
        )


# ========================================================================
# Tests for residual_error_strategy()
# ========================================================================


def test_error_strategy_with_random(
    small_spd_matrix: np.ndarray,
    small_rhs: np.ndarray,
    test_seed: int,
) -> None:
    """Test error strategy with random generation.

    Verifies that the error strategy:
    - Generates error traces successfully
    - Produces consistent shapes
    - Populates all expected fields

    Args:
        small_spd_matrix: Test SPD matrix fixture
        small_rhs: Test RHS vector fixture
        test_seed: Random seed fixture
    """
    rhs, solutions, residuals, error_traces = generate_mixture(
        A=small_spd_matrix,
        b=small_rhs,
        mix={"residual_error": 1.0},
        total=2,
        strategy_overrides={"residual_error": {"residual_iters": 3}},
        seed=test_seed,
        shuffle=False,
    )

    # Verify basic shapes
    assert rhs.shape == (2, 2), "RHS should have 2 samples of dimension 2"
    assert solutions.shape == (2, 2), "Solutions should have 2 samples of dimension 2"

    # Verify error traces are populated
    assert error_traces is not None, "Error traces should be populated"
    assert error_traces.residuals.shape[1] == small_spd_matrix.shape[0]
    assert error_traces.solutions_current.shape[1] == small_spd_matrix.shape[0]
    assert error_traces.errors.shape[1] == small_spd_matrix.shape[0]

    # Verify trace dimensions match
    n_traces = error_traces.residuals.shape[0]
    assert error_traces.solutions_current.shape[0] == n_traces
    assert error_traces.errors.shape[0] == n_traces
    assert error_traces.sample_indices.shape[0] == n_traces
    assert error_traces.iteration_indices.shape[0] == n_traces

    # Verify true_solutions matches solutions
    assert error_traces.true_solutions.shape == solutions.shape
    assert np.allclose(error_traces.true_solutions, solutions)

    # Verify sample indices are valid
    assert error_traces.sample_indices.max() < solutions.shape[0]
    assert np.all(error_traces.sample_indices >= 0)

    # Verify iteration indices are valid
    assert np.all(error_traces.iteration_indices >= 0)
    assert error_traces.iteration_indices.max() <= 3  # residual_iters


def test_error_strategy_with_archive(
    small_spd_matrix: np.ndarray,
    small_rhs: np.ndarray,
    archive_solutions: np.ndarray,
    archive_rhs: np.ndarray,
    test_seed: int,
) -> None:
    """Test error strategy with archive solutions.

    Verifies that the error strategy:
    - Uses archive solutions instead of random generation
    - Computes error traces based on archive data
    - Correctly associates errors with true solutions

    Args:
        small_spd_matrix: Test SPD matrix fixture
        small_rhs: Test RHS vector fixture
        archive_solutions: Pre-computed archive solutions fixture
        archive_rhs: Pre-computed archive RHS vectors fixture
        test_seed: Random seed fixture
    """
    rhs, solutions, residuals, error_traces = generate_mixture(
        A=small_spd_matrix,
        b=small_rhs,
        mix={"residual_error": 1.0},
        total=2,
        strategy_overrides={"residual_error": {"residual_iters": 3}},
        seed=test_seed,
        shuffle=False,
        archive_solutions=archive_solutions,
        archive_rhs=archive_rhs,
    )

    # Verify we got the archive solutions (first 2)
    assert np.allclose(solutions, archive_solutions[:2]), (
        "Solutions should match first 2 archive solutions"
    )
    assert np.allclose(rhs, archive_rhs[:2]), (
        "RHS should match first 2 archive RHS vectors"
    )

    # Verify error traces were collected
    assert error_traces is not None, "Error traces should be populated"
    assert error_traces.residuals.shape[1] == small_spd_matrix.shape[0]

    # Verify true_solutions matches archive solutions
    assert np.allclose(error_traces.true_solutions, archive_solutions[:2]), (
        "True solutions should match archive solutions"
    )


def test_error_vectors_satisfy_equation(
    small_spd_matrix: np.ndarray,
    small_rhs: np.ndarray,
    test_seed: int,
) -> None:
    """Test that error vectors satisfy error_k = x* - x_k.

    This is the KEY mathematical property: for each CG iteration k,
    the error vector should equal the true solution minus the current iterate.

    Args:
        small_spd_matrix: Test SPD matrix fixture
        small_rhs: Test RHS vector fixture
        test_seed: Random seed fixture
    """
    rhs, solutions, residuals, error_traces = generate_mixture(
        A=small_spd_matrix,
        b=small_rhs,
        mix={"residual_error": 1.0},
        total=3,
        strategy_overrides={"residual_error": {"residual_iters": 5}},
        seed=test_seed,
        shuffle=False,
    )

    assert error_traces is not None, "Error traces should be populated"

    # For each trace entry, verify: error_k = x* - x_k
    for i in range(error_traces.residuals.shape[0]):
        sample_idx = error_traces.sample_indices[i]
        x_star = error_traces.true_solutions[sample_idx]
        x_k = error_traces.solutions_current[i]
        error_k = error_traces.errors[i]

        expected_error = x_star - x_k

        assert np.allclose(error_k, expected_error, atol=1e-12), (
            f"Error vector at trace {i} should equal x* - x_k"
        )


def test_residuals_match_current_solutions(
    small_spd_matrix: np.ndarray,
    small_rhs: np.ndarray,
    archive_solutions: np.ndarray,
    archive_rhs: np.ndarray,
    test_seed: int,
) -> None:
    """Test that residuals match current solutions: r_k = b - A @ x_k.

    This ensures that the captured CG states are consistent:
    the residual should equal b - A @ x_k for the current iterate.

    Args:
        small_spd_matrix: Test SPD matrix fixture
        small_rhs: Test RHS vector fixture
        archive_solutions: Pre-computed archive solutions fixture
        archive_rhs: Pre-computed archive RHS vectors fixture
        test_seed: Random seed fixture
    """
    rhs, solutions, residuals, error_traces = generate_mixture(
        A=small_spd_matrix,
        b=small_rhs,
        mix={"residual_error": 1.0},
        total=2,
        strategy_overrides={"residual_error": {"residual_iters": 4}},
        seed=test_seed,
        shuffle=False,
        archive_solutions=archive_solutions,
        archive_rhs=archive_rhs,
    )

    assert error_traces is not None, "Error traces should be populated"

    # For each trace entry, verify: r_k ≈ b - A @ x_k
    for i in range(error_traces.residuals.shape[0]):
        sample_idx = error_traces.sample_indices[i]
        b_vec = rhs[sample_idx]
        x_k = error_traces.solutions_current[i]
        r_k = error_traces.residuals[i]

        expected_residual = b_vec - small_spd_matrix @ x_k

        assert np.allclose(r_k, expected_residual, atol=1e-10), (
            f"Residual at trace {i} should equal b - A @ x_k"
        )


def test_error_strategy_validation(
    small_spd_matrix: np.ndarray,
    small_rhs: np.ndarray,
    test_seed: int,
) -> None:
    """Test error strategy validates sufficient archive samples.

    Verifies that the strategy raises an appropriate error when
    the archive doesn't contain enough solutions.

    Args:
        small_spd_matrix: Test SPD matrix fixture
        small_rhs: Test RHS vector fixture
        test_seed: Random seed fixture
    """
    # Archive with only 1 solution, but we need 3
    insufficient_archive = np.array([[0.5, 0.3]], dtype=np.float64)

    try:
        rhs, solutions, residuals, error_traces = generate_mixture(
            A=small_spd_matrix,
            b=small_rhs,
            mix={"residual_error": 1.0},
            total=3,  # Request more than archive has
            strategy_overrides={"residual_error": {"residual_iters": 3}},
            seed=test_seed,
            shuffle=False,
            archive_solutions=insufficient_archive,
        )
        assert False, "Should have raised ValueError for insufficient archive"
    except ValueError as e:
        assert "Not enough archive solutions" in str(e), (
            "Error message should mention insufficient archive solutions"
        )


def test_error_strategy_in_generate_mixture(
    small_spd_matrix: np.ndarray,
    small_rhs: np.ndarray,
    test_seed: int,
) -> None:
    """Test error strategy works in generate_mixture with multiple strategies.

    Verifies that error strategy can be combined with other strategies
    in a mixed dataset generation.

    Args:
        small_spd_matrix: Test SPD matrix fixture
        small_rhs: Test RHS vector fixture
        test_seed: Random seed fixture
    """
    rhs, solutions, residuals, error_traces = generate_mixture(
        A=small_spd_matrix,
        b=small_rhs,
        mix={"normal": 0.5, "residual_error": 0.5},
        total=10,
        strategy_overrides={"residual_error": {"residual_iters": 3}},
        seed=test_seed,
        shuffle=False,
    )

    # Verify both strategies contributed
    assert rhs.shape == (10, 2), "Should have 10 total samples"
    assert solutions.shape == (10, 2)

    # Verify error traces are present (from residual_error strategy)
    assert error_traces is not None, "Error traces should be populated"

    # Verify error traces have approximately 5 samples worth of data
    # (each sample generates multiple traces across CG iterations)
    unique_samples = np.unique(error_traces.sample_indices)
    assert len(unique_samples) >= 4, (
        "Should have error traces from approximately 5 samples (allowing for rounding)"
    )

    # Verify residuals is None (normal strategy doesn't produce residual traces)
    assert residuals is None, (
        "Residuals should be None when only one strategy produces them"
    )


def test_error_strategy_traces_structure(
    small_spd_matrix: np.ndarray,
    small_rhs: np.ndarray,
    test_seed: int,
) -> None:
    """Test the internal structure of error traces.

    Verifies that:
    - Each sample generates multiple traces (one per CG iteration)
    - Iteration indices are sequential for each sample
    - Sample indices correctly map to true solutions

    Args:
        small_spd_matrix: Test SPD matrix fixture
        small_rhs: Test RHS vector fixture
        test_seed: Random seed fixture
    """
    rhs, solutions, residuals, error_traces = generate_mixture(
        A=small_spd_matrix,
        b=small_rhs,
        mix={"residual_error": 1.0},
        total=3,
        strategy_overrides={"residual_error": {"residual_iters": 4}},
        seed=test_seed,
        shuffle=False,
    )

    assert error_traces is not None, "Error traces should be populated"

    # Verify each sample has multiple traces (CG iterations)
    for sample_idx in range(3):
        mask = error_traces.sample_indices == sample_idx
        sample_traces = error_traces.iteration_indices[mask]

        assert len(sample_traces) > 0, (
            f"Sample {sample_idx} should have at least one trace"
        )

        # Verify iteration indices are sequential starting from 0
        assert sample_traces[0] == 0, (
            f"First iteration for sample {sample_idx} should be 0"
        )
        assert np.all(np.diff(sample_traces) == 1), (
            f"Iteration indices for sample {sample_idx} should be sequential"
        )


def test_error_strategy_with_zero_iterations(
    small_spd_matrix: np.ndarray,
    small_rhs: np.ndarray,
    test_seed: int,
) -> None:
    """Test error strategy behavior with zero CG iterations.

    Verifies that the strategy handles edge case of no CG iterations gracefully.

    Args:
        small_spd_matrix: Test SPD matrix fixture
        small_rhs: Test RHS vector fixture
        test_seed: Random seed fixture
    """
    rhs, solutions, residuals, error_traces = generate_mixture(
        A=small_spd_matrix,
        b=small_rhs,
        mix={"residual_error": 1.0},
        total=2,
        strategy_overrides={"residual_error": {"residual_iters": 0}},  # Zero iterations
        seed=test_seed,
        shuffle=False,
    )

    # Verify basic outputs are still valid
    assert rhs.shape == (2, 2)
    assert solutions.shape == (2, 2)

    # With zero iterations, we should still get error traces
    # (at least the initial state with x_0 = 0 and error = x*)
    assert error_traces is not None, "Error traces should exist even with 0 iterations"
    assert error_traces.residuals.shape[0] >= 2, (
        "Should have at least one trace per sample (initial state)"
    )


# =============================================================================
# EIGENVECTOR STRATEGY TESTS
# =============================================================================


def test_eigenvector_forward_basic(tmp_path: Path) -> None:
    """Test eigenvector forward strategy produces exact samples."""
    A = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    b = np.array([1.0, 0.0], dtype=np.float64)

    rhs, solutions, _, _ = generate_mixture(
        A=A,
        b=b,
        mix={"eigenvector_forward": 1.0},
        total=2,
        seed=42,
        shuffle=False,
    )

    assert rhs.shape == (2, 2)
    assert solutions.shape == (2, 2)

    # Verify b = A @ x for each sample (machine precision)
    for i in range(2):
        residual = rhs[i] - A @ solutions[i]
        rel_residual = np.linalg.norm(residual) / np.linalg.norm(rhs[i])
        assert rel_residual < 1e-14, f"Sample {i}: residual {rel_residual:.2e}"


def test_eigenvector_inverse_machine_precision(tmp_path: Path) -> None:
    """Test eigenvector inverse strategy achieves machine precision."""
    A = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    b = np.array([1.0, 0.0], dtype=np.float64)

    rhs, solutions, _, _ = generate_mixture(
        A=A,
        b=b,
        mix={"eigenvector_inverse": 1.0},
        total=2,
        seed=42,
        shuffle=False,
    )

    # Verify A @ x = b at machine precision
    for i in range(2):
        residual = A @ solutions[i] - rhs[i]
        rel_residual = np.linalg.norm(residual) / np.linalg.norm(rhs[i])
        assert rel_residual < 1e-14, f"Sample {i}: residual {rel_residual:.2e}"


def test_eigenvector_requires_symmetric(tmp_path: Path) -> None:
    """Test eigenvector strategies reject non-symmetric matrices."""
    A = np.array([[4.0, 1.0], [2.0, 3.0]], dtype=np.float64)  # Asymmetric
    b = np.array([1.0, 0.0], dtype=np.float64)

    with pytest.raises(ValueError, match="symmetric"):
        generate_mixture(A=A, b=b, mix={"eigenvector_forward": 1.0}, total=2)


def test_eigenvector_count_exceeds_dimension(tmp_path: Path) -> None:
    """Test that requesting more samples than eigenvectors now works via combinations."""
    A = np.eye(3, dtype=np.float64) * 2.0
    b = np.ones(3, dtype=np.float64)

    # With new implementation, this should work (generates 5 combinations from 3 eigenvectors)
    rhs, solutions, _, _ = generate_mixture(
        A=A, b=b, mix={"eigenvector_forward": 1.0}, total=5
    )

    assert rhs.shape == (5, 3)
    assert solutions.shape == (5, 3)

    # Verify accuracy
    for i in range(5):
        residual = A @ solutions[i] - rhs[i]
        rel_residual = np.linalg.norm(residual) / np.linalg.norm(rhs[i])
        assert rel_residual < 1e-14


def test_eigenvector_selection_modes(tmp_path: Path) -> None:
    """Test different eigenvector eigenvalue range selections."""
    A = np.diag([1.0, 2.0, 3.0, 4.0])
    b = np.ones(4, dtype=np.float64)

    # Test "smallest" mode - use eigenvectors with k smallest eigenvalues
    _, sols_first, _, _ = generate_mixture(
        A=A,
        b=b,
        mix={"eigenvector_forward": 1.0},
        total=2,
        strategy_overrides={
            "eigenvector_forward": {
                "which": "smallest",
                "num_eigenvectors": 2,  # Use k=2 smallest eigenvalues
            }
        },
        shuffle=False,
        seed=42,
    )

    # Test "largest" mode - use eigenvectors with k largest eigenvalues
    _, sols_last, _, _ = generate_mixture(
        A=A,
        b=b,
        mix={"eigenvector_forward": 1.0},
        total=2,
        strategy_overrides={
            "eigenvector_forward": {
                "which": "largest",
                "num_eigenvectors": 2,  # Use k=2 largest eigenvalues
            }
        },
        shuffle=False,
        seed=42,
    )

    # Solutions should be different (different eigenvector subspaces)
    assert not np.allclose(sols_first, sols_last)


def test_mixed_strategy_with_eigenvectors(tmp_path: Path) -> None:
    """Test mixing random and eigenvector strategies."""
    A = np.eye(10, dtype=np.float64) * 2.0
    b = np.ones(10, dtype=np.float64)

    rhs, solutions, _, _ = generate_mixture(
        A=A,
        b=b,
        mix={"random": 0.5, "eigenvector_forward": 0.5},
        total=20,
        seed=42,
    )

    assert rhs.shape == (20, 10)
    assert solutions.shape == (20, 10)

    # All samples should have low residuals
    for i in range(20):
        residual = A @ solutions[i] - rhs[i]
        rel_residual = np.linalg.norm(residual) / np.linalg.norm(rhs[i])
        assert rel_residual < 1e-9, f"Sample {i}: residual {rel_residual:.2e}"


# =============================================================================
# EIGENVECTOR LINEAR COMBINATIONS: Tests for random linear combinations
# =============================================================================


def test_generate_eigenvector_combinations_vectorized(tmp_path: Path) -> None:
    """Test vectorized linear combination generation."""
    # Use 4x4 diagonal matrix for simple eigenvectors
    A = np.diag([1.0, 2.0, 3.0, 4.0])
    eigenvalues, eigenvectors = eigh(A)

    # Select first 3 eigenvectors
    selected = eigenvectors[:, :3]  # Shape (4, 3)

    rng = np.random.default_rng(42)
    combinations = _generate_eigenvector_combinations(selected, num_samples=10, rng=rng)

    # Shape check
    assert combinations.shape == (10, 4), "Output shape mismatch"

    # Each combination should be in span of selected eigenvectors
    # Verify by projecting back onto basis
    for i in range(10):
        # Project onto basis and reconstruct
        reconstruction = selected @ (selected.T @ combinations[i])
        np.testing.assert_allclose(reconstruction, combinations[i], rtol=1e-10)

    # Verify linear independence (combinations should differ)
    for i in range(9):
        assert not np.allclose(combinations[i], combinations[i + 1])


def test_eigenvector_combinations_reproducible(tmp_path: Path) -> None:
    """Test that combinations are reproducible with same seed."""
    A = np.eye(5, dtype=np.float64) * 2.0
    eigenvalues, eigenvectors = eigh(A)

    rng1 = np.random.default_rng(123)
    combinations1 = _generate_eigenvector_combinations(eigenvectors, 5, rng1)

    rng2 = np.random.default_rng(123)
    combinations2 = _generate_eigenvector_combinations(eigenvectors, 5, rng2)

    np.testing.assert_array_equal(combinations1, combinations2)


def test_eigenvector_forward_with_combinations(tmp_path: Path) -> None:
    """Test eigenvector_forward with linear combinations."""
    A = np.diag([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.ones(5, dtype=np.float64)

    rhs, solutions, _, _ = generate_mixture(
        A=A,
        b=b,
        mix={"eigenvector_forward": 1.0},
        total=20,  # Generate 20 samples
        strategy_overrides={
            "eigenvector_forward": {
                "num_eigenvectors": 3,  # From basis of 3
                "which": "smallest",
                "include_eigenvectors": False,  # Only combinations
            }
        },
        seed=42,
        shuffle=False,
    )

    assert rhs.shape == (20, 5)
    assert solutions.shape == (20, 5)

    # Verify machine precision accuracy
    for i in range(20):
        residual = A @ solutions[i] - rhs[i]
        rel_residual = np.linalg.norm(residual) / np.linalg.norm(rhs[i])
        assert rel_residual < 1e-14, f"Sample {i}: residual {rel_residual:.2e}"


def test_eigenvector_forward_include_eigenvectors(tmp_path: Path) -> None:
    """Test including original eigenvectors in output."""
    A = np.diag([1.0, 2.0, 3.0, 4.0])
    b = np.ones(4, dtype=np.float64)

    rhs, solutions, _, _ = generate_mixture(
        A=A,
        b=b,
        mix={"eigenvector_forward": 1.0},
        total=6,  # 2 eigenvectors + 4 combinations
        strategy_overrides={
            "eigenvector_forward": {
                "num_eigenvectors": 2,
                "which": "smallest",
                "include_eigenvectors": True,
            }
        },
        seed=42,
        shuffle=False,
    )

    assert solutions.shape == (6, 4)

    # First 2 should be eigenvectors (diagonal matrix -> standard basis)
    eigenvalues, eigenvectors = eigh(A)
    np.testing.assert_allclose(solutions[:2], eigenvectors[:, :2].T, rtol=1e-10)

    # Remaining 4 should be combinations (different from eigenvectors)
    for i in range(2, 6):
        # Should not match any single eigenvector exactly
        for j in range(4):
            assert not np.allclose(solutions[i], eigenvectors[:, j])


def test_eigenvector_inverse_with_combinations(tmp_path: Path) -> None:
    """Test eigenvector_inverse with combinations as RHS."""
    A = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    b = np.array([1.0, 0.0], dtype=np.float64)

    rhs, solutions, _, _ = generate_mixture(
        A=A,
        b=b,
        mix={"eigenvector_inverse": 1.0},
        total=10,
        strategy_overrides={
            "eigenvector_inverse": {
                "num_eigenvectors": 2,  # Use both eigenvectors
                "include_eigenvectors": False,
            }
        },
        seed=42,
    )

    # Verify solutions are accurate
    for i in range(10):
        residual = A @ solutions[i] - rhs[i]
        rel_residual = np.linalg.norm(residual) / np.linalg.norm(rhs[i])
        assert rel_residual < 1e-14


def test_eigenvector_validation_num_exceeds_dimension(tmp_path: Path) -> None:
    """Test error when num_eigenvectors > matrix dimension."""
    A = np.eye(3, dtype=np.float64) * 2.0
    b = np.ones(3, dtype=np.float64)

    with pytest.raises(ValueError, match="must be positive and ≤"):
        generate_mixture(
            A=A,
            b=b,
            mix={"eigenvector_forward": 1.0},
            total=5,
            strategy_overrides={"eigenvector_forward": {"num_eigenvectors": 5}},  # > 3
        )


def test_eigenvector_validation_include_requires_enough_samples(tmp_path: Path) -> None:
    """Test error when include_eigenvectors=True but samples < num_eigenvectors."""
    A = np.eye(5, dtype=np.float64) * 2.0
    b = np.ones(5, dtype=np.float64)

    with pytest.raises(ValueError, match="must be >="):
        generate_mixture(
            A=A,
            b=b,
            mix={"eigenvector_forward": 1.0},
            total=2,  # Only 2 samples
            strategy_overrides={
                "eigenvector_forward": {
                    "num_eigenvectors": 3,  # But need space for 3 eigenvectors
                    "include_eigenvectors": True,
                }
            },
        )


def test_eigenvector_forward_rhs_computation_change(tmp_path: Path) -> None:
    """Verify RHS is computed as A @ v (not λ * v) for forward strategy."""
    A = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    b = np.array([1.0, 0.0], dtype=np.float64)

    rhs, solutions, _, _ = generate_mixture(
        A=A,
        b=b,
        mix={"eigenvector_forward": 1.0},
        total=2,
        strategy_overrides={
            "eigenvector_forward": {
                "num_eigenvectors": 2,
                "include_eigenvectors": False,
            }
        },
        seed=42,
        shuffle=False,
    )

    # Verify b = A @ x directly (not using eigenvalue equation)
    for i in range(2):
        expected_rhs = A @ solutions[i]
        np.testing.assert_allclose(rhs[i], expected_rhs, rtol=1e-14)


def test_eigenvector_backward_compatible_defaults(tmp_path: Path) -> None:
    """Test that old configs still work without new parameters."""
    A = np.diag([1.0, 2.0, 3.0])
    b = np.ones(3, dtype=np.float64)

    # Old-style config (no new parameters)
    rhs, solutions, _, _ = generate_mixture(
        A=A,
        b=b,
        mix={"eigenvector_forward": 1.0},
        total=3,
        strategy_overrides={"eigenvector_forward": {"selection_mode": "first"}},
        seed=42,
    )

    # Should generate 3 combinations from all 3 eigenvectors
    assert rhs.shape == (3, 3)
    assert solutions.shape == (3, 3)

    # Verify accuracy
    for i in range(3):
        residual = A @ solutions[i] - rhs[i]
        rel_residual = np.linalg.norm(residual) / np.linalg.norm(rhs[i])
        assert rel_residual < 1e-14


# =============================================================================
# PYDANTIC VALIDATION TESTS
# =============================================================================


    def test_pydantic_rejects_unknown_parameters() -> None:
        """Test that Pydantic validation rejects unknown parameters due to extra='forbid'.
    
        Note: The orchestration layer filters out unknown parameters before passing to strategies,
        so this test validates by directly instantiating the config class.
        """
        from neuralls.generation.strategy_configs import KrylovConfig
        from pydantic import ValidationError
    
        # Try to create a config with an unknown parameter directly
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            KrylovConfig(samples=10, unknown_param=123)

def test_pydantic_rejects_invalid_literal_values() -> None:
    """Test that Pydantic validation rejects invalid Literal values."""
    A = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    b = np.array([1.0, 0.0], dtype=np.float64)

    # Try to pass an invalid 'which' value (should be 'smallest', 'largest', or 'both')
    with pytest.raises(ValueError, match="Input should be"):
        generate_mixture(
            A=A,
            b=b,
            mix={"eigenvector_forward": 1.0},
            total=2,
            strategy_overrides={"eigenvector_forward": {"which": "invalid"}},
        )


def test_pydantic_requires_rhs_glob_for_rhs_archive() -> None:
    """Test that rhs_glob is required for rhs_archive strategy."""
    A = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    b = np.array([1.0, 0.0], dtype=np.float64)

    # Try to use rhs_archive without providing rhs_glob
    with pytest.raises(ValueError, match="Field required"):
        generate_mixture(
            A=A,
            b=b,
            mix={"rhs_archive": 1.0},
            total=2,
            strategy_overrides={"rhs_archive": {}},  # Missing rhs_glob
        )


def test_pydantic_requires_solutions_glob_for_solution_archive() -> None:
    """Test that solutions_glob is required for solution_archive strategy."""
    A = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    b = np.array([1.0, 0.0], dtype=np.float64)

    # Try to use solution_archive without providing solutions_glob
    with pytest.raises(ValueError, match="Field required"):
        generate_mixture(
            A=A,
            b=b,
            mix={"solution_archive": 1.0},
            total=2,
            strategy_overrides={"solution_archive": {}},  # Missing solutions_glob
        )


def test_pydantic_validates_residual_iters_type() -> None:
    """Test that Pydantic validates parameter types (residual_iters must be int)."""
    A = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    b = np.array([1.0, 0.0], dtype=np.float64)

    # Try to pass a string for residual_iters (should be int)
    with pytest.raises(ValueError, match="Input should be a valid integer"):
        generate_mixture(
            A=A,
            b=b,
            mix={"residual_error": 1.0},
            total=2,
            strategy_overrides={"residual_error": {"residual_iters": "many"}},
        )


def test_pydantic_validates_krylov_iters_type() -> None:
    """Test that Pydantic validates parameter types (krylov_iters must be int)."""
    A = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    b = np.array([1.0, 0.0], dtype=np.float64)

    # Try to pass a float for krylov_iters (should be int)
    with pytest.raises(ValueError):
        generate_mixture(
            A=A,
            b=b,
            mix={"krylov": 1.0},
            total=2,
            strategy_overrides={"krylov": {"krylov_iters": 15.5}},
        )
