"""Test that data generation maintains A @ x = b consistency after normalization.

This test verifies that the orchestration layer does not double-normalize data
returned by strategies. Strategies receive a normalized matrix and produce
normalized data (RHS and solutions). The orchestration layer should NOT apply
additional normalization to this data.

Regression test for bug where build_dataset() was applying scale.scale_rhs()
and scale.scale_solution() to data that was already normalized by strategies.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.generation.orchestration import build_dataset


@pytest.fixture
def matrix_path() -> str:
    """Path to test matrix."""
    return "/data/projects/graph-cg/data/raw/SpectralData/45x15-displacements/stiffness/subdomain_1_Kaa.txt"


@pytest.fixture
def solutions_glob() -> str:
    """Glob pattern for solution archive."""
    return "/data/projects/graph-cg/data/raw/SpectralData/45x15-displacements/UaVectorsFromSpectral/ua_vector from_spectral_no_realization_*.txt"


def verify_dataset_consistency(dataset_dir: Path) -> tuple[bool, float]:
    """Verify that A @ x == b for all samples in dataset.

    Args:
        dataset_dir: Path to dataset directory.

    Returns:
        Tuple of (is_consistent, max_relative_error).
    """
    data = np.load(dataset_dir / "normalized.npz")
    A = data["matrix"]
    X_rhs = data["rhs"]
    Y_sol = data["solutions"]

    max_error = 0.0
    for i in range(len(X_rhs)):
        b_computed = A @ Y_sol[i]
        b_expected = X_rhs[i]
        residual = np.linalg.norm(b_computed - b_expected)
        norm_b = np.linalg.norm(b_expected)
        relative_error = residual / norm_b if norm_b > 0 else residual
        max_error = max(max_error, relative_error)

    return max_error < 1e-10, max_error


def test_solution_archive_consistency(
    tmp_path: Path, matrix_path: str, solutions_glob: str
) -> None:
    """Test that solution_archive strategy maintains A @ x = b."""
    dataset_dir = build_dataset(
        matrix_path=matrix_path,
        dataset_dir=str(tmp_path / "solution-archive"),
        counts={"solution_archive": 10},
        normalize="matrix",
        shuffle=False,
        seed=42,
        strategy_overrides={
            "solution_archive": {
                "solutions_glob": solutions_glob,
                "shuffle": False,
            }
        },
    )

    is_consistent, max_error = verify_dataset_consistency(Path(dataset_dir))
    assert is_consistent, f"solution_archive: A @ x != b, max error = {max_error:.2e}"


def test_eigenvector_forward_consistency(tmp_path: Path, matrix_path: str) -> None:
    """Test that eigenvector_forward strategy maintains A @ x = b."""
    dataset_dir = build_dataset(
        matrix_path=matrix_path,
        dataset_dir=str(tmp_path / "eigenvector-forward"),
        counts={"eigenvector_forward": 10},
        normalize="matrix",
        shuffle=False,
        seed=42,
    )

    is_consistent, max_error = verify_dataset_consistency(Path(dataset_dir))
    assert is_consistent, (
        f"eigenvector_forward: A @ x != b, max error = {max_error:.2e}"
    )


def test_eigenvector_inverse_consistency(tmp_path: Path, matrix_path: str) -> None:
    """Test that eigenvector_inverse strategy maintains A @ x = b."""
    dataset_dir = build_dataset(
        matrix_path=matrix_path,
        dataset_dir=str(tmp_path / "eigenvector-inverse"),
        counts={"eigenvector_inverse": 10},
        normalize="matrix",
        shuffle=False,
        seed=42,
    )

    is_consistent, max_error = verify_dataset_consistency(Path(dataset_dir))
    assert is_consistent, (
        f"eigenvector_inverse: A @ x != b, max error = {max_error:.2e}"
    )


def test_no_normalization_consistency(
    tmp_path: Path, matrix_path: str, solutions_glob: str
) -> None:
    """Test that consistency holds with normalize='none'."""
    dataset_dir = build_dataset(
        matrix_path=matrix_path,
        dataset_dir=str(tmp_path / "no-normalization"),
        counts={"solution_archive": 5},
        normalize="none",
        shuffle=False,
        seed=42,
        strategy_overrides={
            "solution_archive": {
                "solutions_glob": solutions_glob,
                "shuffle": False,
            }
        },
    )

    is_consistent, max_error = verify_dataset_consistency(Path(dataset_dir))
    assert is_consistent, f"normalize='none': A @ x != b, max error = {max_error:.2e}"


def test_mixed_strategies_consistency(
    tmp_path: Path, matrix_path: str, solutions_glob: str
) -> None:
    """Test that mixing multiple strategies maintains A @ x = b for all samples.

    This verifies that the orchestration correctly handles data from different
    strategies without double-normalization or inconsistencies.
    """
    dataset_dir = build_dataset(
        matrix_path=matrix_path,
        dataset_dir=str(tmp_path / "mixed-strategies"),
        counts={
            "solution_archive": 5,
            "eigenvector_forward": 5,
            "eigenvector_inverse": 5,
        },
        normalize="matrix",
        shuffle=True,
        seed=42,
        strategy_overrides={
            "solution_archive": {
                "solutions_glob": solutions_glob,
                "shuffle": False,
            }
        },
    )

    is_consistent, max_error = verify_dataset_consistency(Path(dataset_dir))
    assert is_consistent, f"mixed strategies: A @ x != b, max error = {max_error:.2e}"

    # Verify we got the expected total count
    data = np.load(Path(dataset_dir) / "normalized.npz")
    assert len(data["rhs"]) == 15, "Expected 15 total samples (5 + 5 + 5)"


def test_mixed_archive_and_synthetic_strategies(
    tmp_path: Path, matrix_path: str, solutions_glob: str
) -> None:
    """Test mixing archive strategies with synthetic strategies.

    This is a key use case: combining real solution archives with synthetic
    data generation strategies.
    """
    dataset_dir = build_dataset(
        matrix_path=matrix_path,
        dataset_dir=str(tmp_path / "archive-and-synthetic"),
        counts={
            "solution_archive": 10,
            "eigenvector_forward": 10,
        },
        normalize="matrix",
        shuffle=True,
        seed=42,
        strategy_overrides={
            "solution_archive": {
                "solutions_glob": solutions_glob,
                "shuffle": False,
            }
        },
    )

    is_consistent, max_error = verify_dataset_consistency(Path(dataset_dir))
    assert is_consistent, f"archive+synthetic: A @ x != b, max error = {max_error:.2e}"

    # Verify total count
    data = np.load(Path(dataset_dir) / "normalized.npz")
    assert len(data["rhs"]) == 20, "Expected 20 total samples (10 + 10)"


def test_large_mixed_dataset_consistency(
    tmp_path: Path, matrix_path: str, solutions_glob: str
) -> None:
    """Test a larger mixed dataset to ensure consistency at scale."""
    dataset_dir = build_dataset(
        matrix_path=matrix_path,
        dataset_dir=str(tmp_path / "large-mixed"),
        counts={
            "solution_archive": 50,
            "eigenvector_forward": 30,
            "eigenvector_inverse": 20,
        },
        normalize="matrix",
        shuffle=True,
        seed=42,
        strategy_overrides={
            "solution_archive": {
                "solutions_glob": solutions_glob,
                "shuffle": True,
                "seed": 42,
            }
        },
    )

    is_consistent, max_error = verify_dataset_consistency(Path(dataset_dir))
    assert is_consistent, (
        f"large mixed dataset: A @ x != b, max error = {max_error:.2e}"
    )

    # Verify total count
    data = np.load(Path(dataset_dir) / "normalized.npz")
    assert len(data["rhs"]) == 100, "Expected 100 total samples (50 + 30 + 20)"


def test_mixed_strategies_with_shuffling(
    tmp_path: Path, matrix_path: str, solutions_glob: str
) -> None:
    """Test that shuffling mixed strategies doesn't break consistency.

    Shuffling should preserve A @ x = b for each individual sample.
    """
    dataset_dir = build_dataset(
        matrix_path=matrix_path,
        dataset_dir=str(tmp_path / "shuffled-mixed"),
        counts={
            "solution_archive": 8,
            "eigenvector_forward": 7,
        },
        normalize="matrix",
        shuffle=True,
        seed=123,
        strategy_overrides={
            "solution_archive": {
                "solutions_glob": solutions_glob,
                "shuffle": True,
                "seed": 456,
            }
        },
    )

    is_consistent, max_error = verify_dataset_consistency(Path(dataset_dir))
    assert is_consistent, f"shuffled mixed: A @ x != b, max error = {max_error:.2e}"


def test_diagonal_normalization_preserves_symmetry_with_mixed_strategies(
    tmp_path: Path, matrix_path: str, solutions_glob: str
) -> None:
    """Test that diagonal normalization preserves matrix symmetry with mixed strategies.

    Diagonal normalization uses symmetric scaling D^(-1/2) A D^(-1/2) which should
    preserve matrix symmetry, unlike Jacobi preconditioning.
    """
    dataset_dir = build_dataset(
        matrix_path=matrix_path,
        dataset_dir=str(tmp_path / "diagonal-symmetric"),
        counts={
            "solution_archive": 5,
            "eigenvector_forward": 5,
            "eigenvector_inverse": 5,
        },
        normalize="diagonal",
        shuffle=False,
        seed=42,
        strategy_overrides={
            "solution_archive": {
                "solutions_glob": solutions_glob,
                "shuffle": False,
            }
        },
    )

    # Verify consistency
    is_consistent, max_error = verify_dataset_consistency(Path(dataset_dir))
    assert is_consistent, (
        f"diagonal normalization: A @ x != b, max error = {max_error:.2e}"
    )

    # Verify matrix symmetry is preserved
    data = np.load(Path(dataset_dir) / "normalized.npz")
    A = data["matrix"]
    max_asymmetry = np.max(np.abs(A - A.T))
    assert max_asymmetry < 1e-10, (
        f"Diagonal normalization did not preserve symmetry: max asymmetry = {max_asymmetry:.2e}"
    )

    # Verify diagonal is normalized to 1
    diag = np.diag(A)
    max_diag_error = np.max(np.abs(diag - 1.0))
    assert max_diag_error < 1e-10, (
        f"Diagonal not normalized to 1: max error = {max_diag_error:.2e}"
    )
