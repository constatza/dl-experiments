"""Tests for scale metadata persistence in data generation."""

import numpy as np
import pytest
from pathlib import Path

from neuralls.generation.orchestration import build_dataset
from neuralls.normalization import load_scale_from_metadata, MatrixScale


@pytest.fixture
def temp_matrix_file(tmp_path: Path):
    """Create a temporary matrix file."""
    matrix_file = tmp_path / "test_matrix.txt"
    # Create a simple 5x5 symmetric positive definite matrix
    matrix = np.array([
        [4.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 4.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 4.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 4.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 4.0]
    ])
    np.savetxt(matrix_file, matrix)
    return matrix_file


def test_scale_metadata_saved_with_matrix_normalization(temp_matrix_file: Path, tmp_path: Path):
    """Test that scale metadata is saved with matrix normalization."""
    output_dir = tmp_path / "output"

    # Build dataset with matrix normalization
    build_dataset(
        matrix_path=str(temp_matrix_file),
        dataset_dir=str(output_dir),
        rhs_path=None,
        counts={"neutral_ones": 1},
        seed=42,
        shuffle=False,
        normalize="matrix",
        strategy_overrides={"neutral_ones": {"samples": 1}},
    )

    # Load the generated dataset
    dataset_file = output_dir / "normalized.npz"
    assert dataset_file.exists()

    data = np.load(dataset_file)

    # Check that metadata fields are present
    assert "normalize_type" in data.files
    assert "spectral_radius_bound" in data.files
    assert "dimension_scale" in data.files

    # Check metadata values
    assert str(data["normalize_type"]) == "matrix"
    assert isinstance(float(data["spectral_radius_bound"]), float)
    assert float(data["spectral_radius_bound"]) > 0
    assert isinstance(float(data["dimension_scale"]), float)
    assert float(data["dimension_scale"]) > 0

    # Test scale reconstruction
    metadata = {
        "spectral_radius_bound": float(data["spectral_radius_bound"]),
        "dimension_scale": float(data["dimension_scale"]),
    }
    scale = load_scale_from_metadata("matrix", metadata)

    assert isinstance(scale, MatrixScale)
    assert scale.spectral_radius_bound == metadata["spectral_radius_bound"]
    assert scale.dimension_scale == metadata["dimension_scale"]


def test_scale_metadata_not_saved_with_none_normalization(temp_matrix_file: Path, tmp_path: Path):
    """Test that scale metadata is not saved when normalization is 'none'."""
    output_dir = tmp_path / "output_none"

    # Build dataset without normalization
    build_dataset(
        matrix_path=str(temp_matrix_file),
        dataset_dir=str(output_dir),
        rhs_path=None,
        counts={"neutral_ones": 1},
        seed=42,
        shuffle=False,
        normalize="none",
        strategy_overrides={"neutral_ones": {"samples": 1}},
    )

    # Load the generated dataset
    dataset_file = output_dir / "normalized.npz"
    assert dataset_file.exists()

    data = np.load(dataset_file)

    # Check that normalize_type is present
    assert "normalize_type" in data.files
    assert str(data["normalize_type"]) == "none"

    # Scale parameters should not be present for "none" normalization
    assert "spectral_radius_bound" not in data.files
    assert "dimension_scale" not in data.files


def test_denormalization_round_trip(temp_matrix_file: Path, tmp_path: Path):
    """Test that we can denormalize using saved metadata."""
    output_dir = tmp_path / "output_roundtrip"

    # Build dataset with matrix normalization
    build_dataset(
        matrix_path=str(temp_matrix_file),
        dataset_dir=str(output_dir),
        rhs_path=None,
        counts={"neutral_ones": 1},
        seed=42,
        shuffle=False,
        normalize="matrix",
        strategy_overrides={"neutral_ones": {"samples": 1}},
    )

    # Load the generated dataset
    dataset_file = output_dir / "normalized.npz"
    data = np.load(dataset_file)

    # Reconstruct scale from metadata
    metadata = {
        "spectral_radius_bound": float(data["spectral_radius_bound"]),
        "dimension_scale": float(data["dimension_scale"]),
    }
    scale = load_scale_from_metadata("matrix", metadata)

    # Get normalized RHS from dataset
    normalized_rhs = data["rhs"][0]

    # Denormalize
    denormalized_rhs = scale.denormalize_rhs(normalized_rhs)

    # The denormalized RHS should be larger than normalized (for typical matrices)
    # since we multiply by composite_scale > 1
    assert scale.composite_scale > 1.0
    assert np.linalg.norm(denormalized_rhs) > np.linalg.norm(normalized_rhs)
