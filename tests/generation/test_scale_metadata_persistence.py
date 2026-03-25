"""Tests for scale metadata persistence in data generation."""

import numpy as np
import pytest
from pathlib import Path

from neuralls.generation.orchestration import build_dataset
from neuralls.io.dataset_storage import (
    load_dataset_manifest,
    load_dense_training_arrays,
    resolve_dataset_paths,
)
from neuralls.normalization import load_scale_from_metadata, MatrixScale
from dlkit.tools.io.sparse import open_sparse_pack


@pytest.fixture
def temp_matrix_file(tmp_path: Path):
    """Create a temporary matrix file."""
    matrix_file = tmp_path / "test_matrix.txt"
    # Create a simple 5x5 symmetric positive definite matrix
    matrix = np.array(
        [
            [4.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 4.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 4.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 4.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 4.0],
        ]
    )
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

    # Load manifest and dense arrays
    manifest = load_dataset_manifest(output_dir)
    rhs, _ = load_dense_training_arrays(output_dir)

    # Check that metadata fields are present
    assert "normalization" in manifest
    scale_metadata = manifest["normalization"]["scale"]
    assert "spectral_radius_bound" in scale_metadata
    assert "dimension_scale" in scale_metadata

    # Check metadata values
    assert str(manifest["normalization"]["type"]) == "matrix"
    assert isinstance(float(scale_metadata["spectral_radius_bound"]), float)
    assert float(scale_metadata["spectral_radius_bound"]) > 0
    assert isinstance(float(scale_metadata["dimension_scale"]), float)
    assert float(scale_metadata["dimension_scale"]) > 0

    # Test scale reconstruction
    metadata = {
        "spectral_radius_bound": float(scale_metadata["spectral_radius_bound"]),
        "dimension_scale": float(scale_metadata["dimension_scale"]),
    }
    scale = load_scale_from_metadata("matrix", metadata)

    assert isinstance(scale, MatrixScale)
    assert scale.spectral_radius_bound == metadata["spectral_radius_bound"]
    assert scale.dimension_scale == metadata["dimension_scale"]
    pack = open_sparse_pack(resolve_dataset_paths(output_dir).matrix_pack_dir)
    assert np.isclose(pack.value_scale, scale.composite_scale)


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

    manifest = load_dataset_manifest(output_dir)

    # Check that normalize_type is present
    assert "normalization" in manifest
    assert str(manifest["normalization"]["type"]) == "none"

    # Scale parameters should not be present for "none" normalization
    scale_metadata = manifest["normalization"]["scale"]
    assert "spectral_radius_bound" not in scale_metadata
    assert "dimension_scale" not in scale_metadata


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

    manifest = load_dataset_manifest(output_dir)
    rhs, _ = load_dense_training_arrays(output_dir)

    # Reconstruct scale from metadata
    metadata = {
        "spectral_radius_bound": float(manifest["normalization"]["scale"]["spectral_radius_bound"]),
        "dimension_scale": float(manifest["normalization"]["scale"]["dimension_scale"]),
    }
    scale = load_scale_from_metadata("matrix", metadata)
    assert isinstance(scale, MatrixScale)

    # Get normalized RHS from dataset
    normalized_rhs = rhs[0]

    # Denormalize
    denormalized_rhs = scale.denormalize_rhs(normalized_rhs)

    # The denormalized RHS should be larger than normalized (for typical matrices)
    # since we multiply by composite_scale > 1
    assert scale.composite_scale > 1.0
    assert np.linalg.norm(denormalized_rhs) > np.linalg.norm(normalized_rhs)
