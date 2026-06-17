"""Tests for generation domain payloads."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neuralls.domain.generation.payloads import GeneratedDatasetPayload


@pytest.fixture
def zarr_path(tmp_path: Path) -> Path:
    """Create a temporary matrix zarr directory."""
    p = tmp_path / "matrix_zarr"
    p.mkdir()
    return p


def test_payload_has_artifact_path_not_arrays(zarr_path: Path) -> None:
    """GeneratedDatasetPayload should carry matrix_artifact_path, not sparse arrays."""
    payload = GeneratedDatasetPayload(
        rhs=np.zeros((2, 4)),
        solutions=np.zeros((2, 4)),
        matrix_artifact_path=zarr_path,
        matrix_size=(4, 4),
        normalization_type="matrix",
        matrix_norm=1.0,
        matrix_norm_type="spectral",
    )
    assert payload.matrix_artifact_path == zarr_path
    assert not hasattr(payload, "indices")
    assert not hasattr(payload, "values")
    assert not hasattr(payload, "nnz_ptr")
