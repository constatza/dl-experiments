"""Tests for generation domain payloads."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neuralls.domain.generation.payloads import GeneratedDatasetPayload


@pytest.fixture
def pack_path(tmp_path: Path) -> Path:
    """Create a temporary matrix pack directory."""
    p = tmp_path / "matrix_coo"
    p.mkdir()
    return p


def test_payload_has_pack_path_not_arrays(pack_path: Path) -> None:
    """GeneratedDatasetPayload should carry matrix_pack_path, not sparse arrays."""
    payload = GeneratedDatasetPayload(
        rhs=np.zeros((2, 4)),
        solutions=np.zeros((2, 4)),
        matrix_pack_path=pack_path,
        matrix_size=(4, 4),
        normalization_type="matrix",
        matrix_norm=1.0,
        matrix_norm_type="spectral",
    )
    assert payload.matrix_pack_path == pack_path
    assert not hasattr(payload, "indices")
    assert not hasattr(payload, "values")
    assert not hasattr(payload, "nnz_ptr")
