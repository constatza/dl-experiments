"""Tests for comparison dataset I/O helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from neuralls.platform.storage.comparison import load_system_extras


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    """Dataset directory with coordinates.zarr/ (3 samples, 2×3) and params.npy."""
    coords = np.arange(18.0, dtype=np.float64).reshape(3, 2, 3)
    zarr.save(str(tmp_path / "coordinates.zarr"), coords)
    np.save(tmp_path / "params.npy", np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    return tmp_path


def test_load_zarr_extra_by_name(dataset_dir: Path) -> None:
    """Load a zarr array extra by name, extracting specified sample."""
    result = load_system_extras(dataset_dir, frozenset({"coordinates"}), sample_index=1)
    assert "coordinates" in result
    assert result["coordinates"].shape == (2, 3)
    np.testing.assert_allclose(result["coordinates"], np.arange(6.0, 12.0).reshape(2, 3))


def test_load_npy_extra_by_name(dataset_dir: Path) -> None:
    """Load a npy file extra by name."""
    result = load_system_extras(dataset_dir, frozenset({"params"}), sample_index=0)
    assert "params" in result
    np.testing.assert_allclose(result["params"], [1.0, 2.0, 3.0, 4.0, 5.0])


def test_missing_extra_silently_absent(dataset_dir: Path) -> None:
    """Missing extras are not included in the result dict."""
    result = load_system_extras(dataset_dir, frozenset({"nonexistent"}), sample_index=0)
    assert "nonexistent" not in result


def test_empty_names_returns_empty_dict(dataset_dir: Path) -> None:
    """Empty names set returns empty dict."""
    result = load_system_extras(dataset_dir, frozenset(), sample_index=0)
    assert result == {}
