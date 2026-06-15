"""Tests for load_training_arrays() — specifically the parameters_zarr detection added here."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from neuralls.platform.storage.training_artifacts import load_training_arrays
from neuralls.shared.constants import PARAMETERS_ZARR_PREFIX


@pytest.fixture
def minimal_dataset(tmp_path: Path) -> Path:
    """Dataset with rhs.zarr, solutions.zarr, and matrix.zarr but no parameters."""
    n, samples = 4, 6
    rng = np.random.default_rng(0)
    zarr.open_array(str(tmp_path / "rhs.zarr"), mode="w", shape=(samples, n), dtype="float64")[
        :
    ] = rng.standard_normal((samples, n))
    zarr.open_array(
        str(tmp_path / "solutions.zarr"), mode="w", shape=(samples, n), dtype="float64"
    )[:] = rng.standard_normal((samples, n))
    zarr.open_array(str(tmp_path / "matrix.zarr"), mode="w", shape=(1, n, n), dtype="float64")[
        0
    ] = np.eye(n)
    return tmp_path


@pytest.fixture
def two_parameters_zarrs(minimal_dataset: Path) -> Path:
    """Adds parameters_0.zarr and parameters_1.zarr to the minimal dataset."""
    samples, p = 6, 3
    rng = np.random.default_rng(1)
    for i in range(2):
        arr = rng.standard_normal((samples, p))
        zarr.open_array(
            str(minimal_dataset / f"{PARAMETERS_ZARR_PREFIX}{i}.zarr"),
            mode="w",
            shape=arr.shape,
            dtype="float64",
        )[:] = arr
    return minimal_dataset


def test_load_training_arrays_detects_parameters_zarr(two_parameters_zarrs: Path) -> None:
    arrays = load_training_arrays(two_parameters_zarrs)

    assert len(arrays.parameters_zarr) == 2
    assert arrays.parameters_zarr[0].stem == f"{PARAMETERS_ZARR_PREFIX}0"
    assert arrays.parameters_zarr[1].stem == f"{PARAMETERS_ZARR_PREFIX}1"


def test_load_training_arrays_returns_empty_parameters_when_absent(
    minimal_dataset: Path,
) -> None:
    arrays = load_training_arrays(minimal_dataset)
    assert arrays.parameters_zarr == ()


def test_load_training_arrays_parameters_sorted_by_index(two_parameters_zarrs: Path) -> None:
    """parameters_zarr must be ordered by index regardless of filesystem traversal order."""
    arrays = load_training_arrays(two_parameters_zarrs)
    stems = [p.stem for p in arrays.parameters_zarr]
    assert stems == sorted(stems)
