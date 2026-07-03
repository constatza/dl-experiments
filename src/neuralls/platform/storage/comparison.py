"""Comparison I/O helpers for solver configuration and system selection."""

from __future__ import annotations

from pathlib import Path

from loguru import logger
import numpy as np
import zarr

from .datasets import (
    load_dense_training_arrays,
    load_matrix_dense_sample,
    load_dataset_manifest,
)


def resolve_system_paths(
    *,
    matrix_path: Path | None,
    rhs_path: Path | None,
    config_matrix: Path | None,
    config_rhs: Path | None,
) -> tuple[Path, Path]:
    """Choose matrix/RHS paths with explicit inputs (no implicit filenames)."""
    matrix = matrix_path or config_matrix
    rhs = rhs_path or config_rhs

    if matrix is None or rhs is None:
        raise ValueError("Matrix/RHS paths must be provided via solver config or caller.")

    return Path(matrix), Path(rhs)


def load_system_extras(
    dataset_dir: Path,
    names: frozenset[str],
    sample_index: int,
) -> dict[str, np.ndarray]:
    """Load optional named side-channel arrays from a comparison dataset directory."""
    extras: dict[str, np.ndarray] = {}
    for name in names:
        zarr_path = dataset_dir / f"{name}.zarr"
        npy_path = dataset_dir / f"{name}.npy"
        if zarr_path.exists():
            arr = zarr.open_array(str(zarr_path), mode="r")
            extras[name] = np.asarray(arr[sample_index] if arr.ndim > 1 else arr[:])
            continue
        if npy_path.exists():
            arr = np.load(npy_path)
            extras[name] = np.asarray(arr[sample_index] if arr.ndim > 1 else arr)
    return extras


def _flatten_rhs_if_needed(arr: np.ndarray, source_path: Path) -> np.ndarray:
    """Handle 1D/column vectors vs multiple RHS entries."""
    # Treat (N, 1) as a single vector, not multiple problems
    is_column_vector = arr.ndim == 2 and arr.shape[1] == 1

    if arr.ndim > 1 and arr.shape[0] > 1 and not is_column_vector:
        logger.debug(
            f"{source_path} contains multiple RHS entries; rhs_sample_index will select one row."
        )

    # Only extract the first row if it's NOT a column vector
    if arr.ndim > 1 and is_column_vector:
        return arr.reshape(-1)

    return arr


def _select_rhs_sample(arr: np.ndarray, rhs_sample_index: int, source_path: Path) -> np.ndarray:
    """Select one RHS sample by index from 1D/2D RHS arrays."""
    is_column_vector = arr.ndim == 2 and arr.shape[1] == 1
    if arr.ndim == 1 or is_column_vector:
        return arr.reshape(-1)

    if rhs_sample_index == -1:
        return arr[0].reshape(-1)

    if rhs_sample_index < -1 or rhs_sample_index >= arr.shape[0]:
        raise IndexError(
            f"rhs_sample_index={rhs_sample_index} out of range for {source_path} "
            f"(available rows: 0..{arr.shape[0] - 1}, or -1 for first row)."
        )
    return arr[rhs_sample_index].reshape(-1)


def load_system_arrays(
    matrix_path: Path,
    rhs_path: Path | None = None,
    rhs_sample_index: int = 0,
    matrix_index: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Load matrix and RHS from explicit file paths (dataset dir, npy, txt)."""
    matrix_path = Path(matrix_path)
    if rhs_path is None:
        raise ValueError("rhs_path is required for loading system arrays.")
    rhs_path = Path(rhs_path)

    def _load_array(path: Path, key: str) -> np.ndarray:
        if path.is_dir():
            load_dataset_manifest(path)
            if key == "matrix":
                return load_matrix_dense_sample(path, sample_index=matrix_index)
            rhs, _ = load_dense_training_arrays(path)
            return np.asarray(rhs, dtype=np.float64)
        match path.suffix:
            case ".npy":
                return np.load(path).astype(np.float64, copy=False)
            case ".npz":
                raise ValueError(
                    f"NPZ inputs are no longer supported ({path}). Use dataset directories or .npy files."
                )
            case _:
                return np.loadtxt(path, dtype=np.float64)

    A = _load_array(matrix_path, "matrix")
    b_raw = _load_array(rhs_path, "rhs")
    b = _flatten_rhs_if_needed(b_raw, rhs_path)
    b = _select_rhs_sample(b, rhs_sample_index, rhs_path)

    return A, b
