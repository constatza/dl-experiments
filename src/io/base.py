"""Base I/O helpers (pure functions, no reshaping or warnings)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_npz_entry(npz_path: str | Path, key: str) -> np.ndarray:
    """Load a single entry from an NPZ without reshaping or warnings."""
    npz_path = Path(npz_path)
    data = np.load(npz_path)
    if key not in data:
        raise KeyError(f"{npz_path} missing key '{key}'")
    return np.asarray(data[key], dtype=np.float64, copy=False)


def load_matrix(matrix_path: str | Path) -> np.ndarray:
    """Load matrix from file (txt, npy, or npz format).

    Args:
        matrix_path: Path to matrix file

    Returns:
        Matrix as numpy float64 array

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    matrix_path = Path(matrix_path)

    if not matrix_path.exists():
        raise FileNotFoundError(f"Matrix file not found: {matrix_path}")

    # NPZ format
    if matrix_path.suffix == ".npz":
        return load_npz_entry(matrix_path, "matrix")

    # NPY binary format
    if matrix_path.suffix == ".npy":
        return np.load(matrix_path).astype(np.float64, copy=False)

    # Text format (default)
    return np.loadtxt(matrix_path, dtype=np.float64)
