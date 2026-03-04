"""Base I/O helpers (pure functions, no reshaping or warnings)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_matrix(matrix_path: str | Path, delimiter: str | None = None) -> np.ndarray:
    """Load matrix from file (txt or npy format).

    Args:
        matrix_path: Path to matrix file
        delimiter: String used to separate values in text files.

    Returns:
        Matrix as numpy float64 array

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    matrix_path = Path(matrix_path)

    if not matrix_path.exists():
        raise FileNotFoundError(f"Matrix file not found: {matrix_path}")

    # NPY binary format
    if matrix_path.suffix == ".npy":
        return np.load(matrix_path).astype(np.float64, copy=False)

    if matrix_path.suffix == ".npz":
        raise ValueError(
            f"NPZ files are no longer supported ({matrix_path}). Use dataset directories or .npy files."
        )

    # Text format (default)
    return np.loadtxt(matrix_path, dtype=np.float64, delimiter=delimiter)
