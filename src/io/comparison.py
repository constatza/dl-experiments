"""Comparison I/O helpers for solver configuration and system selection."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import tomllib
from loguru import logger
import numpy as np

from ..configuration.solver import (
    GeneralSolverParams,
    SolverSpec,
    parse_solver_config,
)
from .base import load_npz_entry as load_npz_entry_raw


def load_solver_config(config_path: str | Path | None) -> tuple[GeneralSolverParams, list[SolverSpec]]:
    """Read solver config TOML from disk."""
    path = (
        Path(config_path)
        if config_path is not None
        else Path(__file__).resolve().parents[2] / "solver-configs" / "default.toml"
    )
    with path.open("rb") as fh:
        cfg = tomllib.load(fh)
    return parse_solver_config(cfg)


def resolve_system_paths(
    *,
    matrix_path: Optional[Path],
    rhs_path: Optional[Path],
    config_matrix: Optional[Path],
    config_rhs: Optional[Path],
) -> tuple[Path, Path]:
    """Choose matrix/RHS paths with explicit inputs (no implicit filenames)."""
    matrix = matrix_path or config_matrix
    rhs = rhs_path or config_rhs or matrix

    if matrix is None or rhs is None:
        raise ValueError("Matrix/RHS paths must be provided via solver config or caller.")

    return Path(matrix), Path(rhs)


def load_npz_entry(npz_path: str | Path, key: str) -> np.ndarray:
    """Load a single entry from an NPZ file, warning when multiple are present."""
    npz_path = Path(npz_path)
    arr = load_npz_entry_raw(npz_path, key)
    if key == "matrix" and arr.ndim > 2:
        logger.warning(f"{npz_path} contains multiple matrices; using the first slice.")
        arr = arr[0]
    if key == "rhs":
        if arr.ndim > 1 and arr.shape[0] > 1:
            logger.warning(f"{npz_path} contains multiple RHS entries; using the first.")
        if arr.ndim > 1:
            arr = arr[0]
        arr = arr.reshape(-1)

    return arr


def load_system_arrays(matrix_path: Path, rhs_path: Path | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """Load matrix and RHS from explicit file paths (npz/npy/txt).

    - npz: use load_npz_entry for "matrix" and "rhs" keys.
    - npy/txt: load directly; RHS is flattened and warns on extra rows.
    """
    matrix_path = Path(matrix_path)
    rhs_path = Path(rhs_path) if rhs_path is not None else matrix_path

    def _load_array(path: Path, key: str) -> np.ndarray:
        match path.suffix:
            case ".npz":
                return load_npz_entry(path, key)
            case ".npy":
                return np.load(path).astype(np.float64, copy=False)
            case _:
                return np.loadtxt(path, dtype=np.float64)

    A = _load_array(matrix_path, "matrix")
    b_raw = _load_array(rhs_path, "rhs")
    if b_raw.ndim > 1 and b_raw.shape[0] > 1:
        logger.warning(f"{rhs_path} contains multiple RHS entries; using the first row.")
        b_raw = b_raw[0]
    b = b_raw.reshape(-1) if b_raw is not None else None

    return A, b
