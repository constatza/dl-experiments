"""Linear system loading, normalization, and diagnostics for the comparison workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import numpy as np
from loguru import logger

from neuralls.composition.comparison.models import ComparisonPaths, LinearSystem
from neuralls.domain.linalg import compute_condition_number
from neuralls.domain.normalization import IScale, create_scale_from_config
from neuralls.domain.solver.utils.validation import validate_matrix, validate_rhs_vector
from neuralls.platform.storage.comparison import load_system_arrays


def _normalize_linear_system(
    matrix: np.ndarray,
    rhs: np.ndarray,
    normalize_system: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply comparison-time normalization to matrix and RHS.

    Args:
        matrix: System matrix A.
        rhs: Right-hand side vector b.
        normalize_system: Scaling mode — one of ``"none"``, ``"matrix"``,
            ``"rhs"``, ``"both"``, ``"diagonal"``, ``"spectral"``.

    Returns:
        Scaled (matrix, rhs) pair.

    Raises:
        ValueError: If normalize_system is not a recognised value.
        TypeError: If create_scale_from_config returns an unexpected type.
    """
    if normalize_system == "none":
        return matrix, rhs
    if normalize_system == "rhs":
        rhs_norm = float(np.linalg.norm(rhs))
        if rhs_norm == 0.0:
            return matrix, rhs
        return matrix, rhs / rhs_norm
    if normalize_system == "both":
        matrix, rhs = _normalize_linear_system(matrix, rhs, "matrix")
        return _normalize_linear_system(matrix, rhs, "rhs")

    strategy = normalize_system
    if strategy not in {"matrix", "diagonal", "spectral"}:
        raise ValueError(
            f"Unsupported normalize_system value: {normalize_system!r}. "
            "Expected one of none, matrix, rhs, both, diagonal, spectral."
        )

    rhs_samples = rhs.reshape(1, -1) if strategy == "spectral" else None
    scale_strategy = cast(Literal["none", "matrix", "spectral", "diagonal"], strategy)
    scale = create_scale_from_config(scale_strategy, matrix, rhs_samples=rhs_samples)
    if scale is None:
        return matrix, rhs
    if isinstance(scale, list):
        if not scale:
            return matrix, rhs
        scale = scale[0]
    if not isinstance(scale, IScale):
        raise TypeError(f"Expected IScale, got {type(scale).__name__}")
    return scale.scale_matrix(matrix), scale.scale_rhs(rhs)


def _load_linear_system(
    paths: ComparisonPaths,
    *,
    rhs_index: int,
    matrix_index: int,
    normalize_system: str,
) -> LinearSystem:
    """Load and validate a linear system from disk.

    Args:
        paths: Resolved comparison paths with matrix and rhs file locations.
        rhs_index: Row index to select from a multi-RHS file; -1 selects row 0.
        matrix_index: Sample index to select from a multi-matrix dataset directory.
        normalize_system: Scaling mode — one of ``"none"``, ``"matrix"``,
            ``"rhs"``, ``"both"``, ``"diagonal"``, ``"spectral"``.

    Returns:
        Validated and scaled LinearSystem.

    Raises:
        ValueError: If validation fails (wrong shape, NaN values, incompatible dimensions).
        FileNotFoundError: If matrix or rhs files don't exist.
    """
    A, b = load_system_arrays(
        paths.matrix, paths.rhs, rhs_index=rhs_index, matrix_index=matrix_index
    )
    A, b = _normalize_linear_system(A, b, normalize_system)
    validate_matrix(A)
    validate_rhs_vector(b, A)
    return LinearSystem(matrix=A, rhs=b)


def _log_matrix_condition_number(
    matrix: np.ndarray,
    *,
    matrix_path: Path,
    display_name: str | None,
) -> None:
    """Log the matrix condition number once per comparison.

    Args:
        matrix: System matrix to evaluate.
        matrix_path: Source path (used for log labels).
        display_name: Optional human-readable comparison label.
    """
    label = display_name or matrix_path.stem or matrix_path.name
    try:
        value = float(compute_condition_number(matrix))
    except np.linalg.LinAlgError as exc:
        logger.warning(
            f"Matrix condition number unavailable: comparison={label} "
            f"matrix={matrix_path.name} error={exc}"
        )
        return
    logger.info(
        f"Matrix condition number: comparison={label} matrix={matrix_path.name} value={value:.4e}"
    )
