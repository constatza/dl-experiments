"""Pure dataset payload types for generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from neuralls.shared.types import LayoutType, ScaleMetadata


@dataclass(frozen=True)
class GeneratedDatasetPayload:
    """Dataset payload produced by generation domain.

    The matrix artifact is written to disk by the accumulator before this
    payload is created. Only dense arrays, the staged artifact path, and
    normalization metadata are carried in memory.

    Attributes:
        rhs: RHS vectors, shape (N, n).
        solutions: Solution vectors, shape (N, n).
        matrix_artifact_path: Path to the finished staged matrix artifact.
        matrix_size: Matrix dimensions (rows, cols).
        normalization_type: Normalization strategy applied.
        matrix_norm: Computed matrix norm value.
        matrix_norm_type: Type of norm used.
        matrix_value_scale: Multiplicative scale applied to matrix values.
        scale_metadata: Optional scale parameters dictionary.
        num_bindings: Number of matrix bindings processed.
    """

    rhs: np.ndarray
    solutions: np.ndarray
    matrix_artifact_path: Path
    matrix_size: tuple[int, int]
    normalization_type: str
    matrix_norm: float
    matrix_norm_type: str
    matrix_value_scale: float = 1.0
    scale_metadata: ScaleMetadata | None = None
    num_bindings: int = 0
    parameters_arrays: tuple[np.ndarray, ...] = ()
    """One array per entry in ``parameters_paths``; shape ``(N, param_dim)``."""
    layout: LayoutType = LayoutType.MANY_MATRICES
    """Physical matrix layout: one row per sample or one shared row for all samples."""
    rhs_kind_codes: np.ndarray | None = None
    """Compact uint8 semantic codes aligned with persisted RHS rows."""
    target_kind_codes: np.ndarray | None = None
    """Compact uint8 semantic codes aligned with persisted target rows."""
    matrix_sample_index: np.ndarray | None = None
    """Per-row physical matrix sample binding aligned with persisted RHS rows."""
