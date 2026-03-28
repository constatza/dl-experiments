"""Pure dataset payload types for generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neuralls.shared.types import ScaleMetadata


@dataclass(frozen=True)
class GeneratedDatasetPayload:
    """In-memory dataset payload produced by the generation domain."""

    rhs: np.ndarray
    solutions: np.ndarray
    indices: np.ndarray
    values: np.ndarray
    nnz_ptr: np.ndarray
    matrix_size: tuple[int, int]
    normalization_type: str
    matrix_norm: float
    matrix_norm_type: str
    matrix_value_scale: float = 1.0
    scale_metadata: ScaleMetadata | None = None
    num_bindings: int = 0


@dataclass
class SparsePackAccumulator:
    """Incrementally accumulate COO sparse payload arrays."""

    indices_parts: list[np.ndarray]
    values_parts: list[np.ndarray]
    nnz_ptr_values: list[int]
    matrix_size: tuple[int, int] | None

    def __init__(self) -> None:
        self.indices_parts = []
        self.values_parts = []
        self.nnz_ptr_values = [0]
        self.matrix_size = None

    def append_dense_matrix(self, matrix: np.ndarray, repeats: int) -> None:
        """Append one dense matrix converted to COO, optionally repeated."""
        rows, cols = np.nonzero(matrix)
        values = np.asarray(matrix[rows, cols], dtype=np.float64)
        indices = np.vstack((rows, cols)).astype(np.int64, copy=False)
        self.append_sparse_components(
            indices=indices,
            values=values,
            size=(int(matrix.shape[0]), int(matrix.shape[1])),
            repeats=repeats,
        )

    def append_sparse_components(
        self,
        *,
        indices: np.ndarray,
        values: np.ndarray,
        size: tuple[int, int],
        repeats: int,
    ) -> None:
        """Append COO components for one matrix sample with optional broadcasting."""
        if repeats < 1:
            raise ValueError(f"repeats must be >= 1, got {repeats}")
        if self.matrix_size is None:
            self.matrix_size = size
        elif self.matrix_size != size:
            raise ValueError(
                f"Matrix size mismatch in sparse accumulator: {size} != {self.matrix_size}"
            )

        idx = np.asarray(indices, dtype=np.int64)
        vals = np.asarray(values, dtype=np.float64)
        if idx.ndim != 2 or idx.shape[0] != 2:
            raise ValueError(f"indices must have shape (2, nnz), got {idx.shape}")
        if vals.ndim != 1:
            raise ValueError(f"values must be 1D, got {vals.shape}")
        if idx.shape[1] != vals.size:
            raise ValueError(f"indices nnz ({idx.shape[1]}) does not match values ({vals.size})")

        nnz = int(vals.size)
        if nnz == 0:
            last = self.nnz_ptr_values[-1]
            for _ in range(repeats):
                self.nnz_ptr_values.append(last)
            return

        if repeats == 1:
            self.indices_parts.append(idx)
            self.values_parts.append(vals)
        else:
            self.indices_parts.append(np.tile(idx, (1, repeats)))
            self.values_parts.append(np.tile(vals, repeats))

        last = self.nnz_ptr_values[-1]
        for _ in range(repeats):
            last += nnz
            self.nnz_ptr_values.append(last)

    def build_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
        """Finalize accumulator into COO payload arrays."""
        if self.matrix_size is None:
            raise ValueError("SparsePackAccumulator is empty.")
        if self.values_parts:
            indices = np.concatenate(self.indices_parts, axis=1)
            values = np.concatenate(self.values_parts, axis=0)
        else:
            indices = np.zeros((2, 0), dtype=np.int64)
            values = np.zeros((0,), dtype=np.float64)
        nnz_ptr = np.asarray(self.nnz_ptr_values, dtype=np.int64)
        return indices, values, nnz_ptr, self.matrix_size
