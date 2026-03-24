"""Dataset storage helpers for split dense arrays + sparse COO matrix packs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from dlkit.tools.io.sparse import SparseFormat, open_sparse_pack, save_sparse_pack

from neuralls.constants import (
    DATASET_MANIFEST_FILENAME,
    MATRIX_COO_DIRNAME,
    RHS_ARRAY_FILENAME,
    SOLUTIONS_ARRAY_FILENAME,
)
from neuralls.generation.data_types import ScaleMetadata


@dataclass(frozen=True)
class DatasetPaths:
    """Resolved canonical dataset artifact paths."""

    root: Path
    manifest_path: Path
    rhs_path: Path
    solutions_path: Path
    matrix_pack_dir: Path


@dataclass
class SparsePackAccumulator:
    """Incrementally accumulate COO sparse payload arrays.

    This helper supports broadcasting one matrix sample across many data rows by
    duplicating sparse payload segments, not dense matrices.
    """

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
            raise ValueError(
                f"indices nnz ({idx.shape[1]}) does not match values ({vals.size})"
            )

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
            # Broadcast sparse payloads (indices/values) for repeated samples.
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


def resolve_dataset_paths(dataset_dir: str | Path) -> DatasetPaths:
    """Resolve canonical dataset artifact locations."""
    root = Path(dataset_dir)
    return DatasetPaths(
        root=root,
        manifest_path=root / DATASET_MANIFEST_FILENAME,
        rhs_path=root / RHS_ARRAY_FILENAME,
        solutions_path=root / SOLUTIONS_ARRAY_FILENAME,
        matrix_pack_dir=root / MATRIX_COO_DIRNAME,
    )


def _dense_to_coo_pack(
    matrix: np.ndarray,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """Convert dense matrix data to duplicated COO pack arrays."""
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")

    if matrix.ndim == 2:
        rows, cols = np.nonzero(matrix)
        values_single = matrix[rows, cols].astype(np.float64, copy=False)
        indices_single = np.vstack((rows, cols)).astype(np.int64, copy=False)
        nnz_single = int(values_single.size)
        size = (int(matrix.shape[0]), int(matrix.shape[1]))
        if nnz_single == 0:
            return (
                np.zeros((2, 0), dtype=np.int64),
                np.zeros((0,), dtype=np.float64),
                np.zeros((n_samples + 1,), dtype=np.int64),
                size,
            )
        indices = np.tile(indices_single, (1, n_samples))
        values = np.tile(values_single, n_samples)
        nnz_ptr = np.arange(
            0,
            nnz_single * n_samples + 1,
            nnz_single,
            dtype=np.int64,
        )
        return indices, values, nnz_ptr, size

    if matrix.ndim != 3:
        raise ValueError(f"matrix must be 2D or 3D, got shape {matrix.shape}")

    if matrix.shape[0] != n_samples:
        raise ValueError(
            f"3D matrix sample count ({matrix.shape[0]}) must equal n_samples ({n_samples})"
        )

    indices_parts: list[np.ndarray] = []
    values_parts: list[np.ndarray] = []
    nnz_ptr = np.zeros((n_samples + 1,), dtype=np.int64)
    for sample_idx in range(n_samples):
        sample = matrix[sample_idx]
        rows, cols = np.nonzero(sample)
        indices_parts.append(np.vstack((rows, cols)).astype(np.int64, copy=False))
        sample_values = sample[rows, cols].astype(np.float64, copy=False)
        values_parts.append(sample_values)
        nnz_ptr[sample_idx + 1] = nnz_ptr[sample_idx] + int(sample_values.size)

    if values_parts:
        indices = np.concatenate(indices_parts, axis=1) if indices_parts else np.zeros((2, 0), dtype=np.int64)
        values = np.concatenate(values_parts, axis=0)
    else:
        indices = np.zeros((2, 0), dtype=np.int64)
        values = np.zeros((0,), dtype=np.float64)

    size = (int(matrix.shape[1]), int(matrix.shape[2]))
    return indices, values, nnz_ptr, size


def save_dataset(
    *,
    dataset_dir: str | Path,
    rhs: np.ndarray,
    solutions: np.ndarray,
    matrix: np.ndarray,
    normalization_type: str,
    matrix_norm: float,
    matrix_norm_type: str,
    matrix_value_scale: float = 1.0,
    scale_metadata: ScaleMetadata | None = None,
) -> DatasetPaths:
    """Persist dataset as split dense arrays + COO sparse matrix pack."""
    paths = resolve_dataset_paths(dataset_dir)
    paths.root.mkdir(parents=True, exist_ok=True)

    rhs_array = np.asarray(rhs, dtype=np.float64)
    solutions_array = np.asarray(solutions, dtype=np.float64)
    if rhs_array.ndim != 2 or solutions_array.ndim != 2:
        raise ValueError(
            f"rhs/solutions must be 2D arrays, got rhs={rhs_array.shape}, solutions={solutions_array.shape}"
        )
    if rhs_array.shape != solutions_array.shape:
        raise ValueError(
            f"rhs and solutions must have identical shape, got {rhs_array.shape} and {solutions_array.shape}"
        )

    n_samples = int(rhs_array.shape[0])
    indices, values, nnz_ptr, size = _dense_to_coo_pack(
        np.asarray(matrix),
        n_samples=n_samples,
    )
    return save_dataset_from_sparse(
        dataset_dir=dataset_dir,
        rhs=rhs_array,
        solutions=solutions_array,
        indices=indices,
        values=values,
        nnz_ptr=nnz_ptr,
        size=size,
        normalization_type=normalization_type,
        matrix_norm=matrix_norm,
        matrix_norm_type=matrix_norm_type,
        matrix_value_scale=matrix_value_scale,
        scale_metadata=scale_metadata,
    )


def save_dataset_from_sparse(
    *,
    dataset_dir: str | Path,
    rhs: np.ndarray,
    solutions: np.ndarray,
    indices: np.ndarray,
    values: np.ndarray,
    nnz_ptr: np.ndarray,
    size: tuple[int, int],
    normalization_type: str,
    matrix_norm: float,
    matrix_norm_type: str,
    matrix_value_scale: float = 1.0,
    scale_metadata: ScaleMetadata | None = None,
) -> DatasetPaths:
    """Persist dataset using pre-built sparse COO payload arrays."""
    paths = resolve_dataset_paths(dataset_dir)
    paths.root.mkdir(parents=True, exist_ok=True)

    rhs_array = np.asarray(rhs, dtype=np.float64)
    solutions_array = np.asarray(solutions, dtype=np.float64)
    if rhs_array.ndim != 2 or solutions_array.ndim != 2:
        raise ValueError(
            f"rhs/solutions must be 2D arrays, got rhs={rhs_array.shape}, solutions={solutions_array.shape}"
        )
    if rhs_array.shape != solutions_array.shape:
        raise ValueError(
            f"rhs and solutions must have identical shape, got {rhs_array.shape} and {solutions_array.shape}"
        )
    expected_samples = int(rhs_array.shape[0])
    nnz_ptr_arr = np.asarray(nnz_ptr, dtype=np.int64)
    if nnz_ptr_arr.ndim != 1:
        raise ValueError(f"nnz_ptr must be 1D, got {nnz_ptr_arr.shape}")
    matrix_samples = int(nnz_ptr_arr.size - 1)
    if matrix_samples < 1:
        raise ValueError("nnz_ptr must encode at least one sparse matrix sample.")
    supports_runtime_broadcast = matrix_samples == 1 and expected_samples >= 1
    if not supports_runtime_broadcast and nnz_ptr_arr.size != expected_samples + 1:
        raise ValueError(
            "nnz_ptr length must be either samples+1 for per-row matrices "
            f"({expected_samples + 1}) or 2 for single-matrix runtime broadcast; "
            f"got {nnz_ptr_arr.shape}"
        )
    value_scale = float(matrix_value_scale)
    if not np.isfinite(value_scale) or value_scale <= 0.0:
        raise ValueError(
            f"matrix_value_scale must be finite and > 0, got {matrix_value_scale}"
        )

    np.save(paths.rhs_path, rhs_array)
    np.save(paths.solutions_path, solutions_array)
    save_sparse_pack(
        path=paths.matrix_pack_dir,
        indices=np.asarray(indices, dtype=np.int64),
        values=np.asarray(values, dtype=np.float64),
        nnz_ptr=nnz_ptr_arr,
        size=size,
        format=SparseFormat.COO,
        dtype=np.dtype(np.float64),
        value_scale=value_scale,
    )

    manifest = {
        "schema": "neuralls.dataset",
        "rhs": {
            "path": RHS_ARRAY_FILENAME,
            "dtype": str(rhs_array.dtype),
            "shape": list(rhs_array.shape),
        },
        "solutions": {
            "path": SOLUTIONS_ARRAY_FILENAME,
            "dtype": str(solutions_array.dtype),
            "shape": list(solutions_array.shape),
        },
        "matrix": {
            "path": MATRIX_COO_DIRNAME,
            "format": "coo_pack",
            "dtype": "float64",
            "n_samples": matrix_samples,
            "size": list(size),
            "value_scale": value_scale,
        },
        "normalization": {
            "type": normalization_type,
            "matrix_norm": float(matrix_norm),
            "matrix_norm_type": matrix_norm_type,
            "scale": scale_metadata or {},
        },
    }
    with paths.manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    return paths


def load_dataset_manifest(dataset_dir: str | Path) -> dict[str, Any]:
    """Load dataset manifest and validate schema marker."""
    paths = resolve_dataset_paths(dataset_dir)
    if not paths.manifest_path.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {paths.manifest_path}")
    with paths.manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict) or manifest.get("schema") != "neuralls.dataset":
        raise ValueError(f"Invalid dataset manifest schema in {paths.manifest_path}")
    return manifest


def load_dense_training_arrays(dataset_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load RHS and solution arrays from dataset directory."""
    paths = resolve_dataset_paths(dataset_dir)
    if not paths.rhs_path.exists() or not paths.solutions_path.exists():
        raise FileNotFoundError(f"Missing dense arrays in dataset directory: {paths.root}")
    rhs = np.load(paths.rhs_path, mmap_mode="r")
    solutions = np.load(paths.solutions_path, mmap_mode="r")
    return np.asarray(rhs, dtype=np.float64), np.asarray(solutions, dtype=np.float64)


def load_matrix_dense_sample(
    dataset_dir: str | Path,
    sample_index: int = 0,
    *,
    denormalize: bool = False,
) -> np.ndarray:
    """Load one sparse matrix sample and convert to dense numpy array."""
    paths = resolve_dataset_paths(dataset_dir)
    reader = open_sparse_pack(paths.matrix_pack_dir)
    matrix_tensor = reader.build_torch_sparse(
        sample_index=sample_index,
        denormalize=denormalize,
    )
    return matrix_tensor.to_dense().detach().cpu().numpy().astype(np.float64, copy=False)
