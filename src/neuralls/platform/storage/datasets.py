"""Dataset storage helpers for split dense arrays + sparse COO matrix packs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from dlkit.io import SparseFormat, open_sparse_pack, save_sparse_pack

from neuralls.domain.generation.payloads import GeneratedDatasetPayload
from neuralls.domain.generation.ports import DatasetWriterPort
from neuralls.shared.constants import (
    DATASET_MANIFEST_FILENAME,
    MATRIX_COO_DIRNAME,
    RHS_ARRAY_FILENAME,
    SOLUTIONS_ARRAY_FILENAME,
)
from neuralls.shared.types import ScaleMetadata


@dataclass(frozen=True)
class DatasetPaths:
    """Resolved canonical dataset artifact paths."""

    root: Path
    manifest_path: Path
    rhs_path: Path
    solutions_path: Path
    matrix_pack_dir: Path


class SparsePackReaderPort(Protocol):
    """Protocol for sparse-pack readers that can materialize one sample tensor."""

    n_samples: int

    def collect(
        self,
        sample_index: int,
        *,
        device: Any | None = None,
        dtype: Any | None = None,
    ) -> Any: ...


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


def as_sparse_pack_reader(reader: Any) -> SparsePackReaderPort:
    """Narrow an external sparse-pack reader to the collect-capable protocol."""
    return cast(SparsePackReaderPort, reader)


def _validate_sparse_inputs(
    indices: np.ndarray, values: np.ndarray, size: tuple[int, int], repeats: int
) -> None:
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    if indices.ndim != 2 or indices.shape[0] != 2:
        raise ValueError(f"indices must have shape (2, nnz), got {indices.shape}")
    if values.ndim != 1:
        raise ValueError(f"values must be 1D, got {values.shape}")
    if indices.shape[1] != values.size:
        raise ValueError(f"indices nnz ({indices.shape[1]}) != values ({values.size})")


def _advance_ptr(ptr: list[int], nnz: int, repeats: int) -> None:
    last = ptr[-1]
    for _ in range(repeats):
        last += nnz
        ptr.append(last)


class DiskBackedSparseAccumulator:
    """Platform-layer accumulator: writes each binding chunk to disk immediately.

    Keeps per-binding sparse chunks on disk until final assembly.
    """

    def __init__(self, staging_dir: Path) -> None:
        staging_dir.mkdir(parents=True, exist_ok=True)
        self._staging_dir = staging_dir
        self._pack_dir = staging_dir / "_pack"
        self._chunk_count = 0
        self._nnz_ptr_values: list[int] = [0]
        self._matrix_size: tuple[int, int] | None = None

    def append_dense_matrix(self, matrix: np.ndarray, repeats: int) -> None:
        rows, cols = np.nonzero(matrix)
        self.append_sparse_components(
            indices=np.vstack((rows, cols)).astype(np.int64, copy=False),
            values=np.asarray(matrix[rows, cols], dtype=np.float64),
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
        _validate_sparse_inputs(indices, values, size, repeats)
        if self._matrix_size is not None and self._matrix_size != size:
            raise ValueError(f"Matrix size mismatch: {size} != {self._matrix_size}")
        self._matrix_size = size

        nnz = int(values.size)
        _advance_ptr(self._nnz_ptr_values, nnz, repeats)

        if nnz == 0:
            return

        np.savez(
            self._staging_dir / f"chunk_{self._chunk_count:06d}.npz",
            indices=np.asarray(indices, dtype=np.int64),
            values=np.asarray(values, dtype=np.float64),
            repeats=np.int64(repeats),
        )
        self._chunk_count += 1

    def build_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
        if self._matrix_size is None:
            raise ValueError("DiskBackedSparseAccumulator is empty.")
        nnz_ptr = np.asarray(self._nnz_ptr_values, dtype=np.int64)
        if int(nnz_ptr[-1]) == 0:
            empty_idx = np.zeros((2, 0), dtype=np.int64)
            empty_val = np.zeros(0, dtype=np.float64)
            save_sparse_pack(
                path=self._pack_dir,
                indices=empty_idx,
                values=empty_val,
                nnz_ptr=nnz_ptr,
                size=self._matrix_size,
                format=SparseFormat.COO,
                dtype=np.dtype(np.float64),
            )
            return empty_idx, empty_val, nnz_ptr, self._matrix_size
        indices_parts: list[np.ndarray] = []
        values_parts: list[np.ndarray] = []
        for i in range(self._chunk_count):
            chunk_path = self._staging_dir / f"chunk_{i:06d}.npz"
            data = np.load(chunk_path)
            indices = np.asarray(data["indices"], dtype=np.int64)
            values = np.asarray(data["values"], dtype=np.float64)
            repeats = int(data["repeats"])
            if repeats == 1:
                indices_parts.append(indices)
                values_parts.append(values)
            else:
                indices_parts.append(np.tile(indices, (1, repeats)))
                values_parts.append(np.tile(values, repeats))
            chunk_path.unlink()

        indices = np.concatenate(indices_parts, axis=1)
        values = np.concatenate(values_parts, axis=0)
        save_sparse_pack(
            path=self._pack_dir,
            indices=indices,
            values=values,
            nnz_ptr=nnz_ptr,
            size=self._matrix_size,
            format=SparseFormat.COO,
            dtype=np.dtype(np.float64),
        )
        return indices, values, nnz_ptr, self._matrix_size

    @property
    def pack_dir(self) -> Path:
        return self._pack_dir


def _write_dataset_manifest(
    paths: DatasetPaths,
    payload: GeneratedDatasetPayload,
) -> None:
    matrix_samples = int(payload.nnz_ptr.size - 1)
    manifest = {
        "schema": "neuralls.dataset",
        "rhs": {
            "path": RHS_ARRAY_FILENAME,
            "dtype": "float64",
            "shape": list(payload.rhs.shape),
        },
        "solutions": {
            "path": SOLUTIONS_ARRAY_FILENAME,
            "dtype": "float64",
            "shape": list(payload.solutions.shape),
        },
        "matrix": {
            "path": MATRIX_COO_DIRNAME,
            "format": "coo_pack",
            "dtype": "float64",
            "n_samples": matrix_samples,
            "size": list(payload.matrix_size),
        },
        "normalization": {
            "type": payload.normalization_type,
            "matrix_norm": float(payload.matrix_norm),
            "matrix_norm_type": payload.matrix_norm_type,
            "scale": payload.scale_metadata or {},
        },
    }
    with paths.manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)


class StreamingSparseDatasetWriter:
    """DatasetWriterPort that persists sparse arrays assembled by DiskBackedSparseAccumulator."""

    def __init__(self, disk_acc: DiskBackedSparseAccumulator) -> None:
        self._disk_acc = disk_acc

    def write_dataset(
        self,
        dataset_dir: Path,
        payload: GeneratedDatasetPayload,
    ) -> None:
        paths = resolve_dataset_paths(dataset_dir)
        paths.root.mkdir(parents=True, exist_ok=True)
        np.save(paths.rhs_path, payload.rhs)
        np.save(paths.solutions_path, payload.solutions)
        if payload.matrix_value_scale != 1.0:
            payload.values[:] *= payload.matrix_value_scale
        save_sparse_pack(
            path=paths.matrix_pack_dir,
            indices=payload.indices,
            values=payload.values,
            nnz_ptr=payload.nnz_ptr,
            size=payload.matrix_size,
            format=SparseFormat.COO,
            dtype=np.dtype(np.float64),
        )
        _write_dataset_manifest(paths, payload)


class SparseDatasetWriter(DatasetWriterPort):
    """Default storage adapter for generated sparse datasets."""

    def write_dataset(
        self,
        dataset_dir: Path,
        payload: GeneratedDatasetPayload,
    ) -> None:
        """Persist a generated dataset payload."""
        save_dataset_from_sparse(
            dataset_dir=dataset_dir,
            rhs=payload.rhs,
            solutions=payload.solutions,
            indices=payload.indices,
            values=payload.values,
            nnz_ptr=payload.nnz_ptr,
            size=payload.matrix_size,
            normalization_type=payload.normalization_type,
            matrix_norm=payload.matrix_norm,
            matrix_norm_type=payload.matrix_norm_type,
            matrix_value_scale=payload.matrix_value_scale,
            scale_metadata=payload.scale_metadata,
        )


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
        indices = (
            np.concatenate(indices_parts, axis=1)
            if indices_parts
            else np.zeros((2, 0), dtype=np.int64)
        )
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
        raise ValueError(f"matrix_value_scale must be finite and > 0, got {matrix_value_scale}")

    np.save(paths.rhs_path, rhs_array)
    np.save(paths.solutions_path, solutions_array)
    save_sparse_pack(
        path=paths.matrix_pack_dir,
        indices=np.asarray(indices, dtype=np.int64),
        values=np.asarray(values, dtype=np.float64) * value_scale,
        nnz_ptr=nnz_ptr_arr,
        size=size,
        format=SparseFormat.COO,
        dtype=np.dtype(np.float64),
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
) -> np.ndarray:
    """Load one sparse matrix sample and convert to dense numpy array."""
    paths = resolve_dataset_paths(dataset_dir)
    reader = as_sparse_pack_reader(open_sparse_pack(paths.matrix_pack_dir))
    matrix_tensor = reader.collect(sample_index=sample_index)
    return matrix_tensor.to_dense().detach().cpu().numpy().astype(np.float64, copy=False)
