"""Dataset storage helpers for dense zarr arrays."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from numpy.typing import NDArray

from neuralls.domain.generation.payloads import GeneratedDatasetPayload
from neuralls.shared.constants import (
    DATASET_MANIFEST_FILENAME,
    MATRIX_ZARR_DIRNAME,
    PARAMETERS_ZARR_PREFIX,
    RHS_ZARR_FILENAME,
    SOLUTIONS_ZARR_FILENAME,
)


@dataclass(frozen=True)
class DatasetPaths:
    """Resolved canonical dataset artifact paths.

    Attributes:
        root: Root directory for the dataset.
        manifest_path: Path to the manifest.json file.
        rhs_path: Path to the rhs.zarr/ directory.
        solutions_path: Path to the solutions.zarr/ directory.
        matrix_zarr_dir: Path to the matrix.zarr/ directory.
    """

    root: Path
    manifest_path: Path
    rhs_path: Path
    solutions_path: Path
    matrix_zarr_dir: Path


class DenseZarrAccumulator:
    """Streams dense matrix slices to matrix.zarr/ during generation.

    Writes each unique matrix directly to disk. In broadcast mode
    (append_dense_matrix called once with repeats=N), stores shape (1, n, n).
    In per-sample mode (multiple calls), stores each repeat as its own slice
    so matrix.zarr[i] corresponds to rhs[i] row-for-row.
    Satisfies DenseAccumulatorPort protocol.

    Args:
        zarr_path: Destination path for the zarr array store.
    """

    def __init__(self, zarr_path: Path) -> None:
        self._zarr_path = Path(zarr_path)
        self._arr: zarr.Array | None = None
        self._size: tuple[int, int] | None = None
        self._n_samples: int = 0
        self._finalized: bool = False

    def append_sparse_components(
        self,
        *,
        indices: NDArray,
        values: NDArray,
        size: tuple[int, int],
        repeats: int,
    ) -> None:
        """Convert sparse COO components to dense and append to the zarr store.

        Args:
            indices: COO indices array, shape (2, nnz).
            values: COO values array, shape (nnz,).
            size: Matrix dimensions (rows, cols).
            repeats: Number of times this matrix is replicated in the store.
        """
        n, m = size
        dense = np.zeros((n, m), dtype=np.float64)
        if values.size > 0:
            dense[indices[0], indices[1]] = values
        self.append_dense_matrix(dense, repeats)

    def append_dense_matrix(self, matrix: NDArray, repeats: int) -> None:
        """Append one dense matrix, optionally repeated, to the zarr store.

        Args:
            matrix: Dense matrix array, shape (n, n).
            repeats: Number of times this matrix is replicated in the store.

        Raises:
            ValueError: If repeats < 1.
        """
        if repeats < 1:
            raise ValueError(f"repeats must be >= 1, got {repeats}")
        n, m = int(matrix.shape[0]), int(matrix.shape[1])
        if self._arr is None:
            self._size = (n, m)
            self._arr = zarr.open_array(
                str(self._zarr_path),
                mode="w",
                shape=(0, n, m),
                chunks=(1, n, m),
                dtype="float64",
            )
        self._arr.resize((self._arr.shape[0] + repeats, n, m))
        self._arr[-repeats:] = np.broadcast_to(matrix[np.newaxis], (repeats, n, m))
        self._n_samples += repeats

    def finalize(self) -> Path:
        """Close the accumulator and return the zarr store path.

        Safe to call multiple times — subsequent calls are no-ops.

        Returns:
            Path to the completed zarr array store directory.
        """
        self._finalized = True
        return self._zarr_path

    @property
    def matrix_size(self) -> tuple[int, int] | None:
        """Matrix dimensions of the first appended sample, or None if empty."""
        return self._size

    @property
    def n_samples(self) -> int:
        """Total logical samples written (sum of all repeats)."""
        return self._n_samples


class DenseDatasetWriter:
    """Writes rhs.zarr/, solutions.zarr/, and manifest.json.

    Satisfies DatasetWriterPort. matrix.zarr/ is already on disk from
    the accumulator and will be moved into place if needed.
    """

    def write_dataset(
        self,
        dataset_dir: Path,
        payload: GeneratedDatasetPayload,
    ) -> None:
        """Persist a generated dataset payload in zarr dense format.

        Args:
            dataset_dir: Destination directory for the dataset.
            payload: Payload produced by the generation domain.
        """
        paths = resolve_dataset_paths(dataset_dir)
        paths.root.mkdir(parents=True, exist_ok=True)

        n = int(payload.rhs.shape[1])

        rhs_arr = zarr.open_array(
            str(paths.rhs_path),
            mode="w",
            shape=payload.rhs.shape,
            chunks=(1, n),
            dtype="float64",
        )
        rhs_arr[:] = payload.rhs

        sol_arr = zarr.open_array(
            str(paths.solutions_path),
            mode="w",
            shape=payload.solutions.shape,
            chunks=(1, n),
            dtype="float64",
        )
        sol_arr[:] = payload.solutions

        pack_src = Path(payload.matrix_zarr_path)
        if pack_src != paths.matrix_zarr_dir:
            if paths.matrix_zarr_dir.exists():
                shutil.rmtree(str(paths.matrix_zarr_dir))
            shutil.move(str(pack_src), str(paths.matrix_zarr_dir))

        mat_arr = zarr.open_array(str(paths.matrix_zarr_dir), mode="r")
        mat_shape = list(mat_arr.shape)
        n_matrix_samples = int(mat_arr.shape[0])
        broadcast = n_matrix_samples == 1

        # Write per-sample parameters arrays (shape N × param_dim each)
        params_manifest: list[dict[str, Any]] = []
        for i, params_arr in enumerate(payload.parameters_arrays):
            if params_arr.size == 0:
                continue
            params_name = f"{PARAMETERS_ZARR_PREFIX}{i}.zarr"
            params_zarr = zarr.open_array(
                str(dataset_dir / params_name),
                mode="w",
                shape=params_arr.shape,
                chunks=(1, params_arr.shape[1]) if params_arr.ndim == 2 else params_arr.shape,
                dtype="float64",
            )
            params_zarr[:] = params_arr
            params_manifest.append(
                {
                    "index": i,
                    "path": params_name,
                    "format": "zarr_dense",
                    "dtype": "float64",
                    "shape": list(params_arr.shape),
                }
            )

        manifest: dict[str, Any] = {
            "schema": "neuralls.dataset.v2",
            "matrix": {
                "path": MATRIX_ZARR_DIRNAME,
                "format": "zarr_dense",
                "dtype": "float64",
                "shape": mat_shape,
                "n_matrix_samples": n_matrix_samples,
                "broadcast": broadcast,
            },
            "rhs": {
                "path": RHS_ZARR_FILENAME,
                "format": "zarr_dense",
                "dtype": "float64",
                "shape": list(payload.rhs.shape),
            },
            "solutions": {
                "path": SOLUTIONS_ZARR_FILENAME,
                "format": "zarr_dense",
                "dtype": "float64",
                "shape": list(payload.solutions.shape),
            },
            "normalization": {
                "type": payload.normalization_type,
                "matrix_norm": float(payload.matrix_norm),
                "matrix_norm_type": payload.matrix_norm_type,
                "scale": payload.scale_metadata or {},
            },
            "params": params_manifest if params_manifest else None,
        }
        with paths.manifest_path.open("w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, sort_keys=True)


def resolve_dataset_paths(dataset_dir: str | Path) -> DatasetPaths:
    """Resolve canonical dataset artifact locations.

    Args:
        dataset_dir: Root directory of the dataset.

    Returns:
        DatasetPaths with all artifact paths resolved.
    """
    root = Path(dataset_dir)
    return DatasetPaths(
        root=root,
        manifest_path=root / DATASET_MANIFEST_FILENAME,
        rhs_path=root / RHS_ZARR_FILENAME,
        solutions_path=root / SOLUTIONS_ZARR_FILENAME,
        matrix_zarr_dir=root / MATRIX_ZARR_DIRNAME,
    )


def load_dataset_manifest(dataset_dir: str | Path) -> dict[str, Any]:
    """Load dataset manifest and validate schema marker.

    Args:
        dataset_dir: Root directory of the dataset.

    Returns:
        Parsed manifest dictionary.

    Raises:
        FileNotFoundError: If manifest.json does not exist.
        ValueError: If the manifest schema is not recognised.
    """
    paths = resolve_dataset_paths(dataset_dir)
    if not paths.manifest_path.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {paths.manifest_path}")
    with paths.manifest_path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if not isinstance(manifest, dict) or not str(manifest.get("schema", "")).startswith(
        "neuralls.dataset"
    ):
        raise ValueError(f"Invalid dataset manifest schema in {paths.manifest_path}")
    return manifest


def load_dense_training_arrays(dataset_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (rhs, solutions) as float64 arrays from rhs.zarr/ and solutions.zarr/.

    Args:
        dataset_dir: Root directory of the dataset.

    Returns:
        Tuple of (rhs, solutions) arrays, both dtype float64.
    """
    paths = resolve_dataset_paths(dataset_dir)
    rhs = np.asarray(zarr.open_array(str(paths.rhs_path), mode="r"), dtype=np.float64)
    solutions = np.asarray(zarr.open_array(str(paths.solutions_path), mode="r"), dtype=np.float64)
    return rhs, solutions


def load_matrix_dense_sample(
    dataset_dir: str | Path,
    sample_index: int = 0,
) -> np.ndarray:
    """Load one matrix sample from matrix.zarr/ as a dense float64 numpy array.

    Args:
        dataset_dir: Root directory of the dataset.
        sample_index: Index of the sample to load.

    Returns:
        Dense float64 array of shape (n, n).
    """
    paths = resolve_dataset_paths(dataset_dir)
    arr = zarr.open_array(str(paths.matrix_zarr_dir), mode="r")
    return np.asarray(arr[sample_index], dtype=np.float64)
