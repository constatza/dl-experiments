"""Write-time dataset storage implementations for generation workflows."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import zarr
from numpy.typing import NDArray

from neuralls.domain.generation.payloads import GeneratedDatasetPayload
from neuralls.domain.generation.ports import DatasetAccumulatorPort
from neuralls.platform.storage.manifest import (
    DatasetArtifact,
    DatasetNormalization,
    make_dataset_manifest,
    save_dataset_manifest,
)
from neuralls.shared.constants import PARAMETERS_ZARR_PREFIX
from neuralls.shared.types import DatasetFormat


@dataclass(frozen=True)
class DatasetArtifactPaths:
    """Resolved physical artifact paths for one dataset format."""

    matrix_path: Path
    rhs_path: Path
    solutions_path: Path
    parameter_paths: tuple[Path, ...]


class GenerationDatasetStorage(Protocol):
    """Small composition-facing write seam for generated datasets."""

    def make_accumulator(self, dataset_dir: Path) -> DatasetAccumulatorPort: ...

    def write_dataset(self, dataset_dir: Path, payload: GeneratedDatasetPayload) -> None: ...


class _StorageErrorFormatter:
    @staticmethod
    def _format_os_error(exc: OSError) -> str:
        details = [f"{type(exc).__name__}: {exc}"]
        if exc.errno is not None:
            details.append(f"errno={exc.errno}")
        winerror = getattr(exc, "winerror", None)
        if winerror is not None:
            details.append(f"winerror={winerror}")
        if exc.filename:
            details.append(f"src={exc.filename}")
        if exc.filename2:
            details.append(f"dst={exc.filename2}")
        return ", ".join(details)

    @staticmethod
    def _permission_hint(exc: OSError) -> str:
        if isinstance(exc, PermissionError) or getattr(exc, "winerror", None) == 5:
            return (
                " This usually means the filesystem blocked an atomic Zarr metadata rename, "
                "which is common on network shares or when another process is holding the file."
            )
        return ""

    def _raise_storage_error(self, operation: str, path: Path, exc: OSError) -> None:
        message = (
            f"{operation} at {path} failed. "
            f"{self._format_os_error(exc)}"
            f"{self._permission_hint(exc)}"
        )
        raise OSError(message) from exc


class DenseZarrAccumulator(_StorageErrorFormatter):
    """Streams dense matrix slices to a staged zarr array during generation."""

    def __init__(self, zarr_path: Path) -> None:
        self._zarr_path = Path(zarr_path)
        self._arr: zarr.Array | None = None
        self._size: tuple[int, int] | None = None
        self._n_samples = 0

    def append_sparse_components(
        self,
        *,
        indices: NDArray,
        values: NDArray,
        size: tuple[int, int],
        repeats: int,
    ) -> None:
        dense = np.zeros(size, dtype=np.float64)
        if values.size > 0:
            dense[indices[0], indices[1]] = values
        self.append_dense_matrix(dense, repeats)

    def append_dense_matrix(self, matrix: NDArray, repeats: int) -> None:
        if repeats < 1:
            raise ValueError(f"repeats must be >= 1, got {repeats}")
        n, m = int(matrix.shape[0]), int(matrix.shape[1])
        if self._arr is None:
            self._size = (n, m)
            try:
                self._arr = zarr.open_array(
                    str(self._zarr_path),
                    mode="w",
                    shape=(0, n, m),
                    chunks=(1, n, m),
                    dtype="float64",
                )
            except OSError as exc:
                self._raise_storage_error("Creating matrix.zarr store", self._zarr_path, exc)
        arr = self._arr
        if arr is None:
            raise RuntimeError("matrix zarr store was not initialized after successful open.")
        try:
            arr.resize((arr.shape[0] + repeats, n, m))
            arr[-repeats:] = np.broadcast_to(matrix[np.newaxis], (repeats, n, m))
        except OSError as exc:
            self._raise_storage_error("Updating matrix.zarr store", self._zarr_path, exc)
        self._n_samples += repeats

    def finalize(self) -> Path:
        return self._zarr_path

    @property
    def matrix_size(self) -> tuple[int, int] | None:
        return self._size

    @property
    def n_samples(self) -> int:
        return self._n_samples


class DenseNpyAccumulator(_StorageErrorFormatter):
    """Accumulates dense matrix samples and stages them as a numpy array."""

    def __init__(self, npy_path: Path) -> None:
        self._npy_path = Path(npy_path)
        self._matrices: list[np.ndarray] = []
        self._size: tuple[int, int] | None = None
        self._n_samples = 0

    def append_sparse_components(
        self,
        *,
        indices: NDArray,
        values: NDArray,
        size: tuple[int, int],
        repeats: int,
    ) -> None:
        dense = np.zeros(size, dtype=np.float64)
        if values.size > 0:
            dense[indices[0], indices[1]] = values
        self.append_dense_matrix(dense, repeats)

    def append_dense_matrix(self, matrix: NDArray, repeats: int) -> None:
        if repeats < 1:
            raise ValueError(f"repeats must be >= 1, got {repeats}")
        n, m = int(matrix.shape[0]), int(matrix.shape[1])
        if self._size is None:
            self._size = (n, m)
        self._matrices.extend(np.array(matrix, copy=True, dtype=np.float64) for _ in range(repeats))
        self._n_samples += repeats

    def finalize(self) -> Path:
        if not self._matrices:
            return self._npy_path
        try:
            np.save(self._npy_path, np.stack(self._matrices, axis=0))
        except OSError as exc:
            self._raise_storage_error("Writing staged matrix numpy array", self._npy_path, exc)
        return self._npy_path

    @property
    def matrix_size(self) -> tuple[int, int] | None:
        return self._size

    @property
    def n_samples(self) -> int:
        return self._n_samples


def _parameter_paths(dataset_dir: Path, suffix: str, count: int) -> tuple[Path, ...]:
    return tuple(dataset_dir / f"{PARAMETERS_ZARR_PREFIX}{index}{suffix}" for index in range(count))


class ZarrGenerationStorage(_StorageErrorFormatter):
    """Write generated datasets into zarr-backed artifacts."""

    format_name = "zarr"

    def make_accumulator(self, dataset_dir: Path) -> DatasetAccumulatorPort:
        return DenseZarrAccumulator(dataset_dir / ".matrix-staging.zarr")

    def artifact_paths(self, dataset_dir: Path, parameter_count: int) -> DatasetArtifactPaths:
        return DatasetArtifactPaths(
            matrix_path=dataset_dir / "matrix.zarr",
            rhs_path=dataset_dir / "rhs.zarr",
            solutions_path=dataset_dir / "solutions.zarr",
            parameter_paths=_parameter_paths(dataset_dir, ".zarr", parameter_count),
        )

    def write_dataset(self, dataset_dir: Path, payload: GeneratedDatasetPayload) -> None:
        paths = self.artifact_paths(dataset_dir, len(payload.parameters_arrays))
        dataset_dir.mkdir(parents=True, exist_ok=True)
        n = int(payload.rhs.shape[1])
        try:
            rhs_arr = zarr.open_array(
                str(paths.rhs_path),
                mode="w",
                shape=payload.rhs.shape,
                chunks=(1, n),
                dtype="float64",
            )
            rhs_arr[:] = payload.rhs
        except OSError as exc:
            self._raise_storage_error("Writing rhs.zarr", paths.rhs_path, exc)

        try:
            sol_arr = zarr.open_array(
                str(paths.solutions_path),
                mode="w",
                shape=payload.solutions.shape,
                chunks=(1, n),
                dtype="float64",
            )
            sol_arr[:] = payload.solutions
        except OSError as exc:
            self._raise_storage_error("Writing solutions.zarr", paths.solutions_path, exc)

        pack_src = Path(payload.matrix_artifact_path)
        if pack_src != paths.matrix_path:
            try:
                if paths.matrix_path.exists():
                    shutil.rmtree(str(paths.matrix_path))
                shutil.move(str(pack_src), str(paths.matrix_path))
            except OSError as exc:
                self._raise_storage_error(
                    "Moving matrix.zarr store into place", paths.matrix_path, exc
                )

        try:
            mat_arr = zarr.open_array(str(paths.matrix_path), mode="r")
        except OSError as exc:
            self._raise_storage_error(
                "Reopening matrix.zarr for manifest inspection", paths.matrix_path, exc
            )

        params_manifest: list[DatasetArtifact] = []
        for index, (params_arr, params_path) in enumerate(
            zip(payload.parameters_arrays, paths.parameter_paths, strict=True)
        ):
            if params_arr.size == 0:
                continue
            try:
                params_zarr = zarr.open_array(
                    str(params_path),
                    mode="w",
                    shape=params_arr.shape,
                    chunks=(1, params_arr.shape[1]) if params_arr.ndim == 2 else params_arr.shape,
                    dtype="float64",
                )
                params_zarr[:] = params_arr
            except OSError as exc:
                self._raise_storage_error(f"Writing {params_path.name}", params_path, exc)
            params_manifest.append(
                DatasetArtifact(
                    path=params_path.name,
                    format=self.format_name,
                    dtype="float64",
                    shape=tuple(int(dim) for dim in params_arr.shape),
                    index=index,
                )
            )

        save_dataset_manifest(
            dataset_dir,
            make_dataset_manifest(
                matrix=DatasetArtifact(
                    path=paths.matrix_path.name,
                    format=self.format_name,
                    dtype="float64",
                    shape=tuple(int(dim) for dim in mat_arr.shape),
                    n_matrix_samples=int(mat_arr.shape[0]),
                    broadcast=int(mat_arr.shape[0]) == 1,
                ),
                rhs=DatasetArtifact(
                    path=paths.rhs_path.name,
                    format=self.format_name,
                    dtype="float64",
                    shape=tuple(int(dim) for dim in payload.rhs.shape),
                ),
                solutions=DatasetArtifact(
                    path=paths.solutions_path.name,
                    format=self.format_name,
                    dtype="float64",
                    shape=tuple(int(dim) for dim in payload.solutions.shape),
                ),
                normalization=DatasetNormalization(
                    type=payload.normalization_type,
                    matrix_norm=float(payload.matrix_norm),
                    matrix_norm_type=payload.matrix_norm_type,
                    scale=dict(payload.scale_metadata or {}),
                ),
                params=tuple(params_manifest),
            ),
        )


class NpyGenerationStorage(_StorageErrorFormatter):
    """Write generated datasets into numpy-backed artifacts."""

    format_name = "npy"

    def make_accumulator(self, dataset_dir: Path) -> DatasetAccumulatorPort:
        return DenseNpyAccumulator(dataset_dir / ".matrix-staging.npy")

    def artifact_paths(self, dataset_dir: Path, parameter_count: int) -> DatasetArtifactPaths:
        return DatasetArtifactPaths(
            matrix_path=dataset_dir / "matrix.npy",
            rhs_path=dataset_dir / "rhs.npy",
            solutions_path=dataset_dir / "solutions.npy",
            parameter_paths=_parameter_paths(dataset_dir, ".npy", parameter_count),
        )

    def write_dataset(self, dataset_dir: Path, payload: GeneratedDatasetPayload) -> None:
        paths = self.artifact_paths(dataset_dir, len(payload.parameters_arrays))
        dataset_dir.mkdir(parents=True, exist_ok=True)
        try:
            np.save(paths.rhs_path, payload.rhs)
            np.save(paths.solutions_path, payload.solutions)
        except OSError as exc:
            self._raise_storage_error("Writing dataset numpy arrays", dataset_dir, exc)

        pack_src = Path(payload.matrix_artifact_path)
        if pack_src != paths.matrix_path:
            try:
                shutil.move(str(pack_src), str(paths.matrix_path))
            except OSError as exc:
                self._raise_storage_error("Moving matrix.npy into place", paths.matrix_path, exc)

        matrix = np.load(paths.matrix_path)
        params_manifest: list[DatasetArtifact] = []
        for index, (params_arr, params_path) in enumerate(
            zip(payload.parameters_arrays, paths.parameter_paths, strict=True)
        ):
            if params_arr.size == 0:
                continue
            try:
                np.save(params_path, params_arr)
            except OSError as exc:
                self._raise_storage_error(f"Writing {params_path.name}", params_path, exc)
            params_manifest.append(
                DatasetArtifact(
                    path=params_path.name,
                    format=self.format_name,
                    dtype="float64",
                    shape=tuple(int(dim) for dim in params_arr.shape),
                    index=index,
                )
            )

        save_dataset_manifest(
            dataset_dir,
            make_dataset_manifest(
                matrix=DatasetArtifact(
                    path=paths.matrix_path.name,
                    format=self.format_name,
                    dtype="float64",
                    shape=tuple(int(dim) for dim in matrix.shape),
                    n_matrix_samples=int(matrix.shape[0]),
                    broadcast=int(matrix.shape[0]) == 1,
                ),
                rhs=DatasetArtifact(
                    path=paths.rhs_path.name,
                    format=self.format_name,
                    dtype="float64",
                    shape=tuple(int(dim) for dim in payload.rhs.shape),
                ),
                solutions=DatasetArtifact(
                    path=paths.solutions_path.name,
                    format=self.format_name,
                    dtype="float64",
                    shape=tuple(int(dim) for dim in payload.solutions.shape),
                ),
                normalization=DatasetNormalization(
                    type=payload.normalization_type,
                    matrix_norm=float(payload.matrix_norm),
                    matrix_norm_type=payload.matrix_norm_type,
                    scale=dict(payload.scale_metadata or {}),
                ),
                params=tuple(params_manifest),
            ),
        )


def make_generation_dataset_storage(dataset_format: DatasetFormat) -> GenerationDatasetStorage:
    """Construct the configured generation storage implementation."""
    match dataset_format:
        case "zarr":
            return ZarrGenerationStorage()
        case "npy":
            return NpyGenerationStorage()
        case _:
            raise ValueError(f"Unknown dataset_format: {dataset_format!r}")
