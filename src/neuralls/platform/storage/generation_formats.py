"""Write-time dataset storage implementations for generation workflows."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Never, Protocol

import h5py
import numpy as np
import zarr
from numpy.typing import NDArray

from neuralls.domain.generation.payloads import GeneratedDatasetPayload
from neuralls.domain.generation.ports import DatasetAccumulatorPort
from neuralls.platform.storage.manifest import DatasetArtifact, DatasetNormalization
from neuralls.platform.storage.manifest_io import make_dataset_manifest, save_dataset_manifest
from neuralls.shared.constants import PARAMETERS_ZARR_PREFIX
from neuralls.shared.types import DatasetFormat, LayoutType

_HDF5_DEFAULT_CHUNK_ROWS: int = 64


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


def _raise_storage_error(operation: str, path: Path, exc: OSError) -> Never:
    """Format and raise a descriptive OSError for a storage operation failure.

    Args:
        operation: Human-readable description of the failed operation.
        path: Path involved in the operation.
        exc: The original OSError raised.

    Raises:
        OSError: Always raised with a descriptive message.
    """
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
    permission_hint = ""
    if isinstance(exc, PermissionError) or winerror == 5:
        permission_hint = (
            " This usually means the filesystem blocked an atomic rename or file lock, "
            "which is common on network shares or when another process is holding the file."
        )
    message = f"{operation} at {path} failed. {', '.join(details)}{permission_hint}"
    raise OSError(message) from exc


class ArtifactReplacer:
    """Filesystem replacement helper for generated dataset artifacts."""

    def replace_file(self, source: Path, target: Path, *, operation: str) -> None:
        try:
            source.replace(target)
        except OSError as exc:
            _raise_storage_error(operation, target, exc)

    def replace_directory(self, source: Path, target: Path, *, operation: str) -> None:
        try:
            if target.exists():
                shutil.rmtree(str(target))
            shutil.move(str(source), str(target))
        except OSError as exc:
            _raise_storage_error(operation, target, exc)

    def remove_path(self, path: Path, *, operation: str) -> None:
        if not path.exists():
            return
        try:
            if path.is_dir():
                shutil.rmtree(str(path))
                return
            path.unlink()
        except OSError as exc:
            _raise_storage_error(operation, path, exc)

    def remove_matching(self, directory: Path, *, prefix: str, suffix: str, operation: str) -> None:
        if not directory.exists():
            return
        for path in directory.iterdir():
            if path.name.startswith(prefix) and path.name.endswith(suffix):
                self.remove_path(path, operation=operation)


class DenseZarrAccumulator:
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
                _raise_storage_error("Creating matrix.zarr store", self._zarr_path, exc)
        arr = self._arr
        if arr is None:
            raise RuntimeError("matrix zarr store was not initialized after successful open.")
        try:
            arr.resize((arr.shape[0] + repeats, n, m))
            arr[-repeats:] = np.broadcast_to(matrix[np.newaxis], (repeats, n, m))
        except OSError as exc:
            _raise_storage_error("Updating matrix.zarr store", self._zarr_path, exc)
        self._n_samples += repeats

    def finalize(self) -> Path:
        return self._zarr_path

    @property
    def matrix_size(self) -> tuple[int, int] | None:
        return self._size

    @property
    def n_samples(self) -> int:
        return self._n_samples


class DenseNpyAccumulator:
    """Accumulates dense matrix samples and stages them as a numpy array.

    Stores (matrix, repeats) pairs instead of expanding repeats into the list.
    Peak accumulation RAM is O(K*n*m) where K = unique matrices, not O(N*n*m).
    Expansion happens once at finalize via np.repeat + np.concatenate.
    """

    def __init__(self, npy_path: Path) -> None:
        self._npy_path = Path(npy_path)
        self._entries: list[tuple[np.ndarray, int]] = []
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
        mat = np.asarray(matrix, dtype=np.float64)
        n, m = int(mat.shape[0]), int(mat.shape[1])
        if self._size is None:
            self._size = (n, m)
        self._entries.append((mat, repeats))
        self._n_samples += repeats

    def finalize(self) -> Path:
        if not self._entries:
            return self._npy_path
        arrays = [np.repeat(mat[np.newaxis], r, axis=0) for mat, r in self._entries]
        try:
            np.save(self._npy_path, np.concatenate(arrays, axis=0))
        except OSError as exc:
            _raise_storage_error("Writing staged matrix numpy array", self._npy_path, exc)
        return self._npy_path

    @property
    def matrix_size(self) -> tuple[int, int] | None:
        return self._size

    @property
    def n_samples(self) -> int:
        return self._n_samples


_ZARR_GROUP_NAME = "dataset.zarr"


def _parameter_paths(dataset_dir: Path, suffix: str, count: int) -> tuple[Path, ...]:
    return tuple(dataset_dir / f"{PARAMETERS_ZARR_PREFIX}{index}{suffix}" for index in range(count))


def _zarr_group_member_paths(group_dir: Path, parameter_count: int) -> DatasetArtifactPaths:
    return DatasetArtifactPaths(
        matrix_path=group_dir / "matrix",
        rhs_path=group_dir / "rhs",
        solutions_path=group_dir / "solutions",
        parameter_paths=tuple(
            group_dir / f"{PARAMETERS_ZARR_PREFIX}{i}" for i in range(parameter_count)
        ),
    )


class ZarrGenerationStorage:
    """Write generated datasets as a zarr group container (dataset.zarr/)."""

    format_name = "zarr"

    def __init__(self, replacer: ArtifactReplacer | None = None) -> None:
        self._replacer = replacer or ArtifactReplacer()

    def make_accumulator(self, dataset_dir: Path) -> DatasetAccumulatorPort:
        return DenseZarrAccumulator(dataset_dir / ".matrix-staging.zarr")

    def artifact_paths(self, dataset_dir: Path, parameter_count: int) -> DatasetArtifactPaths:
        return _zarr_group_member_paths(dataset_dir / _ZARR_GROUP_NAME, parameter_count)

    def write_dataset(self, dataset_dir: Path, payload: GeneratedDatasetPayload) -> None:
        group_dir = dataset_dir / _ZARR_GROUP_NAME
        dataset_dir.mkdir(parents=True, exist_ok=True)
        # mode="w" truncates any existing group — same-format overwrite is always clean
        zarr.open_group(str(group_dir), mode="w")
        paths = _zarr_group_member_paths(group_dir, len(payload.parameters_arrays))
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
            _raise_storage_error("Writing rhs into group", paths.rhs_path, exc)

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
            _raise_storage_error("Writing solutions into group", paths.solutions_path, exc)

        pack_src = Path(payload.matrix_artifact_path)
        if pack_src != paths.matrix_path:
            self._replacer.replace_directory(
                pack_src,
                paths.matrix_path,
                operation="Moving matrix staging into group",
            )

        try:
            mat_arr = zarr.open_array(str(paths.matrix_path), mode="r")
        except OSError as exc:
            _raise_storage_error(
                "Reopening matrix member for manifest inspection", paths.matrix_path, exc
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
                _raise_storage_error(f"Writing {params_path.name} into group", params_path, exc)
            # ponytail: path relative to dataset_dir so manifest resolves from root
            params_manifest.append(
                DatasetArtifact(
                    path=f"{_ZARR_GROUP_NAME}/{params_path.name}",
                    format=self.format_name,
                    dtype="float64",
                    shape=tuple(int(dim) for dim in params_arr.shape),
                    index=index,
                )
            )

        if payload.row_kind_codes is not None:
            row_kind_path = group_dir / "row_kind"
            row_kind_arr = zarr.open_array(
                str(row_kind_path),
                mode="w",
                shape=payload.row_kind_codes.shape,
                chunks=payload.row_kind_codes.shape,
                dtype="uint8",
            )
            row_kind_arr[:] = payload.row_kind_codes
        if payload.matrix_sample_index is not None:
            matrix_sample_index_path = group_dir / "matrix_sample_index"
            matrix_sample_index_arr = zarr.open_array(
                str(matrix_sample_index_path),
                mode="w",
                shape=payload.matrix_sample_index.shape,
                chunks=payload.matrix_sample_index.shape,
                dtype="int64",
            )
            matrix_sample_index_arr[:] = payload.matrix_sample_index

        save_dataset_manifest(
            dataset_dir,
            make_dataset_manifest(
                matrix=DatasetArtifact(
                    path=f"{_ZARR_GROUP_NAME}/matrix",
                    format=self.format_name,
                    dtype="float64",
                    shape=tuple(int(dim) for dim in mat_arr.shape),
                    n_matrix_samples=int(mat_arr.shape[0]),
                    broadcast=payload.layout == LayoutType.BROADCAST_SINGLE,
                    layout=payload.layout,
                    logical_sample_count=int(payload.rhs.shape[0]),
                ),
                rhs=DatasetArtifact(
                    path=f"{_ZARR_GROUP_NAME}/rhs",
                    format=self.format_name,
                    dtype="float64",
                    shape=tuple(int(dim) for dim in payload.rhs.shape),
                ),
                solutions=DatasetArtifact(
                    path=f"{_ZARR_GROUP_NAME}/solutions",
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
                row_kind=DatasetArtifact(
                    path=f"{_ZARR_GROUP_NAME}/row_kind",
                    format=self.format_name,
                    dtype="uint8",
                    shape=tuple(int(dim) for dim in payload.row_kind_codes.shape),
                )
                if payload.row_kind_codes is not None
                else None,
                matrix_sample_index=DatasetArtifact(
                    path=f"{_ZARR_GROUP_NAME}/matrix_sample_index",
                    format=self.format_name,
                    dtype="int64",
                    shape=tuple(int(dim) for dim in payload.matrix_sample_index.shape),
                )
                if payload.matrix_sample_index is not None
                else None,
            ),
        )


class NpyGenerationStorage:
    """Write generated datasets into numpy-backed artifacts."""

    format_name = "npy"

    def __init__(self, replacer: ArtifactReplacer | None = None) -> None:
        self._replacer = replacer or ArtifactReplacer()

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
        self._remove_stale_optional_artifacts(dataset_dir)
        try:
            np.save(paths.rhs_path, payload.rhs)
            np.save(paths.solutions_path, payload.solutions)
            if payload.row_kind_codes is not None:
                np.save(dataset_dir / "row_kind.npy", payload.row_kind_codes)
            if payload.matrix_sample_index is not None:
                np.save(dataset_dir / "matrix_sample_index.npy", payload.matrix_sample_index)
        except OSError as exc:
            _raise_storage_error("Writing dataset numpy arrays", dataset_dir, exc)

        pack_src = Path(payload.matrix_artifact_path)
        if pack_src != paths.matrix_path:
            self._replacer.replace_file(
                pack_src,
                paths.matrix_path,
                operation="Moving matrix.npy into place",
            )

        params_manifest: list[DatasetArtifact] = []
        for index, (params_arr, params_path) in enumerate(
            zip(payload.parameters_arrays, paths.parameter_paths, strict=True)
        ):
            if params_arr.size == 0:
                continue
            try:
                np.save(params_path, params_arr)
            except OSError as exc:
                _raise_storage_error(f"Writing {params_path.name}", params_path, exc)
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
                    shape=(
                        1
                        if payload.layout == LayoutType.BROADCAST_SINGLE
                        else int(payload.rhs.shape[0]),
                        *payload.matrix_size,
                    ),
                    n_matrix_samples=1
                    if payload.layout == LayoutType.BROADCAST_SINGLE
                    else int(payload.rhs.shape[0]),
                    broadcast=payload.layout == LayoutType.BROADCAST_SINGLE,
                    layout=payload.layout,
                    logical_sample_count=int(payload.rhs.shape[0]),
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
                row_kind=DatasetArtifact(
                    path="row_kind.npy",
                    format=self.format_name,
                    dtype="uint8",
                    shape=tuple(int(dim) for dim in payload.row_kind_codes.shape),
                )
                if payload.row_kind_codes is not None
                else None,
                matrix_sample_index=DatasetArtifact(
                    path="matrix_sample_index.npy",
                    format=self.format_name,
                    dtype="int64",
                    shape=tuple(int(dim) for dim in payload.matrix_sample_index.shape),
                )
                if payload.matrix_sample_index is not None
                else None,
            ),
        )

    def _remove_stale_optional_artifacts(self, dataset_dir: Path) -> None:
        self._replacer.remove_path(
            dataset_dir / "row_kind.npy",
            operation="Removing stale row_kind numpy artifact",
        )
        self._replacer.remove_path(
            dataset_dir / "matrix_sample_index.npy",
            operation="Removing stale matrix_sample_index numpy artifact",
        )
        self._replacer.remove_matching(
            dataset_dir,
            prefix=PARAMETERS_ZARR_PREFIX,
            suffix=".npy",
            operation="Removing stale parameter numpy artifact",
        )


HDF5_FILENAME = "dataset.h5"
_HDF5_MATRIX_STAGING_KEY = "matrix"


class DenseHdf5Accumulator:
    """Streams dense matrix slices to a staged HDF5 file during generation."""

    def __init__(self, h5_path: Path) -> None:
        self._h5_path = Path(h5_path)
        self._file: h5py.File | None = None
        self._ds: h5py.Dataset | None = None
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
        if self._file is None:
            self._size = (n, m)
            try:
                self._file = h5py.File(str(self._h5_path), "w")
                self._ds = self._file.create_dataset(
                    _HDF5_MATRIX_STAGING_KEY,
                    shape=(0, n, m),
                    maxshape=(None, n, m),
                    chunks=(1, n, m),
                    dtype="float64",
                )
            except OSError as exc:
                _raise_storage_error("Creating matrix staging HDF5", self._h5_path, exc)
        ds = self._ds
        if ds is None:
            raise RuntimeError("HDF5 matrix staging dataset was not initialized.")
        try:
            current = ds.shape[0]
            ds.resize(current + repeats, axis=0)
            ds[current:] = np.broadcast_to(matrix[np.newaxis], (repeats, n, m))
        except OSError as exc:
            _raise_storage_error("Appending to matrix staging HDF5", self._h5_path, exc)
        self._n_samples += repeats

    def finalize(self) -> Path:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._ds = None
        return self._h5_path

    @property
    def matrix_size(self) -> tuple[int, int] | None:
        return self._size

    @property
    def n_samples(self) -> int:
        return self._n_samples


class Hdf5GenerationStorage:
    """Write generated datasets into a single HDF5 file (dataset.h5)."""

    format_name = "hdf5"

    def __init__(self, replacer: ArtifactReplacer | None = None) -> None:
        self._replacer = replacer or ArtifactReplacer()

    def make_accumulator(self, dataset_dir: Path) -> DatasetAccumulatorPort:
        return DenseHdf5Accumulator(dataset_dir / ".matrix-staging.h5")

    def write_dataset(self, dataset_dir: Path, payload: GeneratedDatasetPayload) -> None:
        dataset_dir.mkdir(parents=True, exist_ok=True)
        h5_path = dataset_dir / HDF5_FILENAME
        staging_path = Path(payload.matrix_artifact_path)

        n_samples = int(payload.rhs.shape[0])
        vec_dim = int(payload.rhs.shape[1])
        chunk_rows = min(_HDF5_DEFAULT_CHUNK_ROWS, n_samples)

        try:
            with h5py.File(str(staging_path), "a") as out:
                mat_shape = tuple(int(d) for d in out["matrix"].shape)
                out.create_dataset(
                    "rhs",
                    data=payload.rhs.astype(np.float64),
                    chunks=(chunk_rows, vec_dim),
                )
                out.create_dataset(
                    "solutions",
                    data=payload.solutions.astype(np.float64),
                    chunks=(chunk_rows, vec_dim),
                )
                if payload.row_kind_codes is not None:
                    out.create_dataset("row_kind", data=payload.row_kind_codes.astype(np.uint8))
                if payload.matrix_sample_index is not None:
                    out.create_dataset(
                        "matrix_sample_index",
                        data=payload.matrix_sample_index.astype(np.int64),
                    )

                params_manifest: list[DatasetArtifact] = []
                for index, params_arr in enumerate(payload.parameters_arrays):
                    if params_arr.size == 0:
                        continue
                    key = f"{PARAMETERS_ZARR_PREFIX}{index}"
                    out.create_dataset(key, data=params_arr.astype(np.float64))
                    params_manifest.append(
                        DatasetArtifact(
                            path=HDF5_FILENAME,
                            format=self.format_name,
                            dtype="float64",
                            shape=tuple(int(d) for d in params_arr.shape),
                            index=index,
                            key=key,
                        )
                    )
        except OSError as exc:
            _raise_storage_error("Writing dataset HDF5", staging_path, exc)

        self._replacer.replace_file(
            staging_path,
            h5_path,
            operation="Replacing dataset HDF5",
        )

        save_dataset_manifest(
            dataset_dir,
            make_dataset_manifest(
                matrix=DatasetArtifact(
                    path=HDF5_FILENAME,
                    format=self.format_name,
                    dtype="float64",
                    shape=mat_shape,
                    n_matrix_samples=int(mat_shape[0]),
                    broadcast=payload.layout == LayoutType.BROADCAST_SINGLE,
                    layout=payload.layout,
                    logical_sample_count=int(payload.rhs.shape[0]),
                    key="matrix",
                ),
                rhs=DatasetArtifact(
                    path=HDF5_FILENAME,
                    format=self.format_name,
                    dtype="float64",
                    shape=tuple(int(d) for d in payload.rhs.shape),
                    key="rhs",
                ),
                solutions=DatasetArtifact(
                    path=HDF5_FILENAME,
                    format=self.format_name,
                    dtype="float64",
                    shape=tuple(int(d) for d in payload.solutions.shape),
                    key="solutions",
                ),
                normalization=DatasetNormalization(
                    type=payload.normalization_type,
                    matrix_norm=float(payload.matrix_norm),
                    matrix_norm_type=payload.matrix_norm_type,
                    scale=dict(payload.scale_metadata or {}),
                ),
                params=tuple(params_manifest),
                row_kind=DatasetArtifact(
                    path=HDF5_FILENAME,
                    format=self.format_name,
                    dtype="uint8",
                    shape=tuple(int(d) for d in payload.row_kind_codes.shape),
                    key="row_kind",
                )
                if payload.row_kind_codes is not None
                else None,
                matrix_sample_index=DatasetArtifact(
                    path=HDF5_FILENAME,
                    format=self.format_name,
                    dtype="int64",
                    shape=tuple(int(d) for d in payload.matrix_sample_index.shape),
                    key="matrix_sample_index",
                )
                if payload.matrix_sample_index is not None
                else None,
            ),
        )


def make_generation_dataset_storage(dataset_format: DatasetFormat) -> GenerationDatasetStorage:
    """Construct the configured generation storage implementation."""
    match dataset_format:
        case "zarr":
            return ZarrGenerationStorage()
        case "npy":
            return NpyGenerationStorage()
        case "hdf5":
            return Hdf5GenerationStorage()
        case _:
            raise ValueError(f"Unknown dataset_format: {dataset_format!r}")
