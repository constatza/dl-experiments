"""Manifest-driven dataset readers and artifact resolution helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import zarr

from neuralls.platform.storage.manifest import DatasetArtifact
from neuralls.platform.storage.manifest_io import read_dataset_manifest
from neuralls.shared.constants import DATASET_MANIFEST_FILENAME
from neuralls.shared.types import LayoutType


@dataclass(frozen=True)
class ResolvedDatasetArtifact:
    """Resolved on-disk artifact contract for one dataset array."""

    path: Path
    format: str
    dtype: str
    shape: tuple[int, ...]
    index: int | None = None
    n_matrix_samples: int | None = None
    broadcast: bool | None = None
    key: str | None = None
    layout: LayoutType | None = None
    logical_sample_count: int | None = None


@dataclass(frozen=True)
class DatasetArtifacts:
    """Explicit resolved dataset contract used by storage readers."""

    root: Path
    manifest_path: Path
    matrix: ResolvedDatasetArtifact
    rhs: ResolvedDatasetArtifact
    solutions: ResolvedDatasetArtifact
    params: tuple[ResolvedDatasetArtifact, ...] = ()
    row_kind: ResolvedDatasetArtifact | None = None
    matrix_sample_index: ResolvedDatasetArtifact | None = None


@dataclass(frozen=True)
class DatasetPaths:
    """Resolved dataset artifact paths from the manifest."""

    root: Path
    manifest_path: Path
    rhs_path: Path
    solutions_path: Path
    matrix_path: Path
    parameter_paths: tuple[Path, ...]

    @property
    def matrix_zarr_dir(self) -> Path:
        """Compatibility alias for older zarr-specific callers."""
        return self.matrix_path


def _resolve_artifact(root: Path, artifact: DatasetArtifact) -> ResolvedDatasetArtifact:
    return ResolvedDatasetArtifact(
        path=root / artifact.path,
        format=artifact.format,
        dtype=artifact.dtype,
        shape=artifact.shape,
        index=artifact.index,
        n_matrix_samples=artifact.n_matrix_samples,
        broadcast=artifact.broadcast,
        key=artifact.key,
        layout=artifact.layout,
        logical_sample_count=artifact.logical_sample_count,
    )


def resolve_dataset_artifacts(dataset_dir: str | Path) -> DatasetArtifacts:
    """Resolve the explicit dataset artifact contract from the manifest."""
    root = Path(dataset_dir)
    try:
        manifest = read_dataset_manifest(root)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Dataset manifest not found in '{root}'. "
            "Re-run the generation pipeline to produce a valid dataset."
        ) from None
    return DatasetArtifacts(
        root=root,
        manifest_path=root / DATASET_MANIFEST_FILENAME,
        matrix=_resolve_artifact(root, manifest.matrix),
        rhs=_resolve_artifact(root, manifest.rhs),
        solutions=_resolve_artifact(root, manifest.solutions),
        params=tuple(_resolve_artifact(root, artifact) for artifact in manifest.params),
        row_kind=_resolve_artifact(root, manifest.row_kind)
        if manifest.row_kind is not None
        else None,
        matrix_sample_index=(
            _resolve_artifact(root, manifest.matrix_sample_index)
            if manifest.matrix_sample_index is not None
            else None
        ),
    )


def resolve_dataset_paths(dataset_dir: str | Path) -> DatasetPaths:
    """Resolve dataset artifact paths from the manifest contract."""
    artifacts = resolve_dataset_artifacts(dataset_dir)
    return DatasetPaths(
        root=artifacts.root,
        manifest_path=artifacts.manifest_path,
        rhs_path=artifacts.rhs.path,
        solutions_path=artifacts.solutions.path,
        matrix_path=artifacts.matrix.path,
        parameter_paths=tuple(artifact.path for artifact in artifacts.params),
    )


def _load_sample(artifact: ResolvedDatasetArtifact, sample_index: int) -> np.ndarray:
    match artifact.format:
        case "npy":
            arr = np.load(artifact.path, mmap_mode="r")
            return np.asarray(arr[sample_index] if arr.ndim > 1 else arr, dtype=np.float64)
        case "zarr":
            arr = zarr.open_array(str(artifact.path), mode="r")
            data = arr[sample_index] if arr.ndim > 1 else arr[:]
            return np.asarray(data, dtype=np.float64)
        case "hdf5":
            if artifact.key is None:
                raise ValueError(f"HDF5 artifact at {artifact.path} is missing a required key")
            with h5py.File(str(artifact.path), "r") as f:
                ds = f[artifact.key]
                data = ds[sample_index] if ds.ndim > 1 else ds[:]  # type: ignore[index]
                return np.asarray(data, dtype=np.float64)
        case _:
            raise ValueError(f"Unsupported format {artifact.format!r} at {artifact.path}")


def _load_resolved(artifact: ResolvedDatasetArtifact) -> np.ndarray:
    match artifact.format:
        case "zarr":
            return np.asarray(zarr.open_array(str(artifact.path), mode="r"), dtype=np.float64)
        case "npy":
            return np.load(artifact.path).astype(np.float64, copy=False)
        case "hdf5":
            if artifact.key is None:
                raise ValueError(f"HDF5 artifact at {artifact.path} is missing a required key")
            with h5py.File(str(artifact.path), "r") as f:
                return np.asarray(f[artifact.key], dtype=np.float64)
        case _:
            raise ValueError(
                f"Unsupported dataset artifact format {artifact.format!r} at {artifact.path}"
            )


def _load_resolved_native(artifact: ResolvedDatasetArtifact) -> np.ndarray:
    """Load one dataset artifact without coercing the dtype."""
    match artifact.format:
        case "zarr":
            return np.asarray(zarr.open_array(str(artifact.path), mode="r"))
        case "npy":
            return np.load(artifact.path)
        case "hdf5":
            if artifact.key is None:
                raise ValueError(f"HDF5 artifact at {artifact.path} is missing a required key")
            with h5py.File(str(artifact.path), "r") as f:
                return np.asarray(f[artifact.key])
        case _:
            raise ValueError(
                f"Unsupported dataset artifact format {artifact.format!r} at {artifact.path}"
            )


def load_dense_training_arrays(dataset_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load dense rhs and solution arrays from a manifest-driven dataset directory."""
    artifacts = resolve_dataset_artifacts(dataset_dir)
    return _load_resolved(artifacts.rhs), _load_resolved(artifacts.solutions)


def load_matrix_dense_sample(dataset_dir: str | Path, sample_index: int = 0) -> np.ndarray:
    """Load one dense matrix sample from a manifest-driven dataset directory."""
    return _load_sample(resolve_dataset_artifacts(dataset_dir).matrix, sample_index)


def read_training_sample_count(paths: DatasetPaths | DatasetArtifacts) -> int:
    """Return the number of training samples without assuming a fixed layout."""
    if isinstance(paths, DatasetArtifacts):
        rhs_n = int(paths.rhs.shape[0])
        sol_n = int(paths.solutions.shape[0])
    else:
        artifacts = resolve_dataset_artifacts(paths.root)
        rhs_n = int(artifacts.rhs.shape[0])
        sol_n = int(artifacts.solutions.shape[0])
    if rhs_n != sol_n:
        raise ValueError(f"RHS and solutions sample counts must match, got {rhs_n} and {sol_n}")
    return rhs_n


def load_parameter_arrays(dataset_dir: str | Path) -> tuple[np.ndarray, ...]:
    """Load all manifest-declared parameter arrays eagerly."""
    artifacts = resolve_dataset_artifacts(dataset_dir)
    return tuple(_load_resolved(a) for a in artifacts.params)


def load_row_kind_codes(dataset_dir: str | Path) -> np.ndarray:
    """Load persisted compact row-kind semantic codes for a dataset."""
    artifact = resolve_dataset_artifacts(dataset_dir).row_kind
    if artifact is None:
        raise ValueError(
            f"Dataset '{dataset_dir}' does not expose row_kind metadata. Regenerate it first."
        )
    return _load_resolved_native(artifact).astype(np.uint8, copy=False).reshape(-1)


def load_optional_row_kind_codes(dataset_dir: str | Path) -> np.ndarray | None:
    """Load row-kind semantic codes when present, otherwise return None."""
    artifact = resolve_dataset_artifacts(dataset_dir).row_kind
    if artifact is None:
        return None
    return _load_resolved_native(artifact).astype(np.uint8, copy=False).reshape(-1)


def load_matrix_sample_index(dataset_dir: str | Path) -> np.ndarray:
    """Load persisted matrix-sample bindings for a dataset."""
    artifact = resolve_dataset_artifacts(dataset_dir).matrix_sample_index
    if artifact is None:
        raise ValueError(
            f"Dataset '{dataset_dir}' does not expose matrix_sample_index metadata. Regenerate it first."
        )
    return _load_resolved_native(artifact).astype(np.int64, copy=False).reshape(-1)


def list_available_matrix_indices(dataset_dir: str | Path) -> tuple[int, ...]:
    """List valid physical matrix indices for a dataset."""
    matrix = resolve_dataset_artifacts(dataset_dir).matrix
    if matrix.n_matrix_samples is not None:
        return tuple(range(int(matrix.n_matrix_samples)))
    if matrix.shape:
        return tuple(range(int(matrix.shape[0])))
    return ()


def load_standard_row_indices(
    dataset_dir: str | Path, require_standard: bool = True
) -> tuple[int, ...]:
    """List row indices that are STANDARD (safe for solver comparison)."""
    from neuralls.shared.enum_codecs import decode_row_kind_array
    from neuralls.shared.types import RowKind

    codes = load_row_kind_codes(dataset_dir)
    kinds = decode_row_kind_array(codes)
    if not require_standard:
        return tuple(range(len(kinds)))
    return tuple(index for index, kind in enumerate(kinds) if kind is RowKind.STANDARD)
