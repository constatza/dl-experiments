"""Manifest-driven dataset readers and artifact resolution helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import zarr

from neuralls.platform.storage.manifest import DatasetArtifact, read_dataset_manifest


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


@dataclass(frozen=True)
class DatasetArtifacts:
    """Explicit resolved dataset contract used by storage readers."""

    root: Path
    manifest_path: Path
    matrix: ResolvedDatasetArtifact
    rhs: ResolvedDatasetArtifact
    solutions: ResolvedDatasetArtifact
    params: tuple[ResolvedDatasetArtifact, ...] = ()


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
        manifest_path=root / "manifest.json",
        matrix=_resolve_artifact(root, manifest.matrix),
        rhs=_resolve_artifact(root, manifest.rhs),
        solutions=_resolve_artifact(root, manifest.solutions),
        params=tuple(_resolve_artifact(root, artifact) for artifact in manifest.params),
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


def _load_resolved(artifact: ResolvedDatasetArtifact) -> np.ndarray:
    match artifact.format:
        case "zarr":
            return np.asarray(zarr.open_array(str(artifact.path), mode="r"), dtype=np.float64)
        case "npy":
            return np.load(artifact.path).astype(np.float64, copy=False)
        case "hdf5":
            with h5py.File(str(artifact.path), "r") as f:
                return np.asarray(f[artifact.key or "data"], dtype=np.float64)
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
    artifacts = resolve_dataset_artifacts(dataset_dir)
    matrix = _load_resolved(artifacts.matrix)
    return np.asarray(matrix[sample_index], dtype=np.float64)


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
