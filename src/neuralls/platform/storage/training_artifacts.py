"""Training artifact IO utilities."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from neuralls.platform.storage.dataset_readers import (
    load_dense_training_arrays,
    read_training_sample_count,
    resolve_dataset_artifacts,
)


@dataclass(frozen=True)
class InMemoryArraySource:
    """Array source carried directly in memory."""

    array: np.ndarray


@dataclass(frozen=True)
class ZarrArraySource:
    """Array source referenced by a zarr path."""

    path: Path


type ArraySource = InMemoryArraySource | ZarrArraySource


@dataclass(frozen=True)
class TrainingArrays:
    """Format-neutral training dataset artifacts and loaded dense tensors."""

    rhs: np.ndarray
    solutions: np.ndarray
    matrix_source: ArraySource
    sample_count: int
    parameter_sources: tuple[ArraySource, ...] = ()

    @property
    def matrix_zarr(self) -> Path:
        """Compatibility view for zarr-backed matrix sources."""
        match self.matrix_source:
            case ZarrArraySource(path=path):
                return path
            case _:
                raise AttributeError("matrix_zarr is only available for zarr-backed datasets.")

    @property
    def parameters_zarr(self) -> tuple[Path, ...]:
        """Compatibility view for zarr-backed parameter sources."""
        paths: list[Path] = []
        for source in self.parameter_sources:
            match source:
                case ZarrArraySource(path=path):
                    paths.append(path)
                case _:
                    raise AttributeError(
                        "parameters_zarr is only available for zarr-backed datasets."
                    )
        return tuple(paths)


def _source_from_artifact(path: Path, format_name: str) -> ArraySource:
    if format_name == "zarr":
        return ZarrArraySource(path=path)
    if format_name == "npy":
        return InMemoryArraySource(array=np.load(path).astype(np.float64, copy=False))
    raise ValueError(f"Unsupported training array format {format_name!r} at {path}")


def load_training_arrays(data_dir: Path) -> TrainingArrays:
    """Resolve training dataset arrays from a manifest-driven dataset directory."""
    artifacts = resolve_dataset_artifacts(data_dir)
    sample_count = read_training_sample_count(artifacts)
    rhs, solutions = load_dense_training_arrays(data_dir)
    return TrainingArrays(
        rhs=rhs,
        solutions=solutions,
        matrix_source=_source_from_artifact(artifacts.matrix.path, artifacts.matrix.format),
        sample_count=sample_count,
        parameter_sources=tuple(
            _source_from_artifact(artifact.path, artifact.format) for artifact in artifacts.params
        ),
    )


def load_array_source_sample(source: ArraySource, sample_index: int) -> np.ndarray:
    """Load one sample from a tagged array source."""
    match source:
        case ZarrArraySource(path=path):
            arr = zarr.open_array(str(path), mode="r")
            data = arr[sample_index] if arr.ndim > 1 else arr[:]
            return np.asarray(data, dtype=np.float64)
        case InMemoryArraySource(array=array):
            if array.ndim > 1:
                return np.asarray(array[sample_index], dtype=np.float64)
            return np.asarray(array, dtype=np.float64)


def coerce_jsonable(value: Any) -> Any:
    """Convert runtime values to JSON-compatible primitives."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): coerce_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [coerce_jsonable(item) for item in value]
    return value


def flatten_numpy_payload(
    payload: Any,
    prefix: str = "",
) -> dict[str, np.ndarray]:
    """Flatten nested numpy payloads into a stable dict of arrays."""
    if isinstance(payload, Mapping):
        flattened: dict[str, np.ndarray] = {}
        for key, value in payload.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(flatten_numpy_payload(value, next_prefix))
        return flattened

    array = np.asarray(payload)
    if array.size == 0:
        return {}
    key = prefix or "value"
    return {key: array}


def save_training_predictions(
    training_result: Any,
    predictions_dir: Path,
    *,
    numpy_payload: Mapping[str, Any] | None = None,
) -> None:
    """Persist training predictions/targets captured by dlkit."""
    if numpy_payload is not None:
        all_numpy = numpy_payload
    else:
        to_numpy = getattr(training_result, "to_numpy", None)
        if not callable(to_numpy):
            return
        all_numpy = to_numpy()
    if not isinstance(all_numpy, Mapping):
        return

    flattened = flatten_numpy_payload(all_numpy)
    if not flattened:
        return

    predictions_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, str] = {}
    for index, (key, value) in enumerate(sorted(flattened.items())):
        filename = f"{index:02d}_{key.replace('.', '__')}.npy"
        np.save(predictions_dir / filename, value)
        manifest[key] = filename

    (predictions_dir / "training_predictions_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
