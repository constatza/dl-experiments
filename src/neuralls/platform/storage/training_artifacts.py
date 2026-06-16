"""Training artifact IO utilities.

Pure functions for serializing, flattening, and persisting training
artifacts (predictions, metrics, numpy arrays) to disk.
No training orchestration logic — decoupled from DLKit and MLflow.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from neuralls.platform.storage.datasets import read_training_sample_count, resolve_dataset_paths
from neuralls.shared.constants import PARAMETERS_ZARR_PREFIX


@dataclass(frozen=True)
class TrainingArrays:
    """Training data artifact paths from dataset storage.

    Attributes:
        rhs: Path to rhs.zarr directory.
        solutions: Path to solutions.zarr directory.
        matrix_zarr: Path to matrix.zarr directory.
        sample_count: Number of samples in rhs/solutions.
        parameters_zarr: Paths to parameters_0.zarr, parameters_1.zarr, … (may be empty).
    """

    rhs: Path
    solutions: Path
    matrix_zarr: Path
    sample_count: int
    parameters_zarr: tuple[Path, ...] = ()


def load_training_arrays(data_dir: Path) -> TrainingArrays:
    """Resolve training artifact paths from dataset directory.

    Args:
        data_dir: Directory containing training dataset files.

    Returns:
        ``TrainingArrays`` with resolved artifact paths and sample count.

    Raises:
        ValueError: If rhs and solutions have mismatched sample counts.
    """
    paths = resolve_dataset_paths(data_dir)
    sample_count = read_training_sample_count(paths)
    parameters_zarr = tuple(sorted(data_dir.glob(f"{PARAMETERS_ZARR_PREFIX}*.zarr")))
    return TrainingArrays(
        rhs=paths.rhs_path,
        solutions=paths.solutions_path,
        matrix_zarr=paths.matrix_zarr_dir,
        sample_count=sample_count,
        parameters_zarr=parameters_zarr,
    )


def coerce_jsonable(value: Any) -> Any:
    """Convert runtime values to JSON-compatible primitives.

    Args:
        value: Any Python value.

    Returns:
        JSON-serializable equivalent.
    """
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
    """Flatten nested numpy payloads into a stable dict of arrays.

    Args:
        payload: Nested mapping or array-like.
        prefix: Key prefix for nested keys (used in recursion).

    Returns:
        Flat ``{key: ndarray}`` dict.
    """
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
    """Persist training predictions/targets captured by dlkit.

    Args:
        training_result: DLKit training result with ``to_numpy()`` method.
        predictions_dir: Directory to write ``.npy`` files and manifest.
        numpy_payload: Optional pre-normalized payload to persist.
    """
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
