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

from neuralls.platform.storage.datasets import load_dense_training_arrays, resolve_dataset_paths


@dataclass(frozen=True)
class TrainingArrays:
    """Training data artifact paths from dataset storage.

    Attributes:
        rhs: Path to rhs.npy
        solutions: Path to solutions.npy
        matrix_pack: Path to matrix_coo sparse pack directory
        sample_count: Number of samples in rhs/solutions
    """

    rhs: Path
    solutions: Path
    matrix_pack: Path
    sample_count: int


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
    rhs, solutions = load_dense_training_arrays(data_dir)
    if rhs.shape[0] != solutions.shape[0]:
        raise ValueError(
            f"RHS and solutions sample counts must match, got {rhs.shape[0]} and {solutions.shape[0]}"
        )
    return TrainingArrays(
        rhs=paths.rhs_path,
        solutions=paths.solutions_path,
        matrix_pack=paths.matrix_pack_dir,
        sample_count=int(rhs.shape[0]),
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
) -> None:
    """Persist training predictions/targets captured by dlkit.

    Args:
        training_result: DLKit training result with ``to_numpy()`` method.
        predictions_dir: Directory to write ``.npy`` files and manifest.
    """
    all_numpy = training_result.to_numpy()
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
