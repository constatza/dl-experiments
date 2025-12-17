"""System data loading utilities.

This module handles loading matrices, RHS vectors, and complete linear systems
from various file formats. All functions are I/O actions with clear contracts.

Functions in this module perform I/O operations and are not pure.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any
from collections.abc import Mapping

import numpy as np

from loguru import logger

from .io.base import load_npz_entry


def extract_model_name(model_config_path: str | Path) -> str:
    """Extract model name from config SESSION.name.

    I/O action - reads model config file.

    Args:
        model_config_path: Path to model config file

    Returns:
        Model name from SESSION.name, or config filename if not set

    Example:
        >>> extract_model_name("configs/ffnn.toml")
        'ffnn'
    """
    model_config_path = Path(model_config_path)

    with open(model_config_path, "rb") as f:
        config = tomllib.load(f)

    session = config.get("SESSION") or {}
    name = session.get("name")

    # If SESSION.name not set, derive from config filename
    if not isinstance(name, str) or not name:
        name = model_config_path.stem

    return name


def derive_checkpoint_path(
    model_template: str | Path,
    data_config: str | Path,
    output_root: str | Path,
) -> Path:
    """Derive checkpoint path from experiment configuration.

    Follows convention: {output_root}/{data_config_stem}/{model_name}/checkpoints/{model_name}.ckpt

    I/O action - reads model config to extract SESSION.name.

    Args:
        model_template: Path to model config (extracts SESSION.name)
        data_config: Path to data config (extracts stem)
        output_root: Base output directory

    Returns:
        Expected checkpoint path following convention

    Example:
        >>> derive_checkpoint_path(
        ...     "configs/ffnn.toml",
        ...     "data-configs/test-solutions.toml",
        ...     "/data/projects/graph-cg/data/output"
        ... )
        PosixPath('/data/projects/graph-cg/data/output/test-solutions/ffnn/checkpoints/ffnn.ckpt')
    """
    model_name = extract_model_name(model_template)
    data_config_name = Path(data_config).stem
    output_root_path = Path(output_root)

    checkpoint_dir = output_root_path / data_config_name / model_name / "checkpoints"
    return checkpoint_dir / f"{model_name}.ckpt"


def _load_matrix_file(matrix_path: str | Path) -> np.ndarray:
    """Load matrix from dataset directory (.npz format).

    I/O action - reads matrix from disk.

    Args:
        matrix_path: Path to dataset directory or matrix file

    Returns:
        Matrix array as float64
    """
    from .io_utils import load_dataset

    matrix_path = Path(matrix_path)

    # If it's a directory, load from normalized.npz
    if matrix_path.is_dir():
        data = load_dataset(matrix_path, variant="normalized")
        return data["matrix"]

    if matrix_path.suffix == ".npz":
        mat = load_npz_entry(matrix_path, "matrix")
        if mat.ndim > 2:
            logger.warning(
                f"{matrix_path} contains multiple matrices; using the first slice."
            )
            mat = mat[0]
        return np.asarray(mat, dtype=np.float64, copy=False)

    # If it's a file path like .../normalized.npz, load from parent dir
    if matrix_path.name == "normalized.npz":
        data = load_dataset(matrix_path.parent, variant="normalized")
        return data["matrix"]

    # Handle .npy binary format or text files
    if matrix_path.suffix == ".npy":
        return np.load(matrix_path).astype(np.float64, copy=False)
    else:
        return np.loadtxt(matrix_path, dtype=np.float64)


def _load_rhs_file(rhs_path: str | Path) -> np.ndarray:
    """Load RHS vector from dataset directory (.npz format).

    I/O action - reads RHS from disk.

    Args:
        rhs_path: Path to dataset directory or RHS file

    Returns:
        RHS vector as float64, reshaped to 1D
    """
    from .io_utils import load_dataset

    rhs_path = Path(rhs_path)

    def _validate_and_flatten(rhs_array: np.ndarray) -> np.ndarray:
        if rhs_array.ndim == 1:
            return rhs_array
        if rhs_array.ndim == 2:
            if rhs_array.shape[1] == 1 or rhs_array.shape[0] == 1:
                return rhs_array.reshape(-1)
            raise ValueError(
                f"RHS array must be 1D or a single column/row vector, got shape {rhs_array.shape}"
            )
        raise ValueError(f"RHS array must be 1D or 2D, got shape {rhs_array.shape}")

    # If it's a directory, load from normalized.npz
    if rhs_path.is_dir():
        data = load_dataset(rhs_path, variant="normalized")
        b = data["rhs"][0]  # First sample as mother RHS
        return _validate_and_flatten(b)

    if rhs_path.suffix == ".npz":
        b = load_npz_entry(rhs_path, "rhs")
        if b.ndim > 1 and b.shape[0] > 1:
            logger.warning(
                f"{rhs_path} contains multiple RHS entries; using the first."
            )
            b = b[0]
        return _validate_and_flatten(b)

    # If it's a file path like .../normalized.npz, load from parent dir
    if rhs_path.name == "normalized.npz":
        data = load_dataset(rhs_path.parent, variant="normalized")
        b = data["rhs"][0]
        return _validate_and_flatten(b)

    # Handle .npy binary format or text files
    if rhs_path.suffix == ".npy":
        b = np.load(rhs_path).astype(np.float64, copy=False)
    else:
        b = np.loadtxt(rhs_path, dtype=np.float64)
    return _validate_and_flatten(b)


def _ensure_rhs_matches_matrix(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Ensure RHS dimension matches matrix, creating fallback if needed.

    Pure function - validates and potentially creates fallback.

    Args:
        A: System matrix
        b: RHS vector

    Returns:
        RHS vector with correct dimension
    """
    if b.shape[0] != A.shape[0]:
        hint = ""
        if b.size == A.size:
            hint = " (loaded data looks like a flattened matrix; check your RHS path)"
        raise ValueError(
            f"RHS length {b.shape[0]} doesn't match matrix size {A.shape[0]}{hint}"
        )
    return b


def load_system_data(
    matrix_path: str | Path, rhs_path: str | Path | None = None
) -> tuple[np.ndarray, np.ndarray | None]:
    """Load matrix and RHS data.

    Orchestrates loading and validation of system components.

    I/O orchestrator - coordinates file loading.

    Args:
        matrix_path: Path to matrix file
        rhs_path: Path to RHS file (optional)

    Returns:
        Tuple of (matrix, rhs) arrays
    """
    A = _load_matrix_file(matrix_path)

    rhs_path = Path(rhs_path) if rhs_path is not None else None
    if rhs_path is None:
        return A, None

    b = _load_rhs_file(rhs_path)
    b = _ensure_rhs_matches_matrix(A, b)

    return A, b


def save_training_data(
    features: np.ndarray,
    targets: np.ndarray,
    features_path: str | Path,
    targets_path: str | Path,
) -> None:
    """Save training data arrays.

    I/O action - writes arrays to disk.

    Args:
        features: Feature array
        targets: Target array
        features_path: Output path for features
        targets_path: Output path for targets
    """
    from .file_operations import ensure_dir

    features_path = Path(features_path)
    targets_path = Path(targets_path)

    ensure_dir(features_path.parent)
    ensure_dir(targets_path.parent)

    # Ensure float64 dtype for all saved data
    np.save(features_path, features.astype(np.float64, copy=False))
    np.save(targets_path, targets.astype(np.float64, copy=False))


def get_latest_checkpoint(
    checkpoint_dir: str | Path, pattern: str = "*.ckpt"
) -> Path | None:
    """Find the most recent checkpoint file.

    I/O action - scans directory for checkpoint files.

    Args:
        checkpoint_dir: Directory containing checkpoints
        pattern: File pattern to match

    Returns:
        Path to latest checkpoint or None
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    checkpoints = list(checkpoint_dir.glob(pattern))
    if not checkpoints:
        return None

    return max(checkpoints, key=lambda p: p.stat().st_mtime)


def load_case_data(
    data_dir: str | Path,
    variant: str = "normalized",
) -> Mapping[str, np.ndarray | dict[str, Any]]:
    """Load all data from a case directory.

    I/O action - loads complete dataset from directory.

    Args:
        data_dir: Path to case directory (e.g., collect-504-norm)
        variant: Dataset variant to load ('normalized', 'raw', or 'test')

    Returns:
        Dictionary with keys: matrix, rhs, solutions, metadata

    Example:
        >>> data = load_case_data("/data/projects/graph-cg/data/processed/collect-504-norm")
        >>> A = data["matrix"]
        >>> R = data["rhs"]
        >>> X = data["solutions"]
    """
    from .io_utils import load_dataset

    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Load dataset using new .npz format
    result = load_dataset(data_dir, variant=variant)

    # Load metadata
    metadata_path = data_dir / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            result["metadata"] = json.load(f)

    return result


def list_available_cases(
    processed_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """List all available data cases in processed directory.

    I/O action - scans directory for case metadata.

    Args:
        processed_dir: Path to processed data directory (defaults to DEFAULT_PROCESSED_DIR)

    Returns:
        List of dictionaries with case information (name, type, dimension, etc.)

    Example:
        >>> cases = list_available_cases()
        >>> for case in cases:
        ...     print(f"{case['name']}: {case['dimension']}-dim, {case['source_type']}")
    """
    from .constants import DEFAULT_PROCESSED_DIR

    if processed_dir is None:
        processed_dir = Path(DEFAULT_PROCESSED_DIR)
    else:
        processed_dir = Path(processed_dir)

    if not processed_dir.exists():
        return []

    cases = []
    for case_dir in processed_dir.iterdir():
        if not case_dir.is_dir():
            continue

        # Check if it has metadata
        metadata_path = case_dir / "metadata.json"
        if not metadata_path.exists():
            continue

        try:
            with metadata_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)

            case_info = {
                "name": case_dir.name,
                "path": str(case_dir),
                "source_type": metadata.get("source_type", "unknown"),
                "dimension": metadata.get("dimension"),
                "num_samples": metadata.get("num_samples"),
                "normalized": metadata.get("normalized", False),
            }

            cases.append(case_info)
        except (json.JSONDecodeError, OSError):
            continue

    return sorted(cases, key=lambda x: (x["source_type"], x["dimension"] or 0))
