"""Common utilities for graph-cg scripts."""

from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np

from dlkit import GeneralSettings
from dlkit.tools.io import load_array, load_sections_config
from dlkit.tools.config.paths_settings import PathsSettings
from dlkit.tools.config.dataset_settings import DatasetSettings
from .validation import validate_file_exists


def load_config(config_path: str | Path) -> GeneralSettings:
    """Load configuration from file.

    Args:
        config_path: Path to config file

    Returns:
        Loaded GeneralSettings
    """
    config_path = validate_file_exists(config_path, "Config file")
    return GeneralSettings.from_toml_file(str(config_path))


def get_paths_from_config(settings: GeneralSettings) -> dict[str, Any]:
    """Extract all relevant paths from config.

    Args:
        settings: Loaded GeneralSettings

    Returns:
        Dictionary with paths
    """
    paths = {}

    # Basic paths from PATHS section
    if settings.PATHS:
        paths.update(
            {
                "matrix_path": getattr(settings.PATHS, "matrix_path", None),
                "rhs_path": getattr(settings.PATHS, "rhs_path", None),
                "output_dir": getattr(settings.PATHS, "output_dir", "./output"),
            }
        )

    # Checkpoint path (MODEL.checkpoint takes priority)
    checkpoint_path = None
    if hasattr(settings, "MODEL") and hasattr(settings.MODEL, "checkpoint"):
        checkpoint_path = getattr(settings.MODEL, "checkpoint", None)
    if (
        checkpoint_path is None
        and settings.PATHS
        and hasattr(settings.PATHS, "checkpoint_path")
    ):
        checkpoint_path = getattr(settings.PATHS, "checkpoint_path", None)
    paths["checkpoint_path"] = checkpoint_path

    # Dataset paths
    if settings.DATASET:
        if settings.DATASET.features:
            paths["features_path"] = settings.DATASET.features[0].path
        if settings.DATASET.targets:
            paths["targets_path"] = settings.DATASET.targets[0].path

    return paths


def get_paths_from_partial_config(config_path: str | Path) -> dict[str, Any]:
    """Extract paths needed for data generation using section loading.

    Args:
        config_path: Path to config file

    Returns:
        Dictionary with paths needed for data generation
    """
    config_path = validate_file_exists(config_path, "Config file")

    # Load only the sections we need - returns pydantic models directly
    sections = load_sections_config(
        str(config_path), {"PATHS": PathsSettings, "DATASET": DatasetSettings}
    )
    paths_cfg = sections["PATHS"]
    dataset_cfg = sections["DATASET"]

    paths = {
        "matrix_path": paths_cfg.matrix_path,
        "rhs_path": paths_cfg.rhs_path,
        "output_dir": paths_cfg.output_dir,
        "features_path": dataset_cfg.features[0].path if dataset_cfg.features else None,
        "targets_path": dataset_cfg.targets[0].path if dataset_cfg.targets else None,
    }

    return paths


def get_solver_params(settings: GeneralSettings) -> dict[str, Any]:
    """Extract solver parameters from config.

    Args:
        settings: Loaded GeneralSettings

    Returns:
        Dictionary with solver parameters
    """
    extras = settings.EXTRAS
    solver_cfg = {}

    if extras is not None:
        extras_dict = extras.model_dump()
        raw_solver = extras_dict.get("solver", {})
        if isinstance(raw_solver, dict):
            solver_cfg = raw_solver

    # Extract with defaults and type conversion
    tolerance = solver_cfg.get("tolerance", 1e-8)
    max_iterations = solver_cfg.get("max_iterations", 30)
    normalize_system = solver_cfg.get("normalize_system", True)
    stopping_criterion = solver_cfg.get("stopping_criterion", "tolerance")

    try:
        tolerance = float(tolerance)
    except (TypeError, ValueError):
        tolerance = 1e-8

    try:
        max_iterations = int(max_iterations)
    except (TypeError, ValueError):
        max_iterations = 30

    if isinstance(normalize_system, str):
        normalize_system = normalize_system.lower() in {"1", "true", "yes", "on"}

    return {
        "tolerance": tolerance,
        "max_iterations": max_iterations,
        "normalize_system": bool(normalize_system),
        "stopping_criterion": str(stopping_criterion),
    }


def ensure_dir(path: str | Path) -> Path:
    """Ensure directory exists, creating it if necessary."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_system_data(
    matrix_path: str | Path, rhs_path: str | Path
) -> tuple[np.ndarray, np.ndarray]:
    """Load matrix and RHS data.

    Args:
        matrix_path: Path to matrix file
        rhs_path: Path to RHS file

    Returns:
        Tuple of (matrix, rhs) arrays
    """
    # Load matrix
    matrix_path = validate_file_exists(matrix_path, "Matrix file")
    if Path(matrix_path).suffix == ".npy":
        A = load_array(matrix_path).numpy()
    else:
        A = np.loadtxt(matrix_path)

    # Load RHS
    rhs_path = validate_file_exists(rhs_path, "RHS file")
    if Path(rhs_path).suffix == ".npy":
        b = load_array(rhs_path).numpy()
    else:
        b = np.loadtxt(rhs_path)

    # Ensure b is 1D
    if b.ndim > 1:
        b = b.reshape(-1)

    if b.shape[0] != A.shape[0]:
        print(
            f"RHS length {b.shape[0]} doesn't match matrix size {A.shape[0]}. Setting RHS to ones."
        )
        b = np.ones(A.shape[0])

    return A, b


def save_training_data(
    features: np.ndarray,
    targets: np.ndarray,
    features_path: str | Path,
    targets_path: str | Path,
) -> None:
    """Save training data arrays.

    Args:
        features: Feature array
        targets: Target array
        features_path: Output path for features
        targets_path: Output path for targets
    """
    features_path = Path(features_path)
    targets_path = Path(targets_path)

    ensure_dir(features_path.parent)
    ensure_dir(targets_path.parent)

    np.save(features_path, features)
    np.save(targets_path, targets)


def get_latest_checkpoint(
    checkpoint_dir: str | Path, pattern: str = "*.ckpt"
) -> Path | None:
    """Find the most recent checkpoint file.

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
