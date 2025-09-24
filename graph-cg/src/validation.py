"""Input validation utilities."""

from __future__ import annotations
from pathlib import Path
import numpy as np


def validate_matrix(A: np.ndarray) -> None:
    """Validate system matrix.

    Args:
        A: Input matrix

    Raises:
        ValueError: If matrix is invalid
    """
    if A.ndim != 2:
        raise ValueError(f"Matrix must be 2D, got {A.ndim}D")

    if A.shape[0] != A.shape[1]:
        raise ValueError(f"Matrix must be square, got shape {A.shape}")

    if not np.isfinite(A).all():
        raise ValueError("Matrix contains non-finite values")


def validate_rhs(b: np.ndarray, A: np.ndarray | None = None) -> None:
    """Validate RHS vector.

    Args:
        b: RHS vector
        A: Optional system matrix for size checking

    Raises:
        ValueError: If RHS is invalid
    """
    if b.ndim > 2:
        raise ValueError(f"RHS must be 1D or 2D, got {b.ndim}D")

    if b.ndim == 2 and b.shape[1] != 1:
        raise ValueError(f"RHS must be a column vector, got shape {b.shape}")

    if not np.isfinite(b).all():
        raise ValueError("RHS contains non-finite values")

    if A is not None and len(b.flatten()) != A.shape[0]:
        raise ValueError(
            f"RHS length {len(b.flatten())} doesn't match matrix size {A.shape[0]}"
        )


def validate_config(config: dict) -> None:
    """Validate configuration dictionary.

    Args:
        config: Configuration dictionary

    Raises:
        ValueError: If config is invalid
    """
    required_sections = ["MODEL", "TRAINING", "DATASET"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Config missing required section: {section}")

    # Validate model section
    model = config["MODEL"]
    if "name" not in model:
        raise ValueError("MODEL section missing 'name' field")

    # Validate dataset section
    dataset = config["DATASET"]
    if "name" not in dataset:
        raise ValueError("DATASET section missing 'name' field")


def validate_file_exists(path: str | Path, description: str = "File") -> Path:
    """Validate that a file exists.

    Args:
        path: File path
        description: Description of the file for error messages

    Returns:
        Validated Path object

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    path = Path(path)
    if not path.exists():
        FileNotFoundError(f"{description} not found: {path}")
    return path


def validate_directory_writable(
    path: str | Path, description: str = "Directory"
) -> Path:
    """Validate that a directory exists and is writable.

    Args:
        path: Directory path
        description: Description for error messages

    Returns:
        Validated Path object

    Raises:
        ValueError: If directory is not writable
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    if not path.is_dir():
        raise ValueError(f"{description} is not a directory: {path}")

    # Test if we can write to the directory
    test_file = path / ".write_test"
    try:
        test_file.touch()
        test_file.unlink()
    except (PermissionError, OSError) as e:
        raise ValueError(f"{description} is not writable: {path}") from e

    return path


def validate_solver_params(tol: float, max_iter: int, stopping_criterion: str) -> None:
    """Validate CG solver parameters.

    Args:
        tol: Tolerance
        max_iter: Maximum iterations
        stopping_criterion: Stopping criterion

    Raises:
        ValueError: If parameters are invalid
    """
    if tol <= 0:
        raise ValueError(f"Tolerance must be positive, got {tol}")

    if max_iter <= 0:
        raise ValueError(f"Max iterations must be positive, got {max_iter}")

    valid_criteria = ["tolerance", "fixed_iterations"]
    if stopping_criterion not in valid_criteria:
        raise ValueError(
            f"Stopping criterion must be one of {valid_criteria}, got {stopping_criterion}"
        )


def validate_noise_params(
    strategy: str, rho: float, dim_idx: int | None = None
) -> None:
    """Validate noise generation parameters.

    Args:
        strategy: Noise strategy name
        rho: Noise parameter
        dim_idx: Dimension index for single_dim strategy

    Raises:
        ValueError: If parameters are invalid
    """
    valid_strategies = [
        "none",
        "global",
        "single_dim",
        "blockwise",
        "worst_case",
        "load_redistribution",
        "missing_data",
        "corrupted_data",
        "extreme_magnitude",
    ]

    if strategy not in valid_strategies:
        raise ValueError(f"Strategy must be one of {valid_strategies}, got {strategy}")

    if strategy != "none" and rho < 0:
        raise ValueError(f"Noise parameter rho must be non-negative, got {rho}")

    if strategy == "single_dim" and dim_idx is not None and dim_idx < 0:
        raise ValueError(f"Dimension index must be non-negative, got {dim_idx}")
