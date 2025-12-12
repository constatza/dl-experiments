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


def validate_rhs_vector(b: np.ndarray | None, A: np.ndarray | None = None) -> None:
    """Validate RHS vector dimensions and finiteness.

    Args:
        b: RHS vector
        A: Optional system matrix for size checking

    Raises:
        ValueError: If RHS is invalid
    """
    if b is None:
        return

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


# Backward compatibility
def validate_rhs(b: np.ndarray | None, A: np.ndarray | None = None) -> None:
    """Validate RHS vector; prefer direct use of validate_rhs_vector."""
    validate_rhs_vector(b, A)


def validate_data_exists(
    data_dir: Path | str,
    required_files: list[str],
) -> None:
    """Validate that required data files exist in a directory.

    This is an action function (performs I/O: file system checks).

    Args:
        data_dir: Directory to check for files.
        required_files: List of filenames that must exist (e.g.,
            ["rhs-samples.npy", "sol-samples.npy"]).

    Raises:
        FileNotFoundError: If any required file is missing, with a descriptive
            error message listing all missing file paths.

    Example:
        >>> validate_data_exists(
        ...     Path("/data/projects/graph-cg/data/processed/generate-90-norm"),
        ...     ["rhs-samples.npy", "sol-samples.npy"],
        ... )
        # Raises FileNotFoundError if any file missing

    Notes:
        - This function has side effects (file system access).
        - Use tmp_path fixture in tests (never tempfile module).
    """
    data_dir = Path(data_dir)
    missing_files = []

    for filename in required_files:
        filepath = data_dir / filename
        if not filepath.exists():
            missing_files.append(str(filepath))

    if missing_files:
        files_str = "\n  - ".join(missing_files)
        raise FileNotFoundError(
            f"Required data files not found in {data_dir}:\n  - {files_str}"
        )


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
        raise FileNotFoundError(f"{description} not found: {path}")
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


def validate_solver_params(
    rtol: float, atol: float, max_iter: int, stopping_criterion: str
) -> None:
    """Validate CG solver parameters.

    Args:
        rtol: Relative tolerance
        atol: Absolute tolerance
        max_iter: Maximum iterations
        stopping_criterion: Stopping criterion

    Raises:
        ValueError: If parameters are invalid
    """
    if rtol <= 0:
        raise ValueError(f"Relative tolerance must be positive, got {rtol}")

    if atol <= 0:
        raise ValueError(f"Absolute tolerance must be positive, got {atol}")

    if max_iter <= 0:
        raise ValueError(f"Max iterations must be positive, got {max_iter}")

    valid_criteria = ["tolerance", "fixed_iterations", "residual_norm"]
    if stopping_criterion not in valid_criteria:
        raise ValueError(
            f"Stopping criterion must be one of {valid_criteria}, got {stopping_criterion}"
        )
