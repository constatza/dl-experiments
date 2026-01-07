"""Dataset loading and tracking utilities.

This module handles dataset I/O operations including loading from .npz files,
tracking dataset files, and managing dataset variants.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any

import numpy as np
from dlkit.tools.io import load_array


def load_numpy_array(path: str | Path) -> np.ndarray:
    """Load numpy array from file, supporting both .npy and text formats.

    I/O action - reads array from disk.

    Args:
        path: Path to array file

    Returns:
        Loaded numpy array as float64
    """
    path = Path(path)

    if path.suffix == ".npy":
        arr = load_array(path).numpy()
    else:
        # Assume text format
        arr = np.loadtxt(path, dtype=np.float64)
    if arr.dtype != np.float64:
        raise ValueError(f"Expected float64 array at {path}, got {arr.dtype}")
    return arr


def load_dataset(
    dataset_dir: str | Path,
    variant: str = "normalized",
) -> dict[str, np.ndarray]:
    """Load dataset from .npz file.

    I/O action - reads dataset from disk.

    Args:
        dataset_dir: Directory containing dataset files
        variant: Dataset variant to load ('normalized', 'raw', or 'comparison')

    Returns:
        Dictionary with keys: 'matrix', 'rhs', 'solutions'

    Raises:
        FileNotFoundError: If requested .npz file doesn't exist
        ValueError: If .npz file missing required arrays
    """
    dataset_dir = Path(dataset_dir)
    filename = f"{variant}.npz"
    filepath = dataset_dir / filename

    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset file '{filename}' not found in {dataset_dir}. "
            f"Available variants: {list_available_variants(dataset_dir)}"
        )

    # Load .npz file
    data = np.load(filepath)

    # Validate required arrays present
    required = ["matrix", "rhs", "solutions"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(
            f"Dataset file '{filename}' missing required arrays: {missing}. "
            f"Found: {list(data.keys())}"
        )

    # Return as dictionary
    matrix = data["matrix"]
    rhs = data["rhs"]
    solutions = data["solutions"]

    for name, arr in (("matrix", matrix), ("rhs", rhs), ("solutions", solutions)):
        if arr.dtype != np.float64:
            raise ValueError(
                f"{filename}:{name} dtype must be float64, got {arr.dtype}"
            )

    return {"matrix": matrix, "rhs": rhs, "solutions": solutions}


def has_comparison_split(dataset_dir: str | Path) -> bool:
    """Check if dataset has a dedicated comparison split.

    I/O action - checks filesystem.

    Args:
        dataset_dir: Directory containing dataset files

    Returns:
        True if comparison.npz exists, False otherwise
    """
    return (Path(dataset_dir) / "comparison.npz").exists()


def list_available_variants(dataset_dir: str | Path) -> list[str]:
    """List available dataset variants in directory.

    I/O action - scans directory.

    Args:
        dataset_dir: Directory containing dataset files

    Returns:
        List of available variant names (e.g., ['normalized', 'raw', 'comparison'])
    """
    dataset_dir = Path(dataset_dir)
    variants = []
    for variant in ["normalized", "raw", "comparison"]:
        if (dataset_dir / f"{variant}.npz").exists():
            variants.append(variant)
    return variants


def save_numpy_array(array: np.ndarray, path: str | Path) -> None:
    """Save numpy array to file.

    I/O action - writes array to disk.

    Args:
        array: Array to save
        path: Output path (.npy format)
    """
    from neuralls.io.filesystem import ensure_dir

    path = Path(path)
    ensure_dir(path.parent)
    np.save(path, array)


class DatasetFileTracker:
    """Track file operations for logging and debugging."""

    def __init__(self):
        """Initialize file tracker."""
        self.operations: list[dict[str, Any]] = []

    def track_read(self, path: str | Path, size: int | None = None) -> None:
        """Track file read operation.

        Args:
            path: File path that was read
            size: File size in bytes (optional)
        """
        from neuralls.io.filesystem import get_file_size

        self.operations.append(
            {
                "operation": "read",
                "path": str(path),
                "size": size or get_file_size(path),
            }
        )

    def track_write(self, path: str | Path, size: int | None = None) -> None:
        """Track file write operation.

        Args:
            path: File path that was written
            size: File size in bytes (optional)
        """
        from neuralls.io.filesystem import get_file_size

        self.operations.append(
            {
                "operation": "write",
                "path": str(path),
                "size": size or get_file_size(path),
            }
        )

    def get_summary(self) -> dict[str, Any]:
        """Get summary of file operations.

        Returns:
            Dictionary with operation statistics
        """
        from neuralls.io.filesystem import format_file_size

        reads = [op for op in self.operations if op["operation"] == "read"]
        writes = [op for op in self.operations if op["operation"] == "write"]

        return {
            "total_operations": len(self.operations),
            "reads": len(reads),
            "writes": len(writes),
            "total_read_size": sum(op.get("size", 0) for op in reads),
            "total_write_size": sum(op.get("size", 0) for op in writes),
        }

    def print_summary(self) -> None:
        """Print file operations summary to stdout."""
        from neuralls.io.filesystem import format_file_size

        summary = self.get_summary()
        print("File Operations Summary:")
        print(f"  Total operations: {summary['total_operations']}")
        print(
            f"  Reads: {summary['reads']} ({format_file_size(summary['total_read_size'])})"
        )
        print(
            f"  Writes: {summary['writes']} ({format_file_size(summary['total_write_size'])})"
        )


# Global file tracker instance
file_tracker = DatasetFileTracker()


def track_files(enabled: bool = True) -> DatasetFileTracker:
    """Get file tracker instance.

    Args:
        enabled: Whether to enable tracking (currently unused)

    Returns:
        Global file tracker instance
    """
    return file_tracker
