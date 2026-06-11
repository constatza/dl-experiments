"""Dataset loading and tracking utilities."""

from __future__ import annotations
from pathlib import Path
from typing import Any

import numpy as np
from dlkit.io import load_array

from neuralls.platform.storage.datasets import (
    load_dataset_manifest,
    load_dense_training_arrays,
    load_matrix_dense_sample,
    resolve_dataset_paths,
)


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
    variant: str = "dataset",
) -> dict[str, np.ndarray]:
    """Load dataset from split-array sparse storage.

    I/O action - reads dataset from disk.

    Args:
        dataset_dir: Directory containing dataset files
        variant: Dataset variant to load ('dataset' only)

    Returns:
        Dictionary with keys: 'matrix', 'rhs', 'solutions'

    Raises:
        FileNotFoundError: If required dataset artifacts are missing
        ValueError: If variant is not supported
    """
    if variant != "dataset":
        raise ValueError(f"Unsupported dataset variant '{variant}'. Only 'dataset' is supported.")

    dataset_dir = Path(dataset_dir)
    load_dataset_manifest(dataset_dir)
    rhs, solutions = load_dense_training_arrays(dataset_dir)
    matrix = load_matrix_dense_sample(dataset_dir, sample_index=0)

    for name, arr in (("rhs", rhs), ("solutions", solutions), ("matrix", matrix)):
        if arr.dtype != np.float64:
            raise ValueError(f"{name} dtype must be float64, got {arr.dtype}")

    return {"matrix": matrix, "rhs": rhs, "solutions": solutions}


def has_comparison_split(dataset_dir: str | Path) -> bool:
    """Check if dataset has a dedicated comparison split.

    I/O action - checks filesystem.

    Args:
        dataset_dir: Directory containing dataset files

    Returns:
        True if a dedicated comparison split exists, False otherwise
    """
    return False


def list_available_variants(dataset_dir: str | Path) -> list[str]:
    """List available dataset variants in directory.

    I/O action - scans directory.

    Args:
        dataset_dir: Directory containing dataset files

    Returns:
        List of available variant names.
    """
    dataset_dir = Path(dataset_dir)
    paths = resolve_dataset_paths(dataset_dir)
    if (
        paths.manifest_path.exists()
        and paths.rhs_path.exists()
        and paths.solutions_path.exists()
        and paths.matrix_zarr_dir.exists()
    ):
        return ["dataset"]
    return []


def save_numpy_array(array: np.ndarray, path: str | Path) -> None:
    """Save numpy array to file.

    I/O action - writes array to disk.

    Args:
        array: Array to save
        path: Output path (.npy format)
    """
    from neuralls.platform.storage.filesystem import ensure_dir

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
        from neuralls.platform.storage.filesystem import get_file_size

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
        from neuralls.platform.storage.filesystem import get_file_size

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
        from neuralls.platform.storage.filesystem import format_file_size

        summary = self.get_summary()
        print("File Operations Summary:")
        print(f"  Total operations: {summary['total_operations']}")
        print(f"  Reads: {summary['reads']} ({format_file_size(summary['total_read_size'])})")
        print(f"  Writes: {summary['writes']} ({format_file_size(summary['total_write_size'])})")


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
