"""Centralized I/O utilities for graph-cg project."""

from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np

from dlkit.tools.io import load_array


def ensure_dir(path: str | Path) -> Path:
    """Ensure directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        Path object for the directory
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_numpy_array(path: str | Path) -> np.ndarray:
    """Load numpy array from file, supporting both .npy and text formats.

    Args:
        path: Path to array file

    Returns:
        Loaded numpy array
    """
    path = Path(path)

    if path.suffix == '.npy':
        return load_array(path).numpy()
    else:
        # Assume text format
        return np.loadtxt(path)


def save_numpy_array(array: np.ndarray, path: str | Path) -> None:
    """Save numpy array to file.

    Args:
        array: Array to save
        path: Output path (.npy format)
    """
    path = Path(path)
    ensure_dir(path.parent)
    np.save(path, array)


def save_text_file(content: str, path: str | Path) -> None:
    """Save text content to file.

    Args:
        content: Text content
        path: Output path
    """
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(content)


def load_text_file(path: str | Path) -> str:
    """Load text content from file.

    Args:
        path: Input path

    Returns:
        Text content
    """
    return Path(path).read_text()


def copy_file(src: str | Path, dst: str | Path) -> None:
    """Copy file from source to destination.

    Args:
        src: Source path
        dst: Destination path
    """
    import shutil

    src = Path(src)
    dst = Path(dst)
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def get_file_size(path: str | Path) -> int:
    """Get file size in bytes.

    Args:
        path: File path

    Returns:
        File size in bytes
    """
    return Path(path).stat().st_size


def file_exists(path: str | Path) -> bool:
    """Check if file exists.

    Args:
        path: File path

    Returns:
        True if file exists
    """
    return Path(path).exists()


def get_latest_checkpoint(checkpoint_dir: str | Path, pattern: str = "*.ckpt") -> Path | None:
    """Find the most recent checkpoint file in directory.

    Args:
        checkpoint_dir: Directory containing checkpoints
        pattern: File pattern to match

    Returns:
        Path to latest checkpoint or None if not found
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    checkpoints = list(checkpoint_dir.glob(pattern))
    if not checkpoints:
        return None

    # Sort by modification time, return most recent
    return max(checkpoints, key=lambda p: p.stat().st_mtime)


def clean_directory(path: str | Path, pattern: str = "*", keep_recent: int = 0) -> None:
    """Clean directory by removing old files.

    Args:
        path: Directory to clean
        pattern: File pattern to match
        keep_recent: Number of most recent files to keep
    """
    path = Path(path)
    if not path.exists():
        return

    files = list(path.glob(pattern))
    if len(files) <= keep_recent:
        return

    # Sort by modification time, remove oldest
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for file_path in files[keep_recent:]:
        file_path.unlink()


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def get_relative_path(path: str | Path, base: str | Path) -> Path:
    """Get relative path from base directory.

    Args:
        path: Target path
        base: Base directory

    Returns:
        Relative path
    """
    return Path(path).relative_to(Path(base))


class FileTracker:
    """Track file operations for logging and debugging."""

    def __init__(self):
        self.operations = []

    def track_read(self, path: str | Path, size: int | None = None) -> None:
        """Track file read operation."""
        self.operations.append({
            "operation": "read",
            "path": str(path),
            "size": size or get_file_size(path)
        })

    def track_write(self, path: str | Path, size: int | None = None) -> None:
        """Track file write operation."""
        self.operations.append({
            "operation": "write",
            "path": str(path),
            "size": size or get_file_size(path)
        })

    def get_summary(self) -> dict[str, Any]:
        """Get summary of file operations."""
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
        """Print file operations summary."""
        summary = self.get_summary()
        print(f"File Operations Summary:")
        print(f"  Total operations: {summary['total_operations']}")
        print(f"  Reads: {summary['reads']} ({format_file_size(summary['total_read_size'])})")
        print(f"  Writes: {summary['writes']} ({format_file_size(summary['total_write_size'])})")


# Global file tracker instance
file_tracker = FileTracker()