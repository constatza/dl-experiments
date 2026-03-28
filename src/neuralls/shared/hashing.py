"""Hashing utilities for workflow caching and change detection.

This module provides pure functions to compute stable hashes for directories,
files, and output states. These hashes are used by Prefect to determine
if a task needs to be re-run.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def compute_directory_hash(directory: Path | str) -> str:
    """Compute stable hash of all Python files in a directory.

    This hash changes when any .py file content changes, enabling
    automatic cache invalidation when source code is modified.

    Args:
        directory: Path to directory to hash

    Returns:
        SHA-1 hash of all Python files (sorted for stability)
    """
    directory = Path(directory)
    if not directory.exists():
        return "missing"

    hasher = hashlib.sha1()

    # Collect all Python files in sorted order (for stability)
    py_files = sorted(directory.rglob("*.py"))

    for py_file in py_files:
        # Include relative path in hash (detect renames)
        relative_path = py_file.relative_to(directory)
        hasher.update(str(relative_path).encode())

        # Include file contents
        with open(py_file, "rb") as f:
            hasher.update(f.read())

    return hasher.hexdigest()


def compute_outputs_state(output_paths: list[Path]) -> str:
    """Compute state of output files for cache invalidation.

    Returns a string indicating which files exist/missing. This becomes
    part of the cache key, so deleting outputs invalidates the cache.

    Args:
        output_paths: List of expected output file paths

    Returns:
        String encoding which files exist (for cache key)
    """
    # Create stable string representation of which files exist
    states = []
    for path in sorted(output_paths):  # Sort for stability
        state = "exists" if path.exists() else "missing"
        states.append(f"{path.name}:{state}")

    return "|".join(states)


def compute_experiment_output_hash(output_dir: Path | str) -> str:
    """Compute hash of output directory state for cache invalidation.

    Uses pathlib to recursively list all files with their sizes and mtimes,
    creating a stable representation of directory contents. Hash changes when
    files are added, removed, or modified.

    Args:
        output_dir: Path to output directory to hash

    Returns:
        SHA-1 hash of directory listing, or "empty" if directory doesn't exist
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return "empty"

    hasher = hashlib.sha1()

    try:
        # Recursively collect all files with stable sorting
        all_files = sorted(output_dir.rglob("*"))

        if not all_files:
            return "empty"

        # Hash each file's relative path, size, and mtime
        for path in all_files:
            if path.is_file():
                # Include relative path (detect renames/moves)
                rel_path = path.relative_to(output_dir)
                hasher.update(str(rel_path).encode())

                # Include size and mtime (detect modifications/deletions)
                try:
                    stat = path.stat()
                    hasher.update(str(stat.st_size).encode())
                    hasher.update(str(int(stat.st_mtime)).encode())
                except OSError:
                    # File might have been deleted between listing and stat
                    hasher.update(b"missing")

    except OSError, PermissionError:
        # If we can't list the directory, return a stable error state
        return "error"

    return hasher.hexdigest()


def compute_data_files_hash(data_dir: Path | str) -> str:
    """Compute content-based hash of data files for cache invalidation.

    Hashes the actual content of data files (not metadata like mtime/size).
    This ensures Prefect cache is invalidated when data content changes.

    Args:
        data_dir: Path to data directory containing dataset artifacts

    Returns:
        SHA-1 hash of data file contents, or "empty" if directory doesn't exist
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return "empty"

    hasher = hashlib.sha1()

    try:
        # Find all data files with stable sorting
        candidates: list[Path] = []
        for pattern in ("*.npy", "matrix_coo/*.npy"):
            candidates.extend(sorted(data_dir.glob(pattern)))

        for filename in ("manifest.json", "metadata.json", "normalization.json"):
            candidate = data_dir / filename
            if candidate.exists():
                candidates.append(candidate)

        all_files = sorted(candidates)

        if not all_files:
            return "empty"

        # Hash file contents (not metadata)
        for file_path in all_files:
            # Include filename for stable ordering
            hasher.update(file_path.name.encode())

            # Hash actual file contents
            with open(file_path, "rb") as f:
                # Read in chunks to handle large files
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)

    except OSError, PermissionError, FileNotFoundError:
        # If we can't read files, return stable error state
        return "error"

    return hasher.hexdigest()


def compute_file_signature(path: Path | str) -> str:
    """Compute a stable SHA-1 signature for a file's contents.

    Args:
        path: Path to the file to hash.

    Returns:
        SHA-1 hash of the file content, or "missing" if file not found.
    """
    file_path = Path(path)
    if not file_path.exists():
        return "missing"

    hasher = hashlib.sha1()
    with open(file_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
