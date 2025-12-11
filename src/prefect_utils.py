"""Utilities for Prefect workflow caching and task management."""

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

    Example:
        >>> src_hash = compute_directory_hash("graph-cg/src")
        >>> # Fix a bug in src/training.py
        >>> new_hash = compute_directory_hash("graph-cg/src")
        >>> assert src_hash != new_hash  # Cache invalidated
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

    Example:
        >>> paths = [Path("checkpoint.ckpt"), Path("plot.png")]
        >>> state = compute_outputs_state(paths)
        >>> # Delete checkpoint
        >>> new_state = compute_outputs_state(paths)
        >>> assert state != new_state  # Cache invalidated
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

    This enables automatic Prefect cache invalidation when outputs are deleted:
    - Delete checkpoints → hash changes → training reruns
    - Delete predictions → hash changes → prediction reruns

    Args:
        output_dir: Path to output directory to hash

    Returns:
        SHA-1 hash of directory listing, or "empty" if directory doesn't exist

    Example:
        >>> checkpoint_dir = Path("output/model/data/checkpoints")
        >>> hash1 = compute_experiment_output_hash(checkpoint_dir)
        >>> # Delete checkpoint.ckpt
        >>> hash2 = compute_experiment_output_hash(checkpoint_dir)
        >>> assert hash1 != hash2  # Cache invalidated
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

    except Exception:
        # If we can't list the directory, return a stable error state
        return "error"

    return hasher.hexdigest()


def compute_data_files_hash(data_dir: Path | str) -> str:
    """Compute content-based hash of data files for cache invalidation.

    Hashes the actual content of data files (not metadata like mtime/size).
    This ensures Prefect cache is invalidated when data content changes.

    Args:
        data_dir: Path to data directory containing .npy files

    Returns:
        SHA-1 hash of data file contents, or "empty" if directory doesn't exist

    Example:
        >>> data_dir = Path("/data/projects/graph-cg/data/processed/generate-90-norm")
        >>> hash1 = compute_data_files_hash(data_dir)
        >>> # Regenerate data with different parameters
        >>> hash2 = compute_data_files_hash(data_dir)
        >>> assert hash1 != hash2  # Cache invalidated, training will re-run
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return "empty"

    hasher = hashlib.sha1()

    try:
        # Find all data files with stable sorting
        candidates = []
        for pattern in ("*.npy", "*.npz"):
            candidates.extend(sorted(data_dir.glob(pattern)))

        for filename in ("metadata.json", "normalization.json"):
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

    except Exception:
        # If we can't read files, return stable error state
        return "error"

    return hasher.hexdigest()
