"""Hashing utilities for workflow caching and change detection."""

from __future__ import annotations

import hashlib
from pathlib import Path


def compute_directory_hash(directory: Path | str) -> str:
    """Compute stable hash of all Python files in a directory.

    This hash changes when any .py file content changes, enabling
    automatic cache invalidation when source code is modified.

    Args:
        directory: Path to directory to hash.

    Returns:
        SHA-1 hash of all Python files (sorted for stability).
    """
    directory = Path(directory)
    if not directory.exists():
        return "missing"

    hasher = hashlib.sha1()
    for py_file in sorted(directory.rglob("*.py")):
        hasher.update(str(py_file.relative_to(directory)).encode())
        with open(py_file, "rb") as f:
            hasher.update(f.read())

    return hasher.hexdigest()
