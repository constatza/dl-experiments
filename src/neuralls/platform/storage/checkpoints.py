"""Checkpoint management utilities.

This module handles checkpoint file operations including finding latest
checkpoints and cleaning old checkpoint files.
"""

from __future__ import annotations
from pathlib import Path


def get_latest_checkpoint(checkpoint_dir: str | Path, pattern: str = "*.ckpt") -> Path | None:
    """Find the most recent checkpoint file in directory.

    I/O action - scans directory and reads file metadata.

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

    I/O action - deletes files from filesystem.

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
