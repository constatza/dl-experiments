"""Storage validation utilities for the platform layer."""

from __future__ import annotations

from pathlib import Path


def validate_data_exists(
    data_dir: Path | str,
    required_files: list[str],
) -> None:
    """Validate that required data files exist in a directory.

    Args:
        data_dir: Directory to check for files.
        required_files: List of filenames that must exist (e.g.,
            ["rhs-samples.npy", "sol-samples.npy"]).

    Raises:
        FileNotFoundError: If any required file is missing, with a descriptive
            error message listing all missing file paths.
    """
    data_dir = Path(data_dir)
    missing_files = [
        str(data_dir / filename)
        for filename in required_files
        if not (data_dir / filename).exists()
    ]
    if missing_files:
        files_str = "\n  - ".join(missing_files)
        raise FileNotFoundError(f"Required data files not found in {data_dir}:\n  - {files_str}")
