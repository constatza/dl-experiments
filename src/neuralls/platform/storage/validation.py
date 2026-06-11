"""Storage validation utilities for the platform layer."""

from __future__ import annotations

from pathlib import Path

from neuralls.platform.storage.datasets import load_dataset_manifest, resolve_dataset_paths
from neuralls.shared.constants import (
    DATASET_MANIFEST_FILENAME,
    MATRIX_ZARR_DIRNAME,
    RHS_ZARR_FILENAME,
)

_SUPPORTED_COMPARISON_FILE_SUFFIXES = {"", ".npy", ".txt"}


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


def build_missing_input_error(path: Path) -> FileNotFoundError:
    """Build a user-facing missing-input error for a comparison dataset path."""
    return FileNotFoundError(f"Comparison input not found: {path}.")


def validate_comparison_matrix_input(path: Path) -> None:
    """Validate one comparison matrix input without executing the workflow."""
    if not path.exists():
        raise build_missing_input_error(path)
    if not path.is_dir():
        if path.suffix not in _SUPPORTED_COMPARISON_FILE_SUFFIXES:
            raise ValueError(
                f"Unsupported comparison matrix input format: {path}. "
                "Use a dataset directory, .npy file, or text matrix file."
            )
        return
    try:
        load_dataset_manifest(path)
    except FileNotFoundError, ValueError:
        manifest_path = path / DATASET_MANIFEST_FILENAME
        values_path = path / "values.npy"
        if not (manifest_path.exists() and values_path.exists()):
            raise ValueError(
                f"Comparison matrix dataset directory is not loadable: {path}. "
                f"Expected a dataset root with {DATASET_MANIFEST_FILENAME} or a sparse-pack "
                "matrix directory containing manifest.json and values.npy."
            ) from None
        return
    matrix_zarr_dir = resolve_dataset_paths(path).matrix_zarr_dir
    if not matrix_zarr_dir.exists():
        raise ValueError(
            f"Comparison matrix dataset directory is missing {MATRIX_ZARR_DIRNAME}: {path}"
        )


def validate_comparison_rhs_input(path: Path) -> None:
    """Validate one comparison RHS input without executing the workflow."""
    if not path.exists():
        raise build_missing_input_error(path)
    if not path.is_dir():
        if path.suffix not in _SUPPORTED_COMPARISON_FILE_SUFFIXES:
            raise ValueError(
                f"Unsupported comparison RHS input format: {path}. "
                "Use a dataset directory, .npy file, or text vector file."
            )
        return
    load_dataset_manifest(path)
    rhs_path = resolve_dataset_paths(path).rhs_path
    if not rhs_path.exists():
        raise ValueError(f"Comparison RHS dataset directory is missing {RHS_ZARR_FILENAME}: {path}")


def validate_comparison_inputs(
    *,
    matrix_path: Path,
    rhs_path: Path,
) -> None:
    """Validate comparison matrix/RHS inputs before opening tracking runs."""
    validate_comparison_matrix_input(matrix_path)
    validate_comparison_rhs_input(rhs_path)
