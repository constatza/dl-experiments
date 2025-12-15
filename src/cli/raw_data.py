"""Utilities for tidying raw data artefacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _load_array(file: Path) -> np.ndarray:
    if file.suffix == ".npy":
        return np.load(file).astype(np.float64, copy=False)
    if file.suffix == ".txt":
        return np.loadtxt(file, dtype=np.float64)
    raise ValueError(f"Unsupported file type: {file.suffix}")


def standardize_raw_filenames(base_dir: Path) -> None:
    """Rename raw files to the ``{kind}-{dimensions}`` convention."""
    for file in sorted(base_dir.glob("*")):
        if file.is_dir():
            continue

        print(f"\nProcessing: {file.name}")

        try:
            data = _load_array(file)
        except Exception as exc:  # noqa: BLE001
            print(f"  Error loading: {exc}")
            continue

        print(f"  Shape: {data.shape}")

        if data.ndim == 1:
            dim_str = str(data.shape[0])
            keyword = "rhs"
        elif data.ndim == 2:
            dim_str = (
                str(data.shape[0])
                if data.shape[0] == data.shape[1]
                else f"{data.shape[0]}x{data.shape[1]}"
            )
            keyword = "matrix"
        else:
            print(f"  Unexpected shape: {data.shape}")
            continue

        new_name = f"{keyword}-{dim_str}{file.suffix}"
        if file.name == new_name:
            print("  ✓ Already standardized")
            continue

        new_path = file.parent / new_name
        if new_path.exists() and new_path != file:
            try:
                existing = _load_array(new_path)
            except Exception as exc:  # noqa: BLE001
                print(f"  ⚠ Warning: {new_name} exists (unable to compare: {exc})")
                continue

            if np.allclose(data, existing):
                print(f"  Duplicate of {new_name}, removing original")
                file.unlink()
            else:
                print(
                    f"  ⚠ Warning: {new_name} already exists with different data, skipping"
                )
            continue

        file.rename(new_path)
        print(f"  ✓ Renamed to {new_name}")

    print("\n=== All files processed ===")
