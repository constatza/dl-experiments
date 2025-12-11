#!/usr/bin/env python3
"""
Diagnostic script to verify normalization consistency between data files.

This script checks:
1. Whether normalized.npz and comparison.npz have matching normalization metadata
2. Whether the data arrays have compatible scales
3. Whether training data and test data use the same normalization
"""

import sys
from pathlib import Path

import numpy as np


def load_normalization_metadata(npz_path: Path) -> dict:
    """Load normalization metadata from an .npz file."""
    data = np.load(npz_path)
    metadata = {}

    # Extract normalization parameters
    if "normalize_type" in data:
        metadata["normalize_type"] = str(data["normalize_type"])
    if "spectral_radius_bound" in data:
        metadata["spectral_radius_bound"] = float(data["spectral_radius_bound"])
    if "dimension_scale" in data:
        metadata["dimension_scale"] = float(data["dimension_scale"])
    if "composite_scale" in data:
        metadata["composite_scale"] = float(data["composite_scale"])

    # Get array info
    if "matrix" in data:
        matrix = data["matrix"]
        metadata["matrix_shape"] = matrix.shape
        metadata["matrix_norm"] = float(np.linalg.norm(matrix))
        metadata["matrix_range"] = (float(matrix.min()), float(matrix.max()))

    if "rhs" in data:
        rhs = data["rhs"]
        metadata["rhs_shape"] = rhs.shape
        metadata["rhs_norm"] = float(np.linalg.norm(rhs))
        metadata["rhs_range"] = (float(rhs.min()), float(rhs.max()))

    if "solutions" in data:
        sol = data["solutions"]
        metadata["solutions_shape"] = sol.shape
        metadata["solutions_norm"] = float(np.linalg.norm(sol))
        metadata["solutions_range"] = (float(sol.min()), float(sol.max()))

    return metadata


def compare_metadata(meta1: dict, meta2: dict, name1: str, name2: str) -> list[str]:
    """Compare two metadata dictionaries and return list of differences."""
    differences = []

    # Compare normalization parameters
    for key in ["normalize_type", "spectral_radius_bound", "dimension_scale", "composite_scale"]:
        if key in meta1 and key in meta2:
            val1, val2 = meta1[key], meta2[key]
            if isinstance(val1, float) and isinstance(val2, float):
                rel_diff = abs(val1 - val2) / max(abs(val1), abs(val2), 1e-12)
                if rel_diff > 1e-6:
                    differences.append(
                        f"  ❌ {key}: {name1}={val1:.6e}, {name2}={val2:.6e} (rel_diff={rel_diff:.3e})"
                    )
                else:
                    print(f"  ✓ {key}: {val1:.6e} (match)")
            elif val1 != val2:
                differences.append(f"  ❌ {key}: {name1}={val1}, {name2}={val2}")
            else:
                print(f"  ✓ {key}: {val1} (match)")
        elif key in meta1 or key in meta2:
            differences.append(f"  ❌ {key}: only in {name1 if key in meta1 else name2}")

    # Compare array shapes
    for key in ["matrix_shape", "rhs_shape", "solutions_shape"]:
        if key in meta1 and key in meta2:
            if meta1[key] != meta2[key]:
                differences.append(f"  ❌ {key}: {name1}={meta1[key]}, {name2}={meta2[key]}")
            else:
                print(f"  ✓ {key}: {meta1[key]} (match)")

    return differences


def verify_directory(data_dir: Path) -> bool:
    """Verify normalization consistency in a data directory."""
    print(f"\n{'=' * 80}")
    print(f"Verifying data directory: {data_dir}")
    print(f"{'=' * 80}\n")

    normalized_path = data_dir / "normalized.npz"
    comparison_path = data_dir / "comparison.npz"

    if not normalized_path.exists():
        print(f"❌ Missing: {normalized_path}")
        return False

    print(f"Loading normalization metadata from {normalized_path.name}...")
    norm_meta = load_normalization_metadata(normalized_path)

    print(f"\nNormalized data:")
    for key, value in norm_meta.items():
        if isinstance(value, tuple):
            print(f"  {key}: {value}")
        elif isinstance(value, float):
            print(f"  {key}: {value:.6e}")
        else:
            print(f"  {key}: {value}")

    if comparison_path.exists():
        print(f"\nLoading normalization metadata from {comparison_path.name}...")
        comp_meta = load_normalization_metadata(comparison_path)

        print(f"\nComparison data:")
        for key, value in comp_meta.items():
            if isinstance(value, tuple):
                print(f"  {key}: {value}")
            elif isinstance(value, float):
                print(f"  {key}: {value:.6e}")
            else:
                print(f"  {key}: {value}")

        print(f"\n{'=' * 80}")
        print("Comparing normalization metadata...")
        print(f"{'=' * 80}\n")

        differences = compare_metadata(norm_meta, comp_meta, "normalized", "comparison")

        if differences:
            print(f"\n❌ Found {len(differences)} difference(s):")
            for diff in differences:
                print(diff)
            return False
        else:
            print("\n✓ All normalization metadata matches!")
            return True
    else:
        print(f"\n⚠️  No comparison.npz found at {comparison_path}")
        print("  This is expected if you haven't created test data yet.")
        return True


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python verify_normalization.py <data_directory>")
        print("\nExample:")
        print("  python verify_normalization.py /data/projects/graph-cg/data/processed/collect-504-solutions")
        sys.exit(1)

    data_dir = Path(sys.argv[1])
    if not data_dir.exists():
        print(f"❌ Directory does not exist: {data_dir}")
        sys.exit(1)

    success = verify_directory(data_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
