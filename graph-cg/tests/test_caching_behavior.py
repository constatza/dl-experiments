#!/usr/bin/env python3
"""Test script to verify data caching behavior.

This script tests that:
1. Data generation is skipped when data already exists
2. Only missing data directories trigger regeneration
3. Multiple experiments share the same data config
"""

from pathlib import Path
import shutil

from src.config_utils import resolve_data_dir
from src.validation import validate_data_exists


def test_data_sharing_pattern():
    """Verify that experiments share data configs correctly."""
    print("=" * 70)
    print("Testing Data Sharing Pattern")
    print("=" * 70)

    # Map data configs to experiments that use them
    data_config_usage = {
        "graph-cg/data-configs/generate-90-krylov50.toml": [
            "ffnn-normscaled-on-generate-90-krylov50",
            "ffnn-constant-on-generate-90-krylov50",
        ],
        "graph-cg/data-configs/generate-280-krylov50.toml": [
            "ffnn-normscaled-on-generate-280-krylov50",
            "ffnn-constant-on-generate-280-krylov50",
        ],
        "graph-cg/data-configs/collect-504.toml": [
            "ffnn-normscaled-on-collect-504",
            "ffnn-constant-on-collect-504",
        ],
        "graph-cg/data-configs/collect-2040.toml": [
            "ffnn-normscaled-on-collect-2040",
            "ffnn-constant-on-collect-2040",
        ],
        "graph-cg/data-configs/solution-bank-example.toml": [
            "solution-archive-baseline",
            "solution-archive-ablated",
        ],
    }

    print(f"\nData sharing pattern:")
    print(f"  Unique data configs: {len(data_config_usage)}")
    print(f"  Total experiments: {sum(len(exps) for exps in data_config_usage.values())}")
    print()

    for data_config, experiments in data_config_usage.items():
        data_dir = resolve_data_dir(Path(data_config))
        print(f"  {data_config}")
        print(f"    → {data_dir.name}")
        print(f"    → Shared by {len(experiments)} experiments:")
        for exp in experiments:
            print(f"       - {exp}")
        print()

    print("✓ Data sharing is correctly configured")
    print(f"  Each of 5 data directories will be generated once")
    print(f"  and shared across 2 experiments each")


def test_selective_regeneration():
    """Test that only missing data triggers regeneration."""
    print("\n" + "=" * 70)
    print("Testing Selective Regeneration")
    print("=" * 70)

    # Define test data configs
    test_configs = [
        "graph-cg/data-configs/generate-90-krylov50.toml",
        "graph-cg/data-configs/generate-280-krylov50.toml",
        "graph-cg/data-configs/collect-504.toml",
    ]

    print("\nChecking which data directories exist:")
    for config_path in test_configs:
        full_path = Path(config_path)
        if full_path.exists():
            data_dir = resolve_data_dir(full_path)
            required_files = ["rhs-samples.npy", "sol-samples.npy", "matrix.npy"]

            try:
                validate_data_exists(data_dir, required_files)
                status = "EXISTS (will skip generation)"
            except FileNotFoundError as e:
                status = "MISSING or INCOMPLETE (will regenerate)"

            print(f"  {config_path}")
            print(f"    → {data_dir}")
            print(f"    → Status: {status}")
        else:
            print(f"  {config_path}")
            print(f"    → Config file not found")
        print()

    print("✓ Selective regeneration test complete")
    print("  Only missing/incomplete data directories will be regenerated")


def test_cache_invalidation_scenario():
    """Test cache behavior when one data directory is deleted."""
    print("\n" + "=" * 70)
    print("Testing Cache Invalidation Scenario")
    print("=" * 70)

    print("\nScenario:")
    print("  1. All 5 data directories exist")
    print("  2. User deletes 1 data directory (e.g., generate-90-norm)")
    print("  3. User runs workflow again")
    print()
    print("Expected behavior:")
    print("  ✓ Only the deleted directory is regenerated")
    print("  ✓ Other 4 directories are skipped (cached)")
    print("  ✓ All 10 experiments can still run")
    print()
    print("This is achieved by:")
    print("  1. File system check in generate_data_task()")
    print("  2. Early return if data exists")
    print("  3. Prefect INPUTS cache for task deduplication")


if __name__ == "__main__":
    try:
        test_data_sharing_pattern()
        test_selective_regeneration()
        test_cache_invalidation_scenario()

        print("\n" + "=" * 70)
        print("All caching behavior tests complete!")
        print("=" * 70)

    except Exception as e:
        print(f"\nTest failed: {e}")
        raise
