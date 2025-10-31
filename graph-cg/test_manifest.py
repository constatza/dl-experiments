#!/usr/bin/env python3
"""Test script to verify experiment manifest functionality."""

from pathlib import Path
import tempfile
from src.experiment_manifest import update_manifest, load_manifest, get_checkpoint_path


def test_manifest():
    """Test the experiment manifest system."""

    # Create temporary experiment directory
    with tempfile.TemporaryDirectory() as tmpdir:
        experiment_dir = Path(tmpdir) / "FFNN" / "collect-504"
        experiment_dir.mkdir(parents=True)

        print(f"Testing in: {experiment_dir}")

        # Stage 1: Data section
        print("\n1. Adding [data] section...")
        update_manifest(
            experiment_dir,
            "data",
            {
                "config_path": "data-configs/collect-504.toml",
                "data_dir": "/data/projects/graph-cg/data/processed/collect-504-norm",
            }
        )

        # Stage 2: Training section
        print("2. Adding [training] section...")
        update_manifest(
            experiment_dir,
            "training",
            {
                "config_path": "configs/ffnn.toml",
                "data_config_path": "data-configs/collect-504.toml",
                "checkpoint_path": "checkpoints/ffnn.ckpt",
            }
        )

        # Stage 3: Inference section
        print("3. Adding [inference] section...")
        update_manifest(
            experiment_dir,
            "inference",
            {
                "checkpoint_path": "checkpoints/ffnn.ckpt",
            }
        )

        # Stage 4: Comparison section
        print("4. Adding [comparison] section...")
        update_manifest(
            experiment_dir,
            "comparison",
            {
                "checkpoint_path": "checkpoints/ffnn.ckpt",
            }
        )

        # Load and display manifest
        print("\n5. Loading complete manifest...")
        manifest = load_manifest(experiment_dir)

        print("\n" + "=" * 60)
        print("EXPERIMENT MANIFEST")
        print("=" * 60)

        for section_name, section_data in manifest.items():
            print(f"\n[{section_name}]")
            for key, value in section_data.items():
                print(f"{key} = {repr(value)}")

        # Verify all sections present
        assert "data" in manifest, "Missing [data] section"
        assert "training" in manifest, "Missing [training] section"
        assert "inference" in manifest, "Missing [inference] section"
        assert "comparison" in manifest, "Missing [comparison] section"

        # Verify content
        assert manifest["data"]["config_path"] == "data-configs/collect-504.toml"
        assert manifest["training"]["checkpoint_path"] == "checkpoints/ffnn.ckpt"
        assert manifest["comparison"]["checkpoint_path"] == "checkpoints/ffnn.ckpt"

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)

        # Show the raw TOML file
        manifest_file = experiment_dir / "experiment.toml"
        print(f"\nRaw TOML content ({manifest_file}):")
        print("-" * 60)
        print(manifest_file.read_text())


if __name__ == "__main__":
    test_manifest()
