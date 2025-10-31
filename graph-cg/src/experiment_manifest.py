"""Minimal experiment manifest management for reproducibility.

This module provides simple functions to create and update experiment.toml files
that track the configuration and artifacts used at each pipeline stage.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

try:
    import tomli_w
except ImportError:
    raise ImportError(
        "tomli_w is required for writing TOML files. "
        "Install with: uv pip install tomli-w"
    )


def update_manifest(
    experiment_dir: Path | str, section: str, data: dict[str, Any]
) -> Path:
    """Update a section in experiment.toml (creates if missing).

    This function implements additive configuration tracking - each pipeline stage
    can update its section without affecting other sections.

    Args:
        experiment_dir: Directory containing the experiment (where experiment.toml lives)
        section: Section name (e.g., "data", "training", "inference", "comparison")
        data: Dictionary of key-value pairs to write to the section

    Returns:
        Path to the updated experiment.toml file

    Example:
        >>> update_manifest(
        ...     Path("/output/FFNN/collect-504"),
        ...     "training",
        ...     {
        ...         "config_path": "configs/ffnn.toml",
        ...         "checkpoint_path": "checkpoints/ffnn.ckpt",
        ...     }
        ... )
    """
    experiment_dir = Path(experiment_dir)
    manifest_path = experiment_dir / "experiment.toml"

    # Load existing manifest or create empty
    if manifest_path.exists():
        with open(manifest_path, "rb") as f:
            manifest = tomllib.load(f)
    else:
        manifest = {}

    # Update section
    manifest[section] = data

    # Ensure parent directory exists
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # Write back
    with open(manifest_path, "wb") as f:
        tomli_w.dump(manifest, f)

    return manifest_path


def load_manifest(experiment_dir: Path | str) -> dict[str, Any]:
    """Load experiment manifest.

    Args:
        experiment_dir: Directory containing experiment.toml

    Returns:
        Dictionary with manifest contents (empty dict if file doesn't exist)

    Example:
        >>> manifest = load_manifest("/output/FFNN/collect-504")
        >>> print(manifest["training"]["checkpoint_path"])
        checkpoints/ffnn.ckpt
    """
    experiment_dir = Path(experiment_dir)
    manifest_path = experiment_dir / "experiment.toml"

    if not manifest_path.exists():
        return {}

    with open(manifest_path, "rb") as f:
        return tomllib.load(f)


def get_checkpoint_path(experiment_dir: Path | str) -> Path | None:
    """Get checkpoint path from manifest (convenience helper).

    Args:
        experiment_dir: Directory containing experiment.toml

    Returns:
        Absolute path to checkpoint, or None if not found

    Example:
        >>> ckpt = get_checkpoint_path("/output/FFNN/collect-504")
        >>> print(ckpt)
        /output/FFNN/collect-504/checkpoints/ffnn.ckpt
    """
    experiment_dir = Path(experiment_dir)
    manifest = load_manifest(experiment_dir)

    training = manifest.get("training")
    if not training:
        return None

    ckpt_rel = training.get("checkpoint_path")
    if not ckpt_rel:
        return None

    # Convert relative path to absolute
    ckpt_path = experiment_dir / ckpt_rel
    return ckpt_path if ckpt_path.exists() else None
