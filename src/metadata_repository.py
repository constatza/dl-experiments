"""Repository for extracting neural preconditioner training metadata."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

from .constants import ConfigKeys


def extract_residual_iters(
    *,
    checkpoint_path: str | Path | None = None,
    data_config_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> int:
    """Extract residual_iters from available metadata sources.

    Priority order:
    1. Experiment manifest (if exists alongside checkpoint)
    2. Data config TOML
    3. Fallback default

    Args:
        checkpoint_path: Path to model checkpoint
        data_config_path: Path to data config TOML
        manifest_path: Optional explicit manifest path

    Returns:
        Number of residual iterations used during training

    Raises:
        ValueError: If no valid source is found
    """
    # Try manifest first (training metadata)
    if manifest_path is not None:
        iters = _extract_from_manifest(Path(manifest_path))
        if iters is not None:
            return iters

    # Try checkpoint directory manifest
    if checkpoint_path is not None:
        ckpt = Path(checkpoint_path)
        if ckpt.parent.name == "checkpoints":
            experiment_dir = ckpt.parent.parent
            manifest = experiment_dir / "manifest.json"
            if manifest.exists():
                iters = _extract_from_manifest(manifest)
                if iters is not None:
                    return iters

    # Fall back to data config
    if data_config_path is not None:
        iters = _extract_from_data_config(Path(data_config_path))
        if iters is not None:
            return iters

    raise ValueError(
        f"Could not extract residual_iters from checkpoint={checkpoint_path}, "
        f"data_config={data_config_path}, manifest={manifest_path}"
    )


def _extract_from_manifest(manifest_path: Path) -> int | None:
    """Extract residual_iters from experiment manifest JSON.

    Args:
        manifest_path: Path to manifest.json

    Returns:
        residual_iters if found, None otherwise
    """
    try:
        with manifest_path.open("r") as f:
            manifest = json.load(f)

        # Check data section
        data_meta = manifest.get("data", {})
        if ConfigKeys.RESIDUAL_ITERS in data_meta:
            return int(data_meta[ConfigKeys.RESIDUAL_ITERS])
        if "residual_iters" in data_meta:
            return int(data_meta["residual_iters"])

        return None
    except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError):
        return None


def _extract_from_data_config(data_config_path: Path) -> int | None:
    """Extract residual_iters from data config TOML.

    Args:
        data_config_path: Path to data config TOML

    Returns:
        Minimum residual_iters across generation strategies if found, None otherwise
    """
    try:
        with data_config_path.open("rb") as f:
            config = tomllib.load(f)

        candidates: list[int] = []

        # Check generation.strategy array
        strategies = config.get("generation", {}).get("strategy", [])
        for strategy in strategies:
            if not isinstance(strategy, dict):
                continue
            if ConfigKeys.RESIDUAL_ITERS in strategy:
                candidates.append(int(strategy[ConfigKeys.RESIDUAL_ITERS]))
            if "residual_iters" in strategy:
                candidates.append(int(strategy["residual_iters"]))

        # Check top-level generation
        generation = config.get("generation", {})
        if isinstance(generation, dict):
            if ConfigKeys.RESIDUAL_ITERS in generation:
                candidates.append(int(generation[ConfigKeys.RESIDUAL_ITERS]))
            if "residual_iters" in generation:
                candidates.append(int(generation["residual_iters"]))

        return min(candidates) if candidates else None
    except (FileNotFoundError, tomllib.TOMLDecodeError, KeyError, ValueError):
        return None
