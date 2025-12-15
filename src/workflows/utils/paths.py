"""Path resolution and manipulation utilities for workflows.

This module encapsulates logic for determining output paths, creating directory
structures, and extracting metadata from config paths. It ensures that path
operations are isolated from business logic.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from src.constants import DEFAULT_OUTPUT_DIR


def extract_model_name(model_config_path: Path | str) -> str:
    """Extract model name from config SESSION.name.
    
    Args:
        model_config_path: Path to the model configuration TOML file.
        
    Returns:
        The session name defined in the config, or the filename stem as fallback.
    """
    model_config_path = Path(model_config_path)
    try:
        with open(model_config_path, "rb") as f:
            config = tomllib.load(f)
        session = config.get("SESSION") or {}
        name = session.get("name")
        if isinstance(name, str) and name:
            return name
        model_cfg = config.get("MODEL") or {}
        model_name = model_cfg.get("name")
        if isinstance(model_name, str) and model_name:
            return model_name
        raise ValueError(
            "Model name missing. Please set [SESSION].name or [MODEL].name in the model config."
        )
    except (FileNotFoundError, tomllib.TOMLDecodeError):
        return model_config_path.stem


def resolve_output_root(paths_cfg: dict[str, str] | None) -> Path:
    """Resolve base output root from experiments config or environment.
    
    Args:
        paths_cfg: Optional dictionary containing path configurations.
        
    Returns:
        Resolved absolute path to the output root directory.
    """
    if paths_cfg:
        configured = paths_cfg.get("output_root") or paths_cfg.get("output_dir")
        if configured:
            return Path(configured).expanduser().resolve()
    
    env_override = os.getenv("GRAPH_CG_OUTPUT_DIR")
    if env_override:
        return Path(env_override).expanduser().resolve()
        
    return DEFAULT_OUTPUT_DIR


def prepare_experiment_outputs(
    model_name: str, data_config_name: str, output_root: Path
) -> dict[str, Path]:
    """Ensure experiment-specific output directories exist.
    
    Creates the following structure:
    output_root/
      data_config_name/
        model_name/
          checkpoints/
          figures/
          metrics/
          predictions/
          
    Args:
        model_name: Name of the model/experiment session.
        data_config_name: Name of the data configuration.
        output_root: Base output directory.
        
    Returns:
        Dictionary containing paths to the created directories.
    """
    output_root.mkdir(parents=True, exist_ok=True)
    experiment_dir = output_root / data_config_name / model_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoints_dir = experiment_dir / "checkpoints"
    figures_dir = experiment_dir / "figures"
    metrics_dir = experiment_dir / "metrics"
    predictions_dir = experiment_dir / "predictions"
    
    for directory in (checkpoints_dir, figures_dir, metrics_dir, predictions_dir):
        directory.mkdir(parents=True, exist_ok=True)
        
    return {
        "experiment_dir": experiment_dir,
        "checkpoints_dir": checkpoints_dir,
        "figures_dir": figures_dir,
        "metrics_dir": metrics_dir,
        "predictions_dir": predictions_dir,
    }
