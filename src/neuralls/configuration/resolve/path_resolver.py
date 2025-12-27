"""Path resolution and validation module.

This module handles all path resolution logic, extracting it from the main loader
for better testability and single responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..data_models import DataConfigFile
from ..models import ModelConfigFile
from ...constants import (
    DEFAULT_FIGURES_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROCESSED_DATA_DIR,
    DEFAULT_PROJECT_ROOT,
    ConfigKeys,
)


@dataclass(frozen=True)
class PathResolutionContext:
    """Encapsulates resolved paths for an experiment."""

    project_root: Path
    processed_root: Path
    output_root: Path
    figures_root: Path


def resolve_project_root(model_cfg: ModelConfigFile) -> Path:
    """Extract project root from model config with fallback.

    Args:
        model_cfg: Validated model configuration.

    Returns:
        Resolved project root path.
    """
    if model_cfg.PATHS.project_root:
        return Path(model_cfg.PATHS.project_root).resolve()
    return DEFAULT_PROJECT_ROOT


def resolve_processed_root(
    data_cfg: DataConfigFile,
    project_root: Path,
) -> Path:
    """Extract processed data directory from data config.

    Args:
        data_cfg: Validated data configuration.
        project_root: Project root path for resolving relative paths.

    Returns:
        Resolved processed data directory path.
    """
    if data_cfg.output.processed_dir:
        path = Path(data_cfg.output.processed_dir)
        if not path.is_absolute():
            path = (project_root / path).resolve()
        return path
    return DEFAULT_PROCESSED_DATA_DIR


def resolve_output_root(
    output_root: Path | str | None,
    project_root: Path,
    fallback: str | None = None,
) -> Path:
    """Resolve output directory with fallbacks.

    Args:
        output_root: Explicit output root path (highest priority).
        project_root: Project root for resolving relative paths.
        fallback: Fallback path string (e.g., from master config).

    Returns:
        Resolved output directory path.
    """
    if output_root:
        return Path(output_root).expanduser().resolve()

    if fallback:
        candidate = Path(fallback)
        if not candidate.is_absolute():
            candidate = (project_root / candidate).resolve()
        return candidate

    return DEFAULT_OUTPUT_DIR


def resolve_config_path(
    experiment_name: str,
    config_filename: str,
    experiment_dir: Path,
    default_dir: Path,
    default_filename: str | None = None,
) -> Path:
    """Resolve config file path with experiment → default fallback.

    Args:
        experiment_name: Experiment identifier for error messages.
        config_filename: Config filename to find in experiment dir.
        experiment_dir: Experiment-specific directory.
        default_dir: Default directory for fallback.
        default_filename: Alternative filename in default dir (optional).

    Returns:
        Resolved config file path.

    Raises:
        FileNotFoundError: If config not found in either location.
    """
    exp_path = experiment_dir / config_filename
    if exp_path.exists():
        return exp_path

    target_default_name = default_filename or config_filename
    default_path = default_dir / target_default_name

    if default_path.exists():
        return default_path

    raise FileNotFoundError(
        f"Config '{config_filename}' not found for '{experiment_name}' "
        f"in '{exp_path.parent}'. Fallback '{target_default_name}' "
        f"also not found in '{default_dir}'"
    )


def build_path_context(
    model_cfg: ModelConfigFile,
    data_cfg: DataConfigFile,
    output_override: Path | str | None = None,
) -> PathResolutionContext:
    """Build complete path resolution context from configs.

    Args:
        model_cfg: Validated model configuration.
        data_cfg: Validated data configuration.
        output_override: Optional explicit output directory override.

    Returns:
        PathResolutionContext with all resolved paths.
    """
    project_root = resolve_project_root(model_cfg)
    processed_root = resolve_processed_root(data_cfg, project_root)
    output_root = resolve_output_root(output_override, project_root)

    return PathResolutionContext(
        project_root=project_root,
        processed_root=processed_root,
        output_root=output_root,
        figures_root=DEFAULT_FIGURES_DIR,
    )
