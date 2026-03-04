"""Path resolution for data generation and experiment artifacts.

Single source of truth for all path logic. No nesting, no wrappers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from neuralls.configuration.data_models import DataConfigFile
from neuralls.constants import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROCESSED_DATA_DIR,
    DEFAULT_PROJECT_ROOT,
)


@dataclass(frozen=True)
class PathContext:
    """Resolved base paths for an experiment.

    This is the ONLY path context object. No hierarchy.

    Attributes:
        project_root: Project root directory.
        output_root: MASTER output root - all artifacts go here.
        processed_root: Where generated datasets live (outside experiments).
    """

    project_root: Path
    output_root: Path
    processed_root: Path

    @property
    def mlflow_tracking_uri(self) -> str:
        """MLflow tracking database URI (derived from output_root)."""
        db_path = self.output_root / "mlruns" / "mlflow.db"
        return f"sqlite:///{db_path.as_posix()}"

    @property
    def mlflow_artifact_location(self) -> str:
        """MLflow artifact storage location (derived from output_root)."""
        return str((self.output_root / "mlartifacts").as_posix())


def resolve_project_root(override: Path | str | None = None) -> Path:
    """Resolve project root with optional override.

    Args:
        override: Optional explicit project root path.

    Returns:
        Resolved absolute project root path.
    """
    if override:
        return Path(override).resolve()
    return DEFAULT_PROJECT_ROOT


def resolve_output_root(override: Path | str | None = None) -> Path:
    """Resolve MASTER output root for all experiments.

    This is THE SINGLE SOURCE OF TRUTH for output paths.
    All MLflow paths are derived from this.

    Priority order:
    1. Explicit override (highest priority)
    2. DEFAULT_OUTPUT_DIR (fallback)

    Args:
        override: Optional explicit output root path.

    Returns:
        Resolved absolute output root path.
    """
    if override:
        return Path(override).resolve()
    return DEFAULT_OUTPUT_DIR


def resolve_processed_root(
    data_cfg: DataConfigFile,
    project_root: Path,
) -> Path:
    """Resolve processed data directory from data config.

    Processed data is where generated datasets live (manifest + split arrays).
    This is SEPARATE from experiment artifacts (which go under output_root).

    Args:
        data_cfg: Validated data configuration.
        project_root: Project root for resolving relative paths.

    Returns:
        Resolved absolute processed data directory.
    """
    data_dir = data_cfg.output.data_dir
    if data_dir is not None:
        path = Path(data_dir)
        if not path.is_absolute():
            path = (project_root / path).resolve()
        return path
    return DEFAULT_PROCESSED_DATA_DIR


def build_path_context(
    data_cfg: DataConfigFile,
    output_override: Path | str | None = None,
    project_override: Path | str | None = None,
) -> PathContext:
    """Build complete path context from data config and overrides.

    This is the MAIN entry point for path resolution.

    Args:
        data_cfg: Validated data configuration.
        output_override: Optional output root override (MASTER path).
        project_override: Optional project root override.

    Returns:
        PathContext with all resolved base paths and MLflow URIs.
    """
    project_root = resolve_project_root(project_override)
    output_root = resolve_output_root(output_override)
    processed_root = resolve_processed_root(data_cfg, project_root)

    return PathContext(
        project_root=project_root,
        output_root=output_root,
        processed_root=processed_root,
    )
