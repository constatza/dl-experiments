"""Path resolution for data generation and experiment artifacts.

Single source of truth for all path logic. No nesting, no wrappers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from neuralls.platform.config.models.data_models import DataConfigFile
from neuralls.platform.config.path_utils import build_sqlite_uri
from neuralls.platform.config.settings import NeurallsSettings
from neuralls.shared.constants import DEFAULT_PROJECT_ROOT


@dataclass(frozen=True)
class PathContext:
    """Resolved base paths for an experiment.

    Attributes:
        project_root: Project root directory.
        output_root: Master output root — all experiment artifacts go here.
        processed_root: Where generated datasets live (outside experiments).
    """

    project_root: Path
    output_root: Path
    processed_root: Path

    @property
    def mlflow_tracking_uri(self) -> str:
        """MLflow tracking database URI (derived from output_root)."""
        db_path = self.output_root / "mlruns" / "mlflow.db"
        return build_sqlite_uri(db_path)

    @property
    def mlflow_artifact_location(self) -> str:
        """MLflow artifact storage location (derived from output_root)."""
        return str((self.output_root / "mlartifacts").resolve())


def build_path_context(
    data_cfg: DataConfigFile,
    settings: NeurallsSettings,
    output_override: Path | None = None,
) -> PathContext:
    """Build complete path context from data config and settings.

    Args:
        data_cfg: Validated data configuration.
        settings: Resolved NeurallsSettings instance.
        output_override: Optional output root override.

    Returns:
        PathContext with all resolved base paths and MLflow URIs.
    """
    output_root = output_override.expanduser().resolve() if output_override else settings.output_dir
    processed_root = (
        data_cfg.output.data_dir
        if data_cfg.output.data_dir is not None
        else settings.processed_dir
    )
    return PathContext(
        project_root=DEFAULT_PROJECT_ROOT,
        output_root=output_root,
        processed_root=processed_root,
    )
