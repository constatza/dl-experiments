"""Workflow package entry point with lazy imports to avoid cycles."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.workflows.reporting import ExperimentResult

__all__ = ["run_experiment_matrix"]


def run_experiment_matrix(
    experiments_config_path: Path,
    *,
    force: bool = False,
    project_root: Path | None = None,
    **kwargs: Any,
) -> list["ExperimentResult"]:
    from src.workflows.runner import run_experiment_matrix as _run

    return _run(
        experiments_config_path,
        force=force,
        project_root=project_root,
        **kwargs,
    )
