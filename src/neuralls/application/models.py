"""Application-layer reporting models for experiment execution results."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TaskResult:
    """Outcome of a single task execution (e.g., training, prediction).

    Args:
        name: Task name.
        artifacts: Paths to produced artifacts.
        metrics: Computed metrics.
        success: Whether the task succeeded.
        error: Error message if failed.
    """

    name: str
    artifacts: list[Path] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    success: bool = True
    error: str | None = None


@dataclass(frozen=True)
class ExperimentResult:
    """Final comprehensive report for a single experiment.

    Args:
        experiment_id: Stable experiment identifier.
        experiment_display_name: Human-readable label.
        status: "Success" or "Failed".
        tasks: Individual task outcomes.
        error: Top-level error message if failed.
    """

    experiment_id: str
    experiment_display_name: str
    status: str
    tasks: list[TaskResult] = field(default_factory=list)
    error: str | None = None

    @property
    def is_success(self) -> bool:
        """Whether the experiment succeeded."""
        return self.status == "Success"
