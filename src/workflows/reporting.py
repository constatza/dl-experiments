"""Reporting models for experiment execution results."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TaskResult:
    """Outcome of a single task execution (e.g., training, prediction)."""
    name: str
    artifacts: list[Path] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    success: bool = True
    error: str | None = None


@dataclass(frozen=True)
class ExperimentResult:
    """Final comprehensive report for a single experiment."""
    experiment_id: str
    status: str  # "Success" or "Failed"
    tasks: list[TaskResult] = field(default_factory=list)
    error: str | None = None
    
    @property
    def is_success(self) -> bool:
        return self.status == "Success"
