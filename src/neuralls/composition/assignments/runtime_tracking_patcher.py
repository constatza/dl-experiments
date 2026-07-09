"""Tracking-only runtime patching for lower-case DLKit jobs."""

from __future__ import annotations

from neuralls.composition.assignments._job_types import AnyJobConfig, TrainLikeJobConfig


def ensure_runtime_tracking(settings: AnyJobConfig) -> AnyJobConfig:
    """Ensure a tracking section exists so runtime wiring has one stable target."""
    if settings.tracking is not None:
        return settings
    return settings.patch({"tracking": {}})


def patch_training_tracking(
    settings: TrainLikeJobConfig,
    *,
    uri: str | None,
) -> TrainLikeJobConfig:
    """Enable MLflow tracking with the resolved tracking URI for training jobs."""
    return settings.patch({"tracking": {"backend": "mlflow", "uri": uri}})
