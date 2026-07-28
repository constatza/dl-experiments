"""Shared lower-case DLKit job types for assignment composition."""

from __future__ import annotations

from dlkit.infrastructure.config.job_config import (
    FitJobConfig,
    InferenceJobConfig,
    SearchJobConfig,
    TrainingJobConfig,
)

type AnyJobConfig = TrainingJobConfig | InferenceJobConfig | SearchJobConfig | FitJobConfig
type TrainLikeJobConfig = TrainingJobConfig | SearchJobConfig
# Job kinds runnable through the assignment/training pipeline (dataset + workspace
# materialization, MLflow tracking). Wider than `TrainLikeJobConfig` — which
# specifically means "has a trainer" — because `FitJobConfig` runs through the same
# pipeline (one-shot, non-gradient fit) without ever having a `training` section.
type TrainableJobConfig = TrainLikeJobConfig | FitJobConfig
