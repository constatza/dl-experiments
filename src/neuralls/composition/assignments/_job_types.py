"""Shared lower-case DLKit job types for assignment composition."""

from __future__ import annotations

from dlkit.infrastructure.config.job_config import (
    InferenceJobConfig,
    SearchJobConfig,
    TrainingJobConfig,
)

type AnyJobConfig = TrainingJobConfig | InferenceJobConfig | SearchJobConfig
type TrainLikeJobConfig = TrainingJobConfig | SearchJobConfig
