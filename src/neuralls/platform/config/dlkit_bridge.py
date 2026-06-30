"""Thin adapter from neuralls job paths to DLKit lower-case job configs."""

from __future__ import annotations

from pathlib import Path

from dlkit.infrastructure.config.factories import load_job
from dlkit.infrastructure.config.job_config import (
    InferenceJobConfig,
    SearchJobConfig,
    TrainingJobConfig,
)

from neuralls.platform.config.settings import NeurallsSettings

type AnyJobSettings = TrainingJobConfig | InferenceJobConfig | SearchJobConfig


def load_job_config(path: Path, settings: NeurallsSettings) -> AnyJobSettings:
    """Load one lower-case DLKit job config through DLKit's native loader.

    Legacy uppercase manifests are not special-cased: they have no ``[run]``
    section, so DLKit's own validation in ``load_job()`` already fails fast
    with ``ConfigValidationError: No run.type found...``.
    """
    del settings
    return load_job(path)
