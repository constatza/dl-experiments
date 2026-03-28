"""Validation tests for strict model_ref configuration."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from neuralls.platform.config.models.preconditioner import (
    LoggedModelRefConfig,
    NeuralPreconditionerConfig,
    PreconditionerType,
    RegisteredModelRefConfig,
)


def test_registered_model_ref_requires_exactly_one_selector() -> None:
    """Registered refs must provide exactly one of alias/version/latest."""
    with pytest.raises(ValidationError):
        NeuralPreconditionerConfig(
            name="neural",
            type=PreconditionerType.NEURAL,
            model_ref=RegisteredModelRefConfig(
                name="dataset-model",
                alias="candidate",
                version=1,
            ),
        )


def test_logged_model_ref_requires_filter_for_latest() -> None:
    """Logged latest lookup requires at least one filter."""
    with pytest.raises(ValidationError):
        NeuralPreconditionerConfig(
            name="neural",
            type=PreconditionerType.NEURAL,
            model_ref=LoggedModelRefConfig(latest=True),
        )


def test_logged_model_ref_with_run_id_is_valid() -> None:
    """run_id selector is valid for logged model refs."""
    cfg = NeuralPreconditionerConfig(
        name="neural",
        type=PreconditionerType.NEURAL,
        model_ref=LoggedModelRefConfig(run_id="abc123"),
    )
    assert cfg.model_ref is not None
    assert cfg.model_ref.source == "logged"
