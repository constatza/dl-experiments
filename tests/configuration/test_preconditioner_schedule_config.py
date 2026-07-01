"""Tests for shared preconditioner scheduling config fields."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from neuralls.platform.config.models.preconditioner import (
    PreconditionerType,
    StandardPreconditionerConfig,
)


def test_start_iter_defaults_to_zero() -> None:
    """Preconditioner schedules activate immediately by default."""
    cfg = StandardPreconditionerConfig(name="jacobi", type=PreconditionerType.JACOBI)
    assert cfg.start_iter == 0


def test_start_iter_accepts_non_negative_value() -> None:
    """Preconditioner schedules accept delayed activation."""
    cfg = StandardPreconditionerConfig(
        name="jacobi",
        type=PreconditionerType.JACOBI,
        start_iter=5,
    )
    assert cfg.start_iter == 5


def test_start_iter_rejects_negative_value() -> None:
    """Preconditioner schedules reject negative activation iterations."""
    with pytest.raises(ValidationError):
        StandardPreconditionerConfig(
            name="jacobi",
            type=PreconditionerType.JACOBI,
            start_iter=-1,
        )
