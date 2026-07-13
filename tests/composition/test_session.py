"""Unit tests for the session-parent-run open/finalize primitive."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from neuralls.composition.tracking.session import session_parent_run


@pytest.fixture
def session_kwargs() -> dict[str, object]:
    """Minimal keyword args for opening a session-parent run."""
    return {
        "tracking_uri": "sqlite:///mlflow.db",
        "artifact_uri": None,
        "run_name": "case | 2026-07-13T00:00:00",
        "tags": {"phase": "session_training"},
        "experiment_name": "some-experiment",
    }


def test_happy_path_finalizes_finished(session_kwargs: dict[str, object]) -> None:
    """No failures reported means the session finalizes as FINISHED."""
    with (
        patch(
            "neuralls.composition.tracking.session.create_session_parent_run",
            return_value="parent-run-1",
        ),
        patch("neuralls.composition.tracking.session.finalize_session_parent_run") as mock_finalize,
    ):
        with session_parent_run(**session_kwargs) as handle:
            assert handle.parent_run_id == "parent-run-1"

    mock_finalize.assert_called_once_with(
        tracking_uri="sqlite:///mlflow.db",
        run_id="parent-run-1",
        status="FINISHED",
    )


def test_mark_failed_finalizes_failed(session_kwargs: dict[str, object]) -> None:
    """An explicit mark_failed() call finalizes the session as FAILED."""
    with (
        patch(
            "neuralls.composition.tracking.session.create_session_parent_run",
            return_value="parent-run-1",
        ),
        patch("neuralls.composition.tracking.session.finalize_session_parent_run") as mock_finalize,
    ):
        with session_parent_run(**session_kwargs) as handle:
            handle.mark_failed()

    mock_finalize.assert_called_once_with(
        tracking_uri="sqlite:///mlflow.db",
        run_id="parent-run-1",
        status="FAILED",
    )


def test_unhandled_exception_finalizes_failed_and_reraises(
    session_kwargs: dict[str, object],
) -> None:
    """An exception escaping the with-block still finalizes FAILED, and propagates."""
    with (
        patch(
            "neuralls.composition.tracking.session.create_session_parent_run",
            return_value="parent-run-1",
        ),
        patch("neuralls.composition.tracking.session.finalize_session_parent_run") as mock_finalize,
    ):
        with pytest.raises(ValueError, match="boom"):
            with session_parent_run(**session_kwargs):
                raise ValueError("boom")

    mock_finalize.assert_called_once_with(
        tracking_uri="sqlite:///mlflow.db",
        run_id="parent-run-1",
        status="FAILED",
    )
