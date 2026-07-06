"""Tests for the dlkit lifecycle-hook -> MLflow parent-tagging bridge."""

from __future__ import annotations

from unittest.mock import patch

from neuralls.platform.dlkit.tracking_hooks import build_parent_link_hooks


def test_build_parent_link_hooks_returns_none_without_parent() -> None:
    """No parent to link means no hooks are built at all."""
    assert build_parent_link_hooks(None) is None


def test_on_run_created_tags_the_child_run_immediately() -> None:
    """The hook must tag the run the instant it's created, not after training finishes."""
    hooks = build_parent_link_hooks("parent-run-1")
    assert hooks is not None
    assert hooks.on_run_created is not None

    with patch("neuralls.platform.dlkit.tracking_hooks.tag_run_parent") as mock_tag:
        hooks.on_run_created("child-run-1", "sqlite:///mlflow.db")

    mock_tag.assert_called_once_with(
        run_id="child-run-1",
        tracking_uri="sqlite:///mlflow.db",
        parent_run_id="parent-run-1",
    )


def test_on_run_created_skips_tagging_without_tracking_uri() -> None:
    """No tracking URI means there's nothing to tag against — a no-op, not an error."""
    hooks = build_parent_link_hooks("parent-run-1")
    assert hooks is not None
    assert hooks.on_run_created is not None

    with patch("neuralls.platform.dlkit.tracking_hooks.tag_run_parent") as mock_tag:
        hooks.on_run_created("child-run-1", None)

    mock_tag.assert_not_called()
