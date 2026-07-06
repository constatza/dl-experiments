"""Bridge between dlkit's training lifecycle hooks and neuralls' MLflow tagging.

dlkit creates its own MLflow run internally during ``execute()`` and has no
built-in way to nest that run under a run created outside of dlkit. It does
expose ``LifecycleHooks.on_run_created``, fired the instant the run exists —
before training starts — which this module uses to tag the child run with its
parent immediately, instead of waiting for training to finish.
"""

from __future__ import annotations

from dlkit.common.hooks import LifecycleHooks

from neuralls.platform.tracking.mlflow_client import tag_run_parent


def build_parent_link_hooks(parent_run_id: str | None) -> LifecycleHooks | None:
    """Build hooks that tag a dlkit-created run with its parent at creation time.

    Args:
        parent_run_id: Parent MLflow run UUID to nest under, or None to skip.

    Returns:
        LifecycleHooks wired to tag the run on creation, or None when there is
        no parent to link.
    """
    if parent_run_id is None:
        return None

    def _on_run_created(run_id: str, tracking_uri: str | None) -> None:
        if tracking_uri is None:
            return
        tag_run_parent(run_id=run_id, tracking_uri=tracking_uri, parent_run_id=parent_run_id)

    return LifecycleHooks(on_run_created=_on_run_created)
