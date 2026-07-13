"""Bridge between dlkit's training lifecycle hooks and neuralls' MLflow tagging.

dlkit creates its own MLflow run internally during ``execute()`` and has no
built-in way to nest that run under a run created outside of dlkit. It does
expose ``LifecycleHooks.on_run_created``, fired the instant the run exists —
before training starts — which this module uses to tag the child run with its
parent immediately, instead of waiting for training to finish.

For a search job, dlkit fires ``on_run_created`` once for the study run, once
per trial run, and once for the best-retrain run. Only the outermost run of
this ``execute()`` call (the study run for a search job) should be tagged to
the external parent: trial/best-retrain runs are already correctly nested
under the study run via MLflow's own active-run stack, and re-tagging them
would overwrite that with a flat link straight to the external parent.
dlkit's ``RunCreatedEvent.is_outermost`` tells us which one that is directly —
no need to infer it from call order ourselves.
"""

from __future__ import annotations

from dlkit.common.hooks import LifecycleHooks, RunCreatedEvent

from neuralls.platform.tracking.mlflow_client import tag_run_parent


def build_parent_link_hooks(parent_run_id: str | None) -> LifecycleHooks | None:
    """Build hooks that tag the outermost dlkit-created run with its parent.

    Args:
        parent_run_id: Parent MLflow run UUID to nest under, or None to skip.

    Returns:
        LifecycleHooks wired to tag only the outermost run created, or None
        when there is no parent to link.
    """
    if parent_run_id is None:
        return None

    def _on_run_created(event: RunCreatedEvent) -> None:
        if not event.is_outermost or event.tracking_uri is None:
            return
        tag_run_parent(
            run_id=event.run_id, tracking_uri=event.tracking_uri, parent_run_id=parent_run_id
        )

    return LifecycleHooks(on_run_created=_on_run_created)
