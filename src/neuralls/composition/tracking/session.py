"""Open/finalize lifecycle for a case-level MLflow session-parent run.

Family 1 only (see module docstring in ``platform/dlkit/tracking_hooks.py``):
callers whose children nest under the parent via explicit ``parent_run_id``
tagging (dlkit's ``on_run_created`` hook), not MLflow's active-run stack.
``create_session_parent_run`` deliberately keeps the parent inactive so
dlkit's own out-of-process run creation never conflicts with it — see its
docstring in ``platform/tracking/mlflow.py``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass

from neuralls.platform.tracking.mlflow import create_session_parent_run, finalize_session_parent_run


@dataclass
class SessionHandle:
    """Mutable handle for reporting a session's aggregate outcome.

    Attributes:
        parent_run_id: The session-parent MLflow run UUID children nest under.
        failed: Whether any item in this session has failed so far.
    """

    parent_run_id: str
    failed: bool = False

    def mark_failed(self) -> None:
        """Record that at least one item in this session failed."""
        self.failed = True


@contextmanager
def session_parent_run(
    *,
    tracking_uri: str,
    artifact_uri: str | None,
    run_name: str,
    tags: Mapping[str, str],
    experiment_name: str,
) -> Iterator[SessionHandle]:
    """Open a session-parent run, yield a handle, finalize on exit.

    Args:
        tracking_uri: MLflow tracking URI.
        artifact_uri: Optional artifact location for a newly created experiment.
        run_name: Display name for the session parent run.
        tags: MLflow tags to attach to the parent run.
        experiment_name: Experiment to create the run under.

    Yields:
        A `SessionHandle` exposing `parent_run_id` and `mark_failed()`.

    Note:
        An exception escaping the `with` block marks the session failed and
        re-raises; it does not swallow the exception.
    """
    parent_run_id = create_session_parent_run(
        tracking_uri=tracking_uri,
        artifact_uri=artifact_uri,
        run_name=run_name,
        tags=tags,
        experiment_name=experiment_name,
    )
    handle = SessionHandle(parent_run_id=parent_run_id)
    try:
        yield handle
    except Exception:
        handle.mark_failed()
        raise
    finally:
        finalize_session_parent_run(
            tracking_uri=tracking_uri,
            run_id=handle.parent_run_id,
            status="FAILED" if handle.failed else "FINISHED",
        )
