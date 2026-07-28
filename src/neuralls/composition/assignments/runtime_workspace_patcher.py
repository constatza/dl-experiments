"""Workspace-only runtime patching for lower-case DLKit jobs."""

from __future__ import annotations

from pathlib import Path

from dlkit.infrastructure.config.job_config import SearchJobConfig, TrainingJobConfig

from neuralls.composition.assignments._job_types import (
    AnyJobConfig,
    TrainableJobConfig,
    TrainLikeJobConfig,
)


def _retained_checkpoint_callback(output_dir: Path) -> dict[str, str]:
    return {
        "name": "RetainedCheckpointCopy",
        "module_path": "neuralls.platform.dlkit.callbacks",
        "destination_dir": str(output_dir / "retained-checkpoints"),
    }


def _merge_callbacks(
    existing_callbacks: object,
    *,
    output_dir: Path,
) -> list[dict[str, str] | object]:
    if isinstance(existing_callbacks, tuple):
        callbacks = list(existing_callbacks)
    else:
        callbacks = []
    retained_callback = _retained_checkpoint_callback(output_dir)
    if retained_callback in callbacks:
        return callbacks
    return [*callbacks, retained_callback]


def patch_runtime_workspace[T: TrainLikeJobConfig](
    settings: T,
    *,
    output_dir: Path,
) -> T:
    """Inject trainer output paths and retained-checkpoint callback wiring."""
    training_cfg = settings.training
    if training_cfg is None or training_cfg.trainer is None:
        raise ValueError("Training jobs require [training].trainer.")

    return settings.patch(
        {
            "training": {
                "trainer": {
                    "default_root_dir": str(output_dir),
                    "callbacks": _merge_callbacks(
                        training_cfg.trainer.callbacks,
                        output_dir=output_dir,
                    ),
                }
            }
        }
    )


def patch_runtime_workspace_for_job[T: TrainableJobConfig](
    settings: T,
    *,
    output_dir: Path,
) -> T:
    """Apply trainer workspace patching only for job kinds that have a trainer.

    `FitJobConfig` has no `training` section by design (see its docstring) —
    there is no trainer-owned output directory or checkpoint callback to
    redirect, so this is a no-op for that job kind rather than hitting
    `patch_runtime_workspace`'s guard clause. Kept as a sibling dispatch
    rather than a modification of `patch_runtime_workspace` itself, so that
    function's contract ("training jobs require [training].trainer") stays
    exact for its existing train-like callers.
    """
    if isinstance(settings, (TrainingJobConfig, SearchJobConfig)):
        return patch_runtime_workspace(settings, output_dir=output_dir)
    return settings


def patch_dataloader_runtime[T: AnyJobConfig](settings: T) -> T:
    """Force conservative dataloader runtime settings for dense zarr execution."""
    data_cfg = settings.data
    if data_cfg is None:
        return settings

    return settings.patch(
        {
            "data": {
                "num_workers": 0,
                "persistent_workers": False,
                "pin_memory": False,
            }
        }
    )
