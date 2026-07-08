"""Workspace-only runtime patching for lower-case DLKit jobs."""

from __future__ import annotations

from pathlib import Path

from neuralls.composition.assignments._job_types import TrainLikeJobConfig


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


def patch_runtime_workspace(
    settings: TrainLikeJobConfig,
    *,
    output_dir: Path,
) -> TrainLikeJobConfig:
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


def patch_dataloader_runtime(settings: TrainLikeJobConfig) -> TrainLikeJobConfig:
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
