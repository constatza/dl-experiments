"""Neuralls-specific Lightning callbacks used by DLKit training jobs."""

from __future__ import annotations

import shutil
from pathlib import Path

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback


def _copy_checkpoint_if_present(source: str | Path | None, destination_dir: Path) -> None:
    """Copy one checkpoint into the retained-checkpoint directory when it exists."""
    if source is None:
        return
    source_path = Path(source)
    if not source_path.exists() or not source_path.is_file():
        return
    destination_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_dir / source_path.name)


class RetainedCheckpointCopy(Callback):
    """Persist trainer checkpoints before external tracking deletes local copies."""

    def __init__(self, destination_dir: str | Path):
        self._destination_dir = Path(destination_dir)

    def _retain_checkpoints(self, trainer: Trainer) -> None:
        checkpoint_callback = getattr(trainer, "checkpoint_callback", None)
        if checkpoint_callback is None:
            return
        _copy_checkpoint_if_present(
            getattr(checkpoint_callback, "best_model_path", None),
            self._destination_dir,
        )
        _copy_checkpoint_if_present(
            getattr(checkpoint_callback, "last_model_path", None),
            self._destination_dir,
        )

    def on_fit_end(self, trainer: Trainer, pl_module: object) -> None:
        del pl_module
        self._retain_checkpoints(trainer)

    def on_test_end(self, trainer: Trainer, pl_module: object) -> None:
        del pl_module
        self._retain_checkpoints(trainer)
