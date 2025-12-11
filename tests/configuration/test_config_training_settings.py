"""Config loading smoke tests for training sections."""

from __future__ import annotations

from pathlib import Path
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

import pytest

from src.configuration import load_config


# Get project root (repo root directory)
PROJECT_ROOT = Path(__file__).parent.parent.parent


@pytest.mark.parametrize(
    "config_path", [
        PROJECT_ROOT / "configs/ffnn.toml",
        PROJECT_ROOT / "configs/linear.toml",
        PROJECT_ROOT / "configs/gnn.toml",
    ]
)
def test_training_sections_round_trip(monkeypatch: pytest.MonkeyPatch, config_path: Path) -> None:
    """Ensure load_config_with_context preserves trainer callbacks/metrics."""

    original_mkdir = Path.mkdir
    original_exists = Path.exists
    original_is_file = Path.is_file
    original_is_dir = Path.is_dir

    def _safe_mkdir(self: Path, mode: int = 0o777, parents: bool = False, exist_ok: bool = False) -> None:
        if str(self).startswith("/data/"):
            return None
        original_mkdir(self, mode=mode, parents=parents, exist_ok=exist_ok)

    def _safe_exists(self: Path) -> bool:
        if str(self).startswith("/data/"):
            return True
        return original_exists(self)

    def _safe_is_file(self: Path) -> bool:
        if str(self).startswith("/data/"):
            return True
        return original_is_file(self)

    def _safe_is_dir(self: Path) -> bool:
        if str(self).startswith("/data/"):
            return True
        return original_is_dir(self)

    monkeypatch.setattr(Path, "mkdir", _safe_mkdir, raising=False)
    monkeypatch.setattr(Path, "exists", _safe_exists, raising=False)
    monkeypatch.setattr(Path, "is_file", _safe_is_file, raising=False)
    monkeypatch.setattr(Path, "is_dir", _safe_is_dir, raising=False)

    with config_path.open("rb") as fh:
        raw_config = tomllib.load(fh)

    raw_training = raw_config.get("TRAINING", {})
    raw_trainer = raw_training.get("trainer", {})
    raw_callbacks = tuple(raw_trainer.get("callbacks", ()))
    expected_callback_names = tuple(cb.get("name") for cb in raw_callbacks)

    raw_metrics = tuple(raw_training.get("metrics", ()))
    expected_metric_names = tuple(m.get("name") for m in raw_metrics)

    settings, _ = load_config(config_path, None)
    training = settings.TRAINING
    assert training is not None, "TRAINING section missing"

    actual_callback_names = tuple(cb.name for cb in training.trainer.callbacks)
    assert actual_callback_names == expected_callback_names

    actual_metric_names = tuple(metric.name for metric in training.metrics)
    assert actual_metric_names == expected_metric_names

    patched_training = training.model_copy(update={"epochs": training.epochs + 2})
    assert patched_training.trainer.callbacks == training.trainer.callbacks
