"""Helpers for enforcing test filesystem scope."""

from __future__ import annotations

from pathlib import Path

ROOT_ARTIFACT_DIRS: frozenset[str] = frozenset(
    {"figures", "mlartifacts", "mlruns", "output", "processed"}
)


def root_artifact_dirs(root: Path) -> set[str]:
    """Return top-level artifact directories present under root."""
    return {name for name in ROOT_ARTIFACT_DIRS if (root / name).is_dir()}


def new_root_artifact_dirs(root: Path, baseline: set[str] | frozenset[str]) -> set[str]:
    """Return artifact directory names created after the baseline snapshot."""
    return root_artifact_dirs(root) - set(baseline)
