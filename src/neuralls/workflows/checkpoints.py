"""Checkpoint resolution utilities."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from neuralls.system_loading import get_latest_checkpoint


def load_checkpoint_from_toml(config_path: Path) -> Path:
    """Read checkpoint.path from a TOML file."""
    with open(config_path, "rb") as fp:
        content = tomllib.load(fp)
    checkpoint_block: Any = content.get("checkpoint")
    candidate = None
    if isinstance(checkpoint_block, dict):
        candidate = checkpoint_block.get("path")
    if candidate is None:
        candidate = content.get("checkpoint_path")
    if not candidate:
        raise ValueError(
            f"Missing checkpoint path in {config_path}. Expected checkpoint.path."
        )
    return Path(candidate).expanduser().resolve()


def _load_solver_checkpoint(solver_config: Path) -> Path | None:
    if solver_config is None:
        return None
    with open(solver_config, "rb") as fp:
        content = tomllib.load(fp)
    # 1) Per-solver entries: use the first neural solver with a checkpoint
    solvers = content.get("solvers") or []
    for entry in solvers:
        if not isinstance(entry, dict):
            continue
        if entry.get("type") != "neural":
            continue
        args = entry.get("args") or {}
        candidate = entry.get("checkpoint_path") or entry.get("checkpoint") or args.get("checkpoint_path")
        if isinstance(candidate, str) and candidate:
            return Path(candidate).expanduser().resolve()
    # 2) Legacy/global neural checkpoint
    neural = content.get("neural") or {}
    candidate = neural.get("checkpoint")
    if isinstance(candidate, str) and candidate:
        return Path(candidate).expanduser().resolve()
    return None


def resolve_checkpoint(
    *,
    explicit: Path | None,
    config_file: Path | None,
    solver_config: Path | None,
    checkpoint_dir: Path,
) -> Path | None:
    """Pick a checkpoint using precedence: solver-provided > explicit > latest in dir."""
    solver_ckpt = _load_solver_checkpoint(solver_config)
    if solver_ckpt and solver_ckpt.exists():
        return solver_ckpt.resolve()

    if explicit and explicit.exists():
        return explicit.resolve()

    config_ckpt = load_checkpoint_from_toml(config_file) if config_file else None
    if config_ckpt and config_ckpt.exists():
        return config_ckpt.resolve()

    latest = get_latest_checkpoint(checkpoint_dir)
    if latest and latest.exists():
        return latest.resolve()

    return None
