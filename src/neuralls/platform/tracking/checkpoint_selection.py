"""Shared checkpoint discovery/selection helpers for run-artifact directories."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from loguru import logger


def checkpoint_selection_role(checkpoint: Path) -> str | None:
    """Classify checkpoint filenames that are safe to prefer over other candidates."""
    match checkpoint.name:
        case "best.ckpt":
            return "best"
        case _:
            return None


def select_preferred_checkpoint(checkpoints: list[Path]) -> tuple[str, Path] | None:
    """Select one preferred checkpoint candidate by explicit filename role."""
    role_candidates: dict[str, list[Path]] = {"best": []}
    for checkpoint in checkpoints:
        match checkpoint_selection_role(checkpoint):
            case "best":
                role_candidates["best"].append(checkpoint)
            case _:
                continue

    match role_candidates:
        case {"best": [best_checkpoint]}:
            return "best", best_checkpoint
        case _:
            return None


def find_single_checkpoint(root: Path) -> Path:
    """Find one unambiguous checkpoint under a local artifact directory."""
    checkpoints = sorted(root.glob("**/*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found under {root}")
    logger.debug(
        "Checkpoint discovery under {} found candidates: {}",
        root,
        [path.as_posix() for path in checkpoints],
    )
    if len(checkpoints) == 1:
        return checkpoints[0]

    groups: dict[tuple[str, str], list[Path]] = {}
    for checkpoint in checkpoints:
        digest = sha256(checkpoint.read_bytes()).hexdigest()
        key = (checkpoint.name, digest)
        groups.setdefault(key, []).append(checkpoint)

    if len(groups) == 1:
        duplicates = next(iter(groups.values()))
        canonical = min(
            duplicates,
            key=lambda path: (len(path.relative_to(root).parts), path.relative_to(root).as_posix()),
        )
        logger.warning(
            "Duplicate checkpoint artifacts found under {}. Using canonical path {}.",
            root,
            canonical,
        )
        return canonical

    match select_preferred_checkpoint(checkpoints):
        case ("best", best_checkpoint):
            logger.warning(
                "Multiple distinct checkpoint artifacts found under {}. Using best checkpoint {}.",
                root,
                best_checkpoint,
            )
            return best_checkpoint
        case _:
            pass

    candidate_list = ", ".join(path.as_posix() for path in checkpoints)
    raise ValueError(f"Multiple distinct checkpoints found under {root}: {candidate_list}")
