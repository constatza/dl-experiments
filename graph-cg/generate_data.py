#!/usr/bin/env python3
"""Generate training data (RHS -> solution) for a linear system using src/ library modules."""

from __future__ import annotations
from pathlib import Path
from typing import Iterable, Tuple, Dict, Optional
import typer

import numpy as np

from src.common import get_paths_from_partial_config, load_system_data, save_training_data
from src.math_utils import normalize_system
from src.data_generation import generate_mixture
from src.validation import validate_matrix, validate_rhs


def generate_training_data(
    config_path: str | Path,
    matrix_path: Optional[str | Path] = None,
    rhs_path: Optional[str | Path] = None,
    total_samples: int = 6000,
    mix: Dict[str, float] = None,
    krylov_iters: int = 15,
    seed: Optional[int] = None,
    normalize: bool = True,
    shuffle: bool = True
) -> Tuple[Path, Path]:
    """Generate training data programmatically.

    Args:
        config_path: Path to config file
        matrix_path: Override matrix path
        rhs_path: Override RHS path
        total_samples: Number of samples to generate
        mix: Strategy mix dictionary
        krylov_iters: CG iterations for krylov strategy
        seed: Random seed
        normalize: Whether to normalize system
        shuffle: Whether to shuffle final dataset

    Returns:
        Tuple of (features_path, targets_path)
    """
    if mix is None:
        mix = {"normal": 0.5, "krylov": 0.5}

    # Load config and get paths using partial loading
    paths = get_paths_from_partial_config(config_path)

    # Use provided paths or get from config
    matrix_file = matrix_path or paths['matrix_path']
    rhs_file = rhs_path or paths['rhs_path']
    features_path = Path(paths['features_path'])
    targets_path = Path(paths['targets_path'])

    # Load and validate system
    A, b = load_system_data(matrix_file, rhs_file)
    validate_matrix(A)
    validate_rhs(b, A)

    # Normalize system if requested
    A, b = normalize_system(A, b, enabled=normalize)

    # Validate mix
    if abs(sum(mix.values()) - 1.0) > 1e-6:
        raise ValueError(f"Mix fractions must sum to 1.0, got {sum(mix.values())}")

    # Generate data
    X, R = generate_mixture(
        A=A,
        b=b,
        mix=mix,
        total=total_samples,
        krylov_iters=krylov_iters,
        seed=seed or 42,
        shuffle=shuffle,
    )

    # Save results (R are RHS features, X are solution targets)
    save_training_data(R, X, features_path, targets_path)

    return features_path, targets_path


def _parse_mix(entries: Iterable[str]) -> dict[str, float]:
    """Parse ["normal:0.7", "krylov:0.3"] into a dict."""
    out: dict[str, float] = {}
    for item in entries:
        if ":" not in item:
            raise typer.BadParameter(f"Invalid mix entry '{item}', expected name:frac")
        name, frac = item.split(":", 1)
        try:
            out[name] = float(frac)
        except ValueError as e:
            raise typer.BadParameter(f"Invalid fraction in '{item}'") from e
    return out


def main(
    config: Path = typer.Option(Path(__file__).parent / "config-ffnn.toml", help="Path to config file"),
    matrix: Path | None = typer.Option(None, help="Override matrix path"),
    rhs: Path | None = typer.Option(None, help="Override RHS path"),
    total_samples: int = typer.Option(6000, min=1, help="Total number of samples to generate"),
    mix: list[str] = typer.Option(
        ["normal:0.5", "krylov:0.5"],
        help="Strategy mix as name:fraction; fractions must sum to 1.0"
    ),
    krylov_iters: int = typer.Option(15, min=1, help="CG iterations for krylov strategy"),
    seed: int | None = typer.Option(None, help="Random seed (None for random)"),
    normalize: bool = typer.Option(True, help="Normalize A and rhs by ||A||_1"),
    shuffle: bool = typer.Option(True, help="Shuffle final dataset"),
):
    """Generate training data using config file."""
    # Parse mix
    mix_dict = _parse_mix(mix)

    print(f"Loading configuration from: {config}")
    print(f"Strategy mix: {mix_dict}")
    print(f"Total samples: {total_samples}")

    try:
        features_path, targets_path = generate_training_data(
            config_path=config,
            matrix_path=matrix,
            rhs_path=rhs,
            total_samples=total_samples,
            mix=mix_dict,
            krylov_iters=krylov_iters,
            seed=seed,
            normalize=normalize,
            shuffle=shuffle
        )

        print(f"Generated training data:")
        print(f"  Features: {features_path}")
        print(f"  Targets: {targets_path}")
        print("Done!")

    except Exception as e:
        print(f"Error: {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    try:
        typer.run(main)
    except KeyboardInterrupt:
        raise SystemExit(130)
