#!/usr/bin/env python3
"""CLI wrapper for dataset statistics inspection."""

from __future__ import annotations

import sys
from pathlib import Path

# Add graph-cg root to Python path so we can import from src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import typer

from src.constants import DEFAULT_PROCESSED_DATA_DIR
from src.cli.analysis import analyze_dataset


def main(
    generated_dir: Path = typer.Option(
        None,
        help="Path to generated dataset directory",
    ),
    collected_dir: Path = typer.Option(
        None,
        help="Path to collected dataset directory",
    ),
):
    # Resolve defaults
    if generated_dir is None:
        generated_dir = (
            DEFAULT_PROCESSED_DATA_DIR / "synthetic-baseline" / "generate-280-krylov50"
        )
    if collected_dir is None:
        collected_dir = DEFAULT_PROCESSED_DATA_DIR / "spectral-baseline" / "collect-504"

    gen_stats = analyze_dataset(generated_dir, "Generated")
    col_stats = analyze_dataset(collected_dir, "Collected")

    print(f"\n{'=' * 70}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 70}")

    print(
        f"\nRHS norm ratio (collected/generated): "
        f"{col_stats['rhs_norm_mean'] / gen_stats['rhs_norm_mean']:.3f}"
    )
    print(
        f"Gradient magnitude ratio (collected/generated): "
        f"{col_stats['gradient_mean'] / gen_stats['gradient_mean']:.3e}"
    )
    print(
        f"Loss at x=0 ratio (collected/generated): "
        f"{col_stats['loss_at_zero_mean'] / gen_stats['loss_at_zero_mean']:.3e}"
    )
    print(
        f"Condition number ratio (collected/generated): "
        f"{col_stats['condition_number'] / gen_stats['condition_number']:.3e}"
    )


if __name__ == "__main__":
    typer.run(main)
