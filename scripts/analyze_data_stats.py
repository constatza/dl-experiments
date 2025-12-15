#!/usr/bin/env python3
"""CLI wrapper for dataset statistics inspection."""

from __future__ import annotations

import sys
from pathlib import Path

# Add graph-cg root to Python path so we can import from src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import typer

from src.cli.analysis import analyze_dataset


def main(
    reference_dir: Path = typer.Option(
        ...,
        "--reference-dir",
        "-r",
        help="Path to reference dataset directory (e.g., generated)",
    ),
    target_dir: Path = typer.Option(
        ...,
        "--target-dir",
        "-t",
        help="Path to target dataset directory (e.g., collected)",
    ),
):
    """Compare statistics between two datasets."""
    if not reference_dir.exists():
        print(f"Error: Reference directory not found: {reference_dir}")
        raise typer.Exit(code=1)
    
    if not target_dir.exists():
        print(f"Error: Target directory not found: {target_dir}")
        raise typer.Exit(code=1)

    gen_stats = analyze_dataset(reference_dir, "Reference")
    col_stats = analyze_dataset(target_dir, "Target")

    print(f"\n{'=' * 70}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 70}")

    print(
        f"\nRHS norm ratio (target/reference): "
        f"{col_stats['rhs_norm_mean'] / gen_stats['rhs_norm_mean']:.3f}"
    )
    print(
        f"Gradient magnitude ratio (target/reference): "
        f"{col_stats['gradient_mean'] / gen_stats['gradient_mean']:.3e}"
    )
    print(
        f"Loss at x=0 ratio (target/reference): "
        f"{col_stats['loss_at_zero_mean'] / gen_stats['loss_at_zero_mean']:.3e}"
    )
    print(
        f"Condition number ratio (target/reference): "
        f"{col_stats['condition_number'] / gen_stats['condition_number']:.3e}"
    )


if __name__ == "__main__":
    typer.run(main)