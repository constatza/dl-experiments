#!/usr/bin/env python3
"""CLI wrapper for preconditioner comparison utilities."""

from __future__ import annotations

import os
from pathlib import Path

import typer

from src.constants import (
    NOISE_STRATEGY_NONE,
    DEFAULT_NOISE_LEVEL,
    DEFAULT_NOISE_SEED,
    EXIT_FAILURE,
    EXIT_KEYBOARD_INTERRUPT,
    SYMBOL_CHECKMARK,
)
from src.cli.comparison import compare_preconditioners

os.environ.setdefault("MPLBACKEND", "Agg")


def parse_combo_spec(spec: str) -> tuple[str, str, str]:
    parts = [part.strip() for part in spec.split(":")]
    if len(parts) == 2:
        warm_name, precond_name = parts
        helper_name = "none"
    elif len(parts) == 3:
        warm_name, precond_name, helper_name = parts
    else:
        raise ValueError("Use WARM:PRECONDITIONER[:HELPER] format, e.g. 'neural_warm_start:neural'.")

    if not warm_name or not precond_name or not helper_name:
        raise ValueError(f"Combination '{spec}' contains empty value(s).")

    return warm_name, precond_name, helper_name


def main(
    config: Path = typer.Option(
        Path(__file__).parent / "configs" / "ffnn.toml", help="Path to TOML config"
    ),
    data_config: Path = typer.Option(
        Path(__file__).parent / "data-configs" / "collect-504.toml",
        help="Path to data config providing dataset metadata",
    ),
    matrix: Path | None = typer.Option(None, help="Override matrix path"),
    rhs: Path | None = typer.Option(None, help="Override RHS path"),
    checkpoint: Path | None = typer.Option(None, help="Override checkpoint path"),
    warm_checkpoint: Path | None = typer.Option(
        None, help="Checkpoint for warm-start model (defaults to --checkpoint)"
    ),
    helper_checkpoint: Path | None = typer.Option(
        None, help="Checkpoint for neural step helper (defaults to --checkpoint)"
    ),
    pca: Path | None = typer.Option(None, help="Path to PCA model"),
    output_dir: Path | None = typer.Option(None, help="Override output directory"),
    noise_strategy: str = typer.Option(NOISE_STRATEGY_NONE, help="Noise strategy"),
    noise_level: float = typer.Option(DEFAULT_NOISE_LEVEL, help="Noise level parameter"),
    noise_seed: int | None = typer.Option(DEFAULT_NOISE_SEED, help="Random seed for noise"),
    figures_dir: Path | None = typer.Option(
        None, help="Override directory for saved plots"
    ),
    breakdown_tol: float | None = typer.Option(
        None, help="Breakdown tolerance for CG denominator checks"
    ),
    combo: list[str] = typer.Option(
        None,
        "--combo",
        "-c",
        help="Add warm/preconditioner[/helper] combo in the form WARM:PRECONDITIONER[:HELPER]. "
        "Repeat for multiple entries.",
    ),
):
    """Compare preconditioner, warm-start, and helper configurations."""

    combination_plan: list[tuple[str, str, str]] | None = None
    if combo:
        try:
            combination_plan = [parse_combo_spec(item) for item in combo]
        except ValueError as exc:
            typer.secho(f"Invalid combination: {exc}", err=True)
            raise typer.Exit(code=EXIT_FAILURE)

    try:
        results = compare_preconditioners(
            config_path=config,
            data_config_path=data_config,
            matrix_path=matrix,
            rhs_path=rhs,
            checkpoint_path=checkpoint,
            warm_start_checkpoint_path=warm_checkpoint,
            step_helper_checkpoint_path=helper_checkpoint,
            pca_path=pca,
            output_dir=output_dir,
            noise_strategy=noise_strategy,
            noise_level=noise_level,
            noise_seed=noise_seed,
            save_plots=True,
            figures_dir=figures_dir,
            breakdown_tol=breakdown_tol,
            custom_combinations=combination_plan,
        )

        print(f"Available preconditioners: {results['preconditioners']}")
        print(f"Available warm starts: {results['warm_starts']}")

        print("\n" + "=" * 60)
        print(results["summary"])
        print("=" * 60)

        recs = results.get("recommendations", {})
        best_overall = recs.get("best_overall") if isinstance(recs, dict) else None
        if best_overall:
            print("\nBest overall combination:")
            print(
                f"  label={best_overall['label']} | iterations={best_overall['iterations']} "
                f"| residual={best_overall['residual']:.3e}"
            )

        best_by_warm = recs.get("best_by_warm_start") if isinstance(recs, dict) else None
        if isinstance(best_by_warm, dict) and best_by_warm:
            print("\nWarm-start suggestions:")
            for warm_name, entry in best_by_warm.items():
                if warm_name == "none":
                    continue
                print(
                    f"  {warm_name}: label={entry['label']} | preconditioner={entry['preconditioner']} "
                    f"| helper={entry['step_helper']} | iterations={entry['iterations']}"
                )

        if results["plot_paths"]:
            print("\nGenerated plots:")
            for plot_type, path in results["plot_paths"].items():
                print(f"  {plot_type}: {path}")

        print(f"\n{SYMBOL_CHECKMARK} Analysis complete!")

    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}")
        raise typer.Exit(code=EXIT_FAILURE)


if __name__ == "__main__":
    try:
        typer.run(main)
    except KeyboardInterrupt:
        raise SystemExit(EXIT_KEYBOARD_INTERRUPT)
