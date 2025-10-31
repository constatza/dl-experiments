#!/usr/bin/env python3
"""Lightweight CLI wrapper around src.cli.prediction.run_inference."""

from __future__ import annotations

import os
from pathlib import Path

import typer
from dlkit.core.postprocessing import summarize

from src.constants import EXIT_FAILURE, EXIT_KEYBOARD_INTERRUPT
from src.cli.prediction import run_inference

os.environ.setdefault("MPLBACKEND", "Agg")


def main(
    config: Path = typer.Option(
        Path(__file__).parent / "configs" / "ffnn.toml", help="Path to config file"
    ),
    data_config: Path = typer.Option(
        Path(__file__).parent / "data-configs" / "collect-504.toml",
        help="Path to data config providing dataset metadata",
    ),
    checkpoint: Path | None = typer.Option(None, help="Override checkpoint path"),
    features: Path | None = typer.Option(
        None, help="Override features path (required if config has no DATASET)"
    ),
    targets: Path | None = typer.Option(
        None, help="Override targets path (required if config has no DATASET)"
    ),
    no_plots: bool = typer.Option(False, help="Skip saving plots"),
    figures_dir: Path | None = typer.Option(
        None, help="Override directory for saved plots"
    ),
):
    """Run inference using a DLKit configuration."""
    print(f"Loading configuration from: {config}")

    try:
        results = run_inference(
            config_path=config,
            data_config_path=data_config,
            checkpoint_path=checkpoint,
            features_path=features,
            targets_path=targets,
            save_plots=not no_plots,
            figures_dir=figures_dir,
        )

        try:
            print(f"Prediction summary: {summarize(results['predictions'])}")
        except Exception:
            pass

        if results["y_true"] is not None and results["y_pred"] is not None:
            print(f"Generated predictions for {len(results['y_true'])} samples")
            if results["plot_path"]:
                print(f"Saved plots to: {results['plot_path']}")
        else:
            print("Could not extract matching prediction/target arrays for plotting.")

        print(f"Inference completed in {results['duration_seconds']:.2f}s")

    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}")
        raise typer.Exit(code=EXIT_FAILURE)


if __name__ == "__main__":
    try:
        typer.run(main)
    except KeyboardInterrupt:
        raise SystemExit(EXIT_KEYBOARD_INTERRUPT)
