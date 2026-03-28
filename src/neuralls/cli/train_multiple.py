#!/usr/bin/env python3
"""Train every experiment in one registry and plot aggregate metrics."""

from __future__ import annotations

from pathlib import Path

import typer

from neuralls.shared.constants import EXIT_FAILURE, EXIT_KEYBOARD_INTERRUPT
from neuralls.composition.experiments.assembler import load_validated_master_config
from neuralls.composition.experiments.multi_training import train_batch, write_metric_report


def main(
    config: Path = typer.Argument(
        ...,
        help="Path to an experiments registry TOML.",
    ),
    metric: str = typer.Option(
        "eval/rel_error",
        help="MLflow metric key to plot across experiments.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        help=(
            "Directory for barplot and label JSON. "
            "Defaults to output_dir/training/ from the experiments TOML."
        ),
    ),
) -> None:
    """Train the registry-defined experiment batch and compare one metric.

    Use this command when you already trust the configs and want a compact
    batch-training report rather than stepping through single runs.
    """
    try:
        cfg, _ = load_validated_master_config(config)
        batch = train_batch(cfg=cfg, configs_dir=config.resolve().parent, output_root=output_dir)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        typer.echo(f"Error during batch training: {exc}", err=True)
        raise typer.Exit(code=EXIT_FAILURE)

    report_dir = write_metric_report(batch, metric=metric, output_dir=output_dir)
    plot_path = report_dir / f"batch_metric_{metric.replace('/', '_')}.png"
    if plot_path.exists():
        typer.echo(f"Saved barplot: {plot_path}")
    else:
        typer.echo(f"No data to plot for metric '{metric}'.", err=True)

    label_json_path = report_dir / "batch_training_labels.json"
    typer.echo(f"Saved label map: {label_json_path}")


def run() -> None:
    """Entry point for pyproject.toml script registration."""
    typer.run(main)


if __name__ == "__main__":
    try:
        typer.run(main)
    except KeyboardInterrupt:
        raise SystemExit(EXIT_KEYBOARD_INTERRUPT)
