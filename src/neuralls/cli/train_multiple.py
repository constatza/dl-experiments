#!/usr/bin/env python3
"""Train every experiment in one registry and plot aggregate metrics."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from neuralls.constants import EXIT_FAILURE, EXIT_KEYBOARD_INTERRUPT
from neuralls.io.loader import load_validated_master_config
from neuralls.plotting import plot_metric_comparison
from neuralls.workflows.multi_training import train_batch


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

    plot_output_dir = output_dir if output_dir is not None else batch.output_dir
    plot_output_dir.mkdir(parents=True, exist_ok=True)

    # Build (label, value) pairs for the requested metric
    labels: list[str] = []
    values: list[float] = []
    missing: list[str] = []

    for result in batch.results:
        if metric in result.metrics:
            labels.append(result.label)
            values.append(result.metrics[metric])
        else:
            missing.append(f"{result.label} ({result.experiment_display_name})")

    if missing:
        typer.echo(
            f"Warning: metric '{metric}' missing for experiments: {', '.join(missing)}",
            err=True,
        )

    # Build legend: label → "experiment_display_name (run: run_id)"
    legend = {
        r.label: (
            f"{r.experiment_display_name} (run: {r.mlflow_run_id})"
            if r.mlflow_run_id
            else r.experiment_display_name
        )
        for r in batch.results
        if r.label in labels
    }

    if labels:
        plot_path = plot_output_dir / f"batch_metric_{metric.replace('/', '_')}.png"
        plot_metric_comparison(
            labels=labels,
            values=values,
            metric_name=metric,
            legend=legend,
            save_path=plot_path,
        )
        typer.echo(f"Saved barplot: {plot_path}")
    else:
        typer.echo(f"No data to plot for metric '{metric}'.", err=True)

    # Save label map JSON
    label_json_path = plot_output_dir / "batch_training_labels.json"
    label_json_path.write_text(json.dumps(batch.label_map, indent=2))
    typer.echo(f"Saved label map: {label_json_path}")


def run() -> None:
    """Entry point for pyproject.toml script registration."""
    typer.run(main)


if __name__ == "__main__":
    try:
        typer.run(main)
    except KeyboardInterrupt:
        raise SystemExit(EXIT_KEYBOARD_INTERRUPT)
