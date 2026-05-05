"""Train every experiment declared in one case config."""

from __future__ import annotations

from pathlib import Path

import typer

from neuralls.cli.options import EnvFileOption, ProfileOption
from neuralls.composition.experiments.assembler import load_validated_case_config
from neuralls.composition.experiments.multi_training import train_batch, write_metric_report
from neuralls.composition.config import load_case_settings
from neuralls.shared.constants import EXIT_FAILURE


def train_case_batch(
    config: Path = typer.Argument(
        ...,
        help="Path to a case config TOML.",
    ),
    metric: str = typer.Option(
        "eval/rel_error",
        help="MLflow metric key to plot across experiments.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        help=("Directory for the batch plot and label JSON. Defaults to output_dir/training/."),
    ),
    env_file: EnvFileOption = None,
    profile: ProfileOption = None,
) -> None:
    """Train the registry-defined experiment batch and emit aggregate reporting."""
    try:
        settings = load_case_settings(config, env_file, profile=profile)
        cfg, _ = load_validated_case_config(config, settings)
        batch = train_batch(
            cfg=cfg,
            configs_dir=config.resolve().parent,
            settings=settings,
            output_root=output_dir,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        typer.echo(f"Error during batch training: {exc}", err=True)
        raise typer.Exit(code=EXIT_FAILURE) from exc

    report_dir = write_metric_report(batch, metric=metric, output_dir=output_dir)
    plot_path = report_dir / f"batch_metric_{metric.replace('/', '_')}.png"
    if plot_path.exists():
        typer.echo(f"Saved barplot: {plot_path}")
    else:
        typer.echo(f"No data to plot for metric '{metric}'.", err=True)

    typer.echo(f"Saved label map: {report_dir / 'batch_training_labels.json'}")
