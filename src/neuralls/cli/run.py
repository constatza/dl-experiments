"""Run the full generate-and-train matrix for one case config."""

from __future__ import annotations

import typer

from neuralls.cli.options import CaseConfigArgument, EnvFileOption, ProfileOption
from neuralls.composition.assignments.training_batch import run_assignment_matrix
from neuralls.composition.config import load_case_settings
from neuralls.shared.constants import DEFAULT_PROJECT_ROOT, EXIT_FAILURE


def run_case_matrix(
    config: CaseConfigArgument,
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force retraining even if a completed MLflow run already exists.",
    ),
    max_epochs: int | None = typer.Option(
        None,
        help="Override max training epochs for every assignment.",
    ),
    env_file: EnvFileOption = None,
    profile: ProfileOption = None,
) -> None:
    """Generate datasets and train every assignment declared in one case config."""
    if not config.exists():
        typer.echo(f"Error: Config file not found: {config}", err=True)
        raise typer.Exit(code=EXIT_FAILURE)

    try:
        settings = load_case_settings(config, env_file, profile=profile)
        typer.echo(f"Running case workflow from: {config}")
        if force:
            typer.echo("Force mode enabled: existing MLflow runs will be ignored.")

        results = run_assignment_matrix(
            case_config_path=config,
            settings=settings,
            force=force,
            project_root=DEFAULT_PROJECT_ROOT,
            max_epochs=max_epochs,
        )
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=EXIT_FAILURE) from exc

    typer.echo("")
    typer.echo("Summary")
    typer.echo("=" * 80)
    typer.echo(f"Completed {len(results)} assignments:")
    for result in results:
        status = "SUCCESS" if result.is_success else "FAILED"
        detail = f" ({result.error})" if result.error else ""
        label = result.assignment_display_name
        if label != result.assignment_id:
            label = f"{label} [{result.assignment_id}]"
        typer.echo(f"  {status} {label}{detail}")

    failures = [result for result in results if not result.is_success]
    if failures:
        raise typer.Exit(code=EXIT_FAILURE)
