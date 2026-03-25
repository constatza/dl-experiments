"""Fast unit tests for run_experiments CLI interface.

These tests mock the run_experiment_matrix workflow to test CLI behavior only:
- Argument parsing
- Error handling
- Output formatting
- Exit codes

For full workflow testing (actual training), see test_run_experiments_integration.py.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import typer
from typer.testing import CliRunner

from neuralls.cli.run_experiments import main
from neuralls.workflows.reporting import ExperimentResult

runner = CliRunner()


def test_run_experiments_help() -> None:
    """Test --help flag displays help text."""
    app = typer.Typer()
    app.command()(main)

    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "train all models" in result.stdout.lower()
    assert "--force" in result.stdout
    assert "--config" in result.stdout


@patch("neuralls.cli.run_experiments.run_experiment_matrix")
def test_run_experiments_missing_config(mock_run, tmp_path: Path) -> None:
    """Test error handling when config file doesn't exist."""
    nonexistent_config = tmp_path / "nonexistent.toml"

    app = typer.Typer()
    app.command()(main)

    result = runner.invoke(app, ["--config", str(nonexistent_config)])

    assert result.exit_code != 0
    assert "error" in result.stdout.lower()
    assert "not found" in result.stdout.lower()
    mock_run.assert_not_called()


@patch("neuralls.cli.run_experiments.run_experiment_matrix")
def test_run_experiments_success(mock_run, tmp_path: Path) -> None:
    """Test successful execution with mocked workflow."""
    # Create dummy config file
    config = tmp_path / "experiments.toml"
    config.write_text(f'output_dir = "{tmp_path / "output"}"\n')

    # Mock successful results
    mock_run.return_value = [
        ExperimentResult(experiment_id="exp1", experiment_display_name="exp1", status="Success"),
        ExperimentResult(experiment_id="exp2", experiment_display_name="exp2", status="Success"),
    ]

    app = typer.Typer()
    app.command()(main)

    result = runner.invoke(app, ["--config", str(config)])

    assert result.exit_code == 0
    assert "✓ exp1" in result.stdout
    assert "✓ exp2" in result.stdout
    assert "Completed 2 experiments" in result.stdout
    mock_run.assert_called_once()


@patch("neuralls.cli.run_experiments.run_experiment_matrix")
def test_run_experiments_with_failures(mock_run, tmp_path: Path) -> None:
    """Test error output formatting and exit code when experiments fail."""
    # Create dummy config file
    config = tmp_path / "experiments.toml"
    config.write_text(f'output_dir = "{tmp_path / "output"}"\n')

    # Mock mixed results (one success, one failure)
    mock_run.return_value = [
        ExperimentResult(experiment_id="exp1", experiment_display_name="exp1", status="Success"),
        ExperimentResult(
            experiment_id="exp2",
            experiment_display_name="exp2",
            status="Failed",
            error="Dataset generation failed",
        ),
    ]

    app = typer.Typer()
    app.command()(main)

    result = runner.invoke(app, ["--config", str(config)])

    assert result.exit_code != 0
    assert "✓ exp1" in result.stdout
    assert "✗ exp2" in result.stdout
    assert "Dataset generation failed" in result.stdout
    mock_run.assert_called_once()


@patch("neuralls.cli.run_experiments.run_experiment_matrix")
def test_run_experiments_force_flag(mock_run, tmp_path: Path) -> None:
    """Test --force flag is passed correctly to workflow."""
    # Create dummy config file
    config = tmp_path / "experiments.toml"
    config.write_text(f'output_dir = "{tmp_path / "output"}"\n')

    # Mock successful results
    mock_run.return_value = [
        ExperimentResult(experiment_id="exp1", experiment_display_name="exp1", status="Success"),
    ]

    app = typer.Typer()
    app.command()(main)

    result = runner.invoke(app, ["--config", str(config), "--force"])

    assert result.exit_code == 0
    assert "Force mode enabled" in result.stdout
    # Verify force=True was passed to run_experiment_matrix
    mock_run.assert_called_once()
    call_kwargs = mock_run.call_args.kwargs
    assert call_kwargs["force"] is True
