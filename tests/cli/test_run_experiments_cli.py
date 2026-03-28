"""Fast unit tests for run_experiments CLI interface.

These tests mock the run_experiment_matrix workflow to test CLI behavior only:
- Argument parsing
- Error handling
- Workflow delegation
- Exit codes

For full workflow testing (actual training), see test_run_experiments_integration.py.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import patch

import typer
from typer.models import OptionInfo
from typer.testing import CliRunner

from neuralls.cli.run_experiments import main
from neuralls.shared.constants import DEFAULT_PROJECT_ROOT
from neuralls.shared.training import ExperimentResult

runner = CliRunner()


def _build_app() -> typer.Typer:
    app = typer.Typer()
    app.command()(main)
    return app


def test_run_experiments_help() -> None:
    """Help path exits cleanly and the CLI declares the expected options."""
    app = _build_app()

    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0

    parameters = inspect.signature(main).parameters
    config = parameters["config"].default
    force = parameters["force"].default
    max_epochs = parameters["max_epochs"].default

    assert isinstance(config, OptionInfo)
    assert config.param_decls == ("--config", "-c")
    assert config.default is ...

    assert isinstance(force, OptionInfo)
    assert force.param_decls == ("--force", "-f")
    assert force.default is False

    assert isinstance(max_epochs, OptionInfo)
    assert max_epochs.default is None


@patch("neuralls.cli.run_experiments.run_experiment_matrix")
def test_run_experiments_missing_config(mock_run, tmp_path: Path) -> None:
    """Missing config fails fast before the workflow is called."""
    nonexistent_config = tmp_path / "nonexistent.toml"

    app = _build_app()

    result = runner.invoke(app, ["--config", str(nonexistent_config)])

    assert result.exit_code != 0
    mock_run.assert_not_called()


@patch("neuralls.cli.run_experiments.run_experiment_matrix")
def test_run_experiments_success(mock_run, tmp_path: Path) -> None:
    """Successful invocation forwards defaults to the workflow."""
    config = tmp_path / "experiments.toml"
    config.write_text(f'output_dir = "{tmp_path / "output"}"\n')

    mock_run.return_value = [
        ExperimentResult(experiment_id="exp1", experiment_display_name="exp1", status="Success"),
        ExperimentResult(experiment_id="exp2", experiment_display_name="exp2", status="Success"),
    ]

    app = _build_app()

    result = runner.invoke(app, ["--config", str(config)])

    assert result.exit_code == 0
    mock_run.assert_called_once()

    call_kwargs = mock_run.call_args.kwargs
    assert call_kwargs["experiments_config_path"] == config
    assert call_kwargs["force"] is False
    assert call_kwargs["project_root"] == DEFAULT_PROJECT_ROOT
    assert call_kwargs["max_epochs"] is None


@patch("neuralls.cli.run_experiments.run_experiment_matrix")
def test_run_experiments_with_failures(mock_run, tmp_path: Path) -> None:
    """Any failed experiment produces a non-zero exit code."""
    config = tmp_path / "experiments.toml"
    config.write_text(f'output_dir = "{tmp_path / "output"}"\n')

    mock_run.return_value = [
        ExperimentResult(experiment_id="exp1", experiment_display_name="exp1", status="Success"),
        ExperimentResult(
            experiment_id="exp2",
            experiment_display_name="exp2",
            status="Failed",
            error="Dataset generation failed",
        ),
    ]

    app = _build_app()

    result = runner.invoke(app, ["--config", str(config)])

    assert result.exit_code != 0
    mock_run.assert_called_once()

    call_kwargs = mock_run.call_args.kwargs
    assert call_kwargs["experiments_config_path"] == config
    assert call_kwargs["force"] is False


@patch("neuralls.cli.run_experiments.run_experiment_matrix")
def test_run_experiments_force_flag(mock_run, tmp_path: Path) -> None:
    """The force flag is forwarded to the workflow."""
    config = tmp_path / "experiments.toml"
    config.write_text(f'output_dir = "{tmp_path / "output"}"\n')

    mock_run.return_value = [
        ExperimentResult(experiment_id="exp1", experiment_display_name="exp1", status="Success"),
    ]

    app = _build_app()

    result = runner.invoke(app, ["--config", str(config), "--force"])

    assert result.exit_code == 0
    mock_run.assert_called_once()

    call_kwargs = mock_run.call_args.kwargs
    assert call_kwargs["experiments_config_path"] == config
    assert call_kwargs["force"] is True
    assert call_kwargs["project_root"] == DEFAULT_PROJECT_ROOT
    assert call_kwargs["max_epochs"] is None
