"""Integration tests for batch comparison CLI.

Tests that the script runs and arguments are parsed correctly.
Detailed logic is tested in src/configuration/loader and src/cli/comparison.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

# Add root to path for script imports

from neuralls.cli.compare_preconditioners import main
from neuralls.configuration.domain import ExperimentWorkspace
from neuralls.workflows.specs import ComparisonSpec, ComparisonOutcome

runner = CliRunner()

def test_script_help():
    """Test that the script help message renders successfully."""
    import typer
    test_app = typer.Typer()
    test_app.command()(main)
    
    result = runner.invoke(test_app, ["--help"])
    assert result.exit_code == 0
    assert "Compare preconditioner methods" in result.stdout

@patch("neuralls.cli.compare_preconditioners.run_batch_comparison")
def test_script_execution_no_experiments(mock_run_batch):
    """Exit with failure when no comparison specs are built."""
    mock_run_batch.side_effect = ValueError("No comparison specs were built.")

    import typer
    test_app = typer.Typer()
    test_app.command()(main)

    result = runner.invoke(test_app, ["--experiments", "dummy.toml", "--solver-config", "solver.toml"])

    assert result.exit_code != 0
    mock_run_batch.assert_called_once()


@patch("neuralls.cli.compare_preconditioners.run_batch_comparison")
def test_script_execution_mock_experiment(mock_run_batch, tmp_path):
    """Happy-path execution with one comparison spec."""
    workspace = ExperimentWorkspace(
        root_dir=tmp_path / "root",
        data_dir=tmp_path / "data",
        checkpoint_dir=tmp_path / "ckpts",
        figures_dir=tmp_path / "figs",
        predictions_dir=tmp_path / "preds",
        run_id="test_model",
    )
    outcome = ComparisonOutcome(
        name="test_exp",
        success=True,
        payload=MagicMock(
            summary="ok",
            preconditioners=["none"],
            recommendations={},
        ),
    )
    mock_run_batch.return_value = [outcome]

    import typer
    test_app = typer.Typer()
    test_app.command()(main)

    result = runner.invoke(test_app, ["--experiments", "dummy.toml", "--solver-config", "solver.toml", "--no-plots"])

    assert result.exit_code == 0
    mock_run_batch.assert_called_once()
