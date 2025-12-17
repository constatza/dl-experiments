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

from scripts.compare_methods import main
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

@patch("scripts.compare_methods.build_batch_comparisons")
def test_script_execution_no_experiments(mock_build_batch):
    """Exit with failure when no comparison specs are built."""
    mock_build_batch.return_value = []

    import typer
    test_app = typer.Typer()
    test_app.command()(main)

    result = runner.invoke(test_app, ["--experiments", "dummy.toml"])

    assert result.exit_code != 0
    mock_build_batch.assert_called_once()


@patch("scripts.compare_methods.run_comparisons")
@patch("scripts.compare_methods.build_batch_comparisons")
def test_script_execution_mock_experiment(mock_build_batch, mock_run, tmp_path):
    """Happy-path execution with one comparison spec."""
    workspace = ExperimentWorkspace(
        root_dir=tmp_path / "root",
        data_dir=tmp_path / "data",
        checkpoint_dir=tmp_path / "ckpts",
        figures_dir=tmp_path / "figs",
        predictions_dir=tmp_path / "preds",
        run_id="test_model",
    )
    spec = ComparisonSpec(
        name="test_exp",
        model_config=Path("m.toml"),
        data_config=Path("d.toml"),
        solver_config=Path("s.toml"),
        workspace=workspace,
        checkpoint=tmp_path / "ckpts" / "ckpt.pt",
    )
    mock_build_batch.return_value = [spec]
    outcome = ComparisonOutcome(name="test_exp", success=True, payload={"summary": "ok"})
    mock_run.return_value = [outcome]

    import typer
    test_app = typer.Typer()
    test_app.command()(main)

    result = runner.invoke(test_app, ["--experiments", "dummy.toml", "--no-plots"])

    assert result.exit_code == 0
    mock_build_batch.assert_called_once()
    mock_run.assert_called_once()
