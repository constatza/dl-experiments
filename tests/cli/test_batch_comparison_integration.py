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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.compare_methods import main

runner = CliRunner()

def test_script_help():
    """Test that the script help message renders successfully."""
    import typer
    test_app = typer.Typer()
    test_app.command()(main)
    
    result = runner.invoke(test_app, ["--help"])
    assert result.exit_code == 0
    assert "Compare preconditioner methods" in result.stdout

@patch("scripts.compare_methods.load_experiments")
def test_script_execution_no_experiments(mock_load):
    """Test script execution when no experiments are returned."""
    mock_load.return_value = []
    
    import typer
    test_app = typer.Typer()
    test_app.command()(main)
    
    result = runner.invoke(test_app, ["--experiments", "dummy.toml"])
    
    assert result.exit_code == 0
    mock_load.assert_called_once()

@patch("scripts.compare_methods.load_experiments")
@patch("scripts.compare_methods.run_single_comparison")
@patch("scripts.compare_methods.get_latest_checkpoint")
def test_script_execution_mock_experiment(mock_get_ckpt, mock_run_single, mock_load, tmp_path):
    """Test script with a mock experiment."""
    # Setup mocks
    mock_context = MagicMock()
    mock_context.training.base_dir = tmp_path
    mock_context.training.checkpoint_dir = tmp_path / "checkpoints"
    mock_context.run_id = "test_run"
    mock_context.data.dataset_id = "test_data"
    
    mock_load.return_value = [
        ("test_exp", MagicMock(), mock_context, Path("m.toml"), Path("d.toml"), Path("s.toml"))
    ]
    
    # Mock checkpoint path to be a MagicMock so we can set exists()
    mock_path = MagicMock(spec=Path)
    mock_path.exists.return_value = True
    mock_path.__str__.return_value = "ckpt.pt"
    mock_get_ckpt.return_value = mock_path
    
    mock_run_single.return_value = {
        "preconditioners": ["none", "neural"],
        "summary": "Mock Summary",
        "results": {},
        "plot_paths": {"convergence": Path("plot.png")}
    }
    
    import typer
    test_app = typer.Typer()
    test_app.command()(main)
    
    result = runner.invoke(test_app, ["--experiments", "dummy.toml", "--no-plots"])
    
    assert result.exit_code == 0
    mock_load.assert_called_once()
    mock_get_ckpt.assert_called()
    mock_run_single.assert_called_once()
