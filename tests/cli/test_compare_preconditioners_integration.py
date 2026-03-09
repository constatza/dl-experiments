"""Integration tests for compare-preconditioners CLI.

Tests that the script runs and arguments are parsed correctly in both
pipeline mode (with --comparison-run) and standalone mode (without).
Detailed logic is tested in tests/workflows/.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from neuralls.cli.compare_preconditioners import main
from neuralls.workflows.results import ComparisonRecommendations, ComparisonResult
from neuralls.workflows.specs import ComparisonOutcome

runner = CliRunner()


def _payload() -> ComparisonResult:
    return ComparisonResult(
        results={},
        summary="ok",
        solver_params=object(),
        preconditioners=("none",),
        recommendations=ComparisonRecommendations(),
    )


@pytest.fixture
def comparison_run_file(tmp_path: Path) -> Path:
    """Minimal comparison_run.json for CLI tests.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Path to a written comparison_run.json file.
    """
    artifact_location = str(tmp_path / "comparisons" / "artifacts")
    payload = {
        "mlflow_run_id": "abc123",
        "mlflow_experiment_id": "0",
        "tracking_uri": f"sqlite:///{tmp_path / 'comparisons' / 'mlflow.db'}",
        "artifact_location": artifact_location,
        "checkpoint_map": {},
        "artifact_uri": "mlartifacts/0/abc123/artifacts",
    }
    path = tmp_path / "comparison_run.json"
    path.write_text(json.dumps(payload))
    return path


@pytest.fixture
def comparison_config_file(tmp_path: Path) -> Path:
    """Minimal comparison config TOML file.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Path to a written comparison.toml file.
    """
    path = tmp_path / "comparison.toml"
    path.write_text(
        "schema_version = 3\n"
        "\n"
        "[general]\n"
        "\n"
        "[general.params]\n"
        "rtol = 1e-6\n"
        "atol = 1e-14\n"
        "max_iterations = 10\n"
        "\n"
        "[general.data]\n"
        f'matrix_path = "{tmp_path}"\n'
        f'rhs_path = "{tmp_path}"\n'
        "\n"
        "[general.tracking]\n"
        f'tracking_uri = "sqlite:///{tmp_path / "comparisons" / "mlflow.db"}"\n'
        f'artifact_location = "{tmp_path / "comparisons" / "artifacts"}"\n'
        "\n"
        "[[preconditioners]]\n"
        'name = "none"\n'
        'type = "identity"\n'
    )
    return path


def test_script_help() -> None:
    """Test that the script help message renders successfully."""
    test_app = typer.Typer()
    test_app.command()(main)

    result = runner.invoke(test_app, ["--help"])
    assert result.exit_code == 0
    assert "Compare preconditioner methods" in result.stdout


@patch("neuralls.cli.compare_preconditioners.run_comparison")
def test_pipeline_mode_with_comparison_run(
    mock_run: MagicMock,
    comparison_run_file: Path,
    comparison_config_file: Path,
) -> None:
    """Pipeline mode: run_comparison called with loaded ComparisonRun.

    Args:
        mock_run: Mocked run_comparison function.
        comparison_run_file: Path to comparison_run.json fixture.
        comparison_config_file: Path to comparison config TOML fixture.
    """
    outcome = ComparisonOutcome(
        name="solver",
        success=True,
        payload=_payload(),
    )
    mock_run.return_value = [outcome]

    test_app = typer.Typer()
    test_app.command()(main)

    result = runner.invoke(
        test_app,
        [
            "--comparison-config", str(comparison_config_file),
            "--comparison-run", str(comparison_run_file),
            "--no-plots",
        ],
    )

    assert result.exit_code == 0
    mock_run.assert_called_once()
    call_args = mock_run.call_args
    assert call_args.args[1].save_plots is False
    # comparison_run should be the loaded ComparisonRun (not None)
    assert call_args.args[2] is not None


@patch("neuralls.cli.compare_preconditioners.run_comparison")
def test_standalone_mode_without_comparison_run(
    mock_run: MagicMock,
    comparison_config_file: Path,
) -> None:
    """Standalone mode: run_comparison called with comparison_run=None.

    Args:
        mock_run: Mocked run_comparison function.
        comparison_config_file: Path to comparison config TOML fixture.
    """
    outcome = ComparisonOutcome(
        name="solver",
        success=True,
        payload=_payload(),
    )
    mock_run.return_value = [outcome]

    test_app = typer.Typer()
    test_app.command()(main)

    result = runner.invoke(
        test_app,
        [
            "--comparison-config", str(comparison_config_file),
            "--no-plots",
        ],
    )

    assert result.exit_code == 0
    mock_run.assert_called_once()
    call_args = mock_run.call_args
    # comparison_run should be None in standalone mode
    assert call_args.args[2] is None


@patch("neuralls.cli.compare_preconditioners.run_comparison")
def test_failure_outcome_exits_nonzero(
    mock_run: MagicMock,
    comparison_config_file: Path,
) -> None:
    """Failed comparison outcome causes non-zero exit code.

    Args:
        mock_run: Mocked run_comparison function.
        comparison_config_file: Path to comparison config TOML fixture.
    """
    outcome = ComparisonOutcome(name="solver", success=False, error="test error")
    mock_run.return_value = [outcome]

    test_app = typer.Typer()
    test_app.command()(main)

    result = runner.invoke(
        test_app,
        ["--comparison-config", str(comparison_config_file), "--no-plots"],
    )

    assert result.exit_code != 0


@patch("neuralls.cli.compare_preconditioners.run_comparison")
def test_invalid_comparison_run_falls_back_to_standalone(
    mock_run: MagicMock,
    tmp_path: Path,
    comparison_config_file: Path,
) -> None:
    """Invalid comparison_run input should warn and continue with cr=None."""
    outcome = ComparisonOutcome(
        name="solver",
        success=True,
        payload=_payload(),
    )
    mock_run.return_value = [outcome]

    bad_cr = tmp_path / "missing-comparison-run.json"
    test_app = typer.Typer()
    test_app.command()(main)

    result = runner.invoke(
        test_app,
        [
            "--comparison-config", str(comparison_config_file),
            "--comparison-run", str(bad_cr),
            "--no-plots",
        ],
    )

    assert result.exit_code == 0
    mock_run.assert_called_once()
    assert mock_run.call_args.args[2] is None
