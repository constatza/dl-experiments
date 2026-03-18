"""Integration tests for compare-preconditioners CLI."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import typer
from typer.testing import CliRunner
import tomli_w

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


def _write_experiments_config(path: Path) -> None:
    payload = {
        "mlflow": {
            "tracking_uri": f"sqlite:///{(path.parent / 'mlruns' / 'mlflow.db').as_posix()}"
        },
        "comparisons": [{"id": "linear", "path": "comparison/linear.toml"}],
    }
    with path.open("wb") as fh:
        tomli_w.dump(payload, fh)


def test_script_help() -> None:
    """The CLI help should render successfully."""
    test_app = typer.Typer()
    test_app.command()(main)
    result = runner.invoke(test_app, ["--help"])
    assert result.exit_code == 0
    assert "experiments registry TOML" in result.stdout
    assert "--comparison-config" not in result.stdout
    assert "--comparison-profile" not in result.stdout


@patch("neuralls.cli.compare_preconditioners.run_comparison_batch")
def test_compare_cli_invokes_batch_workflow(mock_run: MagicMock, tmp_path: Path) -> None:
    """CLI forwards the experiments config path to the batch workflow."""
    experiments_config = tmp_path / "experiments.toml"
    _write_experiments_config(experiments_config)
    mock_run.return_value = [
        ComparisonOutcome(
            comparison_id="solver",
            comparison_display_name="solver",
            success=True,
            payload=_payload(),
        )
    ]
    test_app = typer.Typer()
    test_app.command()(main)

    result = runner.invoke(
        test_app,
        [str(experiments_config)],
    )

    assert result.exit_code == 0
    mock_run.assert_called_once()
    assert mock_run.call_args.args[0] == experiments_config
    assert len(mock_run.call_args.args) == 2


@patch("neuralls.cli.compare_preconditioners.run_comparison_batch")
def test_compare_cli_requires_master_config(mock_run: MagicMock) -> None:
    """The CLI requires the experiments config positional argument."""
    test_app = typer.Typer()
    test_app.command()(main)

    result = runner.invoke(test_app, [])

    assert result.exit_code != 0
    mock_run.assert_not_called()


@patch("neuralls.cli.compare_preconditioners.run_comparison_batch")
def test_failure_outcome_exits_nonzero(mock_run: MagicMock, tmp_path: Path) -> None:
    """Failed comparison outcome causes non-zero exit code."""
    experiments_config = tmp_path / "experiments.toml"
    _write_experiments_config(experiments_config)
    mock_run.return_value = [
        ComparisonOutcome(
            comparison_id="solver",
            comparison_display_name="solver",
            success=False,
            error="test error",
        )
    ]
    test_app = typer.Typer()
    test_app.command()(main)

    result = runner.invoke(
        test_app,
        [str(experiments_config)],
    )

    assert result.exit_code != 0
