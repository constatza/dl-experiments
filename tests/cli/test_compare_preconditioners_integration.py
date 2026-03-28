"""Integration tests for compare-preconditioners CLI."""

from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import MagicMock, patch

import typer
from typer.models import ArgumentInfo
from typer.testing import CliRunner
import tomli_w

from neuralls.cli.compare_preconditioners import main
from neuralls.shared.comparison.specs import ComparisonData, ComparisonGeneral, SolverParams
from neuralls.shared.comparison.results import ComparisonRecommendations, ComparisonResult
from neuralls.shared.comparison.specs import ComparisonOutcome, ComparisonParams

runner = CliRunner()


def _solver_params() -> ComparisonGeneral:
    return ComparisonGeneral(
        params=SolverParams(
            rtol=1.0e-6,
            atol=1.0e-14,
            max_iterations=10,
            stopping_criterion="residual_norm",
            m_max=20,
            breakdown_tol=None,
        ),
        data=ComparisonData(
            matrix_path=Path("/tmp/matrix.npy"),
            rhs_path=Path("/tmp/rhs.npy"),
        ),
    )


def _payload() -> ComparisonResult:
    return ComparisonResult(
        results={},
        summary="ok",
        solver_params=_solver_params(),
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


def _build_app() -> typer.Typer:
    app = typer.Typer()
    app.command()(main)
    return app


def test_script_help() -> None:
    """Help exits successfully and the CLI keeps a required positional config argument."""
    test_app = _build_app()
    result = runner.invoke(test_app, ["--help"])
    assert result.exit_code == 0

    parameters = inspect.signature(main).parameters
    config = parameters["config"].default
    assert isinstance(config, ArgumentInfo)
    assert config.default is ...


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
    test_app = _build_app()

    result = runner.invoke(
        test_app,
        [str(experiments_config)],
    )

    assert result.exit_code == 0
    mock_run.assert_called_once()
    assert mock_run.call_args.args[0] == experiments_config
    assert isinstance(mock_run.call_args.args[1], ComparisonParams)


@patch("neuralls.cli.compare_preconditioners.run_comparison_batch")
def test_compare_cli_requires_master_config(mock_run: MagicMock) -> None:
    """The CLI requires the experiments config positional argument."""
    test_app = _build_app()

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
    test_app = _build_app()

    result = runner.invoke(
        test_app,
        [str(experiments_config)],
    )

    assert result.exit_code != 0
    mock_run.assert_called_once()
