from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

from typer.testing import CliRunner


from neuralls.cli.compare_single import main
from neuralls.configuration.domain import ExperimentWorkspace
from neuralls.workflows.specs import ComparisonSpec, ComparisonOutcome, ComparisonParams

runner = CliRunner()


def test_single_help() -> None:
    import typer

    app = typer.Typer()
    app.command()(main)

    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "compare preconditioner methods" in result.stdout.lower()


@patch("neuralls.cli.compare_single.run_comparisons")
@patch("neuralls.cli.compare_single.build_direct_comparisons")
def test_single_executes(mock_build, mock_run, tmp_path: Path) -> None:
    workspace = ExperimentWorkspace(
        dataset_id="test-dataset",
        run_id="m",
        root_dir=tmp_path / "root",
        data_dir=tmp_path / "data",
    )
    spec = ComparisonSpec(
        name="exp",
        model_config=Path("m.toml"),
        data_config=Path("d.toml"),
        solver_config=Path("s.toml"),
        workspace=workspace,
        checkpoint=tmp_path / "ckpts" / "c.pt",
    )
    mock_build.return_value = [spec]
    outcome = ComparisonOutcome(name="exp", success=True, payload={"summary": "ok"})
    mock_run.return_value = [outcome]

    import typer

    app = typer.Typer()
    app.command()(main)

    result = runner.invoke(
        app,
        [
            "--model-config",
            "m.toml",
            "--data-config",
            "d.toml",
            "--solver-config",
            "s.toml",
            "--no-plots",
        ],
    )

    assert result.exit_code == 0
    mock_build.assert_called_once()
    mock_run.assert_called_once()
