"""Regression tests for test filesystem scope policy."""

from __future__ import annotations

from pathlib import Path

from neuralls.platform.config.loaders import load_comparison_config

from tests.benchmarks.from_file.conftest import DATA_DIR, L_PATH, MATRIX_PATHS, RHS_PATHS
from tests.scope_policy import new_root_artifact_dirs


def test_runtime_roots_stay_under_tmp_path(
    tmp_path: Path,
    runtime_root: Path,
    output_root: Path,
    processed_root: Path,
    figures_root: Path,
    mlflow_tracking_dir: Path,
    mlflow_artifact_dir: Path,
) -> None:
    """All canonical runtime paths should be sandboxed under the test temp dir."""
    expected_root = (tmp_path / "runtime").resolve()

    assert runtime_root == expected_root
    assert output_root == expected_root / "output"
    assert processed_root == expected_root / "processed"
    assert figures_root == expected_root / "figures"
    assert mlflow_tracking_dir == expected_root / "mlruns"
    assert mlflow_artifact_dir == expected_root / "mlartifacts"


def test_scope_guard_detects_new_root_artifact_dir(tmp_path: Path) -> None:
    """New top-level runtime artifacts must be reported by the guard helper."""
    baseline = set()

    leaked = tmp_path / "output"
    leaked.mkdir()

    assert new_root_artifact_dirs(tmp_path, baseline) == {"output"}


def test_benchmark_fixtures_are_vendored_under_tests() -> None:
    """From-file benchmark defaults must resolve under the committed tests tree."""
    assert DATA_DIR.is_dir()
    assert all(DATA_DIR in path.parents for path in MATRIX_PATHS)
    assert all(DATA_DIR in path.parents for path in RHS_PATHS)
    assert DATA_DIR in L_PATH.parents


def test_comparison_fixture_configs_load_from_tests_tree() -> None:
    """Comparison loader coverage should use test-owned fixture configs."""
    comparison_dir = Path(__file__).resolve().parent / "fixtures" / "configs" / "comparison"

    linear = load_comparison_config(comparison_dir / "linear.toml")
    ffnn = load_comparison_config(comparison_dir / "ffnn.toml")

    assert len(linear.preconditioners) == 1
    assert len(ffnn.preconditioners) == 1
