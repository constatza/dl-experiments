"""Tests for modular export package."""

from pathlib import Path

import tomllib

import numpy as np
import pytest

from neuralls.solver.factories import pcg
from neuralls.solver.models.config import SolverConfig
from neuralls.solver.models.result import SolverResult
from neuralls.solver.monitoring.iteration_history import IterationHistory
from neuralls.solver.monitoring.trace_mode import TraceMode
from neuralls.solver.utils.export import (
    ArrayFormat,
    quick_export,
    save_full_config,
    save_iteration_history,
    save_solver_data,
    save_solver_result,
    save_system_arrays,
)


@pytest.fixture
def simple_system():
    """Fixture providing a simple 3x3 system."""
    A = np.eye(3) * 2
    b = np.ones(3)
    x_sol = np.ones(3) * 0.5
    x0 = np.zeros(3)
    return A, b, x_sol, x0


@pytest.fixture
def solver_config():
    """Fixture providing a solver configuration."""
    return SolverConfig(
        algorithm="preconditioned_cg",
        rtol=1e-6,
        atol=1e-10,
        maxiter=100,
        trace_mode=TraceMode.MINIMAL,
    )


@pytest.fixture
def solver_result_minimal():
    """Fixture providing a solver result with minimal history."""
    history = IterationHistory(mode=TraceMode.MINIMAL)
    history.log_iteration(residual_norm=1.0)
    history.log_iteration(residual_norm=0.5)
    history.log_iteration(residual_norm=0.1)

    return SolverResult(
        converged=True,
        iterations=3,
        residual=0.1,
        residual_abs=0.1,
        rhs_norm=1.0,
        breakdown=False,
        tol=1e-6,
        atol=1e-10,
        stopping_criterion="converged",
        iteration_history=history,
    )


@pytest.fixture
def solver_result_full():
    """Fixture providing a solver result with full history."""
    history = IterationHistory(mode=TraceMode.FULL)
    r1 = np.array([1.0, 0.0, 0.0])
    r2 = np.array([0.5, 0.0, 0.0])
    x1 = np.array([0.1, 0.0, 0.0])
    x2 = np.array([0.2, 0.0, 0.0])
    p1 = np.array([1.0, 0.0, 0.0])
    p2 = np.array([0.5, 0.0, 0.0])

    history.log_iteration(residual_norm=1.0, residual=r1, solution=x1, direction=p1)
    history.log_iteration(residual_norm=0.5, residual=r2, solution=x2, direction=p2)

    return SolverResult(
        converged=True,
        iterations=2,
        residual=0.5,
        residual_abs=0.5,
        rhs_norm=1.0,
        breakdown=False,
        tol=1e-6,
        atol=1e-10,
        stopping_criterion="converged",
        iteration_history=history,
    )


class TestSystemArraysExport:
    """Test save_system_arrays function."""

    def test_npz_format(self, tmp_path: Path, simple_system):
        """Test NPZ format export."""
        A, b, x_sol, x0 = simple_system

        save_system_arrays(tmp_path, A, b, x_sol, x0=x0, format=ArrayFormat.NPZ)

        # Verify NPZ file exists
        npz_file = tmp_path / "system.npz"
        assert npz_file.exists()

        # Load and verify contents
        with np.load(npz_file) as data:
            assert "A" in data
            assert "b" in data
            assert "x0" in data
            assert "x_sol" in data
            np.testing.assert_array_equal(data["A"], A)
            np.testing.assert_array_equal(data["b"], b)
            np.testing.assert_array_equal(data["x0"], x0)
            np.testing.assert_array_equal(data["x_sol"], x_sol)

    def test_txt_format(self, tmp_path: Path, simple_system):
        """Test TXT format export."""
        A, b, x_sol, x0 = simple_system

        save_system_arrays(tmp_path, A, b, x_sol, x0=x0, format=ArrayFormat.TXT)

        # Verify TXT files exist
        assert (tmp_path / "A.txt").exists()
        assert (tmp_path / "b.txt").exists()
        assert (tmp_path / "x0.txt").exists()
        assert (tmp_path / "x_sol.txt").exists()

        # Load and verify contents
        A_loaded = np.loadtxt(tmp_path / "A.txt")
        b_loaded = np.loadtxt(tmp_path / "b.txt")
        x0_loaded = np.loadtxt(tmp_path / "x0.txt")
        x_sol_loaded = np.loadtxt(tmp_path / "x_sol.txt")

        np.testing.assert_allclose(A_loaded, A)
        np.testing.assert_allclose(b_loaded, b)
        np.testing.assert_allclose(x0_loaded, x0)
        np.testing.assert_allclose(x_sol_loaded, x_sol)

    def test_both_format(self, tmp_path: Path, simple_system):
        """Test BOTH format export."""
        A, b, x_sol, x0 = simple_system

        save_system_arrays(tmp_path, A, b, x_sol, x0=x0, format=ArrayFormat.BOTH)

        # Verify both NPZ and TXT files exist
        assert (tmp_path / "system.npz").exists()
        assert (tmp_path / "A.txt").exists()
        assert (tmp_path / "b.txt").exists()
        assert (tmp_path / "x0.txt").exists()
        assert (tmp_path / "x_sol.txt").exists()

    def test_none_x0_uses_zeros(self, tmp_path: Path, simple_system):
        """Test that x0=None results in zero initial guess."""
        A, b, x_sol, _ = simple_system

        save_system_arrays(tmp_path, A, b, x_sol, x0=None, format=ArrayFormat.NPZ)

        with np.load(tmp_path / "system.npz") as data:
            expected_x0 = np.zeros_like(b)
            np.testing.assert_array_equal(data["x0"], expected_x0)


class TestIterationHistoryExport:
    """Test save_iteration_history function."""

    def test_minimal_history_npz(self, tmp_path: Path, solver_result_minimal):
        """Test NPZ export of minimal history."""
        history = solver_result_minimal.iteration_history

        save_iteration_history(tmp_path, history, format=ArrayFormat.NPZ)

        # Verify NPZ file exists
        npz_file = tmp_path / "history.npz"
        assert npz_file.exists()

        # Load and verify contents
        with np.load(npz_file) as data:
            assert "residual_norms" in data
            # MINIMAL mode should not have residuals/solutions
            assert "residuals" not in data
            assert "solutions" not in data
            expected_norms = np.array([1.0, 0.5, 0.1])
            np.testing.assert_allclose(data["residual_norms"], expected_norms)

    def test_full_history_npz(self, tmp_path: Path, solver_result_full):
        """Test NPZ export of full history."""
        history = solver_result_full.iteration_history

        save_iteration_history(tmp_path, history, format=ArrayFormat.NPZ)

        # Verify NPZ file exists
        npz_file = tmp_path / "history.npz"
        assert npz_file.exists()

        # Load and verify contents
        with np.load(npz_file) as data:
            assert "residual_norms" in data
            assert "residuals" in data
            assert "solutions" in data
            assert "directions" in data

            expected_norms = np.array([1.0, 0.5])
            np.testing.assert_allclose(data["residual_norms"], expected_norms)
            assert data["residuals"].shape == (2, 3)
            assert data["solutions"].shape == (2, 3)
            assert data["directions"].shape == (2, 3)

    def test_minimal_history_txt(self, tmp_path: Path, solver_result_minimal):
        """Test TXT export of minimal history."""
        history = solver_result_minimal.iteration_history

        save_iteration_history(tmp_path, history, format=ArrayFormat.TXT)

        # Verify only residual_norms.txt exists
        assert (tmp_path / "residual_norms.txt").exists()
        assert not (tmp_path / "residuals.txt").exists()
        assert not (tmp_path / "solutions.txt").exists()


class TestConfigExport:
    """Test configuration export functions."""

    def test_save_full_config(self, tmp_path: Path, solver_config):
        """Test full config export to TOML."""
        data_config = {"matrix_type": "tridiagonal", "size": 100}
        experiment_config = {"project": "test", "purpose": "validation"}

        output_path = tmp_path / "config.toml"
        save_full_config(
            output_path,
            solver_config,
            data_config=data_config,
            experiment_config=experiment_config,
        )

        # Verify file exists
        assert output_path.exists()

        # Load and verify structure
        with output_path.open("rb") as f:
            config = tomllib.load(f)

        assert "solver" in config
        assert "data" in config
        assert "experiment" in config

        assert config["solver"]["algorithm"] == "preconditioned_cg"
        assert config["solver"]["rtol"] == 1e-6
        assert config["data"]["matrix_type"] == "tridiagonal"
        assert config["experiment"]["project"] == "test"


class TestResultsExport:
    """Test results export functions."""

    def test_save_solver_result(self, tmp_path: Path, solver_result_minimal):
        """Test solver result export to TOML."""
        output_path = tmp_path / "results.toml"
        save_solver_result(output_path, solver_result_minimal)

        # Verify file exists
        assert output_path.exists()

        # Load and verify structure
        with output_path.open("rb") as f:
            results = tomllib.load(f)

        assert "outcomes" in results
        assert results["outcomes"]["converged"] is True
        assert results["outcomes"]["iterations"] == 3
        assert results["outcomes"]["final_residual_norm"] == 0.1
        assert results["outcomes"]["rtol"] == 1e-6
        assert results["outcomes"]["stopping_criterion"] == "converged"


class TestSaveSolverData:
    """Test backward-compatible save_solver_data function."""

    def test_txt_format_default(
        self, tmp_path: Path, simple_system, solver_config, solver_result_minimal
    ):
        """Test backward compatibility with TXT as default format."""
        A, b, x_sol, x0 = simple_system

        save_solver_data(
            output_dir=tmp_path,
            name_stem="test_run",
            A=A,
            b=b,
            x0=x0,
            x_sol=x_sol,
            result=solver_result_minimal,
            solver_config=solver_config,
        )

        target_dir = tmp_path / "test_run"
        assert target_dir.exists()

        # Verify TXT files (default format)
        assert (target_dir / "A.txt").exists()
        assert (target_dir / "b.txt").exists()
        assert (target_dir / "config.toml").exists()
        assert (target_dir / "results.toml").exists()

    def test_npz_format(
        self, tmp_path: Path, simple_system, solver_config, solver_result_minimal
    ):
        """Test NPZ format option."""
        A, b, x_sol, x0 = simple_system

        save_solver_data(
            output_dir=tmp_path,
            name_stem="test_run",
            A=A,
            b=b,
            x0=x0,
            x_sol=x_sol,
            result=solver_result_minimal,
            solver_config=solver_config,
            format=ArrayFormat.NPZ,
        )

        target_dir = tmp_path / "test_run"
        assert target_dir.exists()

        # Verify NPZ files
        assert (target_dir / "system.npz").exists()
        assert (target_dir / "history.npz").exists()
        assert (target_dir / "config.toml").exists()
        assert (target_dir / "results.toml").exists()

        # Verify no TXT files
        assert not (target_dir / "A.txt").exists()
        assert not (target_dir / "b.txt").exists()


class TestQuickExport:
    """Test convenience quick_export function."""

    def test_quick_export_npz(
        self, tmp_path: Path, simple_system, solver_config, solver_result_minimal
    ):
        """Test quick export convenience function."""
        A, b, x_sol, x0 = simple_system

        quick_export(
            result=solver_result_minimal,
            config=solver_config,
            A=A,
            b=b,
            x_sol=x_sol,
            output_dir=tmp_path,
            name_stem="quick_test",
            x0=x0,
            format=ArrayFormat.NPZ,
        )

        target_dir = tmp_path / "quick_test"
        assert target_dir.exists()

        # Verify all files created
        assert (target_dir / "system.npz").exists()
        assert (target_dir / "history.npz").exists()
        assert (target_dir / "config.toml").exists()
        assert (target_dir / "results.toml").exists()


class TestNPZFileSizes:
    """Test that NPZ format is significantly smaller than TXT."""

    def test_npz_smaller_than_txt(self, tmp_path: Path):
        """Verify NPZ is significantly smaller than TXT for large arrays."""
        # Create larger system for meaningful comparison
        size = 100
        A = np.eye(size) * 2
        b = np.ones(size)
        x_sol = np.ones(size)

        npz_dir = tmp_path / "npz"
        txt_dir = tmp_path / "txt"
        npz_dir.mkdir()
        txt_dir.mkdir()

        # Save in both formats
        save_system_arrays(npz_dir, A, b, x_sol, format=ArrayFormat.NPZ)
        save_system_arrays(txt_dir, A, b, x_sol, format=ArrayFormat.TXT)

        # Compare file sizes
        npz_size = (npz_dir / "system.npz").stat().st_size
        txt_size = sum(
            (txt_dir / f).stat().st_size
            for f in ["A.txt", "b.txt", "x0.txt", "x_sol.txt"]
        )

        # NPZ should be significantly smaller (expect ~10x but require at least 2x)
        assert npz_size < txt_size / 2, (
            f"NPZ ({npz_size} bytes) should be at least 2x smaller than "
            f"TXT ({txt_size} bytes)"
        )
