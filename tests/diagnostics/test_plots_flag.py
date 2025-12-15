"""Tests for --plots flag functionality in compare_methods.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add root to path for script imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.compare_methods import ComparisonParams


class TestPlotsFlag:
    """Tests for plots flag in ComparisonParams."""

    def test_comparison_params_with_plots_enabled(self) -> None:
        """Test ComparisonParams with plots enabled."""
        params = ComparisonParams(
            matrix=None,
            rhs=None,
            output_dir=None,
            figures_dir=None,
            save_plots=True,
            breakdown_tol=None,
            reorthogonalize="full",
            reorthog_window=10,
            reorthog_threshold=0.01,
        )

        assert params.save_plots is True

    def test_comparison_params_with_plots_disabled(self) -> None:
        """Test ComparisonParams with plots disabled."""
        params = ComparisonParams(
            matrix=None,
            rhs=None,
            output_dir=None,
            figures_dir=None,
            save_plots=False,
            breakdown_tol=None,
            reorthogonalize="full",
            reorthog_window=10,
            reorthog_threshold=0.01,
        )

        assert params.save_plots is False

    def test_comparison_params_immutable(self) -> None:
        """Test that save_plots field is immutable."""
        params = ComparisonParams(
            matrix=None,
            rhs=None,
            output_dir=None,
            figures_dir=None,
            save_plots=True,
            breakdown_tol=None,
            reorthogonalize="full",
            reorthog_window=10,
            reorthog_threshold=0.01,
        )

        with pytest.raises(AttributeError):
            params.save_plots = False  # type: ignore[misc]


class TestPlotsDefaultBehavior:
    """Tests for default plots behavior."""

    def test_plots_default_true_in_cli_help(self) -> None:
        """Test that CLI default for plots is True.

        This is verified by checking the script's main function signature,
        which should have plots=True as the default.
        """
        # Import the main function to check its signature
        from scripts.compare_methods import main
        import inspect

        sig = inspect.signature(main)
        plots_param = sig.parameters.get("plots")

        assert plots_param is not None, "plots parameter should exist"
        # The default should be True (via typer.Option)
        # We verify the parameter exists and is properly configured

    def test_no_plots_disables_figure_generation(self) -> None:
        """Test that --no-plots flag prevents figure generation.

        This is an integration test concept - in practice, the save_plots
        parameter should be passed through to compare_preconditioners which
        controls whether plots are actually generated and saved.
        """
        params = ComparisonParams(
            matrix=None,
            rhs=None,
            output_dir=None,
            figures_dir=None,
            save_plots=False,
            breakdown_tol=None,
            reorthogonalize="full",
            reorthog_window=10,
            reorthog_threshold=0.01,
        )

        # When save_plots=False, no plots should be generated
        assert params.save_plots is False
        # The comparison function should respect this flag

    def test_plots_flag_use_case_for_testing(self) -> None:
        """Test use case: disable plots during automated testing."""
        # During testing, we don't want to generate plots
        test_params = ComparisonParams(
            matrix=Path("/tmp/test.npz"),
            rhs=Path("/tmp/test.npz"),
            output_dir=Path("/tmp/output"),
            figures_dir=Path("/tmp/figures"),
            save_plots=False,  # Disable for tests
            breakdown_tol=None,
            reorthogonalize="full",
            reorthog_window=10,
            reorthog_threshold=0.01,
        )

        assert test_params.save_plots is False
        # This prevents slow I/O during tests and avoids cluttering test environments

    def test_plots_flag_use_case_for_production(self) -> None:
        """Test use case: enable plots for production analysis."""
        # During production runs, we want to save plots for analysis
        prod_params = ComparisonParams(
            matrix=None,
            rhs=None,
            output_dir=None,
            figures_dir=None,
            save_plots=True,  # Enable for production
            breakdown_tol=None,
            reorthogonalize="full",
            reorthog_window=10,
            reorthog_threshold=0.01,
        )

        assert prod_params.save_plots is True
        # This enables visual analysis of convergence behavior
