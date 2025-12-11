"""Tests for solver_params module.

Tests cover SolverParams dataclass and parameter extraction from configs
and GeneralSettings objects.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.configuration.solver import (
    SolverParams,
    extract_solver_params_from_config,
    get_solver_params,
)


@pytest.fixture
def solver_config_full() -> dict[str, Any]:
    """Complete solver configuration dictionary.

    Returns:
        Dictionary with solver and data_generation sections.
    """
    return {
        "solver": {
            "tolerance": 1e-8,
            "max_iterations": 30,
            "stopping_criterion": "tolerance",
        },
        "data_generation": {
            "normalize": "matrix",
        },
    }


@pytest.fixture
def solver_config_partial() -> dict[str, Any]:
    """Partial solver configuration (missing some sections).

    Returns:
        Dictionary with only solver section.
    """
    return {
        "solver": {
            "tolerance": 1e-6,
            "max_iterations": 50,
        },
    }


@pytest.fixture
def solver_config_empty() -> dict[str, Any]:
    """Empty solver configuration.

    Returns:
        Empty dictionary (will use all defaults).
    """
    return {}


@pytest.fixture
def solver_params_expected() -> SolverParams:
    """Expected SolverParams for full config.

    Returns:
        SolverParams with standard test values.
    """
    return SolverParams(
        solver_type="flexible_cg",
        rtol=1e-8,
        atol=1e-14,
        max_iterations=30,
        normalize_system="matrix",
        stopping_criterion="tolerance",
        m_max=10,
        beta_max=1e10,
        eps_curv=1e-14,
        eps_breakdown=1e-14,
        m_replacement=50,
        gamma_div=1e10,
    )


class TestSolverParamsDataclass:
    """Tests for SolverParams dataclass."""

    def test_solver_params_creation(self) -> None:
        """Test SolverParams can be created with required fields."""
        params = SolverParams(
            solver_type="flexible_cg",
            rtol=1e-8,
            atol=1e-14,
            max_iterations=30,
            normalize_system="matrix",
            stopping_criterion="tolerance",
            m_max=10,
            beta_max=1e10,
            eps_curv=1e-14,
            eps_breakdown=1e-14,
            m_replacement=50,
            gamma_div=1e10,
        )

        assert params.rtol == 1e-8
        assert params.atol == 1e-14
        assert params.max_iterations == 30
        assert params.normalize_system == "matrix"
        assert params.stopping_criterion == "tolerance"

    def test_solver_params_to_dict(
        self, solver_params_expected: SolverParams
    ) -> None:
        """Test to_dict() conversion.

        Args:
            solver_params_expected: Expected SolverParams.
        """
        params_dict = solver_params_expected.to_dict()

        assert isinstance(params_dict, dict)
        assert params_dict["rtol"] == 1e-8
        assert params_dict["atol"] == 1e-14
        assert params_dict["max_iterations"] == 30
        assert params_dict["normalize_system"] == "matrix"
        assert params_dict["stopping_criterion"] == "tolerance"

    def test_solver_params_frozen(self) -> None:
        """Test that SolverParams is immutable (frozen)."""
        params = SolverParams(
            solver_type="flexible_cg",
            rtol=1e-8,
            atol=1e-14,
            max_iterations=30,
            normalize_system="matrix",
            stopping_criterion="tolerance",
            m_max=10,
            beta_max=1e10,
            eps_curv=1e-14,
            eps_breakdown=1e-14,
            m_replacement=50,
            gamma_div=1e10,
        )

        with pytest.raises(AttributeError):
            params.rtol = 1e-6  # type: ignore[misc]

    def test_solver_params_normalize_system_as_bool(self) -> None:
        """Test that normalize_system can be boolean or string."""
        params_bool = SolverParams(
            solver_type="flexible_cg",
            rtol=1e-8,
            atol=1e-14,
            max_iterations=30,
            normalize_system=True,
            stopping_criterion="tolerance",
            m_max=10,
            beta_max=1e10,
            eps_curv=1e-14,
            eps_breakdown=1e-14,
            m_replacement=50,
            gamma_div=1e10,
        )

        assert params_bool.normalize_system is True

        params_str = SolverParams(
            solver_type="flexible_cg",
            rtol=1e-8,
            atol=1e-14,
            max_iterations=30,
            normalize_system="matrix",
            stopping_criterion="tolerance",
            m_max=10,
            beta_max=1e10,
            eps_curv=1e-14,
            eps_breakdown=1e-14,
            m_replacement=50,
            gamma_div=1e10,
        )

        assert params_str.normalize_system == "matrix"


class TestExtractSolverParamsFromConfig:
    """Tests for extract_solver_params_from_config function."""

    def test_extract_from_full_config(
        self, solver_config_full: dict[str, Any]
    ) -> None:
        """Test extraction from complete config.

        Args:
            solver_config_full: Complete solver config.
        """
        params = extract_solver_params_from_config(solver_config_full)

        assert params.rtol == 1e-8
        assert params.max_iterations == 30
        assert params.normalize_system == "matrix"
        assert params.stopping_criterion == "tolerance"

    def test_extract_from_partial_config(
        self, solver_config_partial: dict[str, Any]
    ) -> None:
        """Test extraction with missing sections uses defaults.

        Args:
            solver_config_partial: Partial solver config.
        """
        params = extract_solver_params_from_config(solver_config_partial)

        # Explicit values from partial config
        assert params.rtol == 1e-6
        assert params.max_iterations == 50

        # Defaults for missing sections
        assert params.normalize_system == "matrix"
        assert params.stopping_criterion == "residual_norm"

    def test_extract_from_empty_config(
        self, solver_config_empty: dict[str, Any]
    ) -> None:
        """Test extraction from empty config uses all defaults.

        Args:
            solver_config_empty: Empty solver config.
        """
        params = extract_solver_params_from_config(solver_config_empty)

        assert params.rtol == 1e-6
        assert params.max_iterations == 100
        assert params.normalize_system == "matrix"
        assert params.stopping_criterion == "residual_norm"

    def test_extract_preserves_all_fields(
        self, solver_config_full: dict[str, Any]
    ) -> None:
        """Test all fields are correctly extracted.

        Args:
            solver_config_full: Complete solver config.
        """
        params = extract_solver_params_from_config(solver_config_full)

        # All fields should be set
        assert params.rtol is not None
        assert params.atol is not None
        assert params.max_iterations is not None
        assert params.normalize_system is not None
        assert params.stopping_criterion is not None

    def test_extract_with_different_normalize_values(
        self,
    ) -> None:
        """Test extraction with different normalize values."""
        config_rhs = {
            "solver": {"tolerance": 1e-8, "max_iterations": 30},
            "data_generation": {"normalize": "rhs"},
        }

        params = extract_solver_params_from_config(config_rhs)
        assert params.normalize_system == "rhs"

        config_false = {
            "solver": {"tolerance": 1e-8, "max_iterations": 30},
            "data_generation": {"normalize": False},
        }

        params = extract_solver_params_from_config(config_false)
        assert params.normalize_system is False


class TestGetSolverParamsFromSettings:
    """Tests for get_solver_params function with GeneralSettings."""

    def test_get_solver_params_from_settings_with_extras(
        self,
    ) -> None:
        """Test extraction from settings.EXTRAS with solver params."""
        # Create mock settings object
        class MockSettings:
            def __init__(self) -> None:
                self.EXTRAS = {
                    "solver": {
                        "tolerance": 1e-10,
                        "max_iterations": 100,
                        "stopping_criterion": "fixed_iterations",
                    },
                    "data_generation": {
                        "normalize": "rhs",
                    },
                }

        settings = MockSettings()  # type: ignore[assignment]
        params = get_solver_params(settings)

        assert params.rtol == 1e-10
        assert params.max_iterations == 100
        assert params.stopping_criterion == "fixed_iterations"
        assert params.normalize_system == "rhs"

    def test_get_solver_params_from_settings_empty_extras(
        self,
    ) -> None:
        """Test extraction with empty EXTRAS uses defaults."""
        class MockSettings:
            def __init__(self) -> None:
                self.EXTRAS = {}

        settings = MockSettings()  # type: ignore[assignment]
        params = get_solver_params(settings)

        # Should use defaults
        assert params.rtol == 1e-6
        assert params.max_iterations == 100
        assert params.normalize_system == "matrix"
        assert params.stopping_criterion == "residual_norm"

    def test_get_solver_params_from_settings_no_extras(
        self,
    ) -> None:
        """Test extraction when settings has no EXTRAS attribute."""
        class MockSettings:
            pass

        settings = MockSettings()  # type: ignore[assignment]
        params = get_solver_params(settings)

        # Should use defaults
        assert params.rtol == 1e-6
        assert params.max_iterations == 100

    def test_get_solver_params_backwards_compat_old_structure(
        self,
    ) -> None:
        """Test backwards compatibility with old EXTRAS structure."""
        class MockSettings:
            def __init__(self) -> None:
                self.EXTRAS = {
                    "solver": {
                        "tolerance": 1e-7,
                    },
                    "data_generation": {
                        "normalize": "both",
                    },
                }

        settings = MockSettings()  # type: ignore[assignment]
        params = get_solver_params(settings)

        # Old values should be used
        assert params.rtol == 1e-7
        assert params.normalize_system == "both"

        # Missing values use defaults
        assert params.max_iterations == 100
        assert params.stopping_criterion == "residual_norm"

    def test_get_solver_params_partial_extras(
        self,
    ) -> None:
        """Test extraction when EXTRAS has partial solver section."""
        class MockSettings:
            def __init__(self) -> None:
                self.EXTRAS = {
                    "solver": {
                        "tolerance": 1e-9,
                        # max_iterations missing - should use default
                    },
                    # data_generation section missing - should use default
                }

        settings = MockSettings()  # type: ignore[assignment]
        params = get_solver_params(settings)

        assert params.rtol == 1e-9
        assert params.max_iterations == 100  # default
        assert params.normalize_system == "matrix"  # default

    def test_get_solver_params_round_trip(
        self, solver_config_full: dict[str, Any]
    ) -> None:
        """Test that params extracted from config match settings extraction.

        Args:
            solver_config_full: Complete solver config.
        """
        # Extract from config
        params_from_config = extract_solver_params_from_config(solver_config_full)

        # Create settings with equivalent EXTRAS
        class MockSettings:
            def __init__(self) -> None:
                self.EXTRAS = solver_config_full

        settings = MockSettings()  # type: ignore[assignment]
        params_from_settings = get_solver_params(settings)

        # Should be equivalent
        assert params_from_config.rtol == params_from_settings.rtol
        assert params_from_config.atol == params_from_settings.atol
        assert params_from_config.max_iterations == params_from_settings.max_iterations
        assert (
            params_from_config.normalize_system
            == params_from_settings.normalize_system
        )
        assert (
            params_from_config.stopping_criterion
            == params_from_settings.stopping_criterion
        )
