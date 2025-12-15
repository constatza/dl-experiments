"""Test count-based generation plan parsing."""

from __future__ import annotations

import pytest

from src.generation.plan import (
    GenerationPlan,
    StrategySpec,
    _canonicalize_strategy_name,
    parse_generation_plan,
)


class TestCanonicalization:
    """Test strategy name normalization."""

    def test_normalize_dashes_to_underscores(self) -> None:
        assert _canonicalize_strategy_name("rhs-archive") == "rhs_archive"

    def test_lowercase_conversion(self) -> None:
        assert _canonicalize_strategy_name("RHS_ARCHIVE") == "rhs_archive"

    def test_legacy_alias_rhs_bank(self) -> None:
        assert _canonicalize_strategy_name("rhs_bank") == "rhs_archive"

    def test_legacy_alias_rhs_repository(self) -> None:
        assert _canonicalize_strategy_name("rhs_repository") == "rhs_archive"

    def test_legacy_alias_solution_bank(self) -> None:
        assert _canonicalize_strategy_name("solution_bank") == "solution_archive"

    def test_legacy_alias_solution_repository(self) -> None:
        assert _canonicalize_strategy_name("solution_repository") == "solution_archive"

    def test_preserve_standard_names(self) -> None:
        assert _canonicalize_strategy_name("random") == "random"
        assert _canonicalize_strategy_name("krylov") == "krylov"


class TestStrategySpec:
    """Test StrategySpec validation."""

    def test_valid_positive_count(self) -> None:
        spec = StrategySpec("random", "random", 1000, {})
        assert spec.samples == 1000

    def test_valid_all_count(self) -> None:
        spec = StrategySpec("solution_archive", "solution_archive", -1, {})
        assert spec.samples == -1

    def test_valid_skip_count(self) -> None:
        spec = StrategySpec("random", "random", 0, {})
        assert spec.samples == 0

    def test_invalid_negative_count(self) -> None:
        with pytest.raises(ValueError, match="invalid samples count: -5"):
            StrategySpec("random", "random", -5, {})

    def test_invalid_large_negative_count(self) -> None:
        with pytest.raises(ValueError, match="invalid samples count: -100"):
            StrategySpec("krylov", "krylov", -100, {})

    def test_immutability(self) -> None:
        spec = StrategySpec("random", "random", 100, {})
        with pytest.raises(AttributeError):
            spec.samples = 200  # type: ignore

    def test_options_stored(self) -> None:
        options = {"krylov_iters": 15, "scale": 2.0}
        spec = StrategySpec("krylov", "krylov", 500, options)
        assert spec.options["krylov_iters"] == 15
        assert spec.options["scale"] == 2.0


class TestGenerationPlan:
    """Test GenerationPlan validation."""

    def test_empty_strategies_raises(self) -> None:
        with pytest.raises(ValueError, match="At least one generation strategy"):
            GenerationPlan(strategies={})

    def test_all_zero_samples_raises(self) -> None:
        strategies = {
            "random": StrategySpec("random", "random", 0, {}),
            "krylov": StrategySpec("krylov", "krylov", 0, {}),
        }
        with pytest.raises(ValueError, match="At least one strategy must have samples"):
            GenerationPlan(strategies=strategies)

    def test_valid_plan_with_positive_samples(self) -> None:
        strategies = {
            "random": StrategySpec("random", "random", 100, {}),
        }
        plan = GenerationPlan(strategies=strategies)
        assert len(plan.strategies) == 1

    def test_valid_plan_with_all_samples(self) -> None:
        strategies = {
            "solution_archive": StrategySpec(
                "solution_archive", "solution_archive", -1, {}
            ),
        }
        plan = GenerationPlan(strategies=strategies)
        assert plan.solution_archive is not None

    def test_rhs_archive_property(self) -> None:
        strategies = {
            "rhs_archive": StrategySpec("rhs_archive", "rhs_archive", 100, {}),
        }
        plan = GenerationPlan(strategies=strategies)
        assert plan.rhs_archive is not None
        assert plan.rhs_archive.samples == 100

    def test_rhs_archive_property_none(self) -> None:
        strategies = {
            "random": StrategySpec("random", "random", 100, {}),
        }
        plan = GenerationPlan(strategies=strategies)
        assert plan.rhs_archive is None

    def test_solution_archive_property(self) -> None:
        strategies = {
            "solution_archive": StrategySpec(
                "solution_archive", "solution_archive", -1, {}
            ),
        }
        plan = GenerationPlan(strategies=strategies)
        assert plan.solution_archive is not None
        assert plan.solution_archive.samples == -1

    def test_solution_archive_property_none(self) -> None:
        strategies = {
            "krylov": StrategySpec("krylov", "krylov", 500, {}),
        }
        plan = GenerationPlan(strategies=strategies)
        assert plan.solution_archive is None

    def test_synthetic_property(self) -> None:
        strategies = {
            "random": StrategySpec("random", "random", 500, {}),
            "krylov": StrategySpec("krylov", "krylov", 300, {}),
            "rhs_archive": StrategySpec("rhs_archive", "rhs_archive", 200, {}),
        }
        plan = GenerationPlan(strategies=strategies)
        assert "random" in plan.synthetic
        assert "krylov" in plan.synthetic
        assert "rhs_archive" not in plan.synthetic

    def test_synthetic_property_excludes_both_archives(self) -> None:
        strategies = {
            "random": StrategySpec("random", "random", 100, {}),
            "rhs_archive": StrategySpec("rhs_archive", "rhs_archive", 50, {}),
            "solution_archive": StrategySpec(
                "solution_archive", "solution_archive", 50, {}
            ),
        }
        plan = GenerationPlan(strategies=strategies)
        synthetic = plan.synthetic
        assert "random" in synthetic
        assert "rhs_archive" not in synthetic
        assert "solution_archive" not in synthetic

    def test_immutability(self) -> None:
        strategies = {
            "random": StrategySpec("random", "random", 100, {}),
        }
        plan = GenerationPlan(strategies=strategies)
        with pytest.raises(AttributeError):
            plan.strategies = {}  # type: ignore


class TestParsing:
    """Test parse_generation_plan()."""

    def test_missing_strategy_array_raises(self) -> None:
        config = {}
        with pytest.raises(ValueError, match="Missing.*generation.strategy"):
            parse_generation_plan(config)

    def test_empty_strategy_array_raises(self) -> None:
        config = {"strategy": []}
        with pytest.raises(ValueError, match="At least one.*strategy.*block"):
            parse_generation_plan(config)

    def test_strategy_not_array_raises(self) -> None:
        config = {"strategy": "not_an_array"}
        with pytest.raises(ValueError, match="must be an array of tables"):
            parse_generation_plan(config)

    def test_strategy_entry_not_table_raises(self) -> None:
        config = {"strategy": ["not_a_table"]}
        with pytest.raises(ValueError, match="entry at index 0 must be a table"):
            parse_generation_plan(config)

    def test_missing_name_raises(self) -> None:
        config = {"strategy": [{"samples": 100}]}
        with pytest.raises(ValueError, match="missing a non-empty 'name'"):
            parse_generation_plan(config)

    def test_empty_name_raises(self) -> None:
        config = {"strategy": [{"name": "", "samples": 100}]}
        with pytest.raises(ValueError, match="missing a non-empty 'name'"):
            parse_generation_plan(config)

    def test_missing_samples_raises(self) -> None:
        config = {"strategy": [{"name": "random"}]}
        with pytest.raises(ValueError, match="missing 'samples' field"):
            parse_generation_plan(config)

    def test_non_integer_samples_raises(self) -> None:
        config = {"strategy": [{"name": "random", "samples": "many"}]}
        with pytest.raises(ValueError, match="non-integer 'samples'"):
            parse_generation_plan(config)

    def test_float_samples_converts_to_int(self) -> None:
        config = {"strategy": [{"name": "random", "samples": 100.0}]}
        plan = parse_generation_plan(config)
        assert plan.strategies["random"].samples == 100

    def test_invalid_negative_samples_raises(self) -> None:
        config = {"strategy": [{"name": "random", "samples": -5}]}
        with pytest.raises(ValueError, match="invalid samples=-5"):
            parse_generation_plan(config)

    def test_single_strategy_success(self) -> None:
        config = {"strategy": [{"name": "random", "samples": 1000}]}
        plan = parse_generation_plan(config)
        assert "random" in plan.strategies
        assert plan.strategies["random"].samples == 1000

    def test_multiple_strategies_success(self) -> None:
        config = {
            "strategy": [
                {"name": "random", "samples": 500},
                {"name": "krylov", "samples": 300, "krylov_iters": 15},
            ]
        }
        plan = parse_generation_plan(config)
        assert len(plan.strategies) == 2
        assert plan.strategies["krylov"].options["krylov_iters"] == 15

    def test_duplicate_strategies_merge_counts(self) -> None:
        config = {
            "strategy": [
                {"name": "random", "samples": 500},
                {"name": "random", "samples": 300},
            ]
        }
        plan = parse_generation_plan(config)
        assert plan.strategies["random"].samples == 800

    def test_duplicate_all_count_wins(self) -> None:
        config = {
            "strategy": [
                {
                    "name": "solution_archive",
                    "samples": 500,
                    "solutions_glob": "/a/*.txt",
                },
                {
                    "name": "solution_archive",
                    "samples": -1,
                    "solutions_glob": "/b/*.txt",
                },
            ]
        }
        plan = parse_generation_plan(config)
        assert plan.strategies["solution_archive"].samples == -1

    def test_duplicate_all_count_first_wins(self) -> None:
        config = {
            "strategy": [
                {
                    "name": "solution_archive",
                    "samples": -1,
                    "solutions_glob": "/a/*.txt",
                },
                {
                    "name": "solution_archive",
                    "samples": 500,
                    "solutions_glob": "/b/*.txt",
                },
            ]
        }
        plan = parse_generation_plan(config)
        assert plan.strategies["solution_archive"].samples == -1

    def test_duplicate_with_zero_uses_max(self) -> None:
        config = {
            "strategy": [
                {"name": "random", "samples": 0},
                {"name": "random", "samples": 300},
            ]
        }
        plan = parse_generation_plan(config)
        assert plan.strategies["random"].samples == 300

    def test_duplicate_both_zero_stays_zero(self) -> None:
        config = {
            "strategy": [
                {"name": "random", "samples": 0},
                {"name": "random", "samples": 0},
                {"name": "krylov", "samples": 100},  # Keep plan valid
            ]
        }
        plan = parse_generation_plan(config)
        assert plan.strategies["random"].samples == 0

    def test_options_extraction(self) -> None:
        config = {
            "strategy": [
                {
                    "name": "cg_residual_error",
                    "samples": 10000,
                    "residual_iters": 50,
                    "solutions_glob": "/path/*.txt",
                }
            ]
        }
        plan = parse_generation_plan(config)
        spec = plan.strategies["cg_residual_error"]
        assert spec.options["residual_iters"] == 50
        assert spec.options["solutions_glob"] == "/path/*.txt"
        assert "samples" not in spec.options
        assert "name" not in spec.options

    def test_options_merge_on_duplicate(self) -> None:
        config = {
            "strategy": [
                {"name": "random", "samples": 500, "scale": 1.0, "seed": 42},
                {"name": "random", "samples": 300, "scale": 2.0, "offset": 10},
            ]
        }
        plan = parse_generation_plan(config)
        spec = plan.strategies["random"]
        assert spec.samples == 800
        assert spec.options["scale"] == 2.0  # Later value wins
        assert spec.options["seed"] == 42
        assert spec.options["offset"] == 10

    def test_legacy_alias_parsing(self) -> None:
        config = {
            "strategy": [
                {"name": "rhs-bank", "samples": 100},
            ]
        }
        plan = parse_generation_plan(config)
        assert "rhs_archive" in plan.strategies
        assert plan.rhs_archive is not None

    def test_archive_strategies_identified(self) -> None:
        config = {
            "strategy": [
                {"name": "rhs_archive", "samples": 100},
                {"name": "solution_archive", "samples": -1},
                {"name": "random", "samples": 500},
            ]
        }
        plan = parse_generation_plan(config)
        assert plan.rhs_archive is not None
        assert plan.solution_archive is not None
        assert len(plan.synthetic) == 1
        assert "random" in plan.synthetic


class TestIntegration:
    """Integration tests with realistic configs."""

    def test_realistic_mixed_config(self) -> None:
        """Test a realistic mixed strategy configuration."""
        config = {
            "strategy": [
                {
                    "name": "cg_residual_error",
                    "samples": 10000,
                    "residual_iters": 50,
                    "solutions_glob": "/data/solutions/*.txt",
                },
                {
                    "name": "random",
                    "samples": 5000,
                },
            ]
        }
        plan = parse_generation_plan(config)

        assert len(plan.strategies) == 2
        assert plan.strategies["cg_residual_error"].samples == 10000
        assert plan.strategies["random"].samples == 5000
        assert plan.solution_archive is None
        assert plan.rhs_archive is None
        assert len(plan.synthetic) == 2

    def test_pure_solution_archive_config(self) -> None:
        """Test pure solution archive collection."""
        config = {
            "strategy": [
                {
                    "name": "solution_archive",
                    "samples": -1,
                    "solutions_glob": "/data/ua_vectors/*.txt",
                }
            ]
        }
        plan = parse_generation_plan(config)

        assert len(plan.strategies) == 1
        assert plan.solution_archive is not None
        assert plan.solution_archive.samples == -1
        assert len(plan.synthetic) == 0

    def test_skip_strategy_with_zero(self) -> None:
        """Test skipping a strategy with samples=0."""
        config = {
            "strategy": [
                {"name": "random", "samples": 0},
                {"name": "krylov", "samples": 1000, "krylov_iters": 15},
            ]
        }
        plan = parse_generation_plan(config)

        # Both strategies present, but random is skipped
        assert len(plan.strategies) == 2
        assert plan.strategies["random"].samples == 0
        assert plan.strategies["krylov"].samples == 1000


class TestPydanticValidation:
    """Test Pydantic validation of strategy configurations."""

    def test_pydantic_rejects_unknown_parameters(self) -> None:
        """Test that unknown parameters are rejected when configs are validated."""
        from src.generation.strategy_configs import KrylovConfig
        from pydantic import ValidationError

        # Try to create a config with an unknown parameter
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            KrylovConfig(samples=10, unknown_param=123)

    def test_pydantic_rejects_invalid_literal_values(self) -> None:
        """Test that invalid Literal values are rejected."""
        from src.generation.strategy_configs import EigenvectorForwardConfig
        from pydantic import ValidationError

        # Try to pass an invalid 'which' value
        with pytest.raises(ValidationError, match="Input should be"):
            EigenvectorForwardConfig(samples=10, which="invalid")

    def test_pydantic_validates_which_accepts_valid_values(self) -> None:
        """Test that valid 'which' values are accepted."""
        from src.generation.strategy_configs import EigenvectorForwardConfig

        # All three valid values should work
        config1 = EigenvectorForwardConfig(samples=10, which="smallest")
        assert config1.which == "smallest"

        config2 = EigenvectorForwardConfig(samples=10, which="largest")
        assert config2.which == "largest"

        config3 = EigenvectorForwardConfig(samples=10, which="random")
        assert config3.which == "random"

    def test_pydantic_requires_rhs_glob(self) -> None:
        """Test that rhs_glob is required for RhsArchiveConfig."""
        from src.generation.strategy_configs import RhsArchiveConfig
        from pydantic import ValidationError

        # Try to create config without rhs_glob
        with pytest.raises(ValidationError, match="Field required"):
            RhsArchiveConfig(samples=10)

    def test_pydantic_requires_solutions_glob(self) -> None:
        """Test that solutions_glob is required for SolutionArchiveConfig."""
        from src.generation.strategy_configs import SolutionArchiveConfig
        from pydantic import ValidationError

        # Try to create config without solutions_glob
        with pytest.raises(ValidationError, match="Field required"):
            SolutionArchiveConfig(samples=10)

    def test_pydantic_validates_type_residual_iters(self) -> None:
        """Test that residual_iters must be int."""
        from src.generation.strategy_configs import ResidualErrorConfig
        from pydantic import ValidationError

        # Try to pass a string for residual_iters
        with pytest.raises(ValidationError, match="Input should be a valid integer"):
            ResidualErrorConfig(samples=10, residual_iters="many")

    def test_pydantic_validates_type_krylov_iters(self) -> None:
        """Test that krylov_iters must be int."""
        from src.generation.strategy_configs import KrylovConfig
        from pydantic import ValidationError

        # Pydantic will coerce float to int, but invalid types should fail
        with pytest.raises(ValidationError, match="Input should be a valid integer"):
            KrylovConfig(samples=10, krylov_iters="invalid")

    def test_pydantic_frozen_prevents_mutation(self) -> None:
        """Test that frozen=True prevents mutation of config objects."""
        from src.generation.strategy_configs import KrylovConfig
        from pydantic import ValidationError

        config = KrylovConfig(samples=10, krylov_iters=15)

        # Try to modify a field (Pydantic raises ValidationError for frozen models)
        with pytest.raises(ValidationError, match="Instance is frozen"):
            config.krylov_iters = 20

    def test_pydantic_accepts_valid_configs(self) -> None:
        """Test that valid configurations are accepted."""
        from src.generation.strategy_configs import (
            KrylovConfig,
            ResidualErrorConfig,
            EigenvectorForwardConfig,
            RandomNormalConfig,
        )

        # All these should succeed
        krylov = KrylovConfig(samples=100, krylov_iters=20, seed=42)
        assert krylov.samples == 100
        assert krylov.krylov_iters == 20

        residual = ResidualErrorConfig(samples=50, residual_iters=10)
        assert residual.samples == 50
        assert residual.residual_iters == 10

        eigenvector = EigenvectorForwardConfig(
            samples=30, which="largest", include_eigenvectors=True
        )
        assert eigenvector.which == "largest"
        assert eigenvector.include_eigenvectors is True

        random_normal = RandomNormalConfig(samples=200, target_rhs_scale=2.5)
        assert random_normal.target_rhs_scale == 2.5
