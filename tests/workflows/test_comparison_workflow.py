"""Tests for neuralls.workflows.comparison.

Covers pure helpers:
- _validate_neural_preconditioner(): type-gate and presence checks
- _resolve_preconditioner(): three dispatch cases + KeyError
- _resolve_neural_preconditioners(): map over a mixed list

Covers orchestrated function (load_solver_config + compare_preconditioners mocked):
- run_comparison(): pipeline mode (with ComparisonRun) and standalone mode (without)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from typing import cast

from neuralls.configuration.comparison import ComparisonsTrackingConfig
from neuralls.configuration.preconditioner import (
    NeuralPreconditionerConfig,
    PreconditionerType,
    StandardPreconditionerConfig,
)
from neuralls.workflows.comparison import (
    _resolve_neural_preconditioners,
    _resolve_preconditioner,
    _validate_neural_preconditioner,
    run_comparison,
)
from neuralls.workflows.comparison_run import ComparisonRun
from neuralls.workflows.specs import ComparisonOutcome, ComparisonParams

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------

EXP_ID_ALPHA: str = "alpha-experiment"
EXP_ID_BETA: str = "beta-experiment"
MISSING_EXP_ID: str = "ghost-experiment"
SOLVER_STEM: str = "default"
OUTPUT_ROOT_STR: str = "/tmp/test-output"

# Module paths for patching (import locations inside comparison.py)
_LOAD_SOLVER_CONFIG = "neuralls.workflows.comparison.load_solver_config"
_COMPARE_PRECONDITIONERS = "neuralls.workflows.comparison.compare_preconditioners"
_MLFLOW_MODULE = "neuralls.workflows.comparison.mlflow"
_SETUP_TRACKING = "neuralls.workflows.comparison.setup_comparison_tracking"
_SAVE_COMPARISON_TOML = "neuralls.workflows.comparison._save_comparison_toml"


# ===========================================================================
# _validate_neural_preconditioner
# ===========================================================================


class TestValidateNeuralPreconditioner:
    """Tests for the pure _validate_neural_preconditioner() guard."""

    def test_non_neural_spec_passes_without_error(
        self,
        jacobi_spec: StandardPreconditionerConfig,
    ) -> None:
        """Non-neural specs always pass validation regardless of missing fields.

        Args:
            jacobi_spec: Standard Jacobi spec from conftest fixture.
        """
        _validate_neural_preconditioner(jacobi_spec)  # must not raise

    def test_neural_spec_with_checkpoint_path_passes(
        self,
        neural_spec_with_checkpoint: NeuralPreconditionerConfig,
    ) -> None:
        """Neural spec with explicit checkpoint_path passes validation.

        Args:
            neural_spec_with_checkpoint: Neural spec with checkpoint_path set.
        """
        _validate_neural_preconditioner(neural_spec_with_checkpoint)  # must not raise

    def test_neural_spec_with_experiment_passes(
        self,
        neural_spec_with_experiment: NeuralPreconditionerConfig,
    ) -> None:
        """Neural spec with experiment reference passes validation.

        Args:
            neural_spec_with_experiment: Neural spec with experiment set.
        """
        _validate_neural_preconditioner(neural_spec_with_experiment)  # must not raise

    def test_neural_spec_with_neither_raises_value_error(
        self,
        neural_spec_invalid: NeuralPreconditionerConfig,
    ) -> None:
        """Neural spec missing both checkpoint_path and experiment raises ValueError.

        Args:
            neural_spec_invalid: Neural spec with neither field set.
        """
        with pytest.raises(ValueError, match="checkpoint_path.*experiment|experiment.*checkpoint_path"):
            _validate_neural_preconditioner(neural_spec_invalid)

    def test_error_message_includes_solver_name(
        self,
        neural_spec_invalid: NeuralPreconditionerConfig,
    ) -> None:
        """ValueError message includes the solver name for diagnostics.

        Args:
            neural_spec_invalid: Neural spec with neither field set.
        """
        with pytest.raises(ValueError) as exc_info:
            _validate_neural_preconditioner(neural_spec_invalid)
        assert neural_spec_invalid.name in str(exc_info.value)


# ===========================================================================
# _resolve_preconditioner
# ===========================================================================


class TestResolvePreconditioner:
    """Tests for the pure _resolve_preconditioner() dispatcher."""

    def test_non_neural_spec_returned_unchanged(
        self,
        jacobi_spec: StandardPreconditionerConfig,
        comparison_run: ComparisonRun,
    ) -> None:
        """Non-neural specs are returned as the exact same object (no copy).

        Args:
            jacobi_spec: Standard Jacobi spec.
            comparison_run: ComparisonRun used for resolution context.
        """
        resolved = _resolve_preconditioner(jacobi_spec, comparison_run)
        assert resolved is jacobi_spec

    def test_neural_spec_with_checkpoint_returned_unchanged(
        self,
        neural_spec_with_checkpoint: NeuralPreconditionerConfig,
        comparison_run: ComparisonRun,
    ) -> None:
        """Neural spec with explicit checkpoint_path is returned without modification.

        Args:
            neural_spec_with_checkpoint: Neural spec with checkpoint_path set.
            comparison_run: ComparisonRun used for resolution context.
        """
        resolved = _resolve_preconditioner(neural_spec_with_checkpoint, comparison_run)
        assert resolved is neural_spec_with_checkpoint

    def test_neural_spec_with_experiment_resolves_checkpoint(
        self,
        neural_spec_with_experiment: NeuralPreconditionerConfig,
        comparison_run: ComparisonRun,
        checkpoint_alpha: Path,
    ) -> None:
        """Neural spec with experiment reference gets checkpoint_path injected.

        Args:
            neural_spec_with_experiment: Neural spec referencing EXP_ID_ALPHA.
            comparison_run: ComparisonRun with alpha in checkpoint_map.
            checkpoint_alpha: Expected resolved checkpoint path.
        """
        resolved = _resolve_preconditioner(neural_spec_with_experiment, comparison_run)
        assert resolved.checkpoint_path == checkpoint_alpha

    def test_resolved_spec_preserves_original_name(
        self,
        neural_spec_with_experiment: NeuralPreconditionerConfig,
        comparison_run: ComparisonRun,
    ) -> None:
        """Resolved neural spec retains its original name field.

        Args:
            neural_spec_with_experiment: Neural spec with experiment set.
            comparison_run: ComparisonRun used for resolution context.
        """
        resolved = _resolve_preconditioner(neural_spec_with_experiment, comparison_run)
        assert resolved.name == neural_spec_with_experiment.name

    def test_neural_spec_with_experiment_raises_key_error_when_missing(
        self,
        comparison_run: ComparisonRun,
    ) -> None:
        """Raises KeyError when the referenced experiment is not in checkpoint_map.

        Args:
            comparison_run: ComparisonRun without the requested experiment.
        """
        spec = NeuralPreconditionerConfig(
            name="ghost-neural",
            type=PreconditionerType.NEURAL,
            experiment=MISSING_EXP_ID,
        )
        with pytest.raises(KeyError, match=MISSING_EXP_ID):
            _resolve_preconditioner(spec, comparison_run)

    def test_neural_spec_without_either_raises_value_error(
        self,
        neural_spec_invalid: NeuralPreconditionerConfig,
        comparison_run: ComparisonRun,
    ) -> None:
        """Raises ValueError for neural spec missing both fields during resolution.

        Args:
            neural_spec_invalid: Neural spec with neither checkpoint_path nor experiment.
            comparison_run: ComparisonRun used for resolution context.
        """
        with pytest.raises(ValueError):
            _resolve_preconditioner(neural_spec_invalid, comparison_run)


# ===========================================================================
# _resolve_neural_preconditioners
# ===========================================================================


class TestResolveNeuralPreconditioners:
    """Tests for the pure _resolve_neural_preconditioners() map function."""

    def test_non_neural_specs_pass_through_unchanged(
        self,
        jacobi_spec: StandardPreconditionerConfig,
        comparison_run: ComparisonRun,
    ) -> None:
        """Non-neural specs pass through the map without modification.

        Args:
            jacobi_spec: Standard Jacobi spec.
            comparison_run: ComparisonRun for resolution context.
        """
        resolved = _resolve_neural_preconditioners([jacobi_spec], comparison_run)
        assert resolved == [jacobi_spec]

    def test_mixed_list_all_resolved(
        self,
        jacobi_spec: StandardPreconditionerConfig,
        neural_spec_with_checkpoint: NeuralPreconditionerConfig,
        neural_spec_with_experiment: NeuralPreconditionerConfig,
        comparison_run: ComparisonRun,
        checkpoint_alpha: Path,
    ) -> None:
        """Mixed list: non-neural unchanged, both neural variants correctly resolved.

        Args:
            jacobi_spec: Standard Jacobi spec (unchanged).
            neural_spec_with_checkpoint: Neural spec with explicit path (unchanged).
            neural_spec_with_experiment: Neural spec resolved from checkpoint_map.
            comparison_run: ComparisonRun containing alpha experiment.
            checkpoint_alpha: Expected resolved path for neural_spec_with_experiment.
        """
        specs: list[Any] = [
            jacobi_spec,
            neural_spec_with_checkpoint,
            neural_spec_with_experiment,
        ]
        resolved = _resolve_neural_preconditioners(specs, comparison_run)

        assert len(resolved) == 3
        assert resolved[0] is jacobi_spec
        assert resolved[1] is neural_spec_with_checkpoint
        resolved_neural = cast(NeuralPreconditionerConfig, resolved[2])
        assert resolved_neural.checkpoint_path == checkpoint_alpha

    def test_output_length_matches_input_length(
        self,
        jacobi_spec: StandardPreconditionerConfig,
        comparison_run: ComparisonRun,
    ) -> None:
        """Output list length always matches input list length.

        Args:
            jacobi_spec: Standard Jacobi spec used to fill input list.
            comparison_run: ComparisonRun for resolution context.
        """
        specs = [jacobi_spec] * 5
        resolved = _resolve_neural_preconditioners(specs, comparison_run)
        assert len(resolved) == 5

    def test_empty_list_returns_empty_list(
        self,
        comparison_run: ComparisonRun,
    ) -> None:
        """Empty input list returns an empty output list.

        Args:
            comparison_run: ComparisonRun for resolution context (unused).
        """
        resolved = _resolve_neural_preconditioners([], comparison_run)
        assert resolved == []


# ===========================================================================
# run_comparison — shared helpers
# ===========================================================================


def _make_mock_comp_run(run_id: str = "comp-run-id") -> MagicMock:
    """Build a MagicMock for the MLflow run returned by mlflow.start_run context.

    Args:
        run_id: Fake MLflow run UUID to embed.

    Returns:
        MagicMock with .info.run_id set.
    """
    mock_run = MagicMock()
    mock_run.info.run_id = run_id
    return mock_run


def _configure_mock_mlflow(mock_mlflow: MagicMock, run_id: str = "comp-run-id") -> None:
    """Configure a mock mlflow module so start_run works as a context manager.

    Also sets get_artifact_uri to return a valid plain-path string so that
    ``_artifact_uri_to_local_path()`` can parse it without error.

    Args:
        mock_mlflow: The patched mlflow module mock.
        run_id: Fake run UUID to embed in the returned run object.
    """
    mock_run = _make_mock_comp_run(run_id=run_id)
    mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
    mock_mlflow.start_run.return_value.__exit__.return_value = False
    mock_mlflow.get_artifact_uri.return_value = f"mlartifacts/0/{run_id}/artifacts"


def _make_mock_solver_config(
    *,
    comparisons: ComparisonsTrackingConfig | None,
    solvers: list[Any] | None = None,
) -> MagicMock:
    """Build a MagicMock representing a loaded solver config.

    Args:
        comparisons: ComparisonsTrackingConfig to assign to mock.general.comparisons.
        solvers: List of solver spec objects to assign to mock.solvers.

    Returns:
        MagicMock configured with .general (including .comparisons) and .solvers.
    """
    mock_cfg = MagicMock()
    mock_cfg.general.comparisons = comparisons
    mock_cfg.solvers = solvers if solvers is not None else []
    return mock_cfg


# ===========================================================================
# run_comparison — pipeline mode (comparison_run provided)
# ===========================================================================


class TestRunComparisonPipelineMode:
    """Tests for run_comparison() in pipeline mode (comparison_run provided)."""

    def test_success_path_returns_single_success_outcome(
        self,
        comparison_run: ComparisonRun,
        tmp_path: Path,
    ) -> None:
        """On happy path, returns a list with one ComparisonOutcome(success=True).

        Args:
            comparison_run: ComparisonRun fixture.
            tmp_path: Pytest temporary directory.
        """
        solver_path = tmp_path / f"{SOLVER_STEM}.toml"
        solver_path.touch()

        mock_cfg = _make_mock_solver_config(comparisons=None)
        mock_result = MagicMock()

        with (
            patch(_LOAD_SOLVER_CONFIG, return_value=mock_cfg),
            patch(_COMPARE_PRECONDITIONERS, return_value=mock_result),
            patch(_MLFLOW_MODULE) as mock_mlflow,
            patch(_SETUP_TRACKING),
            patch(_SAVE_COMPARISON_TOML),
        ):
            _configure_mock_mlflow(mock_mlflow)
            outcomes = run_comparison(solver_path, ComparisonParams(), comparison_run)

        assert len(outcomes) == 1
        assert outcomes[0].success is True
        assert outcomes[0].error is None
        assert outcomes[0].payload is mock_result

    def test_outcome_name_is_solver_stem(
        self,
        comparison_run: ComparisonRun,
        tmp_path: Path,
    ) -> None:
        """ComparisonOutcome.name equals the solver config file stem.

        Args:
            comparison_run: ComparisonRun fixture.
            tmp_path: Pytest temporary directory.
        """
        solver_path = tmp_path / f"{SOLVER_STEM}.toml"
        solver_path.touch()

        mock_cfg = _make_mock_solver_config(comparisons=None)

        with (
            patch(_LOAD_SOLVER_CONFIG, return_value=mock_cfg),
            patch(_COMPARE_PRECONDITIONERS, return_value=MagicMock()),
            patch(_MLFLOW_MODULE) as mock_mlflow,
            patch(_SETUP_TRACKING),
            patch(_SAVE_COMPARISON_TOML),
        ):
            _configure_mock_mlflow(mock_mlflow)
            outcomes = run_comparison(solver_path, ComparisonParams(), comparison_run)

        assert outcomes[0].name == SOLVER_STEM

    def test_setup_tracking_called_with_comparison_run_uri(
        self,
        comparison_run: ComparisonRun,
        tmp_path: Path,
    ) -> None:
        """setup_comparison_tracking is called with ComparisonRun's tracking coords.

        Args:
            comparison_run: ComparisonRun fixture with tracking_uri set.
            tmp_path: Pytest temporary directory.
        """
        solver_path = tmp_path / "cfg.toml"
        solver_path.touch()

        mock_cfg = _make_mock_solver_config(comparisons=None)

        with (
            patch(_LOAD_SOLVER_CONFIG, return_value=mock_cfg),
            patch(_COMPARE_PRECONDITIONERS, return_value=MagicMock()),
            patch(_MLFLOW_MODULE) as mock_mlflow,
            patch(_SETUP_TRACKING) as mock_setup,
            patch(_SAVE_COMPARISON_TOML),
        ):
            _configure_mock_mlflow(mock_mlflow)
            run_comparison(solver_path, ComparisonParams(), comparison_run)

        mock_setup.assert_called_once_with(
            comparison_run.tracking_uri, comparison_run.artifact_location
        )

    def test_comparison_run_id_in_tags(
        self,
        comparison_run: ComparisonRun,
        tmp_path: Path,
    ) -> None:
        """MLflow start_run is called with batch_run_id tag from ComparisonRun.

        Args:
            comparison_run: ComparisonRun fixture.
            tmp_path: Pytest temporary directory.
        """
        solver_path = tmp_path / "cfg.toml"
        solver_path.touch()

        mock_cfg = _make_mock_solver_config(comparisons=None)

        with (
            patch(_LOAD_SOLVER_CONFIG, return_value=mock_cfg),
            patch(_COMPARE_PRECONDITIONERS, return_value=MagicMock()),
            patch(_MLFLOW_MODULE) as mock_mlflow,
            patch(_SETUP_TRACKING),
            patch(_SAVE_COMPARISON_TOML),
        ):
            _configure_mock_mlflow(mock_mlflow)
            run_comparison(solver_path, ComparisonParams(), comparison_run)

        call_kwargs = mock_mlflow.start_run.call_args.kwargs
        assert call_kwargs["tags"]["batch_run_id"] == comparison_run.mlflow_run_id
        assert call_kwargs["tags"]["phase"] == "comparison"

    def test_file_not_found_produces_failure_outcome(
        self,
        comparison_run: ComparisonRun,
        tmp_path: Path,
    ) -> None:
        """FileNotFoundError from load_solver_config yields success=False outcome.

        Args:
            comparison_run: ComparisonRun fixture.
            tmp_path: Pytest temporary directory.
        """
        solver_path = tmp_path / "missing.toml"

        with patch(_LOAD_SOLVER_CONFIG, side_effect=FileNotFoundError("no such file")):
            outcomes = run_comparison(solver_path, ComparisonParams(), comparison_run)

        assert outcomes[0].success is False
        assert "no such file" in (outcomes[0].error or "")

    def test_key_error_from_missing_experiment_produces_failure_outcome(
        self,
        comparison_run: ComparisonRun,
        tmp_path: Path,
    ) -> None:
        """KeyError for unresolved experiment reference yields success=False.

        Args:
            comparison_run: ComparisonRun missing the requested experiment.
            tmp_path: Pytest temporary directory.
        """
        solver_path = tmp_path / f"{SOLVER_STEM}.toml"
        solver_path.touch()

        ghost_spec = NeuralPreconditionerConfig(
            name="ghost",
            type=PreconditionerType.NEURAL,
            experiment=MISSING_EXP_ID,
        )
        mock_cfg = _make_mock_solver_config(comparisons=None, solvers=[ghost_spec])

        with patch(_LOAD_SOLVER_CONFIG, return_value=mock_cfg):
            outcomes = run_comparison(solver_path, ComparisonParams(), comparison_run)

        assert outcomes[0].success is False
        assert MISSING_EXP_ID in (outcomes[0].error or "")


# ===========================================================================
# run_comparison — standalone mode (comparison_run=None)
# ===========================================================================


class TestRunComparisonStandaloneMode:
    """Tests for run_comparison() in standalone mode (no ComparisonRun)."""

    @pytest.fixture
    def tracking_config(self, tmp_path: Path) -> ComparisonsTrackingConfig:
        """ComparisonsTrackingConfig with tmp_path-based URI and artifact location.

        Args:
            tmp_path: Pytest temporary directory.

        Returns:
            Frozen ComparisonsTrackingConfig for standalone mode.
        """
        return ComparisonsTrackingConfig(
            tracking_uri=f"sqlite:///{tmp_path / 'comparisons.db'}",
            artifact_location=str(tmp_path / "comparisons" / "artifacts"),
        )

    def test_success_path_returns_single_success_outcome(
        self,
        tmp_path: Path,
        jacobi_spec: StandardPreconditionerConfig,
        tracking_config: ComparisonsTrackingConfig,
    ) -> None:
        """On happy path, returns a list with one ComparisonOutcome(success=True).

        Args:
            tmp_path: Pytest temporary directory.
            jacobi_spec: Non-neural spec that passes validation.
            tracking_config: ComparisonsTrackingConfig fixture.
        """
        solver_path = tmp_path / f"{SOLVER_STEM}.toml"
        solver_path.touch()

        mock_cfg = _make_mock_solver_config(
            comparisons=tracking_config, solvers=[jacobi_spec]
        )
        mock_result = MagicMock()

        with (
            patch(_LOAD_SOLVER_CONFIG, return_value=mock_cfg),
            patch(_COMPARE_PRECONDITIONERS, return_value=mock_result),
            patch(_MLFLOW_MODULE) as mock_mlflow,
            patch(_SETUP_TRACKING),
            patch(_SAVE_COMPARISON_TOML),
        ):
            _configure_mock_mlflow(mock_mlflow)
            outcomes = run_comparison(solver_path, ComparisonParams())

        assert len(outcomes) == 1
        assert outcomes[0].success is True
        assert outcomes[0].payload is mock_result

    def test_setup_tracking_called_with_comparisons_config(
        self,
        tmp_path: Path,
        jacobi_spec: StandardPreconditionerConfig,
        tracking_config: ComparisonsTrackingConfig,
    ) -> None:
        """setup_comparison_tracking is called with comparisons config coords.

        Args:
            tmp_path: Pytest temporary directory.
            jacobi_spec: Non-neural spec that passes validation.
            tracking_config: ComparisonsTrackingConfig fixture.
        """
        solver_path = tmp_path / "cfg.toml"
        solver_path.touch()

        mock_cfg = _make_mock_solver_config(
            comparisons=tracking_config, solvers=[jacobi_spec]
        )

        with (
            patch(_LOAD_SOLVER_CONFIG, return_value=mock_cfg),
            patch(_COMPARE_PRECONDITIONERS, return_value=MagicMock()),
            patch(_MLFLOW_MODULE) as mock_mlflow,
            patch(_SETUP_TRACKING) as mock_setup,
            patch(_SAVE_COMPARISON_TOML),
        ):
            _configure_mock_mlflow(mock_mlflow)
            run_comparison(solver_path, ComparisonParams())

        mock_setup.assert_called_once_with(
            tracking_config.tracking_uri, tracking_config.artifact_location
        )

    def test_missing_comparisons_config_produces_failure_outcome(
        self,
        tmp_path: Path,
        jacobi_spec: StandardPreconditionerConfig,
    ) -> None:
        """Missing [general.comparisons] yields success=False in standalone mode.

        Args:
            tmp_path: Pytest temporary directory.
            jacobi_spec: Non-neural spec that passes validation.
        """
        solver_path = tmp_path / f"{SOLVER_STEM}.toml"
        solver_path.touch()

        mock_cfg = _make_mock_solver_config(comparisons=None, solvers=[jacobi_spec])

        with patch(_LOAD_SOLVER_CONFIG, return_value=mock_cfg):
            outcomes = run_comparison(solver_path, ComparisonParams())

        assert outcomes[0].success is False
        assert outcomes[0].error is not None

    def test_neural_spec_without_checkpoint_produces_failure_outcome(
        self,
        tmp_path: Path,
        tracking_config: ComparisonsTrackingConfig,
    ) -> None:
        """Neural spec without checkpoint_path yields success=False in standalone mode.

        Args:
            tmp_path: Pytest temporary directory.
            tracking_config: ComparisonsTrackingConfig fixture.
        """
        solver_path = tmp_path / f"{SOLVER_STEM}.toml"
        solver_path.touch()

        spec = NeuralPreconditionerConfig(
            name="neural-no-ckpt",
            type=PreconditionerType.NEURAL,
            experiment=EXP_ID_ALPHA,
        )
        mock_cfg = _make_mock_solver_config(
            comparisons=tracking_config, solvers=[spec]
        )

        with patch(_LOAD_SOLVER_CONFIG, return_value=mock_cfg):
            outcomes = run_comparison(solver_path, ComparisonParams())

        assert outcomes[0].success is False
        assert outcomes[0].error is not None

    def test_phase_tag_is_direct_comparison(
        self,
        tmp_path: Path,
        jacobi_spec: StandardPreconditionerConfig,
        tracking_config: ComparisonsTrackingConfig,
    ) -> None:
        """MLflow start_run is called with phase=direct_comparison tag.

        Args:
            tmp_path: Pytest temporary directory.
            jacobi_spec: Non-neural spec.
            tracking_config: ComparisonsTrackingConfig fixture.
        """
        solver_path = tmp_path / "cfg.toml"
        solver_path.touch()

        mock_cfg = _make_mock_solver_config(
            comparisons=tracking_config, solvers=[jacobi_spec]
        )

        with (
            patch(_LOAD_SOLVER_CONFIG, return_value=mock_cfg),
            patch(_COMPARE_PRECONDITIONERS, return_value=MagicMock()),
            patch(_MLFLOW_MODULE) as mock_mlflow,
            patch(_SETUP_TRACKING),
            patch(_SAVE_COMPARISON_TOML),
        ):
            _configure_mock_mlflow(mock_mlflow)
            run_comparison(solver_path, ComparisonParams())

        call_kwargs = mock_mlflow.start_run.call_args.kwargs
        assert call_kwargs["tags"]["phase"] == "direct_comparison"

    def test_runtime_error_from_comparison_produces_failure_outcome(
        self,
        tmp_path: Path,
        jacobi_spec: StandardPreconditionerConfig,
        tracking_config: ComparisonsTrackingConfig,
    ) -> None:
        """RuntimeError from compare_preconditioners yields success=False.

        Args:
            tmp_path: Pytest temporary directory.
            jacobi_spec: Non-neural spec.
            tracking_config: ComparisonsTrackingConfig fixture.
        """
        solver_path = tmp_path / "cfg.toml"
        solver_path.touch()

        mock_cfg = _make_mock_solver_config(
            comparisons=tracking_config, solvers=[jacobi_spec]
        )

        with (
            patch(_LOAD_SOLVER_CONFIG, return_value=mock_cfg),
            patch(_COMPARE_PRECONDITIONERS, side_effect=RuntimeError("solver blew up")),
            patch(_MLFLOW_MODULE) as mock_mlflow,
            patch(_SETUP_TRACKING),
            patch(_SAVE_COMPARISON_TOML),
        ):
            _configure_mock_mlflow(mock_mlflow)
            outcomes = run_comparison(solver_path, ComparisonParams())

        assert outcomes[0].success is False
        assert "solver blew up" in (outcomes[0].error or "")
