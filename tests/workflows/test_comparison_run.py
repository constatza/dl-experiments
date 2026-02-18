"""Tests for neuralls.workflows.comparison_run.

Covers:
- make_comparison_run(): pure constructor from training results
- resolve_checkpoint_from_run(): pure checkpoint lookup
- save_comparison_run() + load_comparison_run(): JSON round-trip
- load_comparison_run(): FileNotFoundError on missing file

Property tests (Hypothesis):
- checkpoint_map length always equals len(results)
- JSON round-trip is an identity transform on ComparisonRun
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from neuralls.workflows.comparison_run import (
    ComparisonRun,
    load_comparison_run,
    make_comparison_run,
    resolve_checkpoint_from_run,
    save_comparison_run,
    setup_comparison_tracking,
)
from neuralls.workflows.multi_training import TrainingRunResult

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------

PARENT_RUN_ID: str = "aaaabbbb-0000-1111-2222-ccccddddeeee"
TRACKING_URI: str = "sqlite:///mlruns.db"
EXP_ID_ALPHA: str = "alpha-experiment"
EXP_ID_BETA: str = "beta-experiment"
MISSING_EXP_ID: str = "does-not-exist"
MLFLOW_EXPERIMENT_ID: str = "42"
ARTIFACT_URI: str = "mlartifacts/42/aaaabbbb-0000-1111-2222-ccccddddeeee/artifacts"


# ===========================================================================
# make_comparison_run
# ===========================================================================


class TestMakeComparisonRun:
    """Tests for the pure make_comparison_run() constructor."""

    def test_checkpoint_map_keys_match_experiment_ids(
        self,
        training_results: list[TrainingRunResult],
    ) -> None:
        """checkpoint_map keys are exactly the experiment_id values from results.

        Args:
            training_results: Two-element list of TrainingRunResult fixtures.
        """
        run = make_comparison_run(
            PARENT_RUN_ID, TRACKING_URI, training_results, ARTIFACT_URI
        )
        expected_keys = {r.experiment_id for r in training_results}
        assert set(run.checkpoint_map.keys()) == expected_keys

    def test_checkpoint_map_values_are_checkpoint_paths(
        self,
        training_results: list[TrainingRunResult],
    ) -> None:
        """checkpoint_map values equal the checkpoint_path from each result.

        Args:
            training_results: Two-element list of TrainingRunResult fixtures.
        """
        run = make_comparison_run(
            PARENT_RUN_ID, TRACKING_URI, training_results, ARTIFACT_URI
        )
        for result in training_results:
            assert run.checkpoint_map[result.experiment_id] == result.checkpoint_path

    def test_mlflow_run_id_set_from_parent_run_id(
        self,
        training_results: list[TrainingRunResult],
    ) -> None:
        """mlflow_run_id field equals the provided parent_run_id.

        Args:
            training_results: Two-element list of TrainingRunResult fixtures.
        """
        run = make_comparison_run(
            PARENT_RUN_ID, TRACKING_URI, training_results, ARTIFACT_URI
        )
        assert run.mlflow_run_id == PARENT_RUN_ID

    def test_tracking_uri_preserved(
        self,
        training_results: list[TrainingRunResult],
    ) -> None:
        """tracking_uri field equals the provided tracking_uri argument.

        Args:
            training_results: Two-element list of TrainingRunResult fixtures.
        """
        run = make_comparison_run(
            PARENT_RUN_ID, TRACKING_URI, training_results, ARTIFACT_URI
        )
        assert run.tracking_uri == TRACKING_URI

    def test_artifact_uri_preserved(
        self,
        training_results: list[TrainingRunResult],
    ) -> None:
        """artifact_uri field equals the provided artifact_uri argument.

        Args:
            training_results: Two-element list of TrainingRunResult fixtures.
        """
        run = make_comparison_run(
            PARENT_RUN_ID, TRACKING_URI, training_results, ARTIFACT_URI
        )
        assert run.artifact_uri == ARTIFACT_URI

    def test_empty_results_produce_empty_checkpoint_map(self) -> None:
        """Empty results list produces a ComparisonRun with an empty checkpoint_map."""
        run = make_comparison_run(PARENT_RUN_ID, TRACKING_URI, [], ARTIFACT_URI)
        assert run.checkpoint_map == {}

    def test_returns_frozen_dataclass(
        self,
        training_results: list[TrainingRunResult],
    ) -> None:
        """Returned ComparisonRun is a frozen dataclass (immutable).

        Args:
            training_results: Two-element list of TrainingRunResult fixtures.
        """
        run = make_comparison_run(
            PARENT_RUN_ID, TRACKING_URI, training_results, ARTIFACT_URI
        )
        with pytest.raises((AttributeError, TypeError)):
            run.mlflow_run_id = "mutation-attempt"  # type: ignore[misc]


# ===========================================================================
# resolve_checkpoint_from_run
# ===========================================================================


class TestResolveCheckpointFromRun:
    """Tests for the pure resolve_checkpoint_from_run() lookup."""

    def test_returns_correct_path_for_known_experiment(
        self,
        comparison_run: ComparisonRun,
        checkpoint_alpha: Path,
    ) -> None:
        """Returns the correct checkpoint path for a known experiment ID.

        Args:
            comparison_run: ComparisonRun with alpha and beta in checkpoint_map.
            checkpoint_alpha: Expected alpha checkpoint path.
        """
        spec = _make_spec(experiment=EXP_ID_ALPHA)
        result = resolve_checkpoint_from_run(spec, comparison_run)
        assert result == checkpoint_alpha

    def test_returns_correct_path_for_second_experiment(
        self,
        comparison_run: ComparisonRun,
        checkpoint_beta: Path,
    ) -> None:
        """Returns the correct checkpoint path for the second experiment.

        Args:
            comparison_run: ComparisonRun with alpha and beta in checkpoint_map.
            checkpoint_beta: Expected beta checkpoint path.
        """
        spec = _make_spec(experiment=EXP_ID_BETA)
        result = resolve_checkpoint_from_run(spec, comparison_run)
        assert result == checkpoint_beta

    def test_raises_key_error_for_missing_experiment(
        self,
        comparison_run: ComparisonRun,
    ) -> None:
        """Raises KeyError when spec.experiment is not in checkpoint_map.

        Args:
            comparison_run: ComparisonRun with alpha and beta in checkpoint_map.
        """
        spec = _make_spec(experiment=MISSING_EXP_ID)
        with pytest.raises(KeyError, match=MISSING_EXP_ID):
            resolve_checkpoint_from_run(spec, comparison_run)

    def test_error_message_lists_available_experiments(
        self,
        comparison_run: ComparisonRun,
    ) -> None:
        """KeyError message includes the available experiments for diagnostics.

        Args:
            comparison_run: ComparisonRun with alpha and beta in checkpoint_map.
        """
        spec = _make_spec(experiment=MISSING_EXP_ID)
        with pytest.raises(KeyError) as exc_info:
            resolve_checkpoint_from_run(spec, comparison_run)
        assert EXP_ID_ALPHA in str(exc_info.value) or EXP_ID_BETA in str(exc_info.value)


# ===========================================================================
# save_comparison_run / load_comparison_run  (round-trip)
# ===========================================================================


class TestSaveLoadComparisonRun:
    """Tests for the JSON serialisation round-trip."""

    def test_round_trip_preserves_mlflow_run_id(
        self,
        comparison_run: ComparisonRun,
        tmp_path: Path,
    ) -> None:
        """mlflow_run_id survives a save + load cycle.

        Args:
            comparison_run: Source ComparisonRun.
            tmp_path: Pytest temporary directory.
        """
        path = tmp_path / "comparison_run.json"
        save_comparison_run(comparison_run, path)
        loaded = load_comparison_run(path)
        assert loaded.mlflow_run_id == comparison_run.mlflow_run_id

    def test_round_trip_preserves_tracking_uri(
        self,
        comparison_run: ComparisonRun,
        tmp_path: Path,
    ) -> None:
        """tracking_uri survives a save + load cycle.

        Args:
            comparison_run: Source ComparisonRun.
            tmp_path: Pytest temporary directory.
        """
        path = tmp_path / "comparison_run.json"
        save_comparison_run(comparison_run, path)
        loaded = load_comparison_run(path)
        assert loaded.tracking_uri == comparison_run.tracking_uri

    def test_round_trip_preserves_artifact_uri(
        self,
        comparison_run: ComparisonRun,
        tmp_path: Path,
    ) -> None:
        """artifact_uri (str) survives a save + load cycle.

        Args:
            comparison_run: Source ComparisonRun.
            tmp_path: Pytest temporary directory.
        """
        path = tmp_path / "comparison_run.json"
        save_comparison_run(comparison_run, path)
        loaded = load_comparison_run(path)
        assert loaded.artifact_uri == comparison_run.artifact_uri

    def test_round_trip_preserves_checkpoint_map_keys(
        self,
        comparison_run: ComparisonRun,
        tmp_path: Path,
    ) -> None:
        """checkpoint_map keys survive a save + load cycle.

        Args:
            comparison_run: Source ComparisonRun.
            tmp_path: Pytest temporary directory.
        """
        path = tmp_path / "comparison_run.json"
        save_comparison_run(comparison_run, path)
        loaded = load_comparison_run(path)
        assert set(loaded.checkpoint_map.keys()) == set(comparison_run.checkpoint_map.keys())

    def test_round_trip_preserves_checkpoint_map_paths(
        self,
        comparison_run: ComparisonRun,
        tmp_path: Path,
    ) -> None:
        """checkpoint_map Path values survive a save + load cycle.

        Args:
            comparison_run: Source ComparisonRun.
            tmp_path: Pytest temporary directory.
        """
        path = tmp_path / "comparison_run.json"
        save_comparison_run(comparison_run, path)
        loaded = load_comparison_run(path)
        for key in comparison_run.checkpoint_map:
            assert loaded.checkpoint_map[key] == comparison_run.checkpoint_map[key]

    def test_save_produces_valid_json_file(
        self,
        comparison_run: ComparisonRun,
        tmp_path: Path,
    ) -> None:
        """save_comparison_run writes a parseable JSON file.

        Args:
            comparison_run: Source ComparisonRun.
            tmp_path: Pytest temporary directory.
        """
        path = tmp_path / "comparison_run.json"
        save_comparison_run(comparison_run, path)
        raw = json.loads(path.read_text())
        assert isinstance(raw, dict)
        for field in ("mlflow_run_id", "mlflow_experiment_id", "tracking_uri", "artifact_location", "checkpoint_map", "artifact_uri"):
            assert field in raw

    def test_save_serialises_paths_as_strings(
        self,
        comparison_run: ComparisonRun,
        tmp_path: Path,
    ) -> None:
        """Paths in checkpoint_map are serialised as strings in JSON.

        Args:
            comparison_run: Source ComparisonRun.
            tmp_path: Pytest temporary directory.
        """
        path = tmp_path / "comparison_run.json"
        save_comparison_run(comparison_run, path)
        raw = json.loads(path.read_text())
        assert isinstance(raw["artifact_uri"], str)
        for v in raw["checkpoint_map"].values():
            assert isinstance(v, str)

    def test_load_raises_file_not_found_for_missing_file(
        self,
        tmp_path: Path,
    ) -> None:
        """load_comparison_run raises FileNotFoundError when file does not exist.

        Args:
            tmp_path: Pytest temporary directory.
        """
        absent_path = tmp_path / "nonexistent.json"
        with pytest.raises(FileNotFoundError):
            load_comparison_run(absent_path)

    def test_loaded_checkpoint_map_contains_path_objects(
        self,
        comparison_run: ComparisonRun,
        tmp_path: Path,
    ) -> None:
        """Loaded checkpoint_map values are Path objects, not raw strings.

        Args:
            comparison_run: Source ComparisonRun.
            tmp_path: Pytest temporary directory.
        """
        path = tmp_path / "comparison_run.json"
        save_comparison_run(comparison_run, path)
        loaded = load_comparison_run(path)
        for v in loaded.checkpoint_map.values():
            assert isinstance(v, Path)

    def test_loaded_artifact_uri_is_string(
        self,
        comparison_run: ComparisonRun,
        tmp_path: Path,
    ) -> None:
        """Loaded artifact_uri is a str, not a Path object.

        Args:
            comparison_run: Source ComparisonRun.
            tmp_path: Pytest temporary directory.
        """
        path = tmp_path / "comparison_run.json"
        save_comparison_run(comparison_run, path)
        loaded = load_comparison_run(path)
        assert isinstance(loaded.artifact_uri, str)

    def test_round_trip_with_empty_checkpoint_map(
        self,
        tmp_path: Path,
    ) -> None:
        """Empty checkpoint_map survives a save + load cycle.

        Args:
            tmp_path: Pytest temporary directory.
        """
        run = ComparisonRun(
            mlflow_run_id=PARENT_RUN_ID,
            mlflow_experiment_id=MLFLOW_EXPERIMENT_ID,
            tracking_uri=TRACKING_URI,
            artifact_location=str(tmp_path / "comparisons" / "artifacts"),
            checkpoint_map={},
            artifact_uri=ARTIFACT_URI,
        )
        path = tmp_path / "empty.json"
        save_comparison_run(run, path)
        loaded = load_comparison_run(path)
        assert loaded.checkpoint_map == {}


# ===========================================================================
# Hypothesis property tests
# ===========================================================================


@given(
    experiment_ids=st.lists(
        st.text(
            alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), whitelist_characters="-_"),
            min_size=1,
            max_size=32,
        ),
        min_size=0,
        max_size=6,
        unique=True,
    )
)
@settings(max_examples=40, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_checkpoint_map_length_equals_results_count(
    experiment_ids: list[str],
    tmp_path: Path,
) -> None:
    """checkpoint_map always has the same number of entries as results.

    Args:
        experiment_ids: Hypothesis-generated list of unique experiment ID strings.
        tmp_path: Pytest temporary directory.
    """
    results = [
        TrainingRunResult(
            label=str(i),
            experiment_id=exp_id,
            checkpoint_path=tmp_path / f"{exp_id}.ckpt",
            mlflow_run_id=None,
            metrics={},
        )
        for i, exp_id in enumerate(experiment_ids)
    ]
    run = make_comparison_run(PARENT_RUN_ID, TRACKING_URI, results, ARTIFACT_URI)
    assert len(run.checkpoint_map) == len(results)


@given(
    run_id=st.text(min_size=1, max_size=64, alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), whitelist_characters="-")),
    tracking_uri=st.text(min_size=1, max_size=128, alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), whitelist_characters=":/._-")),
)
@settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_save_load_round_trip_is_identity(
    run_id: str,
    tracking_uri: str,
    tmp_path: Path,
) -> None:
    """save_comparison_run followed by load_comparison_run is a lossless identity.

    Args:
        run_id: Hypothesis-generated MLflow run UUID string.
        tracking_uri: Hypothesis-generated tracking URI string.
        tmp_path: Pytest temporary directory.
    """
    ckpt = tmp_path / "model.ckpt"
    ckpt.touch()

    original = ComparisonRun(
        mlflow_run_id=run_id,
        mlflow_experiment_id="0",
        tracking_uri=tracking_uri,
        artifact_location=str(tmp_path / "comparisons"),
        checkpoint_map={"exp-1": ckpt},
        artifact_uri=ARTIFACT_URI,
    )
    path = tmp_path / "run.json"
    save_comparison_run(original, path)
    restored = load_comparison_run(path)

    assert restored.mlflow_run_id == original.mlflow_run_id
    assert restored.tracking_uri == original.tracking_uri
    assert restored.artifact_uri == original.artifact_uri
    assert restored.checkpoint_map == original.checkpoint_map


# ===========================================================================
# Private helpers (module-level, not fixtures)
# ===========================================================================


class _SpecStub:
    """Minimal stub for a solver spec with an experiment attribute.

    Attributes:
        experiment: Experiment ID string used by resolve_checkpoint_from_run.
    """

    def __init__(self, experiment: str) -> None:
        self.experiment = experiment


def _make_spec(*, experiment: str) -> _SpecStub:
    """Create a minimal spec stub for resolve_checkpoint_from_run tests.

    Args:
        experiment: Experiment ID to embed in the stub.

    Returns:
        _SpecStub with the given experiment attribute.
    """
    return _SpecStub(experiment=experiment)


# ===========================================================================
# ComparisonRun fields: mlflow_experiment_id, artifact_uri, artifact_location
# ===========================================================================


class TestComparisonRunNewFields:
    """Tests for mlflow_experiment_id, artifact_uri, and artifact_location fields."""

    def test_mlflow_experiment_id_stored(self, comparison_run: ComparisonRun) -> None:
        """mlflow_experiment_id field is preserved on the ComparisonRun.

        Args:
            comparison_run: Fixture with MLFLOW_EXPERIMENT_ID set.
        """
        assert comparison_run.mlflow_experiment_id == MLFLOW_EXPERIMENT_ID

    def test_artifact_uri_stored(self, comparison_run: ComparisonRun, artifact_uri: str) -> None:
        """artifact_uri field equals the configured artifact URI.

        Args:
            comparison_run: Fixture with artifact_uri set.
            artifact_uri: Expected artifact URI string.
        """
        assert comparison_run.artifact_uri == artifact_uri

    def test_artifact_location_stored(
        self, comparison_run: ComparisonRun, artifact_location: str
    ) -> None:
        """artifact_location field equals the configured location.

        Args:
            comparison_run: Fixture with artifact_location set.
            artifact_location: Expected artifact location string.
        """
        assert comparison_run.artifact_location == artifact_location

    def test_round_trip_preserves_mlflow_experiment_id(
        self,
        comparison_run: ComparisonRun,
        tmp_path: Path,
    ) -> None:
        """mlflow_experiment_id survives a save + load cycle.

        Args:
            comparison_run: Source ComparisonRun.
            tmp_path: Pytest temporary directory.
        """
        path = tmp_path / "cr.json"
        save_comparison_run(comparison_run, path)
        loaded = load_comparison_run(path)
        assert loaded.mlflow_experiment_id == comparison_run.mlflow_experiment_id

    def test_round_trip_preserves_artifact_uri(
        self,
        comparison_run: ComparisonRun,
        tmp_path: Path,
    ) -> None:
        """artifact_uri survives a save + load cycle.

        Args:
            comparison_run: Source ComparisonRun.
            tmp_path: Pytest temporary directory.
        """
        path = tmp_path / "cr.json"
        save_comparison_run(comparison_run, path)
        loaded = load_comparison_run(path)
        assert loaded.artifact_uri == comparison_run.artifact_uri

    def test_round_trip_preserves_artifact_location(
        self,
        comparison_run: ComparisonRun,
        tmp_path: Path,
    ) -> None:
        """artifact_location survives a save + load cycle.

        Args:
            comparison_run: Source ComparisonRun.
            tmp_path: Pytest temporary directory.
        """
        path = tmp_path / "cr.json"
        save_comparison_run(comparison_run, path)
        loaded = load_comparison_run(path)
        assert loaded.artifact_location == comparison_run.artifact_location


# ===========================================================================
# setup_comparison_tracking
# ===========================================================================


class TestSetupComparisonTracking:
    """Tests for setup_comparison_tracking() with mocked MLflow."""

    def test_sets_tracking_uri(self, tmp_path: Path) -> None:
        """Calls mlflow.set_tracking_uri with the provided URI.

        Args:
            tmp_path: Pytest temporary directory.
        """
        uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
        location = str(tmp_path / "artifacts")

        with (
            patch("neuralls.workflows.comparison_run.mlflow") as mock_mlflow,
            patch("neuralls.workflows.comparison_run.MlflowClient") as mock_client_cls,
        ):
            mock_client_cls.return_value.get_experiment_by_name.return_value = object()
            setup_comparison_tracking(uri, location)

        mock_mlflow.set_tracking_uri.assert_called_once_with(uri)

    def test_sets_experiment_name(self, tmp_path: Path) -> None:
        """Calls mlflow.set_experiment with the fixed experiment name.

        Args:
            tmp_path: Pytest temporary directory.
        """
        uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
        location = str(tmp_path / "artifacts")

        with (
            patch("neuralls.workflows.comparison_run.mlflow") as mock_mlflow,
            patch("neuralls.workflows.comparison_run.MlflowClient") as mock_client_cls,
        ):
            mock_client_cls.return_value.get_experiment_by_name.return_value = object()
            setup_comparison_tracking(uri, location)

        mock_mlflow.set_experiment.assert_called_once_with("neuralls-comparisons")

    def test_creates_experiment_when_missing(self, tmp_path: Path) -> None:
        """Creates the experiment when it does not exist in the DB.

        Args:
            tmp_path: Pytest temporary directory.
        """
        uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
        location = str(tmp_path / "artifacts")

        with (
            patch("neuralls.workflows.comparison_run.mlflow"),
            patch("neuralls.workflows.comparison_run.MlflowClient") as mock_client_cls,
        ):
            mock_client = mock_client_cls.return_value
            mock_client.get_experiment_by_name.return_value = None
            setup_comparison_tracking(uri, location)

        mock_client.create_experiment.assert_called_once_with(
            "neuralls-comparisons", artifact_location=location
        )

    def test_skips_create_when_experiment_exists(self, tmp_path: Path) -> None:
        """Does not create the experiment when it already exists.

        Args:
            tmp_path: Pytest temporary directory.
        """
        uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
        location = str(tmp_path / "artifacts")

        with (
            patch("neuralls.workflows.comparison_run.mlflow"),
            patch("neuralls.workflows.comparison_run.MlflowClient") as mock_client_cls,
        ):
            mock_client = mock_client_cls.return_value
            mock_client.get_experiment_by_name.return_value = object()
            setup_comparison_tracking(uri, location)

        mock_client.create_experiment.assert_not_called()
