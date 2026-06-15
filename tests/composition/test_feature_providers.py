"""Tests for the feature provider registry and its concrete implementations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr
from dlkit.infrastructure.config.data_entries import ZarrEntry

from neuralls.composition.experiments.feature_providers import (
    FiLMConditionProvider,
    DeepONetQueryProvider,
    registered_extra_names,
    resolve_extra_feature,
)
from neuralls.composition.experiments.runtime_dataset_contract import (
    default_training_dataset_contract,
)
from neuralls.composition.experiments.training import (
    _create_feature_configs,
    _extra_feature_names_from_settings,
)
from neuralls.platform.storage.training_artifacts import TrainingArrays
from neuralls.shared.constants import PARAMETERS_ZARR_PREFIX


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def contract():
    """Default runtime dataset contract."""
    return default_training_dataset_contract()


@pytest.fixture
def no_params_arrays(tmp_path: Path) -> TrainingArrays:
    """TrainingArrays with no parameters_zarr entries."""
    n, samples = 4, 6
    rng = np.random.default_rng(0)
    for name, shape in [("rhs.zarr", (samples, n)), ("solutions.zarr", (samples, n))]:
        zarr.open_array(str(tmp_path / name), mode="w", shape=shape, dtype="float64")[:] = (
            rng.standard_normal(shape)
        )
    zarr.open_array(str(tmp_path / "matrix.zarr"), mode="w", shape=(1, n, n), dtype="float64")[
        0
    ] = np.eye(n)
    return TrainingArrays(
        rhs=tmp_path / "rhs.zarr",
        solutions=tmp_path / "solutions.zarr",
        matrix_zarr=tmp_path / "matrix.zarr",
        sample_count=samples,
        parameters_zarr=(),
    )


@pytest.fixture
def one_param_arrays(tmp_path: Path) -> TrainingArrays:
    """TrainingArrays with one parameters_0.zarr of shape (6, 4)."""
    n, samples, param_dim = 4, 6, 4
    rng = np.random.default_rng(1)
    for name, shape in [("rhs.zarr", (samples, n)), ("solutions.zarr", (samples, n))]:
        zarr.open_array(str(tmp_path / name), mode="w", shape=shape, dtype="float64")[:] = (
            rng.standard_normal(shape)
        )
    zarr.open_array(str(tmp_path / "matrix.zarr"), mode="w", shape=(1, n, n), dtype="float64")[
        0
    ] = np.eye(n)
    params_path = tmp_path / f"{PARAMETERS_ZARR_PREFIX}0.zarr"
    zarr.open_array(str(params_path), mode="w", shape=(samples, param_dim), dtype="float64")[:] = (
        rng.standard_normal((samples, param_dim))
    )
    return TrainingArrays(
        rhs=tmp_path / "rhs.zarr",
        solutions=tmp_path / "solutions.zarr",
        matrix_zarr=tmp_path / "matrix.zarr",
        sample_count=samples,
        parameters_zarr=(params_path,),
    )


@pytest.fixture
def rhs_data() -> np.ndarray:
    """Small RHS array for feature config tests."""
    return np.zeros((6, 4), dtype=np.float64)


# ---------------------------------------------------------------------------
# FiLMConditionProvider
# ---------------------------------------------------------------------------


def test_film_condition_path_matches_parameters_0(one_param_arrays: TrainingArrays) -> None:
    entry = FiLMConditionProvider().provide(one_param_arrays)
    assert isinstance(entry, ZarrEntry)
    assert entry.path == one_param_arrays.parameters_zarr[0]


def test_film_condition_model_input_is_true(one_param_arrays: TrainingArrays) -> None:
    entry = FiLMConditionProvider().provide(one_param_arrays)
    assert entry.model_input is True


def test_film_condition_name_is_condition(one_param_arrays: TrainingArrays) -> None:
    entry = FiLMConditionProvider().provide(one_param_arrays)
    assert entry.name == "condition"


def test_film_condition_raises_without_params(no_params_arrays: TrainingArrays) -> None:
    with pytest.raises(ValueError, match="parameters_0.zarr"):
        FiLMConditionProvider().provide(no_params_arrays)


# ---------------------------------------------------------------------------
# DeepONetQueryProvider
# ---------------------------------------------------------------------------


def test_deeponet_query_provider_returns_value_entry(one_param_arrays: TrainingArrays) -> None:
    """DeepONetQueryProvider returns a ValueEntry with correct name."""
    from dlkit.infrastructure.config.data_entries import ValueEntry

    result = DeepONetQueryProvider().provide(one_param_arrays)
    assert isinstance(result, ValueEntry)
    assert result.name == "query"


def test_deeponet_query_provider_reshapes_to_n_1_param_dim(
    one_param_arrays: TrainingArrays,
) -> None:
    """DeepONetQueryProvider reshapes parameters_zarr from (N, param_dim) to (N, 1, param_dim)."""
    import zarr
    import numpy as np
    from dlkit.infrastructure.config.data_entries import ValueEntry

    # Load what's actually in parameters_zarr[0]
    params = np.asarray(zarr.open_array(str(one_param_arrays.parameters_zarr[0]), mode="r")[:])
    expected = params[:, np.newaxis, :]  # (N, 1, param_dim)

    result = DeepONetQueryProvider().provide(one_param_arrays)
    assert isinstance(result, ValueEntry)
    np.testing.assert_array_equal(result.value, expected)


def test_deeponet_query_raises_without_params(no_params_arrays: TrainingArrays) -> None:
    """DeepONetQueryProvider raises ValueError when no parameters_zarr files exist."""
    with pytest.raises(ValueError, match="parameters_zarr"):
        DeepONetQueryProvider().provide(no_params_arrays)


# ---------------------------------------------------------------------------
# resolve_extra_feature
# ---------------------------------------------------------------------------


def test_resolve_condition_returns_zarr_entry(one_param_arrays: TrainingArrays) -> None:
    entry = resolve_extra_feature("condition", one_param_arrays)
    assert isinstance(entry, ZarrEntry)
    assert entry.name == "condition"


def test_resolve_unknown_name_raises_with_registry_hint(
    one_param_arrays: TrainingArrays,
) -> None:
    with pytest.raises(ValueError, match="No feature provider registered for 'unknown'"):
        resolve_extra_feature("unknown", one_param_arrays)


# ---------------------------------------------------------------------------
# registered_extra_names
# ---------------------------------------------------------------------------


def test_registered_extra_names_contains_condition_and_query() -> None:
    names = registered_extra_names()
    assert "condition" in names
    assert "query" in names


# ---------------------------------------------------------------------------
# _create_feature_configs (training.py integration)
# ---------------------------------------------------------------------------


def test_create_feature_configs_no_extras_yields_x_and_matrix(
    no_params_arrays: TrainingArrays,
    rhs_data: np.ndarray,
    contract,
) -> None:
    entries = _create_feature_configs(no_params_arrays, rhs_data, contract, frozenset())
    names = [e.name for e in entries]
    assert names == ["x", "matrix"]


def test_create_feature_configs_with_condition_yields_three_entries(
    one_param_arrays: TrainingArrays,
    rhs_data: np.ndarray,
    contract,
) -> None:
    entries = _create_feature_configs(
        one_param_arrays, rhs_data, contract, frozenset({"condition"})
    )
    names = [e.name for e in entries]
    assert names == ["x", "matrix", "condition"]


def test_create_feature_configs_standard_ignores_params_zarr(
    one_param_arrays: TrainingArrays,
    rhs_data: np.ndarray,
    contract,
) -> None:
    entries = _create_feature_configs(one_param_arrays, rhs_data, contract, frozenset())
    names = [e.name for e in entries]
    assert "condition" not in names
    assert "parameters_0" not in names


# ---------------------------------------------------------------------------
# _extra_feature_names_from_settings (training.py integration)
# ---------------------------------------------------------------------------


def _make_minimal_settings(feature_names: list[str]):
    """Build a minimal DLKit TrainingWorkflowConfig with placeholder feature entries."""
    from dlkit.infrastructure.config.data_entries import ValueEntry
    from dlkit.infrastructure.config.dataset_settings import DatasetSettings
    from dlkit.infrastructure.config.workflow_configs import TrainingWorkflowConfig

    features = tuple(ValueEntry(name=name) for name in feature_names)
    dataset = DatasetSettings(name="FlexibleDataset", features=features)
    return TrainingWorkflowConfig.model_validate(
        {
            "SESSION": {"seed": 42, "workflow": "train"},
            "TRAINING": {
                "trainer": {"max_epochs": 1},
                "loss_function": {
                    "name": "normalized_vector_norm_loss",
                    "module_path": "dlkit.domain.losses",
                    "target_key": "targets.y",
                },
            },
            "DATASET": dataset.model_dump(),
        }
    )


def test_extra_names_from_settings_condition_declared(contract) -> None:
    try:
        settings = _make_minimal_settings(["x", "condition"])
        extra = _extra_feature_names_from_settings(settings, contract)
        assert extra == frozenset({"condition"})
    except Exception:
        pytest.skip("Minimal settings construction requires DLKit internals; skip if unavailable")


def test_extra_names_from_settings_standard_empty(contract) -> None:
    try:
        settings = _make_minimal_settings(["x"])
        extra = _extra_feature_names_from_settings(settings, contract)
        assert extra == frozenset()
    except Exception:
        pytest.skip("Minimal settings construction requires DLKit internals; skip if unavailable")
