"""Tests for neuralls.workflows.comparison (schema_version=3)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from neuralls.configuration.preconditioner import (
    NeuralPreconditionerConfig,
    PreconditionerType,
    StandardPreconditionerConfig,
)
from neuralls.solver.models.result import CGComparisonResult
from neuralls.workflows.comparison_artifacts import (
    coerce_comparison_result_payload,
    extract_array_artifacts,
    serialize_comparison_payload,
)
from neuralls.workflows.comparison import (
    _resolve_neural_preconditioners,
    _resolve_preconditioner,
    _validate_neural_preconditioner,
    run_comparison,
)
from neuralls.workflows.comparison_run import ComparisonRun
from neuralls.workflows.results import (
    ComparisonRecommendations,
    ComparisonResult,
    PlotPaths,
    RankedRecommendation,
)
from neuralls.workflows.specs import ComparisonOutcome, ComparisonParams

_LOAD_COMPARISON_CONFIG = "neuralls.workflows.comparison.load_comparison_config"
_COMPARE_PRECONDITIONERS = "neuralls.workflows.comparison.compare_preconditioners"
_MLFLOW_MODULE = "neuralls.workflows.comparison.mlflow"
_SETUP_TRACKING = "neuralls.workflows.comparison.setup_comparison_tracking"
_RESOLVE_PRECONDITIONER_MODELS = "neuralls.workflows.comparison.resolve_preconditioner_models"


def _make_mock_comp_run(run_id: str = "comp-run-id") -> MagicMock:
    mock_run = MagicMock()
    mock_run.info.run_id = run_id
    return mock_run


def _configure_mock_mlflow(mock_mlflow: MagicMock, run_id: str = "comp-run-id") -> None:
    mock_run = _make_mock_comp_run(run_id=run_id)
    mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
    mock_mlflow.start_run.return_value.__exit__.return_value = False
    mock_mlflow.get_artifact_uri.return_value = f"mlartifacts/0/{run_id}/artifacts"


def _mock_cfg(
    *,
    tracking_uri: str = "sqlite:////tmp/comparisons.db",
    artifact_location: str = "/tmp/mlartifacts",
    model_tracking_uri: str = "sqlite:////tmp/models.db",
    preconditioners: list[object] | None = None,
) -> MagicMock:
    cfg = MagicMock()
    cfg.run_name = None
    cfg.general.tracking.tracking_uri = tracking_uri
    cfg.general.tracking.artifact_location = artifact_location
    cfg.general.tracking.experiment_name = "neuralls-comparisons"
    cfg.general.model_store.tracking_uri = model_tracking_uri
    cfg.general.data.dataset_alias = None
    cfg.preconditioners = tuple(preconditioners or [])
    return cfg


def _typed_comparison_result() -> ComparisonResult:
    return ComparisonResult(
        results={
            "none": CGComparisonResult(
                x=np.array([1.0, 2.0]),
                converged=True,
                iterations=2,
                residual=1.0e-8,
                residual_abs=1.0e-9,
                residual_history=[1.0, 1.0e-8],
                residual_history_abs=[1.0, 1.0e-9],
                preconditioner="none",
                initial_guess=np.zeros(2),
                exact_error=None,
                rhs_norm=1.0,
                breakdown=False,
            )
        },
        summary="ok",
        solver_params=object(),
        plot_paths=PlotPaths(convergence=Path("/tmp/convergence.png")),
        preconditioners=("none",),
        condition_numbers={"none": 1.0},
        recommendations=ComparisonRecommendations(
            ranked=(
                RankedRecommendation(
                    label="none",
                    iterations=2,
                    residual=1.0e-8,
                    residual_abs=1.0e-9,
                    breakdown=False,
                ),
            ),
            overall_best=RankedRecommendation(
                label="none",
                iterations=2,
                residual=1.0e-8,
                residual_abs=1.0e-9,
                breakdown=False,
            ),
        ),
    )


def test_validate_neural_preconditioner_requires_model_ref() -> None:
    spec = NeuralPreconditionerConfig(name="neural", type=PreconditionerType.NEURAL)
    try:
        _validate_neural_preconditioner(spec)
    except ValueError as exc:
        assert "model_ref" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_validate_neural_preconditioner_rejects_checkpoint_path() -> None:
    spec = NeuralPreconditionerConfig(
        name="neural",
        type=PreconditionerType.NEURAL,
        checkpoint_path=Path("/tmp/model.ckpt"),
        model_ref={"source": "logged", "run_id": "run-1"},
    )
    try:
        _validate_neural_preconditioner(spec)
    except ValueError as exc:
        assert "checkpoint_path/experiment" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_resolve_preconditioner_keeps_non_neural_unchanged(
    comparison_run: ComparisonRun,
) -> None:
    spec = StandardPreconditionerConfig(name="jacobi", type=PreconditionerType.JACOBI)
    assert _resolve_preconditioner(spec, comparison_run) is spec


def test_resolve_neural_preconditioners_validates_all(
    comparison_run: ComparisonRun,
) -> None:
    specs = [
        StandardPreconditionerConfig(name="none", type=PreconditionerType.IDENTITY),
        NeuralPreconditionerConfig(
            name="neural",
            type=PreconditionerType.NEURAL,
            model_ref={"source": "logged", "run_id": "run-1"},
        ),
    ]
    resolved = _resolve_neural_preconditioners(specs, comparison_run)
    assert len(resolved) == 2


def test_run_comparison_pipeline_mode_success(
    comparison_run: ComparisonRun,
    tmp_path: Path,
) -> None:
    comparison_config = tmp_path / "comparison.toml"
    comparison_config.touch()
    cfg = _mock_cfg(
        preconditioners=[
            StandardPreconditionerConfig(name="none", type=PreconditionerType.IDENTITY),
            NeuralPreconditionerConfig(
                name="neural",
                type=PreconditionerType.NEURAL,
                model_ref={"source": "logged", "run_id": "run-1"},
            ),
        ]
    )
    payload = MagicMock()

    with (
        patch(_LOAD_COMPARISON_CONFIG, return_value=cfg),
        patch(_COMPARE_PRECONDITIONERS, return_value=payload),
        patch(_RESOLVE_PRECONDITIONER_MODELS, side_effect=lambda **kwargs: kwargs["specs"]),
        patch(_MLFLOW_MODULE) as mock_mlflow,
        patch(_SETUP_TRACKING) as mock_setup_tracking,
    ):
        _configure_mock_mlflow(mock_mlflow)
        outcomes = run_comparison(comparison_config, ComparisonParams(), comparison_run)

    assert len(outcomes) == 1
    assert outcomes[0].success is True
    assert outcomes[0].payload is None
    mock_setup_tracking.assert_called_once_with(
        tracking_uri=cfg.general.tracking.tracking_uri,
        artifact_location=cfg.general.tracking.artifact_location,
        experiment_name=cfg.general.tracking.experiment_name,
    )
    _, start_kwargs = mock_mlflow.start_run.call_args
    assert start_kwargs["tags"]["batch_run_id"] == comparison_run.mlflow_run_id


def test_extract_array_artifacts_detaches_numpy_data() -> None:
    payload, array_artifacts = extract_array_artifacts(_typed_comparison_result())
    serialized = serialize_comparison_payload(payload)

    assert serialized["results"]["none"]["iterations"] == 2
    assert serialized["results"]["none"]["residual"] == 1.0e-8
    x_ref = serialized["results"]["none"]["x"]
    assert x_ref == {
        "path": "arrays/results/none/x.npy",
        "shape": [2],
        "dtype": "float64",
    }
    artifact_paths = {artifact.reference.path for artifact in array_artifacts}
    assert Path("arrays/results/none/x.npy") in artifact_paths
    np.testing.assert_array_equal(
        next(
            artifact.values
            for artifact in array_artifacts
            if artifact.reference.path == Path("arrays/results/none/x.npy")
        ),
        np.array([1.0, 2.0]),
    )


def test_coerce_comparison_result_payload_uses_safe_defaults_for_magicmock() -> None:
    payload = coerce_comparison_result_payload(MagicMock())

    assert payload.summary == ""
    assert payload.preconditioners == ()
    assert payload.condition_numbers == {}
    assert payload.plot_paths == PlotPaths()
    assert payload.recommendations == ComparisonRecommendations()
    assert payload.results == {}


def test_run_comparison_standalone_requires_tracking(tmp_path: Path) -> None:
    comparison_config = tmp_path / "comparison.toml"
    comparison_config.touch()
    cfg = _mock_cfg(preconditioners=[StandardPreconditionerConfig(name="none", type=PreconditionerType.IDENTITY)])
    cfg.general.tracking = None

    with patch(_LOAD_COMPARISON_CONFIG, return_value=cfg):
        outcomes = run_comparison(comparison_config, ComparisonParams(), None)
    assert outcomes[0].success is False
    assert outcomes[0].error is not None
    assert "general.tracking is required" in outcomes[0].error


def test_run_comparison_pipeline_also_requires_tracking(
    comparison_run: ComparisonRun,
    tmp_path: Path,
) -> None:
    comparison_config = tmp_path / "comparison.toml"
    comparison_config.touch()
    cfg = _mock_cfg(preconditioners=[StandardPreconditionerConfig(name="none", type=PreconditionerType.IDENTITY)])
    cfg.general.tracking = None

    with patch(_LOAD_COMPARISON_CONFIG, return_value=cfg):
        outcomes = run_comparison(comparison_config, ComparisonParams(), comparison_run)
    assert outcomes[0].success is False
    assert outcomes[0].error is not None
    assert "general.tracking is required" in outcomes[0].error


def test_run_comparison_rejects_experiments_config_path(tmp_path: Path) -> None:
    comparison_config = tmp_path / "comparison.toml"
    comparison_config.touch()
    outcomes = run_comparison(
        comparison_config,
        ComparisonParams(),
        comparison_run=None,
        experiments_config_path=tmp_path / "experiments.toml",
    )
    assert outcomes == [
        ComparisonOutcome(
            name=comparison_config.stem,
            success=False,
            error="experiments_config_path is not supported in comparison schema_version=3.",
        )
    ]
