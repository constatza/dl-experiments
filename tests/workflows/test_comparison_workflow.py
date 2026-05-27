"""Tests for neuralls.composition.experiments.comparison_batch."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import tomli_w

from neuralls.composition.comparison.models import ComparisonOutcome, ComparisonParams
from neuralls.composition.experiments.comparison_batch import (
    _resolve_neural_preconditioners,
    _run_comparison_from_config,
    _validate_neural_preconditioner,
    run_comparison_batch,
)
from neuralls.domain.solver.models.config import ComparisonData, ComparisonGeneral, SolverParams
from neuralls.domain.solver.models.result import (
    CGComparisonResult,
    ComparisonRecommendations,
    ComparisonResult,
    PlotPaths,
    RankedRecommendation,
)
from neuralls.platform.config.models.experiments import ComparisonRegistryEntry
from neuralls.platform.config.models.preconditioner import (
    LoggedModelRefConfig,
    NeuralPreconditionerConfig,
    PreconditionerConfig,
    PreconditionerType,
    RegisteredModelRefConfig,
    StandardPreconditionerConfig,
)
from neuralls.platform.config.resolution import build_sqlite_tracking_uri
from neuralls.platform.config.settings import NeurallsSettings
from neuralls.platform.reporting.artifacts import (
    coerce_comparison_result_payload,
    extract_array_artifacts,
    serialize_comparison_payload,
)
from neuralls.platform.tracking.comparison_tracking import (
    log_comparison_result_metrics,
)
from neuralls.platform.tracking.mlflow import sanitize_metric_key_segment


def test_resolve_comparison_config_importable_from_composition() -> None:
    """resolve_comparison_config must live in composition, not platform."""
    from neuralls.composition.comparison.config_assembler import resolve_comparison_config  # noqa: F401


_RESOLVE_COMPARISON_CONFIG = (
    "neuralls.composition.comparison.config_assembler.resolve_comparison_config"
)
_COMPARE_PRECONDITIONERS = (
    "neuralls.composition.experiments.comparison_batch.compare_preconditioners"
)
_MLFLOW_MODULE = "neuralls.composition.experiments.comparison_batch.mlflow"
_COMPARISON_TRACKING_MLFLOW_MODULE = "neuralls.platform.tracking.comparison_tracking.mlflow"
_SETUP_TRACKING = "neuralls.composition.experiments.comparison_batch.setup_comparison_tracking"
_LOG_COMPARISON_ARTIFACTS = (
    "neuralls.composition.experiments.comparison_batch.log_comparison_artifacts_to_mlflow"
)


def _write_comparison_config(path: Path) -> None:
    matrix_path = path.parent / "matrix.npy"
    rhs_path = path.parent / "rhs.npy"
    path.write_text(
        "\n".join(
            [
                "[general]",
                "",
                "[general.params]",
                "rtol = 1.0e-6",
                "atol = 1.0e-14",
                "max_iterations = 10",
                'stopping_criterion = "residual_norm"',
                "m_max = 20",
                "",
                "[general.data]",
                f'matrix_path = "{matrix_path.as_posix()}"',
                f'rhs_path = "{rhs_path.as_posix()}"',
                "",
                "[[preconditioners]]",
                'name = "none"',
                'type = "identity"',
            ]
        ),
        encoding="utf-8",
    )


def _write_experiments_config(
    path: Path,
    *,
    with_comparisons: bool = False,
    comparison_name: str = "Comparisons",
) -> None:
    payload: dict[str, object] = {
        "mlflow": {"tracking_uri": build_sqlite_tracking_uri(path.parent / "mlruns" / "mlflow.db")},
        "names": {
            "training": "neuralls-training",
            "comparison": comparison_name,
        },
        "comparison_defaults": {
            "rtol": 1.0e-6,
            "atol": 1.0e-14,
            "max_iterations": 10,
            "stopping_criterion": "residual_norm",
            "m_max": 20,
            "preconditioners": [{"name": "none", "type": "identity"}],
        },
    }
    if with_comparisons:
        (path.parent / "datasets").mkdir(exist_ok=True)
        _write_dataset_config(path.parent / "datasets" / "solutions.toml", "solutions")
        _write_dataset_config(path.parent / "datasets" / "gaussian-rhs.toml", "gaussian-rhs")
        payload["datasets"] = [
            {"id": "solutions", "path": "datasets/solutions.toml"},
            {"id": "gaussian-rhs", "path": "datasets/gaussian-rhs.toml"},
        ]
        payload["comparisons"] = [
            {"id": "a", "matrix_dataset": "solutions", "rhs_dataset": "gaussian-rhs"},
            {"id": "b", "matrix_dataset": "solutions", "rhs_dataset": "gaussian-rhs"},
        ]
    with path.open("wb") as fh:
        tomli_w.dump(payload, fh)


def _write_dataset_config(path: Path, dataset_id: str) -> None:
    """Write a minimal dataset config stub for registry resolution."""
    path.write_text(
        "\n".join(
            [
                f'id = "{dataset_id}"',
                "",
                "[output]",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _configure_mock_mlflow(mock_mlflow: MagicMock, run_id: str = "comp-run-id") -> None:
    mock_run = MagicMock()
    mock_run.info.run_id = run_id
    mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
    mock_mlflow.start_run.return_value.__exit__.return_value = False
    mock_mlflow.get_artifact_uri.return_value = f"mlartifacts/0/{run_id}/artifacts"


def _mock_cfg(
    *,
    matrix_path: Path | None = None,
    rhs_path: Path | None = None,
    preconditioners: list[PreconditionerConfig] | None = None,
) -> MagicMock:
    cfg = MagicMock()
    cfg.general.data.matrix_path = matrix_path
    cfg.general.data.rhs_path = rhs_path
    cfg.general.data.dataset_alias = None
    cfg.preconditioners = tuple(preconditioners or [])
    return cfg


def _solver_params(tmp_path: Path) -> ComparisonGeneral:
    return ComparisonGeneral(
        params=SolverParams(
            rtol=1.0e-6,
            atol=1.0e-14,
            max_iterations=10,
            stopping_criterion="residual_norm",
            m_max=20,
            breakdown_tol=None,
        ),
        data=ComparisonData(
            matrix_path=tmp_path / "matrix.npy",
            rhs_path=tmp_path / "rhs.npy",
        ),
    )


def _write_system_inputs(tmp_path: Path) -> tuple[Path, Path]:
    """Create minimal matrix/RHS arrays used by comparison preflight."""
    matrix_path = tmp_path / "matrix.npy"
    rhs_path = tmp_path / "rhs.npy"
    np.save(matrix_path, np.eye(2, dtype=np.float64))
    np.save(rhs_path, np.array([[1.0, 0.0]], dtype=np.float64))
    return matrix_path, rhs_path


def _logged_ref(run_id: str) -> LoggedModelRefConfig:
    return LoggedModelRefConfig(run_id=run_id)


def _registered_ref(name: str, alias: str) -> RegisteredModelRefConfig:
    return RegisteredModelRefConfig(name=name, alias=alias)


def _typed_comparison_result(plot_path: Path) -> ComparisonResult:
    return ComparisonResult(
        results={
            "none": CGComparisonResult(
                x=np.array([1.0, 2.0]),
                converged=True,
                iterations=2,
                residual=1.0e-8,
                residual_abs=1.0e-9,
                residual_history_rel=[1.0, 1.0e-8],
                residual_history_abs=[1.0, 1.0e-9],
                preconditioner="none",
                initial_guess=np.zeros(2),
                exact_error=None,
                rhs_norm=1.0,
                breakdown=False,
            )
        },
        summary="ok",
        solver_params=_solver_params(plot_path.parent),
        plot_paths=PlotPaths(convergence=plot_path),
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


def _make_entry(comparison_id: str = "test") -> ComparisonRegistryEntry:
    """Build a minimal ComparisonRegistryEntry for direct _run_comparison_from_config calls."""
    return ComparisonRegistryEntry(
        id=comparison_id,
        matrix_dataset="solutions",
        rhs_dataset="gaussian-rhs",
    )


def _make_settings(tmp_path: Path) -> NeurallsSettings:
    """Build a NeurallsSettings backed by tmp_path."""
    return NeurallsSettings(
        _env_file=[],
        processed_dir=tmp_path / "processed",
        output_dir=tmp_path / "output",
    )


def test_comparison_data_model_tracks_explicit_normalize_in_fields_set() -> None:
    """model_fields_set must distinguish explicit normalize_system from default."""
    from neuralls.platform.config.models.comparison import ComparisonDataModel

    default_model = ComparisonDataModel()
    assert "normalize_system" not in default_model.model_fields_set

    explicit_model = ComparisonDataModel(normalize_system="matrix")
    assert "normalize_system" in explicit_model.model_fields_set


def test_validate_neural_preconditioner_requires_model_ref() -> None:
    spec = NeuralPreconditionerConfig(name="neural", type=PreconditionerType.NEURAL)
    try:
        _validate_neural_preconditioner(spec)
    except ValueError as exc:
        assert "model_ref" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_validate_neural_preconditioner_rejects_checkpoint_path(tmp_path: Path) -> None:
    spec = NeuralPreconditionerConfig(
        name="neural",
        type=PreconditionerType.NEURAL,
        checkpoint_path=tmp_path / "model.ckpt",
        model_ref=_logged_ref("run-1"),
    )
    try:
        _validate_neural_preconditioner(spec)
    except ValueError as exc:
        assert "checkpoint_path/experiment" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_resolve_neural_preconditioners_validates_all() -> None:
    specs = [
        StandardPreconditionerConfig(name="none", type=PreconditionerType.IDENTITY),
        NeuralPreconditionerConfig(
            name="neural",
            type=PreconditionerType.NEURAL,
            model_ref=_logged_ref("run-1"),
        ),
    ]
    resolved = _resolve_neural_preconditioners(specs)
    assert len(resolved) == 2


def test_extract_array_artifacts_detaches_numpy_data(tmp_path: Path) -> None:
    payload, array_artifacts = extract_array_artifacts(
        _typed_comparison_result(tmp_path / "convergence.png")
    )
    serialized = serialize_comparison_payload(payload)
    assert serialized["results"]["none"]["iterations"] == 2
    assert serialized["results"]["none"]["residual"] == 1.0e-8
    assert serialized["results"]["none"]["x"] == {
        "path": "arrays/results/none/x.npy",
        "shape": [2],
        "dtype": "float64",
    }
    artifact_paths = {artifact.reference.path for artifact in array_artifacts}
    assert Path("arrays/results/none/x.npy") in artifact_paths


def test_coerce_comparison_result_payload_uses_safe_defaults_for_magicmock() -> None:
    payload = coerce_comparison_result_payload(MagicMock())
    assert payload.summary == ""
    assert payload.preconditioners == ()
    assert payload.condition_numbers == {}
    assert payload.plot_paths == PlotPaths()
    assert payload.recommendations == ComparisonRecommendations()
    assert payload.results == {}


def test_run_comparison_injects_master_topology(tmp_path: Path) -> None:
    experiments_config = tmp_path / "experiments.toml"
    _write_experiments_config(experiments_config, comparison_name="CustomComparison")
    matrix_path, rhs_path = _write_system_inputs(tmp_path)
    cfg = _mock_cfg(
        matrix_path=matrix_path,
        rhs_path=rhs_path,
        preconditioners=[
            StandardPreconditionerConfig(name="none", type=PreconditionerType.IDENTITY)
        ],
    )
    payload = MagicMock()
    entry = _make_entry()
    settings = _make_settings(tmp_path)

    with (
        patch(_COMPARE_PRECONDITIONERS, return_value=payload),
        patch(_MLFLOW_MODULE) as mock_mlflow,
        patch(_COMPARISON_TRACKING_MLFLOW_MODULE, mock_mlflow),
        patch(_SETUP_TRACKING) as mock_setup_tracking,
        patch(_LOG_COMPARISON_ARTIFACTS),
    ):
        _configure_mock_mlflow(mock_mlflow)
        outcomes = _run_comparison_from_config(cfg, entry, experiments_config, settings)

    assert outcomes[0].success is True
    mock_setup_tracking.assert_called_once()
    assert mock_setup_tracking.call_args.kwargs["experiment_name"] == "CustomComparison"


def test_sanitize_mlflow_metric_segment_replaces_invalid_characters() -> None:
    assert (
        sanitize_metric_key_segment(
            "Residual-Error 50 Gaussian 93x31 | Scale-Equivariant Constant-Width 3000"
        )
        == "Residual-Error_50_Gaussian_93x31_Scale-Equivariant_Constant-Width_3000"
    )


def test_log_comparison_metrics_uses_mlflow_safe_metric_names(tmp_path: Path) -> None:
    result = ComparisonResult(
        results={
            "Residual-Error 50 Gaussian 93x31 | Scale-Equivariant Constant-Width 3000": (
                CGComparisonResult(
                    x=np.array([1.0, 2.0]),
                    converged=True,
                    iterations=2,
                    residual=1.0e-8,
                    residual_abs=1.0e-9,
                    residual_history_rel=[1.0, 1.0e-8],
                    residual_history_abs=[1.0, 1.0e-9],
                    preconditioner="neural",
                    initial_guess=np.zeros(2),
                    exact_error=None,
                    rhs_norm=1.0,
                    breakdown=False,
                )
            )
        },
        summary="ok",
        solver_params=_solver_params(tmp_path),
        preconditioners=("neural",),
        condition_numbers={
            "Residual-Error 50 Gaussian 93x31 | Scale-Equivariant Constant-Width 3000": 1.0
        },
        recommendations=ComparisonRecommendations(),
    )

    with patch(_COMPARISON_TRACKING_MLFLOW_MODULE) as mock_mlflow:
        log_comparison_result_metrics(
            result,
            child_run_tags={
                "Residual-Error 50 Gaussian 93x31 | Scale-Equivariant Constant-Width 3000": {}
            },
        )

    logged_names = [call.args[0] for call in mock_mlflow.log_metric.call_args_list[:4]]
    assert logged_names == [
        "condition_number/Residual-Error_50_Gaussian_93x31_Scale-Equivariant_Constant-Width_3000",
        "iterations/Residual-Error_50_Gaussian_93x31_Scale-Equivariant_Constant-Width_3000",
        "final_residual/Residual-Error_50_Gaussian_93x31_Scale-Equivariant_Constant-Width_3000",
        "converged/Residual-Error_50_Gaussian_93x31_Scale-Equivariant_Constant-Width_3000",
    ]


def test_run_comparison_stages_plot_paths_before_logging(tmp_path: Path) -> None:
    """Comparison artifacts should record uploaded figures paths, not temp absolutes."""
    external_plots = tmp_path / "external-plots"
    external_plots.mkdir()
    experiments_config = tmp_path / "experiments.toml"
    _write_experiments_config(experiments_config)
    convergence = external_plots / "convergence.png"
    condition_numbers = external_plots / "condition_numbers.png"
    convergence.write_text("convergence", encoding="utf-8")
    condition_numbers.write_text("condition", encoding="utf-8")
    matrix_path, rhs_path = _write_system_inputs(tmp_path)
    cfg = _mock_cfg(
        matrix_path=matrix_path,
        rhs_path=rhs_path,
        preconditioners=[
            StandardPreconditionerConfig(name="none", type=PreconditionerType.IDENTITY)
        ],
    )
    payload = ComparisonResult(
        results={},
        summary="ok",
        solver_params=_solver_params(tmp_path),
        preconditioners=("none",),
        plot_paths=PlotPaths(
            convergence=convergence,
            condition_numbers=condition_numbers,
        ),
        recommendations=ComparisonRecommendations(),
    )
    logged_files: set[str] = set()
    comparison_json: dict[str, object] = {}
    entry = _make_entry()
    settings = _make_settings(tmp_path)

    def _capture_work_root(tracking_uri: str, run_id: str, work_root: Path) -> None:  # noqa: ARG001
        nonlocal logged_files, comparison_json
        logged_files = {
            item.relative_to(work_root).as_posix()
            for item in work_root.rglob("*")
            if item.is_file()
        }
        comparison_json = json.loads((work_root / "comparison.json").read_text(encoding="utf-8"))

    with (
        patch(_COMPARE_PRECONDITIONERS, return_value=payload),
        patch(_MLFLOW_MODULE) as mock_mlflow,
        patch(_COMPARISON_TRACKING_MLFLOW_MODULE, mock_mlflow),
        patch(_SETUP_TRACKING),
        patch(_LOG_COMPARISON_ARTIFACTS, side_effect=_capture_work_root),
    ):
        _configure_mock_mlflow(mock_mlflow)
        outcomes = _run_comparison_from_config(cfg, entry, experiments_config, settings)

    assert outcomes[0].success is True
    assert "figures/convergence.png" in logged_files
    assert "figures/condition_numbers.png" in logged_files
    plot_paths = comparison_json["plot_paths"]
    assert isinstance(plot_paths, dict)
    assert plot_paths["convergence"] == "figures/convergence.png"
    assert plot_paths["condition_numbers"] == "figures/condition_numbers.png"


def test_run_comparison_warns_and_continues_when_neural_resolution_fails(
    tmp_path: Path,
) -> None:
    experiments_config = tmp_path / "experiments.toml"
    _write_experiments_config(experiments_config)
    matrix_path, rhs_path = _write_system_inputs(tmp_path)
    cfg = _mock_cfg(
        matrix_path=matrix_path,
        rhs_path=rhs_path,
        preconditioners=[
            StandardPreconditionerConfig(name="none", type=PreconditionerType.IDENTITY),
            NeuralPreconditionerConfig(
                name="missing-neural",
                type=PreconditionerType.NEURAL,
                model_ref=_registered_ref("MissingFFNN", "solutions"),
            ),
        ],
    )
    payload = MagicMock()
    entry = _make_entry()
    settings = _make_settings(tmp_path)

    with (
        patch(_COMPARE_PRECONDITIONERS, return_value=payload),
        patch(_MLFLOW_MODULE) as mock_mlflow,
        patch(_COMPARISON_TRACKING_MLFLOW_MODULE, mock_mlflow),
        patch(_SETUP_TRACKING),
        patch(_LOG_COMPARISON_ARTIFACTS),
        patch(
            "neuralls.composition.experiments.comparison_batch.resolve_preconditioner_models_with_warnings",
            return_value=MagicMock(
                specs=[StandardPreconditionerConfig(name="none", type=PreconditionerType.IDENTITY)],
                warnings=(
                    "Skipping neural preconditioner 'missing-neural': Registered model 'MissingFFNN' not found",
                ),
            ),
        ),
    ):
        _configure_mock_mlflow(mock_mlflow)
        outcomes = _run_comparison_from_config(cfg, entry, experiments_config, settings)

    assert outcomes[0].success is True
    assert len(outcomes[0].warnings) == 1
    assert "Skipping neural preconditioner 'missing-neural'" in outcomes[0].warnings[0]


def test_run_comparison_fails_if_all_preconditioners_are_skipped(
    tmp_path: Path,
) -> None:
    experiments_config = tmp_path / "experiments.toml"
    _write_experiments_config(experiments_config)
    matrix_path, rhs_path = _write_system_inputs(tmp_path)
    cfg = _mock_cfg(
        matrix_path=matrix_path,
        rhs_path=rhs_path,
        preconditioners=[
            NeuralPreconditionerConfig(
                name="missing-neural",
                type=PreconditionerType.NEURAL,
                model_ref=_registered_ref("MissingFFNN", "solutions"),
            ),
        ],
    )
    entry = _make_entry()
    settings = _make_settings(tmp_path)

    with (
        patch(_MLFLOW_MODULE) as mock_mlflow,
        patch(_COMPARISON_TRACKING_MLFLOW_MODULE, mock_mlflow),
        patch(_SETUP_TRACKING),
        patch(
            "neuralls.composition.experiments.comparison_batch.resolve_preconditioner_models_with_warnings",
            return_value=MagicMock(
                specs=[],
                warnings=(
                    "Skipping neural preconditioner 'missing-neural': Registered model 'MissingFFNN' not found",
                ),
            ),
        ),
    ):
        _configure_mock_mlflow(mock_mlflow)
        outcomes = _run_comparison_from_config(cfg, entry, experiments_config, settings)

    assert outcomes[0].success is False
    assert "No runnable preconditioners remain" in (outcomes[0].error or "")


def test_run_comparison_ignores_unrelated_broken_experiments(
    tmp_path: Path,
) -> None:
    (tmp_path / "datasets").mkdir()
    (tmp_path / "models").mkdir()
    (tmp_path / "datasets" / "valid-dataset.toml").write_text(
        'id = "valid-dataset"\n',
        encoding="utf-8",
    )
    (tmp_path / "models" / "valid-model.toml").write_text(
        "[MODEL]\nname = 'NormScaledLinearFFNN'\nmodule_path = 'dlkit.nn'\n",
        encoding="utf-8",
    )
    experiments_config = tmp_path / "experiments.toml"
    experiments_config.write_text(
        f"""
[mlflow]
tracking_uri = "{build_sqlite_tracking_uri(tmp_path / "mlflow.db")}"

[[datasets]]
id = "valid-dataset"
path = "datasets/valid-dataset.toml"

[[models]]
id = "valid-model"
path = "models/valid-model.toml"

[[experiments]]
id = "valid-exp"
dataset = "valid-dataset"
model = "valid-model"
        """.strip(),
        encoding="utf-8",
    )
    matrix_path, rhs_path = _write_system_inputs(tmp_path)
    cfg = _mock_cfg(
        matrix_path=matrix_path,
        rhs_path=rhs_path,
        preconditioners=[
            NeuralPreconditionerConfig(
                name="registered-neural",
                type=PreconditionerType.NEURAL,
                model_ref=_registered_ref(
                    "NormScaledLinearFFNN",
                    "residuals-100",
                ),
            ),
        ],
    )
    payload = MagicMock()
    entry = _make_entry()
    settings = _make_settings(tmp_path)

    with (
        patch(_COMPARE_PRECONDITIONERS, return_value=payload),
        patch(_MLFLOW_MODULE) as mock_mlflow,
        patch(_COMPARISON_TRACKING_MLFLOW_MODULE, mock_mlflow),
        patch(_SETUP_TRACKING),
        patch(_LOG_COMPARISON_ARTIFACTS),
        patch(
            "neuralls.composition.experiments.comparison_batch.resolve_preconditioner_models_with_warnings",
            return_value=MagicMock(specs=cfg.preconditioners, warnings=()),
        ) as mock_resolve_specs,
    ):
        _configure_mock_mlflow(mock_mlflow)
        outcomes = _run_comparison_from_config(cfg, entry, experiments_config, settings)

    assert outcomes[0].success is True
    assert mock_resolve_specs.call_args.kwargs["experiment_contexts"] is None


def test_run_comparison_fails_before_tracking_when_matrix_input_is_missing(
    tmp_path: Path,
) -> None:
    experiments_config = tmp_path / "experiments.toml"
    _write_experiments_config(experiments_config)
    missing_matrix = tmp_path / "missing-matrix.npy"
    _, rhs_path = _write_system_inputs(tmp_path)
    cfg = _mock_cfg(
        matrix_path=missing_matrix,
        rhs_path=rhs_path,
        preconditioners=[
            StandardPreconditionerConfig(name="none", type=PreconditionerType.IDENTITY)
        ],
    )
    entry = _make_entry()
    settings = _make_settings(tmp_path)

    with (
        patch(_MLFLOW_MODULE) as mock_mlflow,
        patch(_COMPARISON_TRACKING_MLFLOW_MODULE, mock_mlflow),
        patch(_SETUP_TRACKING) as mock_setup_tracking,
    ):
        outcomes = _run_comparison_from_config(cfg, entry, experiments_config, settings)

    assert outcomes[0].success is False
    assert f"Comparison input not found: {missing_matrix}" in (outcomes[0].error or "")
    mock_setup_tracking.assert_not_called()
    mock_mlflow.start_run.assert_not_called()


def test_run_comparison_missing_known_dataset_raises_file_not_found(
    tmp_path: Path,
) -> None:
    experiments_config = tmp_path / "experiments.toml"
    _write_experiments_config(experiments_config)
    missing_dataset = tmp_path / "solutions"
    rhs_path = tmp_path / "rhs.npy"
    np.save(rhs_path, np.array([[1.0, 0.0]], dtype=np.float64))
    cfg = _mock_cfg(
        matrix_path=missing_dataset,
        rhs_path=rhs_path,
        preconditioners=[
            StandardPreconditionerConfig(name="none", type=PreconditionerType.IDENTITY)
        ],
    )
    entry = _make_entry()
    settings = _make_settings(tmp_path)

    outcomes = _run_comparison_from_config(cfg, entry, experiments_config, settings)

    assert outcomes[0].success is False
    assert str(missing_dataset) in (outcomes[0].error or "")


def test_run_comparison_batch_preserves_declared_order(tmp_path: Path) -> None:
    experiments_config = tmp_path / "experiments.toml"
    _write_experiments_config(experiments_config, with_comparisons=True)

    def _fake_run_from_config(
        cfg: object,
        entry: ComparisonRegistryEntry,
        experiments_config_path: Path,
        settings: NeurallsSettings,
    ) -> list[ComparisonOutcome]:
        return [
            ComparisonOutcome(
                comparison_id=entry.id,
                comparison_display_name=entry.effective_display_name,
                success=True,
            )
        ]

    with (
        patch(
            "neuralls.composition.experiments.comparison_batch._run_comparison_from_config",
            side_effect=_fake_run_from_config,
        ) as mock_run,
        patch(
            "neuralls.composition.experiments.comparison_batch.resolve_comparison_config",
            return_value=_mock_cfg(),
        ),
    ):
        outcomes = run_comparison_batch(experiments_config, ComparisonParams())

    assert [outcome.comparison_id for outcome in outcomes] == ["a", "b"]
    assert [call.args[1].id for call in mock_run.call_args_list] == ["a", "b"]


def test_log_comparison_metrics_logs_scalar_metrics_per_preconditioner(
    tmp_path: Path,
) -> None:
    """_log_comparison_metrics must emit one metric set per preconditioner."""
    result = _typed_comparison_result(tmp_path / "conv.png")

    with patch(_COMPARISON_TRACKING_MLFLOW_MODULE) as mock_mlflow:
        log_comparison_result_metrics(result, child_run_tags={"none": {}})

    metric_calls = {call.args[0]: call.args[1] for call in mock_mlflow.log_metric.call_args_list}
    assert metric_calls["condition_number/none"] == 1.0
    assert metric_calls["iterations/none"] == 2
    assert metric_calls["final_residual/none"] == pytest.approx(1.0e-8)
    assert metric_calls["converged/none"] == 1
    mock_mlflow.log_param.assert_called_once_with("best_preconditioner", "none")


def test_run_comparison_logs_scalar_metrics_to_mlflow(tmp_path: Path) -> None:
    """compare workflow must log per-preconditioner metrics to the parent run."""
    experiments_config = tmp_path / "experiments.toml"
    _write_experiments_config(experiments_config)
    matrix_path, rhs_path = _write_system_inputs(tmp_path)
    cfg = _mock_cfg(
        matrix_path=matrix_path,
        rhs_path=rhs_path,
        preconditioners=[
            StandardPreconditionerConfig(name="none", type=PreconditionerType.IDENTITY)
        ],
    )
    plot_path = tmp_path / "conv.png"
    plot_path.write_text("x", encoding="utf-8")
    payload = _typed_comparison_result(plot_path)
    entry = _make_entry()
    settings = _make_settings(tmp_path)

    with (
        patch(_COMPARE_PRECONDITIONERS, return_value=payload),
        patch(_MLFLOW_MODULE) as mock_mlflow,
        patch(_COMPARISON_TRACKING_MLFLOW_MODULE, mock_mlflow),
        patch(_SETUP_TRACKING),
        patch(_LOG_COMPARISON_ARTIFACTS),
    ):
        _configure_mock_mlflow(mock_mlflow)
        outcomes = _run_comparison_from_config(cfg, entry, experiments_config, settings)

    assert outcomes[0].success is True
    logged_metric_names = {call.args[0] for call in mock_mlflow.log_metric.call_args_list}
    assert "condition_number/none" in logged_metric_names
    assert "iterations/none" in logged_metric_names
    assert "final_residual/none" in logged_metric_names
    assert "converged/none" in logged_metric_names


def test_log_comparison_result_metrics_logs_child_scalar_metrics(tmp_path: Path) -> None:
    """Each child run must log scalar summary metrics for its preconditioner."""
    result = _typed_comparison_result(tmp_path / "conv.png")
    captured_metrics: list[tuple[str, float]] = []

    def _fake_start_run(*args: object, **kwargs: object) -> MagicMock:
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=MagicMock())
        ctx.__exit__ = MagicMock(return_value=False)
        return ctx

    with patch(_COMPARISON_TRACKING_MLFLOW_MODULE) as mock_mlflow:
        mock_mlflow.start_run.side_effect = _fake_start_run
        mock_mlflow.log_metric.side_effect = lambda k, v, **kw: captured_metrics.append((k, v))
        log_comparison_result_metrics(result, child_run_tags={"none": {}})

    metric_map = {k: v for k, v in captured_metrics}
    assert metric_map["condition_number"] == 1.0
    assert metric_map["iterations"] == 2
    assert metric_map["final_residual"] == pytest.approx(1.0e-8)
    assert metric_map["converged"] == 1
