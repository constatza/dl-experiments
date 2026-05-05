"""Tests for neuralls.composition.experiments.comparison_batch."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import tomli_w

from neuralls.domain.solver.models.config import ComparisonData, ComparisonGeneral, SolverParams
from neuralls.platform.config.mlflow import build_sqlite_tracking_uri
from neuralls.platform.config.models.preconditioner import (
    LoggedModelRefConfig,
    NeuralPreconditionerConfig,
    PreconditionerType,
    PreconditionerConfig,
    RegisteredModelRefConfig,
    StandardPreconditionerConfig,
)
from neuralls.domain.solver.models.result import CGComparisonResult
from neuralls.composition.experiments.comparison_batch import (
    _resolve_neural_preconditioners,
    _validate_neural_preconditioner,
    run_comparison,
    run_comparison_batch,
)
from neuralls.platform.reporting.artifacts import (
    coerce_comparison_result_payload,
    extract_array_artifacts,
    serialize_comparison_payload,
)
from neuralls.domain.solver.models.result import (
    ComparisonRecommendations,
    ComparisonResult,
    PlotPaths,
    RankedRecommendation,
)
from neuralls.composition.comparison.models import ComparisonOutcome, ComparisonParams

_LOAD_COMPARISON_CONFIG = "neuralls.composition.experiments.comparison_batch.load_comparison_config"
_COMPARE_PRECONDITIONERS = (
    "neuralls.composition.experiments.comparison_batch.compare_preconditioners"
)
_MLFLOW_MODULE = "neuralls.composition.experiments.comparison_batch.mlflow"
_SETUP_TRACKING = "neuralls.composition.experiments.comparison_batch.setup_comparison_tracking"


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
                f'matrix_path = "{matrix_path}"',
                f'rhs_path = "{rhs_path}"',
                "",
                "[[preconditioners]]",
                'name = "none"',
                'type = "identity"',
            ]
        ),
        encoding="utf-8",
    )


def _write_experiments_config(path: Path, *, with_comparisons: bool = False) -> None:
    payload: dict[str, object] = {
        "mlflow": {"tracking_uri": build_sqlite_tracking_uri(path.parent / "mlruns" / "mlflow.db")},
        "names": {
            "training": "neuralls-training",
            "comparison": "Comparisons",
        },
    }
    if with_comparisons:
        payload["comparisons"] = [
            {"id": "a", "path": "comparison-a.toml"},
            {"id": "b", "path": "comparison-b.toml"},
        ]
        _write_comparison_config(path.parent / "comparison-a.toml")
        _write_comparison_config(path.parent / "comparison-b.toml")
    with path.open("wb") as fh:
        tomli_w.dump(payload, fh)


def _configure_mock_mlflow(mock_mlflow: MagicMock, run_id: str = "comp-run-id") -> None:
    mock_run = MagicMock()
    mock_run.info.run_id = run_id
    mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
    mock_mlflow.start_run.return_value.__exit__.return_value = False
    mock_mlflow.get_artifact_uri.return_value = f"mlartifacts/0/{run_id}/artifacts"


def _mock_cfg(
    *,
    preconditioners: list[PreconditionerConfig] | None = None,
) -> MagicMock:
    cfg = MagicMock()
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
    comparison_config = tmp_path / "comparison.toml"
    experiments_config = tmp_path / "experiments.toml"
    comparison_config.touch()
    _write_experiments_config(experiments_config)
    cfg = _mock_cfg(
        preconditioners=[
            StandardPreconditionerConfig(name="none", type=PreconditionerType.IDENTITY)
        ]
    )
    payload = MagicMock()

    with (
        patch(_LOAD_COMPARISON_CONFIG, return_value=cfg),
        patch(_COMPARE_PRECONDITIONERS, return_value=payload),
        patch(_MLFLOW_MODULE) as mock_mlflow,
        patch(_SETUP_TRACKING) as mock_setup_tracking,
    ):
        _configure_mock_mlflow(mock_mlflow)
        outcomes = run_comparison(
            comparison_config,
            ComparisonParams(),
            experiments_config_path=experiments_config,
        )

    assert outcomes[0].success is True
    mock_setup_tracking.assert_called_once()
    assert mock_setup_tracking.call_args.kwargs["experiment_name"] == "Comparisons"


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
    comparison_config = tmp_path / "comparison.toml"
    comparison_config.touch()
    cfg = _mock_cfg(
        preconditioners=[
            StandardPreconditionerConfig(name="none", type=PreconditionerType.IDENTITY)
        ]
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

    def _capture_logged_artifacts(path: str) -> None:
        nonlocal logged_files, comparison_json
        root = Path(path)
        logged_files = {
            item.relative_to(root).as_posix() for item in root.rglob("*") if item.is_file()
        }
        comparison_json = json.loads((root / "comparison.json").read_text(encoding="utf-8"))

    with (
        patch(_LOAD_COMPARISON_CONFIG, return_value=cfg),
        patch(_COMPARE_PRECONDITIONERS, return_value=payload),
        patch(_MLFLOW_MODULE) as mock_mlflow,
        patch(_SETUP_TRACKING),
    ):
        _configure_mock_mlflow(mock_mlflow)
        mock_mlflow.log_artifacts.side_effect = _capture_logged_artifacts
        outcomes = run_comparison(
            comparison_config,
            ComparisonParams(),
            experiments_config_path=experiments_config,
        )

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
    comparison_config = tmp_path / "comparison.toml"
    experiments_config = tmp_path / "experiments.toml"
    comparison_config.touch()
    _write_experiments_config(experiments_config)
    cfg = _mock_cfg(
        preconditioners=[
            StandardPreconditionerConfig(name="none", type=PreconditionerType.IDENTITY),
            NeuralPreconditionerConfig(
                name="missing-neural",
                type=PreconditionerType.NEURAL,
                model_ref=_registered_ref("MissingFFNN", "solutions"),
            ),
        ]
    )
    payload = MagicMock()

    with (
        patch(_LOAD_COMPARISON_CONFIG, return_value=cfg),
        patch(_COMPARE_PRECONDITIONERS, return_value=payload),
        patch(_MLFLOW_MODULE) as mock_mlflow,
        patch(_SETUP_TRACKING),
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
        outcomes = run_comparison(
            comparison_config,
            ComparisonParams(),
            experiments_config_path=experiments_config,
        )

    assert outcomes[0].success is True
    assert len(outcomes[0].warnings) == 1
    assert "Skipping neural preconditioner 'missing-neural'" in outcomes[0].warnings[0]


def test_run_comparison_fails_if_all_preconditioners_are_skipped(
    tmp_path: Path,
) -> None:
    comparison_config = tmp_path / "comparison.toml"
    experiments_config = tmp_path / "experiments.toml"
    comparison_config.touch()
    _write_experiments_config(experiments_config)
    cfg = _mock_cfg(
        preconditioners=[
            NeuralPreconditionerConfig(
                name="missing-neural",
                type=PreconditionerType.NEURAL,
                model_ref=_registered_ref("MissingFFNN", "solutions"),
            ),
        ]
    )

    with (
        patch(_LOAD_COMPARISON_CONFIG, return_value=cfg),
        patch(_MLFLOW_MODULE) as mock_mlflow,
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
        outcomes = run_comparison(
            comparison_config,
            ComparisonParams(),
            experiments_config_path=experiments_config,
        )

    assert outcomes[0].success is False
    assert "No runnable preconditioners remain" in (outcomes[0].error or "")


def test_run_comparison_ignores_unrelated_broken_experiments(
    tmp_path: Path,
) -> None:
    comparison_config = tmp_path / "comparison.toml"
    comparison_config.touch()
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
    cfg = _mock_cfg(
        preconditioners=[
            NeuralPreconditionerConfig(
                name="registered-neural",
                type=PreconditionerType.NEURAL,
                model_ref=_registered_ref(
                    "NormScaledLinearFFNN",
                    "residuals-100",
                ),
            ),
        ]
    )
    payload = MagicMock()

    with (
        patch(_LOAD_COMPARISON_CONFIG, return_value=cfg),
        patch(_COMPARE_PRECONDITIONERS, return_value=payload),
        patch(_MLFLOW_MODULE) as mock_mlflow,
        patch(_SETUP_TRACKING),
        patch(
            "neuralls.composition.experiments.comparison_batch.resolve_preconditioner_models_with_warnings",
            return_value=MagicMock(specs=cfg.preconditioners, warnings=()),
        ) as mock_resolve_specs,
    ):
        _configure_mock_mlflow(mock_mlflow)
        outcomes = run_comparison(
            comparison_config,
            ComparisonParams(),
            experiments_config_path=experiments_config,
        )

    assert outcomes[0].success is True
    assert mock_resolve_specs.call_args.kwargs["experiment_contexts"] is None


def test_run_comparison_batch_preserves_declared_order(tmp_path: Path) -> None:
    experiments_config = tmp_path / "experiments.toml"
    _write_experiments_config(experiments_config, with_comparisons=True)

    def _fake_run_comparison(**kwargs: object) -> list[ComparisonOutcome]:
        return [
            ComparisonOutcome(
                comparison_id=str(kwargs["comparison_id"]),
                comparison_display_name=str(kwargs["comparison_display_name"]),
                success=True,
            )
        ]

    with patch(
        "neuralls.composition.experiments.comparison_batch.run_comparison",
        side_effect=_fake_run_comparison,
    ) as mock_run:
        outcomes = run_comparison_batch(experiments_config, ComparisonParams())

    assert [outcome.comparison_id for outcome in outcomes] == ["a", "b"]
    assert [call.kwargs["comparison_id"] for call in mock_run.call_args_list] == ["a", "b"]
