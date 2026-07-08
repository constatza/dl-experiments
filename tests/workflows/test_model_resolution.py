"""Tests for strict model_ref resolution helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from neuralls.platform.config.models.preconditioner import (
    LoggedModelRefConfig,
    NeuralPreconditionerConfig,
    PreconditionerType,
    RegisteredModelRefConfig,
    StandardPreconditionerConfig,
)
from neuralls.composition.assignments.model_resolution import (
    ModelResolution,
    PreconditionerResolutionResult,
    _download_checkpoint_for_run,
    _find_single_checkpoint,
    build_neural_download_dirname,
    resolve_model_ref,
    resolve_preconditioner_models,
    resolve_preconditioner_models_with_warnings,
)


def _tracking_uri(tmp_path: Path) -> str:
    """Build an absolute SQLite URI under the test temp directory."""
    return f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"


def test_resolve_preconditioner_models_keeps_non_neural_specs(tmp_path: Path) -> None:
    """Non-neural preconditioners pass through unchanged."""
    jacobi = StandardPreconditionerConfig(name="jacobi", type=PreconditionerType.JACOBI)
    resolved = resolve_preconditioner_models(
        specs=[jacobi],
        tracking_uri=_tracking_uri(tmp_path),
        download_root=tmp_path,
    )
    assert resolved == [jacobi]


def test_build_neural_download_dirname_prefers_assignment_id() -> None:
    """Stable assignment ids should drive local artifact directory naming."""
    neural = NeuralPreconditionerConfig(
        name="Residual-Error 100 | Embedded SPD",
        type=PreconditionerType.NEURAL,
        assignment="residuals-100-embedded-spd",
        model_ref=LoggedModelRefConfig(run_id="run-1"),
    )
    assert build_neural_download_dirname(neural) == "residuals-100-embedded-spd"


def test_build_neural_download_dirname_sanitizes_display_name_when_needed() -> None:
    """Fallback naming should strip Windows-invalid punctuation from labels."""
    neural = NeuralPreconditionerConfig(
        name='Residual-Error 100 Gaussian | Embedded SPD:?"<>',
        type=PreconditionerType.NEURAL,
        model_ref=LoggedModelRefConfig(run_id="run-1"),
    )
    dirname = build_neural_download_dirname(neural)
    assert dirname == "Residual-Error_100_Gaussian_Embedded_SPD"
    assert "|" not in dirname
    assert ":" not in dirname
    assert '"' not in dirname


@patch("neuralls.composition.assignments.model_resolution.resolve_model_ref")
def test_resolve_preconditioner_models_sets_resolved_checkpoint(
    mock_resolve_model_ref,
    tmp_path: Path,
) -> None:
    """Neural specs receive resolved_checkpoint_path from resolver."""
    checkpoint = tmp_path / "resolved.ckpt"
    checkpoint.write_text("checkpoint")
    mock_resolve_model_ref.return_value = ModelResolution(
        model_uri="runs:/run-1/model",
        run_id="run-1",
        checkpoint_path=checkpoint,
    )

    neural = NeuralPreconditionerConfig(
        name="neural",
        type=PreconditionerType.NEURAL,
        model_ref=LoggedModelRefConfig(run_id="run-1"),
    )

    resolved = resolve_preconditioner_models(
        specs=[neural],
        tracking_uri=_tracking_uri(tmp_path),
        download_root=tmp_path,
    )
    assert len(resolved) == 1
    resolved_neural = resolved[0]
    assert isinstance(resolved_neural, NeuralPreconditionerConfig)
    assert resolved_neural.type == PreconditionerType.NEURAL
    assert resolved_neural.checkpoint_path == checkpoint
    assert resolved_neural.resolved_checkpoint_path == checkpoint


@patch("neuralls.composition.assignments.model_resolution.resolve_model_ref")
def test_resolve_preconditioner_models_keeps_explicit_checkpoint(
    mock_resolve_model_ref,
    tmp_path: Path,
) -> None:
    """Neural specs with explicit checkpoint_path bypass model_ref resolution."""
    checkpoint = tmp_path / "explicit.ckpt"
    checkpoint.write_text("checkpoint")
    neural = NeuralPreconditionerConfig(
        name="neural",
        type=PreconditionerType.NEURAL,
        checkpoint_path=checkpoint,
    )

    resolved = resolve_preconditioner_models(
        specs=[neural],
        tracking_uri=_tracking_uri(tmp_path),
        download_root=tmp_path,
    )

    mock_resolve_model_ref.assert_not_called()
    assert len(resolved) == 1
    resolved_neural = resolved[0]
    assert isinstance(resolved_neural, NeuralPreconditionerConfig)
    assert resolved_neural.checkpoint_path == checkpoint
    assert resolved_neural.resolved_checkpoint_path == checkpoint


def test_resolve_preconditioner_models_requires_model_source(tmp_path: Path) -> None:
    """Neural specs without checkpoint_path/model_ref are invalid for resolution."""
    neural = NeuralPreconditionerConfig(
        name="neural",
        type=PreconditionerType.NEURAL,
    )

    with pytest.raises(ValueError, match="requires either checkpoint_path or model_ref"):
        resolve_preconditioner_models(
            specs=[neural],
            tracking_uri=_tracking_uri(tmp_path),
            download_root=tmp_path,
        )


@patch("neuralls.composition.assignments.model_resolution.resolve_model_ref")
def test_resolve_preconditioner_models_with_warnings_skips_unresolved_neural(
    mock_resolve_model_ref,
    tmp_path: Path,
) -> None:
    """Comparison-specific resolution can skip unresolved neural specs."""
    mock_resolve_model_ref.side_effect = ValueError("Registered model 'MissingFFNN' not found")
    jacobi = StandardPreconditionerConfig(name="jacobi", type=PreconditionerType.JACOBI)
    neural = NeuralPreconditionerConfig(
        name="missing-neural",
        type=PreconditionerType.NEURAL,
        model_ref=RegisteredModelRefConfig(name="MissingFFNN", alias="solutions"),
    )

    resolved = resolve_preconditioner_models_with_warnings(
        specs=[jacobi, neural],
        tracking_uri=_tracking_uri(tmp_path),
        download_root=tmp_path,
        skip_unresolved=True,
    )

    assert isinstance(resolved, PreconditionerResolutionResult)
    assert resolved.specs == [jacobi]
    assert len(resolved.warnings) == 1
    assert "Skipping neural preconditioner 'missing-neural'" in resolved.warnings[0]


@patch("neuralls.composition.assignments.model_resolution.MlflowClient")
@patch("neuralls.composition.assignments.model_resolution.search_registered_models")
def test_resolve_model_ref_dataset_placeholder_requires_dataset_alias(
    mock_client_cls,
    mock_search_registered_models,
    tmp_path: Path,
) -> None:
    """Using alias=@dataset requires a provided dataset alias context."""
    mock_client_cls.return_value = object()
    mock_search_registered_models.return_value = [object()]
    neural = NeuralPreconditionerConfig(
        name="neural",
        type=PreconditionerType.NEURAL,
        model_ref=RegisteredModelRefConfig(
            name="NormScaledLinearFFNN",
            alias="@dataset",
        ),
    )
    with pytest.raises(ValueError, match="requires general\\.data\\.dataset_alias"):
        resolve_model_ref(
            spec=neural,
            tracking_uri=_tracking_uri(tmp_path),
            destination=tmp_path,
        )


@patch("neuralls.composition.assignments.model_resolution._download_checkpoint_for_run")
@patch("neuralls.composition.assignments.model_resolution.search_registered_models")
@patch("neuralls.composition.assignments.model_resolution.MlflowClient")
def test_resolve_model_ref_normalizes_explicit_at_alias(
    mock_client_cls,
    mock_search_registered_models,
    mock_download_checkpoint,
    tmp_path: Path,
) -> None:
    """Registered alias values strip @ prefix before MLflow lookup."""
    mock_search_registered_models.return_value = [object()]
    mock_download_checkpoint.return_value = tmp_path / "resolved.ckpt"
    client = mock_client_cls.return_value
    client.get_model_version_by_alias.return_value.run_id = "run-1"

    neural = NeuralPreconditionerConfig(
        name="neural",
        type=PreconditionerType.NEURAL,
        model_ref=RegisteredModelRefConfig(
            name="NormScaledLinearFFNN",
            alias="@solutions",
        ),
    )
    resolve_model_ref(
        spec=neural,
        tracking_uri=_tracking_uri(tmp_path),
        destination=tmp_path,
    )

    client.get_model_version_by_alias.assert_called_once_with(
        "NormScaledLinearFFNN",
        "solutions",
    )


def test_find_single_checkpoint_collapses_nested_duplicate_copy(tmp_path: Path) -> None:
    """Nested duplicate artifact copies should resolve to one canonical checkpoint."""
    root = tmp_path / "run-download"
    duplicate_root = root / "checkpoints"
    root.mkdir(parents=True)
    duplicate_root.mkdir(parents=True)
    primary = root / "model.ckpt"
    duplicate = duplicate_root / "model.ckpt"
    primary.write_bytes(b"same-checkpoint")
    duplicate.write_bytes(b"same-checkpoint")

    resolved = _find_single_checkpoint(root)

    assert resolved == primary


def test_find_single_checkpoint_rejects_distinct_candidates(tmp_path: Path) -> None:
    """Multiple distinct checkpoint files in one run root should fail loudly."""
    root = tmp_path / "run-download"
    root.mkdir(parents=True)
    (root / "first.ckpt").write_bytes(b"first")
    (root / "second.ckpt").write_bytes(b"second")

    with pytest.raises(ValueError, match="Multiple distinct checkpoints found"):
        _find_single_checkpoint(root)


def test_find_single_checkpoint_stays_within_provided_root(tmp_path: Path) -> None:
    """Checkpoint discovery must not inspect sibling download roots."""
    target_root = tmp_path / "run-a"
    sibling_root = tmp_path / "run-b"
    target_root.mkdir(parents=True)
    sibling_root.mkdir(parents=True)
    expected = target_root / "model.ckpt"
    expected.write_bytes(b"target")
    (sibling_root / "other.ckpt").write_bytes(b"sibling")

    resolved = _find_single_checkpoint(target_root)

    assert resolved == expected


@patch("neuralls.composition.assignments.model_resolution.MlflowClient")
def test_download_checkpoint_for_run_collapses_nested_duplicate_artifacts(
    mock_client_cls,
    tmp_path: Path,
) -> None:
    """Downloaded duplicate trees from one run should resolve to one checkpoint."""
    download_root = tmp_path / "download"
    nested = download_root / "checkpoints"
    nested.mkdir(parents=True)
    primary = download_root / "model.ckpt"
    duplicate = nested / "model.ckpt"
    primary.write_bytes(b"same-checkpoint")
    duplicate.write_bytes(b"same-checkpoint")
    client = mock_client_cls.return_value
    client.download_artifacts.return_value = str(download_root)

    resolved = _download_checkpoint_for_run(
        client=client,
        run_id="run-1",
        destination=tmp_path / "out",
        fallback_artifact_path="model",
    )

    assert resolved == primary


@patch("neuralls.composition.assignments.model_resolution.MlflowClient")
def test_download_checkpoint_for_run_raises_on_ambiguous_distinct_artifacts(
    mock_client_cls,
    tmp_path: Path,
) -> None:
    """Downloaded distinct checkpoint candidates from one run should raise."""
    download_root = tmp_path / "download"
    download_root.mkdir(parents=True)
    (download_root / "first.ckpt").write_bytes(b"first")
    (download_root / "second.ckpt").write_bytes(b"second")
    client = mock_client_cls.return_value
    client.download_artifacts.return_value = str(download_root)

    with pytest.raises(ValueError, match="Multiple distinct checkpoints found"):
        _download_checkpoint_for_run(
            client=client,
            run_id="run-1",
            destination=tmp_path / "out",
            fallback_artifact_path="model",
        )
