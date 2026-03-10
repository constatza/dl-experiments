"""Tests for strict model_ref resolution helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from neuralls.configuration.preconditioner import (
    NeuralPreconditionerConfig,
    PreconditionerType,
    StandardPreconditionerConfig,
)
from neuralls.workflows.model_resolution import (
    ModelResolution,
    PreconditionerResolutionResult,
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


@patch("neuralls.workflows.model_resolution.resolve_model_ref")
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
        model_ref={"source": "logged", "run_id": "run-1"},
    )

    resolved = resolve_preconditioner_models(
        specs=[neural],
        tracking_uri=_tracking_uri(tmp_path),
        download_root=tmp_path,
    )
    assert len(resolved) == 1
    resolved_neural = resolved[0]
    assert resolved_neural.type == PreconditionerType.NEURAL
    assert resolved_neural.checkpoint_path == checkpoint
    assert resolved_neural.resolved_checkpoint_path == checkpoint


@patch("neuralls.workflows.model_resolution.resolve_model_ref")
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


@patch("neuralls.workflows.model_resolution.resolve_model_ref")
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
        model_ref={"source": "registered", "name": "MissingFFNN", "alias": "solutions"},
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


@patch("neuralls.workflows.model_resolution.MlflowClient")
@patch("neuralls.workflows.model_resolution.search_registered_models")
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
        model_ref={"source": "registered", "name": "NormScaledLinearFFNN", "alias": "@dataset"},
    )
    with pytest.raises(ValueError, match="requires general\\.data\\.dataset_alias"):
        resolve_model_ref(
            spec=neural,
            tracking_uri=_tracking_uri(tmp_path),
            destination=tmp_path,
        )


@patch("neuralls.workflows.model_resolution._download_checkpoint_for_run")
@patch("neuralls.workflows.model_resolution.search_registered_models")
@patch("neuralls.workflows.model_resolution.MlflowClient")
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
        model_ref={"source": "registered", "name": "NormScaledLinearFFNN", "alias": "@solutions"},
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
