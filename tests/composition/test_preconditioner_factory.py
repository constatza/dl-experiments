"""Unit tests for preconditioner factory function.

Tests factory creation, error handling, and edge cases independent of solver internals.
Follows project principles:
- Use fixtures for all test data
- Use tmp_path for temporary files (never tempfile)
- Type hints throughout
- Focus on our own logic, not torchalg internals
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.testing import assert_close
from torchalg.preconditioners.base import PreconditionerContext
from torchalg.preconditioners.implementations import (
    Identity,
    ILUPreconditioner,
    JacobiPreconditioner,
    NeuralPreconditioner,
    ScheduledPreconditioner,
)
from torchalg.preconditioners.ports import ExtraInputPredictorPort, PredictorAdapter

from neuralls.composition.preconditioners.factory import (
    create_preconditioner,
    create_scheduled_preconditioner,
    PreconditionerScheduleConfig,
)
from neuralls.platform.config.models.preconditioner import (
    NeuralPreconditionerConfig,
    PreconditionerType,
    StandardPreconditionerConfig,
)

# ==============================================================================
# Mock Predictor and Adapter for Neural Preconditioner Testing
# ==============================================================================


class MockPredictor(ExtraInputPredictorPort):
    """Lightweight mock predictor for testing."""

    def __init__(self) -> None:
        """Initialize mock predictor."""
        self.cleaned_up = False
        self.apply_count = 0

    @property
    def required_inputs(self) -> tuple[str, ...]:
        """Return empty tuple — this mock needs no extra inputs."""
        return ()

    def apply(self, residual: torch.Tensor, **extra_inputs: torch.Tensor) -> torch.Tensor:
        """Mock apply: returns half of input."""
        self.apply_count += 1
        return residual * 0.5

    def cleanup(self) -> None:
        """Mark as cleaned up."""
        self.cleaned_up = True

    def __enter__(self) -> MockPredictor:
        """Enter context manager."""
        return self

    def __exit__(self, *args: object) -> None:
        """Exit context manager with cleanup."""
        self.cleanup()


class MockAdapter(PredictorAdapter):
    """Lightweight mock adapter for testing."""

    def __init__(self) -> None:
        """Initialize mock adapter."""
        self.predictor = MockPredictor()

    def create_predictor(
        self,
        checkpoint_path: Path,
        config_path: Path | None = None,
        data_config_path: Path | None = None,
    ) -> ExtraInputPredictorPort:
        """Create mock predictor, raising if the checkpoint is missing."""
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return self.predictor


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def well_conditioned_matrix() -> torch.Tensor:
    """Well-conditioned 4x4 SPD matrix for testing (condition number ~2)."""
    return torch.diag(torch.tensor([4.0, 3.0, 2.0, 2.0], dtype=torch.float64))


@pytest.fixture
def dense_spd_matrix() -> torch.Tensor:
    """Dense 5x5 SPD tridiagonal matrix for ILU testing."""
    n = 5
    return (
        2 * torch.eye(n, dtype=torch.float64)
        + torch.diag(-torch.ones(n - 1, dtype=torch.float64), 1)
        + torch.diag(-torch.ones(n - 1, dtype=torch.float64), -1)
    )


@pytest.fixture
def residual_vector() -> torch.Tensor:
    """4D residual vector."""
    return torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)


@pytest.fixture
def mock_checkpoint(tmp_path: Path) -> Path:
    """Create mock checkpoint file."""
    checkpoint = tmp_path / "mock_checkpoint.ckpt"
    checkpoint.write_text("mock checkpoint data")
    return checkpoint


# ==============================================================================
# Factory Tests - Identity
# ==============================================================================


def test_factory_creates_identity_preconditioner(well_conditioned_matrix: torch.Tensor) -> None:
    """Factory creates Identity preconditioner for identity type."""
    config = StandardPreconditionerConfig(name="identity", type=PreconditionerType.IDENTITY)

    precond = create_preconditioner(well_conditioned_matrix, config)

    assert isinstance(precond, Identity)


def test_factory_creates_identity_for_none_type(well_conditioned_matrix: torch.Tensor) -> None:
    """Factory creates Identity for 'none' type alias."""
    config = StandardPreconditionerConfig(name="none", type=PreconditionerType.NONE)

    precond = create_preconditioner(well_conditioned_matrix, config)

    assert isinstance(precond, Identity)


def test_identity_preconditioner_returns_copy(
    well_conditioned_matrix: torch.Tensor, residual_vector: torch.Tensor
) -> None:
    """Identity preconditioner returns an equal-valued clone of the input."""
    config = StandardPreconditionerConfig(name="identity", type=PreconditionerType.IDENTITY)

    precond = create_preconditioner(well_conditioned_matrix, config)
    result = precond.apply(residual_vector)

    assert_close(result, residual_vector)
    assert result is not residual_vector


# ==============================================================================
# Factory Tests - Jacobi
# ==============================================================================


def test_factory_creates_jacobi_preconditioner(well_conditioned_matrix: torch.Tensor) -> None:
    """Factory creates JacobiPreconditioner for jacobi type."""
    config = StandardPreconditionerConfig(name="jacobi", type=PreconditionerType.JACOBI)

    precond = create_preconditioner(well_conditioned_matrix, config)

    assert isinstance(precond, JacobiPreconditioner)


def test_jacobi_preconditioner_applies_diagonal_scaling(
    well_conditioned_matrix: torch.Tensor, residual_vector: torch.Tensor
) -> None:
    """JacobiPreconditioner divides by the diagonal: z_i = r_i / A_ii."""
    config = StandardPreconditionerConfig(name="jacobi", type=PreconditionerType.JACOBI)

    precond = create_preconditioner(well_conditioned_matrix, config)
    result = precond.apply(residual_vector)

    expected = residual_vector / torch.diagonal(well_conditioned_matrix)
    assert_close(result, expected)


def test_jacobi_preserves_dtype(well_conditioned_matrix: torch.Tensor) -> None:
    """JacobiPreconditioner preserves float64 dtype."""
    config = StandardPreconditionerConfig(name="jacobi", type=PreconditionerType.JACOBI)

    precond = create_preconditioner(well_conditioned_matrix, config)
    residual = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    result = precond.apply(residual)

    assert result.dtype == torch.float64


# ==============================================================================
# Factory Tests - ILU
# ==============================================================================


def test_factory_creates_ilu_preconditioner(dense_spd_matrix: torch.Tensor) -> None:
    """Factory creates ILUPreconditioner for ilu type."""
    config = StandardPreconditionerConfig(name="ilu", type=PreconditionerType.ILU)

    precond = create_preconditioner(dense_spd_matrix, config)

    assert isinstance(precond, ILUPreconditioner)


def test_ilu_preconditioner_accepts_dense_matrix(dense_spd_matrix: torch.Tensor) -> None:
    """ILUPreconditioner applies without error on a dense tridiagonal matrix."""
    config = StandardPreconditionerConfig(name="ilu", type=PreconditionerType.ILU)

    precond = create_preconditioner(dense_spd_matrix, config)

    residual_5d = torch.ones(5, dtype=torch.float64)
    result = precond.apply(residual_5d)

    assert result.shape == (5,)
    assert result.dtype == torch.float64


def test_ilu_preserves_dtype(dense_spd_matrix: torch.Tensor) -> None:
    """ILUPreconditioner preserves float64 dtype."""
    config = StandardPreconditionerConfig(name="ilu", type=PreconditionerType.ILU)

    precond = create_preconditioner(dense_spd_matrix, config)
    residual = torch.ones(5, dtype=torch.float64)
    result = precond.apply(residual)

    assert result.dtype == torch.float64


# ==============================================================================
# Factory Tests - Neural
# ==============================================================================


def test_factory_creates_neural_preconditioner(
    well_conditioned_matrix: torch.Tensor, mock_checkpoint: Path
) -> None:
    """Factory creates NeuralPreconditioner for neural type."""
    mock_adapter = MockAdapter()

    config = NeuralPreconditionerConfig(
        name="neural",
        type=PreconditionerType.NEURAL,
        checkpoint_path=mock_checkpoint,
    )

    precond = create_preconditioner(well_conditioned_matrix, config, adapter=mock_adapter)

    assert isinstance(precond, NeuralPreconditioner)


def test_neural_preconditioner_with_custom_adapter(
    well_conditioned_matrix: torch.Tensor, mock_checkpoint: Path, residual_vector: torch.Tensor
) -> None:
    """NeuralPreconditioner delegates to the injected adapter's predictor (DIP)."""
    mock_adapter = MockAdapter()

    config = NeuralPreconditionerConfig(
        name="neural",
        type=PreconditionerType.NEURAL,
        checkpoint_path=mock_checkpoint,
    )

    precond = create_preconditioner(well_conditioned_matrix, config, adapter=mock_adapter)
    result = precond.apply(residual_vector)

    expected = residual_vector * 0.5
    assert_close(result, expected)


def test_neural_preconditioner_checkpoint_not_found(
    well_conditioned_matrix: torch.Tensor,
    tmp_path: Path,
) -> None:
    """NeuralPreconditioner surfaces a missing checkpoint as FileNotFoundError."""
    mock_adapter = MockAdapter()
    missing_checkpoint = tmp_path / "missing" / "checkpoint.ckpt"

    config = NeuralPreconditionerConfig(
        name="neural",
        type=PreconditionerType.NEURAL,
        checkpoint_path=missing_checkpoint,
    )

    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        create_preconditioner(well_conditioned_matrix, config, adapter=mock_adapter)


def test_neural_preconditioner_cleanup_on_delete(
    well_conditioned_matrix: torch.Tensor, mock_checkpoint: Path, residual_vector: torch.Tensor
) -> None:
    """Neural preconditioner cleans up its predictor when cleanup() is called."""
    mock_adapter = MockAdapter()

    config = NeuralPreconditionerConfig(
        name="neural",
        type=PreconditionerType.NEURAL,
        checkpoint_path=mock_checkpoint,
    )

    precond = create_preconditioner(well_conditioned_matrix, config, adapter=mock_adapter)

    assert isinstance(precond, NeuralPreconditioner)
    predictor = mock_adapter.predictor
    assert not predictor.cleaned_up
    _ = precond.apply(residual_vector)

    precond.cleanup()

    assert predictor.cleaned_up


# ==============================================================================
# Factory Tests - Error Handling
# ==============================================================================


def test_factory_rejects_unsupported_type(well_conditioned_matrix: torch.Tensor) -> None:
    """Factory raises ValueError for unsupported preconditioner type."""
    config = StandardPreconditionerConfig(name="invalid", type=PreconditionerType.IDENTITY)
    config = config.model_copy(update={"type": "invalid_type"})

    with pytest.raises(ValueError, match="Unsupported preconditioner type"):
        create_preconditioner(well_conditioned_matrix, config)


def test_factory_requires_neural_config_for_neural_type(
    well_conditioned_matrix: torch.Tensor,
) -> None:
    """Factory requires NeuralPreconditionerConfig for neural type."""
    config = StandardPreconditionerConfig(name="neural", type=PreconditionerType.IDENTITY)
    config = config.model_copy(update={"type": PreconditionerType.NEURAL})

    with pytest.raises(TypeError, match="Neural type requires NeuralPreconditionerConfig"):
        create_preconditioner(well_conditioned_matrix, config)


# ==============================================================================
# Scheduled Preconditioner Builder Tests
# ==============================================================================


def test_create_scheduled_preconditioner_no_scheduling() -> None:
    """Builder returns primary unchanged when no scheduling is needed."""
    primary = Identity()
    schedule = PreconditionerScheduleConfig(limit_iters=-1)
    result = create_scheduled_preconditioner(primary, schedule)

    assert result is primary


def test_create_scheduled_preconditioner_with_limit() -> None:
    """Builder wraps the preconditioner when limit_iters is specified."""
    primary = Identity()
    schedule = PreconditionerScheduleConfig(limit_iters=10)
    result = create_scheduled_preconditioner(primary, schedule)

    assert isinstance(result, ScheduledPreconditioner)
    assert result._limit_iters == 10
    assert result._start_iter == 0


def test_create_scheduled_preconditioner_with_start_iter() -> None:
    """Builder forwards delayed activation to ScheduledPreconditioner."""
    primary = Identity()
    schedule = PreconditionerScheduleConfig(start_iter=7, limit_iters=10)
    result = create_scheduled_preconditioner(primary, schedule)

    assert isinstance(result, ScheduledPreconditioner)
    assert result._start_iter == 7
    assert result._limit_iters == 10


def test_create_scheduled_preconditioner_wraps_unlimited_delayed_schedule() -> None:
    """Delayed unlimited schedules are still wrapped."""
    primary = Identity()
    schedule = PreconditionerScheduleConfig(start_iter=3, limit_iters=-1)
    result = create_scheduled_preconditioner(primary, schedule)

    assert isinstance(result, ScheduledPreconditioner)
    assert result._start_iter == 3
    assert result._limit_iters is None


def test_create_scheduled_preconditioner_delayed_schedule_uses_fallback_before_start(
    well_conditioned_matrix: torch.Tensor,
    residual_vector: torch.Tensor,
) -> None:
    """Delayed factory schedules use the identity fallback before activation."""
    primary = JacobiPreconditioner(well_conditioned_matrix)
    schedule = PreconditionerScheduleConfig(start_iter=2, limit_iters=-1)
    result = create_scheduled_preconditioner(primary, schedule)

    early_ctx = PreconditionerContext(iteration=1, residual_norm=1.0, rhs_norm=1.0)
    active_ctx = PreconditionerContext(iteration=2, residual_norm=1.0, rhs_norm=1.0)

    assert_close(result.apply(residual_vector, early_ctx), residual_vector)
    assert_close(
        result.apply(residual_vector, active_ctx),
        torch.tensor([0.25, 2 / 3, 1.5, 2.0], dtype=torch.float64),
    )


def test_create_scheduled_preconditioner_default_fallback() -> None:
    """Identity is used as the default fallback."""
    primary = Identity()
    schedule = PreconditionerScheduleConfig(limit_iters=5)
    result = create_scheduled_preconditioner(primary, schedule)

    ctx = PreconditionerContext(iteration=10, residual_norm=1.0, rhs_norm=1.0)
    r = torch.tensor([1.0, 2.0], dtype=torch.float64)
    z = result.apply(r, ctx)

    assert_close(z, r)


def test_create_scheduled_preconditioner_with_identity_fallback() -> None:
    """Explicit Identity fallback type behaves the same as the default."""
    primary = Identity()
    schedule = PreconditionerScheduleConfig(
        limit_iters=5,
        fallback=PreconditionerType.IDENTITY,
    )
    result = create_scheduled_preconditioner(primary, schedule)

    ctx = PreconditionerContext(iteration=10, residual_norm=1.0, rhs_norm=1.0)
    r = torch.tensor([1.0, 2.0], dtype=torch.float64)
    z = result.apply(r, ctx)

    assert_close(z, r)
