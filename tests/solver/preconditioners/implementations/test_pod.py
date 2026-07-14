"""Tests for POD-2G preconditioner components."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.sparse import diags as sparse_diags


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def poisson_1d() -> NDArray:
    """1D Poisson matrix: tridiagonal [-1, 2, -1], size 20.

    Returns:
        20x20 dense SPD tridiagonal matrix.
    """
    n = 20
    return sparse_diags([-1, 2, -1], [-1, 0, 1], shape=(n, n), format="csr").toarray()


@pytest.fixture
def poisson_rhs(poisson_1d: NDArray) -> NDArray:
    """Right-hand side vector of ones matching poisson_1d size.

    Args:
        poisson_1d: The 20x20 Poisson matrix fixture.

    Returns:
        Vector of ones with length 20.
    """
    return np.ones(poisson_1d.shape[0])


@pytest.fixture
def poisson_snapshots(poisson_1d: NDArray) -> NDArray:
    """Snapshot ensemble spanning the Poisson matrix's solution space.

    Solves ``poisson_1d @ x = b_i`` for several random RHS vectors, so a
    full-rank POD basis recovers the solution space exactly and a truncated
    basis approximates it.

    Args:
        poisson_1d: The 20x20 Poisson matrix fixture.

    Returns:
        Snapshot ensemble, shape (15, 20) — one solution vector per row.
    """
    rng = np.random.default_rng(0)
    n = poisson_1d.shape[0]
    rhs_batch = rng.standard_normal((15, n))
    return np.linalg.solve(poisson_1d, rhs_batch.T).T


@pytest.fixture
def pod_basis(poisson_snapshots: NDArray) -> NDArray:
    """POD basis (rank 10) built from poisson_snapshots.

    Args:
        poisson_snapshots: Snapshot ensemble fixture.

    Returns:
        Φ_r, shape (20, 10).
    """
    from neuralls.domain.solver.preconditioners.implementations.pod import compute_pod_basis

    return compute_pod_basis(poisson_snapshots, rank=10)


# ---------------------------------------------------------------------------
# compute_pod_basis
# ---------------------------------------------------------------------------


class TestComputePodBasis:
    def test_output_shape(self, poisson_snapshots: NDArray) -> None:
        """Basis shape must be (n_dofs, rank)."""
        from neuralls.domain.solver.preconditioners.implementations.pod import compute_pod_basis

        basis = compute_pod_basis(poisson_snapshots, rank=10)
        assert basis.shape == (20, 10)

    def test_columns_orthonormal(self, pod_basis: NDArray) -> None:
        """Φ_rᵀΦ_r must equal the identity (snapshot-method POD property)."""
        gram = pod_basis.T @ pod_basis
        np.testing.assert_allclose(gram, np.eye(10), atol=1e-8)

    def test_rank_exceeds_samples_raises(self, poisson_snapshots: NDArray) -> None:
        """rank > n_samples must raise ValueError."""
        from neuralls.domain.solver.preconditioners.implementations.pod import compute_pod_basis

        with pytest.raises(ValueError, match="rank"):
            compute_pod_basis(poisson_snapshots, rank=poisson_snapshots.shape[0] + 1)

    def test_dtype_preserved_float32(self, poisson_snapshots: NDArray) -> None:
        """float32 input must produce a float32 basis (no silent upcast)."""
        from neuralls.domain.solver.preconditioners.implementations.pod import compute_pod_basis

        basis = compute_pod_basis(poisson_snapshots.astype(np.float32), rank=10)
        assert basis.dtype == np.float32

    def test_dtype_preserved_float64(self, poisson_snapshots: NDArray) -> None:
        """float64 input must produce a float64 basis (no silent downcast)."""
        from neuralls.domain.solver.preconditioners.implementations.pod import compute_pod_basis

        basis = compute_pod_basis(poisson_snapshots.astype(np.float64), rank=10)
        assert basis.dtype == np.float64

    def test_explicit_dtype_override(self, poisson_snapshots: NDArray) -> None:
        """An explicit dtype argument must override the input array's dtype."""
        from neuralls.domain.solver.preconditioners.implementations.pod import compute_pod_basis

        basis = compute_pod_basis(poisson_snapshots.astype(np.float64), rank=10, dtype=np.float32)
        assert basis.dtype == np.float32

    def test_energy_threshold_selects_fewer_modes_than_full_rank(
        self, poisson_snapshots: NDArray
    ) -> None:
        """A energy threshold < 1.0 must retain strictly fewer modes than the full rank."""
        from neuralls.domain.solver.preconditioners.implementations.pod import compute_pod_basis

        basis = compute_pod_basis(poisson_snapshots, rank=0.999)
        assert 0 < basis.shape[1] < poisson_snapshots.shape[0]

    def test_energy_threshold_one_retains_full_rank(self, poisson_snapshots: NDArray) -> None:
        """rank=1.0 (100% energy) must retain min(n_samples, n_dofs) modes."""
        from neuralls.domain.solver.preconditioners.implementations.pod import compute_pod_basis

        basis = compute_pod_basis(poisson_snapshots, rank=1.0)
        assert basis.shape[1] == min(poisson_snapshots.shape)

    def test_energy_threshold_out_of_range_raises(self, poisson_snapshots: NDArray) -> None:
        """An energy threshold outside (0, 1] must raise ValueError."""
        from neuralls.domain.solver.preconditioners.implementations.pod import compute_pod_basis

        with pytest.raises(ValueError, match="energy threshold"):
            compute_pod_basis(poisson_snapshots, rank=1.5)


# ---------------------------------------------------------------------------
# DenseTransferOperator
# ---------------------------------------------------------------------------


class TestDenseTransferOperator:
    def test_prolongate_shape(self, pod_basis: NDArray) -> None:
        """Prolongate maps coarse (rank) → fine (n_dofs)."""
        from neuralls.domain.solver.preconditioners.implementations.pod import (
            DenseTransferOperator,
        )

        transfer = DenseTransferOperator(pod_basis)
        fine = transfer.prolongate(np.ones(10))
        assert fine.shape == (20,)

    def test_restrict_shape(self, pod_basis: NDArray) -> None:
        """Restrict maps fine (n_dofs) → coarse (rank)."""
        from neuralls.domain.solver.preconditioners.implementations.pod import (
            DenseTransferOperator,
        )

        transfer = DenseTransferOperator(pod_basis)
        coarse = transfer.restrict(np.ones(20))
        assert coarse.shape == (10,)

    def test_restrict_prolongate_consistency(self, pod_basis: NDArray) -> None:
        """restrict(prolongate(x)) == x since Φ_rᵀΦ_r = I for orthonormal Φ_r."""
        from neuralls.domain.solver.preconditioners.implementations.pod import (
            DenseTransferOperator,
        )

        transfer = DenseTransferOperator(pod_basis)
        x = np.arange(10, dtype=np.float64)
        roundtrip = transfer.restrict(transfer.prolongate(x))
        np.testing.assert_allclose(roundtrip, x, atol=1e-8)


# ---------------------------------------------------------------------------
# PODCoarseningStrategy
# ---------------------------------------------------------------------------


class TestPODCoarseningStrategy:
    def test_returns_smaller_matrix(self, poisson_1d: NDArray, poisson_snapshots: NDArray) -> None:
        """Coarse matrix must have fewer dofs than the fine matrix."""
        from neuralls.domain.solver.preconditioners.implementations.pod import (
            PODCoarseningStrategy,
        )

        A_c, _ = PODCoarseningStrategy(poisson_snapshots, rank=10).build_transfer(poisson_1d)
        assert A_c.shape == (10, 10)

    def test_coarse_matrix_spd(self, poisson_1d: NDArray, poisson_snapshots: NDArray) -> None:
        """Galerkin coarse matrix (Φ_rᵀAΦ_r) must be SPD for SPD fine matrix."""
        from neuralls.domain.solver.preconditioners.implementations.pod import (
            PODCoarseningStrategy,
        )

        A_c, _ = PODCoarseningStrategy(poisson_snapshots, rank=10).build_transfer(poisson_1d)
        eigenvalues = np.linalg.eigvalsh(A_c)
        assert np.all(eigenvalues > 0)

    def test_transfer_shapes_consistent(
        self, poisson_1d: NDArray, poisson_snapshots: NDArray
    ) -> None:
        """Prolongate/restrict sizes must match the coarse and fine matrix shapes."""
        from neuralls.domain.solver.preconditioners.implementations.pod import (
            PODCoarseningStrategy,
        )

        A_c, transfer = PODCoarseningStrategy(poisson_snapshots, rank=10).build_transfer(poisson_1d)
        assert transfer.prolongate(np.ones(A_c.shape[0])).shape == (poisson_1d.shape[0],)
        assert transfer.restrict(np.ones(poisson_1d.shape[0])).shape == (A_c.shape[0],)


# ---------------------------------------------------------------------------
# POD2GPreconditioner
# ---------------------------------------------------------------------------


class TestPOD2GPreconditioner:
    def test_apply_returns_correct_shape(
        self, poisson_1d: NDArray, poisson_rhs: NDArray, poisson_snapshots: NDArray
    ) -> None:
        """apply() output must match residual shape."""
        from neuralls.domain.solver.preconditioners.implementations.pod import (
            POD2GPreconditioner,
        )

        precond = POD2GPreconditioner(poisson_1d, snapshots=poisson_snapshots, rank=10)
        z = precond.apply(poisson_rhs)
        assert z.shape == poisson_rhs.shape

    def test_requires_flexible_cg_false(
        self, poisson_1d: NDArray, poisson_snapshots: NDArray
    ) -> None:
        """POD-2G is a linear preconditioner and must not request FCG."""
        from neuralls.domain.solver.preconditioners.implementations.pod import (
            POD2GPreconditioner,
        )

        precond = POD2GPreconditioner(poisson_1d, snapshots=poisson_snapshots, rank=10)
        assert precond.requires_flexible_cg is False


# ---------------------------------------------------------------------------
# Integration: POD-2G inside FCG solver
# ---------------------------------------------------------------------------


class TestPOD2GWithFCGSolver:
    def test_pod2g_preconditioned_fcg_converges(
        self, poisson_1d: NDArray, poisson_rhs: NDArray, poisson_snapshots: NDArray
    ) -> None:
        """POD-2G-preconditioned FCG must converge on 1D Poisson."""
        from neuralls.domain.solver.factories import flexible_cg
        from neuralls.domain.solver.preconditioners.implementations.pod import (
            POD2GPreconditioner,
        )

        precond = POD2GPreconditioner(poisson_1d, snapshots=poisson_snapshots, rank=15)
        x, info = flexible_cg(poisson_1d, poisson_rhs, preconditioner=precond, rtol=1e-8)
        assert info.converged, f"FCG+POD-2G did not converge: {info}"
        np.testing.assert_allclose(poisson_1d @ x, poisson_rhs, rtol=1e-6)


# ---------------------------------------------------------------------------
# Config model parsing
# ---------------------------------------------------------------------------


class TestPOD2GConfigParsing:
    def test_parse_pod_coarsening_config(self) -> None:
        """AMGPreconditionerConfig must parse with coarsening.method='pod'."""
        from neuralls.platform.config.models.preconditioner import (
            AMGPreconditionerConfig,
            PODCoarseningConfig,
            parse_preconditioner_config,
        )

        cfg = parse_preconditioner_config(
            {
                "name": "pod2g",
                "type": "amg",
                "coarsening": {"method": "pod", "snapshots_glob": "/data/sol_*.txt"},
            }
        )
        assert isinstance(cfg, AMGPreconditionerConfig)
        assert isinstance(cfg.coarsening, PODCoarseningConfig)
        assert cfg.coarsening.rank == 8
        assert cfg.n_levels == 2
        assert cfg.smoother_omega == pytest.approx(0.67)

    def test_parse_pod_coarsening_config_custom_rank(self) -> None:
        """rank override must be respected."""
        from neuralls.platform.config.models.preconditioner import (
            AMGPreconditionerConfig,
            PODCoarseningConfig,
            parse_preconditioner_config,
        )

        cfg = parse_preconditioner_config(
            {
                "name": "pod2g",
                "type": "amg",
                "coarsening": {
                    "method": "pod",
                    "snapshots_glob": "/data/sol_*.txt",
                    "rank": 12,
                },
            }
        )
        assert isinstance(cfg, AMGPreconditionerConfig)
        assert isinstance(cfg.coarsening, PODCoarseningConfig)
        assert cfg.coarsening.rank == 12

    def test_parse_pod_coarsening_config_energy_threshold_rank(self) -> None:
        """A float rank must parse as an energy-capture threshold."""
        from neuralls.platform.config.models.preconditioner import (
            AMGPreconditionerConfig,
            PODCoarseningConfig,
            parse_preconditioner_config,
        )

        cfg = parse_preconditioner_config(
            {
                "name": "pod2g",
                "type": "amg",
                "coarsening": {
                    "method": "pod",
                    "snapshots_glob": "/data/sol_*.txt",
                    "rank": 0.9999,
                },
            }
        )
        assert isinstance(cfg, AMGPreconditionerConfig)
        assert isinstance(cfg.coarsening, PODCoarseningConfig)
        assert cfg.coarsening.rank == pytest.approx(0.9999)

    def test_pod_coarsening_energy_threshold_out_of_range_raises(self) -> None:
        """A float rank outside (0, 1] must fail validation."""
        from pydantic import ValidationError

        from neuralls.platform.config.models.preconditioner import parse_preconditioner_config

        with pytest.raises(ValidationError, match="energy threshold"):
            parse_preconditioner_config(
                {
                    "name": "pod2g",
                    "type": "amg",
                    "coarsening": {
                        "method": "pod",
                        "snapshots_glob": "/data/sol_*.txt",
                        "rank": 1.5,
                    },
                }
            )

    def test_pod_coarsening_without_snapshots_glob_raises(self) -> None:
        """coarsening.method='pod' without snapshots_glob must fail validation
        (the field is required on PODCoarseningConfig — invalid state is
        unrepresentable rather than caught by a cross-field validator)."""
        from pydantic import ValidationError

        from neuralls.platform.config.models.preconditioner import parse_preconditioner_config

        with pytest.raises(ValidationError, match="snapshots_glob"):
            parse_preconditioner_config(
                {"name": "pod2g", "type": "amg", "coarsening": {"method": "pod"}}
            )

    def test_parse_aggregation_coarsening_config(self) -> None:
        """coarsening.method='aggregation' must parse to AggregationCoarseningConfig.

        There is no implicit default (see TestAMGConfigParsing.test_coarsening_is_required
        in test_amg.py) — every AMG config must state its coarsening strategy explicitly.
        """
        from neuralls.platform.config.models.preconditioner import (
            AggregationCoarseningConfig,
            AMGPreconditionerConfig,
            parse_preconditioner_config,
        )

        cfg = parse_preconditioner_config(
            {"name": "amg", "type": "amg", "coarsening": {"method": "aggregation"}}
        )
        assert isinstance(cfg, AMGPreconditionerConfig)
        assert isinstance(cfg.coarsening, AggregationCoarseningConfig)


# ---------------------------------------------------------------------------
# Factory wiring (end-to-end: config -> loaded snapshots -> preconditioner)
# ---------------------------------------------------------------------------


class TestPOD2GFactory:
    def test_create_preconditioner_loads_snapshots_and_builds_pod_amg(
        self, tmp_path, poisson_1d: NDArray, poisson_rhs: NDArray, poisson_snapshots: NDArray
    ) -> None:
        """create_preconditioner must load snapshot files and build an AMGPreconditioner
        wired with PODCoarseningStrategy when coarsening='pod'."""
        from neuralls.composition.preconditioners.factory import create_preconditioner
        from neuralls.domain.solver.preconditioners.implementations.amg import AMGPreconditioner
        from neuralls.platform.config.models.preconditioner import (
            parse_preconditioner_config,
        )

        for i, snapshot in enumerate(poisson_snapshots):
            np.savetxt(tmp_path / f"sol_{i:02d}.txt", snapshot)

        cfg = parse_preconditioner_config(
            {
                "name": "pod2g",
                "type": "amg",
                "coarsening": {
                    "method": "pod",
                    "snapshots_glob": str(tmp_path / "sol_*.txt"),
                    "rank": 10,
                },
            }
        )
        precond = create_preconditioner(poisson_1d, cfg)
        assert isinstance(precond, AMGPreconditioner)
        z = precond.apply(poisson_rhs)
        assert z.shape == poisson_rhs.shape
