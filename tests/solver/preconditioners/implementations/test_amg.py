"""Tests for AMG preconditioner components."""

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
def small_spd() -> NDArray:
    """Small 6x6 SPD matrix for unit tests.

    Returns:
        6x6 diagonal matrix (well-conditioned, trivial aggregation).
    """
    return np.diag([4.0, 3.0, 4.0, 3.0, 4.0, 3.0])


@pytest.fixture
def amg_preconditioner(poisson_1d: NDArray):
    """Classical AMGPreconditioner on the 1D Poisson matrix.

    Args:
        poisson_1d: 20x20 Poisson matrix fixture.

    Returns:
        AMGPreconditioner instance with VCycle + JacobiSmoother.
    """
    from neuralls.domain.solver.preconditioners.implementations.amg import (
        AMGPreconditioner,
        JacobiSmoother,
        SparseAggregationCoarsening,
        VCycle,
    )

    return AMGPreconditioner(
        poisson_1d,
        coarsening=SparseAggregationCoarsening(),
        cycle=VCycle(smoother=JacobiSmoother()),
        n_levels=2,
    )


@pytest.fixture
def context():
    """Dummy PreconditionerContext at iteration 0.

    Returns:
        PreconditionerContext with iteration=0.
    """
    from neuralls.domain.solver.preconditioners.base import PreconditionerContext

    return PreconditionerContext(iteration=0, residual_norm=1.0, rhs_norm=1.0)


# ---------------------------------------------------------------------------
# JacobiSmoother
# ---------------------------------------------------------------------------


class TestJacobiSmoother:
    def test_reduces_residual(self, small_spd: NDArray) -> None:
        """Smoothing should reduce the residual norm."""
        from neuralls.domain.solver.preconditioners.implementations.amg import JacobiSmoother

        smoother = JacobiSmoother(omega=0.67)
        rhs = np.ones(6)
        x = np.zeros(6)
        x_smooth = smoother.smooth(small_spd, rhs, x, steps=5)
        r_before = np.linalg.norm(rhs - small_spd @ x)
        r_after = np.linalg.norm(rhs - small_spd @ x_smooth)
        assert r_after < r_before

    def test_output_shape(self, small_spd: NDArray) -> None:
        """Output shape must match input shape."""
        from neuralls.domain.solver.preconditioners.implementations.amg import JacobiSmoother

        smoother = JacobiSmoother()
        rhs = np.ones(6)
        x = smoother.smooth(small_spd, rhs, np.zeros(6), steps=1)
        assert x.shape == (6,)

    def test_zero_steps_returns_copy(self, small_spd: NDArray) -> None:
        """Zero smoothing steps should return the initial iterate unchanged."""
        from neuralls.domain.solver.preconditioners.implementations.amg import JacobiSmoother

        smoother = JacobiSmoother()
        x0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        rhs = np.ones(6)
        x = smoother.smooth(small_spd, rhs, x0, steps=0)
        np.testing.assert_array_equal(x, x0)


# ---------------------------------------------------------------------------
# SparseTransferOperator
# ---------------------------------------------------------------------------


class TestSparseTransferOperator:
    @pytest.fixture
    def transfer(self):
        """2:1 aggregation transfer operator (8 fine → 4 coarse).

        Returns:
            SparseTransferOperator with piecewise-constant P.
        """
        from scipy.sparse import csr_matrix

        from neuralls.domain.solver.preconditioners.implementations.amg import (
            SparseTransferOperator,
        )

        # P: 8 fine nodes grouped in pairs → 4 coarse nodes
        rows = np.arange(8)
        cols = rows // 2
        P = csr_matrix((np.ones(8), (rows, cols)), shape=(8, 4))
        return SparseTransferOperator(P)

    def test_prolongate_shape(self, transfer) -> None:
        """Prolongate maps coarse (4) → fine (8)."""
        coarse = np.ones(4)
        fine = transfer.prolongate(coarse)
        assert fine.shape == (8,)

    def test_restrict_shape(self, transfer) -> None:
        """Restrict maps fine (8) → coarse (4)."""
        fine = np.ones(8)
        coarse = transfer.restrict(fine)
        assert coarse.shape == (4,)

    def test_restrict_prolongate_consistency(self, transfer) -> None:
        """Restriction should be Galerkin: R = P^T, so restrict(ones) sums pairs."""
        fine = np.ones(8)
        coarse = transfer.restrict(fine)
        # Each coarse dof aggregates 2 fine dofs → sum = 2
        np.testing.assert_allclose(coarse, np.full(4, 2.0))


# ---------------------------------------------------------------------------
# SparseAggregationCoarsening
# ---------------------------------------------------------------------------


class TestSparseAggregationCoarsening:
    def test_returns_smaller_matrix(self, poisson_1d: NDArray) -> None:
        """Coarse matrix must have fewer dofs than fine matrix."""
        from neuralls.domain.solver.preconditioners.implementations.amg import (
            SparseAggregationCoarsening,
        )

        A_c, _ = SparseAggregationCoarsening().build_transfer(poisson_1d)
        assert A_c.shape[0] < poisson_1d.shape[0]

    def test_coarse_matrix_spd(self, poisson_1d: NDArray) -> None:
        """Galerkin coarse matrix (P^T A P) must be SPD for SPD fine matrix."""
        from neuralls.domain.solver.preconditioners.implementations.amg import (
            SparseAggregationCoarsening,
        )

        A_c, _ = SparseAggregationCoarsening().build_transfer(poisson_1d)
        eigenvalues = np.linalg.eigvalsh(A_c)
        assert np.all(eigenvalues > 0)

    def test_transfer_shapes_consistent(self, poisson_1d: NDArray) -> None:
        """Prolongate output (fine) and restrict output (coarse) sizes must match."""
        from neuralls.domain.solver.preconditioners.implementations.amg import (
            SparseAggregationCoarsening,
        )

        A_c, transfer = SparseAggregationCoarsening().build_transfer(poisson_1d)
        n_fine = poisson_1d.shape[0]
        n_coarse = A_c.shape[0]
        assert transfer.prolongate(np.ones(n_coarse)).shape == (n_fine,)
        assert transfer.restrict(np.ones(n_fine)).shape == (n_coarse,)


# ---------------------------------------------------------------------------
# VCycle
# ---------------------------------------------------------------------------


class TestVCycle:
    def test_output_shape(self, poisson_1d: NDArray, poisson_rhs: NDArray) -> None:
        """V-cycle output must match rhs shape."""
        from neuralls.domain.solver.preconditioners.implementations.amg import (
            AMGPreconditioner,
            JacobiSmoother,
            SparseAggregationCoarsening,
            VCycle,
        )

        precond = AMGPreconditioner(
            poisson_1d,
            coarsening=SparseAggregationCoarsening(),
            cycle=VCycle(smoother=JacobiSmoother()),
            n_levels=2,
        )
        z = precond.apply(poisson_rhs)
        assert z.shape == poisson_rhs.shape

    def test_output_dtype_float64(self, poisson_1d: NDArray, poisson_rhs: NDArray) -> None:
        """V-cycle output must be float64."""
        from neuralls.domain.solver.preconditioners.implementations.amg import (
            AMGPreconditioner,
            JacobiSmoother,
            SparseAggregationCoarsening,
            VCycle,
        )

        precond = AMGPreconditioner(
            poisson_1d,
            coarsening=SparseAggregationCoarsening(),
            cycle=VCycle(smoother=JacobiSmoother()),
        )
        z = precond.apply(poisson_rhs)
        assert z.dtype == np.float64


# ---------------------------------------------------------------------------
# AMGPreconditioner
# ---------------------------------------------------------------------------


class TestAMGPreconditioner:
    def test_apply_returns_correct_shape(self, amg_preconditioner, poisson_rhs: NDArray) -> None:
        """apply() output must match residual shape."""
        z = amg_preconditioner.apply(poisson_rhs)
        assert z.shape == poisson_rhs.shape

    def test_apply_accepts_context(self, amg_preconditioner, poisson_rhs: NDArray, context) -> None:
        """apply() must accept PreconditionerContext without error."""
        z = amg_preconditioner.apply(poisson_rhs, context)
        assert z.shape == poisson_rhs.shape

    def test_lazy_hierarchy_build(self, poisson_1d: NDArray) -> None:
        """Hierarchy must be None before first apply, populated after."""
        from neuralls.domain.solver.preconditioners.implementations.amg import (
            AMGPreconditioner,
            JacobiSmoother,
            SparseAggregationCoarsening,
            VCycle,
        )

        precond = AMGPreconditioner(
            poisson_1d,
            coarsening=SparseAggregationCoarsening(),
            cycle=VCycle(smoother=JacobiSmoother()),
        )
        assert precond._hierarchy is None
        precond.apply(np.ones(poisson_1d.shape[0]))
        assert precond._hierarchy is not None

    def test_linear_requires_flexible_cg_false(self, poisson_1d: NDArray) -> None:
        """Classical AMG (linear=True) must not request FCG."""
        from neuralls.domain.solver.preconditioners.implementations.amg import (
            AMGPreconditioner,
            JacobiSmoother,
            SparseAggregationCoarsening,
            VCycle,
        )

        precond = AMGPreconditioner(
            poisson_1d,
            coarsening=SparseAggregationCoarsening(),
            cycle=VCycle(smoother=JacobiSmoother()),
            linear=True,
        )
        assert precond.requires_flexible_cg is False

    def test_nonlinear_requires_flexible_cg_true(self, poisson_1d: NDArray) -> None:
        """AMG with linear=False must request FCG (neural AMG case)."""
        from neuralls.domain.solver.preconditioners.implementations.amg import (
            AMGPreconditioner,
            JacobiSmoother,
            SparseAggregationCoarsening,
            VCycle,
        )

        precond = AMGPreconditioner(
            poisson_1d,
            coarsening=SparseAggregationCoarsening(),
            cycle=VCycle(smoother=JacobiSmoother()),
            linear=False,
        )
        assert precond.requires_flexible_cg is True

    def test_extra_input_names_empty_for_classical(self, amg_preconditioner) -> None:
        """Classical AMG needs no extra inputs."""
        assert amg_preconditioner.extra_input_names == ()

    def test_bind_inputs_noop_for_classical(self, amg_preconditioner, poisson_rhs: NDArray) -> None:
        """bind_inputs on classical AMG must not raise and must still work."""
        amg_preconditioner.bind_inputs(positions=np.zeros(20))
        z = amg_preconditioner.apply(poisson_rhs)
        assert z.shape == poisson_rhs.shape

    def test_bind_inputs_invalidates_hierarchy(self, poisson_1d: NDArray) -> None:
        """bind_inputs must invalidate the cached hierarchy."""
        from neuralls.domain.solver.preconditioners.implementations.amg import (
            AMGPreconditioner,
            JacobiSmoother,
            SparseAggregationCoarsening,
            VCycle,
        )

        precond = AMGPreconditioner(
            poisson_1d,
            coarsening=SparseAggregationCoarsening(),
            cycle=VCycle(smoother=JacobiSmoother()),
        )
        precond.apply(np.ones(poisson_1d.shape[0]))
        assert precond._hierarchy is not None
        precond.bind_inputs()
        assert precond._hierarchy is None

    def test_hierarchy_has_correct_n_levels(self, poisson_1d: NDArray) -> None:
        """Built hierarchy must have exactly n_levels MultigridLevel entries."""
        from neuralls.domain.solver.preconditioners.implementations.amg import (
            AMGPreconditioner,
            JacobiSmoother,
            SparseAggregationCoarsening,
            VCycle,
        )

        precond = AMGPreconditioner(
            poisson_1d,
            coarsening=SparseAggregationCoarsening(),
            cycle=VCycle(smoother=JacobiSmoother()),
            n_levels=2,
        )
        precond.apply(np.ones(poisson_1d.shape[0]))
        assert precond._hierarchy is not None
        assert len(precond._hierarchy.levels) == 2

    def test_coarsest_level_has_no_transfer(self, poisson_1d: NDArray) -> None:
        """The coarsest level must have transfer=None."""
        from neuralls.domain.solver.preconditioners.implementations.amg import (
            AMGPreconditioner,
            JacobiSmoother,
            SparseAggregationCoarsening,
            VCycle,
        )

        precond = AMGPreconditioner(
            poisson_1d,
            coarsening=SparseAggregationCoarsening(),
            cycle=VCycle(smoother=JacobiSmoother()),
        )
        precond.apply(np.ones(poisson_1d.shape[0]))
        assert precond._hierarchy is not None
        assert precond._hierarchy.levels[-1].transfer is None


# ---------------------------------------------------------------------------
# Integration: AMG inside FCG solver
# ---------------------------------------------------------------------------


class TestAMGWithFCGSolver:
    def test_amg_preconditioned_fcg_converges(
        self, poisson_1d: NDArray, poisson_rhs: NDArray
    ) -> None:
        """AMG-preconditioned FCG must converge on 1D Poisson."""
        from neuralls.domain.solver.factories import flexible_cg
        from neuralls.domain.solver.preconditioners.implementations.amg import (
            AMGPreconditioner,
            JacobiSmoother,
            SparseAggregationCoarsening,
            VCycle,
        )

        precond = AMGPreconditioner(
            poisson_1d,
            coarsening=SparseAggregationCoarsening(),
            cycle=VCycle(smoother=JacobiSmoother()),
        )
        x, info = flexible_cg(poisson_1d, poisson_rhs, preconditioner=precond, rtol=1e-8)
        assert info.converged, f"FCG+AMG did not converge: {info}"
        np.testing.assert_allclose(poisson_1d @ x, poisson_rhs, rtol=1e-6)

    def test_amg_fewer_iters_than_unpreconditioned(
        self, poisson_1d: NDArray, poisson_rhs: NDArray
    ) -> None:
        """AMG preconditioning must reduce iteration count vs. no preconditioning."""
        from neuralls.domain.solver.factories import flexible_cg
        from neuralls.domain.solver.preconditioners.implementations.amg import (
            AMGPreconditioner,
            JacobiSmoother,
            SparseAggregationCoarsening,
            VCycle,
        )

        _, info_unprecond = flexible_cg(poisson_1d, poisson_rhs, rtol=1e-8)
        precond = AMGPreconditioner(
            poisson_1d,
            coarsening=SparseAggregationCoarsening(),
            cycle=VCycle(smoother=JacobiSmoother()),
        )
        _, info_amg = flexible_cg(poisson_1d, poisson_rhs, preconditioner=precond, rtol=1e-8)
        assert info_amg.iterations < info_unprecond.iterations


# ---------------------------------------------------------------------------
# Interface: context always passed (no ContextualPreconditioner split)
# ---------------------------------------------------------------------------


class TestUniformContextInterface:
    def test_all_preconditioners_accept_context(self, poisson_1d: NDArray) -> None:
        """Every preconditioner must accept a PreconditionerContext without error."""
        from neuralls.domain.solver.preconditioners import (
            Identity,
            JacobiPreconditioner,
            PreconditionerContext,
        )
        from neuralls.domain.solver.preconditioners.implementations.amg import (
            AMGPreconditioner,
            JacobiSmoother,
            SparseAggregationCoarsening,
            VCycle,
        )

        ctx = PreconditionerContext(iteration=5, residual_norm=0.1, rhs_norm=1.0)
        r = np.ones(poisson_1d.shape[0])
        for precond in [
            Identity(),
            JacobiPreconditioner(poisson_1d),
            AMGPreconditioner(
                poisson_1d,
                coarsening=SparseAggregationCoarsening(),
                cycle=VCycle(smoother=JacobiSmoother()),
            ),
        ]:
            z = precond.apply(r, ctx)
            assert z.shape == r.shape


# ---------------------------------------------------------------------------
# Config model parsing
# ---------------------------------------------------------------------------


class TestAMGConfigParsing:
    def test_parse_amg_config(self) -> None:
        """AMGPreconditionerConfig must parse from dict with type='amg'."""
        from neuralls.platform.config.models.preconditioner import (
            AMGPreconditionerConfig,
            parse_preconditioner_config,
        )

        cfg = parse_preconditioner_config({"name": "amg", "type": "amg"})
        assert isinstance(cfg, AMGPreconditionerConfig)
        assert cfg.n_levels == 2
        assert cfg.smoother_omega == pytest.approx(0.67)

    def test_parse_amg_config_custom_levels(self) -> None:
        """n_levels override must be respected."""
        from neuralls.platform.config.models.preconditioner import (
            AMGPreconditionerConfig,
            parse_preconditioner_config,
        )

        cfg = parse_preconditioner_config({"name": "amg", "type": "amg", "n_levels": 3})
        assert isinstance(cfg, AMGPreconditionerConfig)
        assert cfg.n_levels == 3


# ---------------------------------------------------------------------------
# NeuralPreconditioner extra_input_names fallback
# ---------------------------------------------------------------------------


class TestNeuralPreconditionerFallback:
    def test_extra_input_names_uses_config_when_set(self) -> None:
        """When extra_input_names is non-empty in config, it overrides predictor.required_inputs."""
        from unittest.mock import MagicMock

        from neuralls.domain.solver.preconditioners.implementations.neural import (
            NeuralPreconditioner,
        )
        from neuralls.domain.solver.preconditioners.ports import ExtraInputPredictorPort

        mock_adapter = MagicMock()
        mock_predictor = MagicMock(spec=ExtraInputPredictorPort)
        mock_predictor.required_inputs = ("matrix",)
        mock_adapter.create_predictor.return_value = mock_predictor

        precond = NeuralPreconditioner(
            checkpoint_path=MagicMock(),
            adapter=mock_adapter,
            extra_input_names=("positions",),
        )
        assert precond.extra_input_names == ("positions",)

    def test_extra_input_names_falls_back_to_predictor(self) -> None:
        """When extra_input_names is empty in config, fall back to predictor.required_inputs."""
        from unittest.mock import MagicMock

        from neuralls.domain.solver.preconditioners.implementations.neural import (
            NeuralPreconditioner,
        )
        from neuralls.domain.solver.preconditioners.ports import ExtraInputPredictorPort

        mock_adapter = MagicMock()
        mock_predictor = MagicMock(spec=ExtraInputPredictorPort)
        mock_predictor.required_inputs = ("matrix", "positions")
        mock_adapter.create_predictor.return_value = mock_predictor

        precond = NeuralPreconditioner(
            checkpoint_path=MagicMock(),
            adapter=mock_adapter,
            extra_input_names=(),
        )
        assert precond.extra_input_names == ("matrix", "positions")


# ---------------------------------------------------------------------------
# SmootherBase hierarchy
# ---------------------------------------------------------------------------


class TestSmootherBase:
    def test_jacobi_smoother_is_smoother_base(self) -> None:
        """JacobiSmoother must be a SmootherBase instance (nominative hierarchy)."""
        from neuralls.domain.solver.preconditioners.implementations.amg import (
            JacobiSmoother,
            SmootherBase,
        )

        assert isinstance(JacobiSmoother(), SmootherBase)

    def test_jacobi_smoother_satisfies_multigrid_smoother_protocol(
        self, small_spd: NDArray
    ) -> None:
        """JacobiSmoother must satisfy the MultigridSmoother protocol (structural typing)."""
        from neuralls.domain.solver.preconditioners.implementations.amg import JacobiSmoother

        smoother = JacobiSmoother()
        rhs = np.ones(6)
        x0 = np.zeros(6)
        result = smoother.smooth(small_spd, rhs, x0, steps=1)
        assert result.shape == x0.shape


# ---------------------------------------------------------------------------
# WCycle
# ---------------------------------------------------------------------------


class TestWCycle:
    def test_output_shape(self, poisson_1d: NDArray, poisson_rhs: NDArray) -> None:
        """W-cycle output must match rhs shape."""
        from neuralls.domain.solver.preconditioners.implementations.amg import (
            AMGPreconditioner,
            JacobiSmoother,
            SparseAggregationCoarsening,
            WCycle,
        )

        precond = AMGPreconditioner(
            poisson_1d,
            coarsening=SparseAggregationCoarsening(),
            cycle=WCycle(smoother=JacobiSmoother()),
            n_levels=2,
        )
        z = precond.apply(poisson_rhs)
        assert z.shape == poisson_rhs.shape

    def test_output_dtype_float64(self, poisson_1d: NDArray, poisson_rhs: NDArray) -> None:
        """W-cycle output must be float64."""
        from neuralls.domain.solver.preconditioners.implementations.amg import (
            AMGPreconditioner,
            JacobiSmoother,
            SparseAggregationCoarsening,
            WCycle,
        )

        precond = AMGPreconditioner(
            poisson_1d,
            coarsening=SparseAggregationCoarsening(),
            cycle=WCycle(smoother=JacobiSmoother()),
            n_levels=2,
        )
        z = precond.apply(poisson_rhs)
        assert z.dtype == np.float64

    def test_wcycle_residual_not_larger_than_vcycle(
        self, poisson_1d: NDArray, poisson_rhs: NDArray
    ) -> None:
        """W-cycle must not produce a larger residual than V-cycle on the same rhs.

        Theory: W-cycle does strictly more coarse-grid work (γ=2 vs γ=1), so its
        residual after one cycle should be ≤ that of V-cycle for SPD problems
        (Briggs et al. 2000, §3.3).
        """
        from neuralls.domain.solver.preconditioners.implementations.amg import (
            AMGPreconditioner,
            JacobiSmoother,
            SparseAggregationCoarsening,
            VCycle,
            WCycle,
        )

        smoother = JacobiSmoother()
        coarsening = SparseAggregationCoarsening()

        v_precond = AMGPreconditioner(
            poisson_1d, coarsening=coarsening, cycle=VCycle(smoother), n_levels=2
        )
        w_precond = AMGPreconditioner(
            poisson_1d, coarsening=coarsening, cycle=WCycle(smoother), n_levels=2
        )

        e_v = v_precond.apply(poisson_rhs)
        e_w = w_precond.apply(poisson_rhs)

        res_v = np.linalg.norm(poisson_rhs - poisson_1d @ e_v)
        res_w = np.linalg.norm(poisson_rhs - poisson_1d @ e_w)
        assert res_w <= res_v + 1e-10  # W-cycle never worse than V-cycle


# ---------------------------------------------------------------------------
# Preset classes: VCycleAMG and WCycleAMG
# ---------------------------------------------------------------------------


@pytest.fixture(
    params=[
        pytest.param("VCycleAMG", id="VCycleAMG"),
        pytest.param("WCycleAMG", id="WCycleAMG"),
    ]
)
def amg_preset_class(request: pytest.FixtureRequest):
    """Parametrize tests over both preset AMG variant classes.

    Args:
        request: Pytest fixture request carrying the class name.

    Returns:
        The preset AMG class (VCycleAMG or WCycleAMG).
    """
    import neuralls.domain.solver.preconditioners.implementations.amg as amg_pkg

    return getattr(amg_pkg, request.param)


class TestAMGPresets:
    def test_apply_returns_correct_shape(
        self, amg_preset_class, poisson_1d: NDArray, poisson_rhs: NDArray
    ) -> None:
        """apply() output must match residual shape for both preset classes."""
        precond = amg_preset_class(poisson_1d, n_levels=2)
        z = precond.apply(poisson_rhs)
        assert z.shape == poisson_rhs.shape

    def test_apply_returns_float64(
        self, amg_preset_class, poisson_1d: NDArray, poisson_rhs: NDArray
    ) -> None:
        """apply() must return float64 for both preset classes."""
        precond = amg_preset_class(poisson_1d, n_levels=2)
        assert precond.apply(poisson_rhs).dtype == np.float64

    def test_fcg_converges(
        self, amg_preset_class, poisson_1d: NDArray, poisson_rhs: NDArray
    ) -> None:
        """FCG preconditioned with either preset must converge on 1D Poisson."""
        from neuralls.domain.solver.factories import flexible_cg

        precond = amg_preset_class(poisson_1d, n_levels=2)
        x, info = flexible_cg(poisson_1d, poisson_rhs, preconditioner=precond, rtol=1e-8)
        assert info.converged, f"{amg_preset_class.__name__}+FCG did not converge: {info}"
        np.testing.assert_allclose(poisson_1d @ x, poisson_rhs, rtol=1e-6)

    def test_spd_preservation(self, amg_preset_class, poisson_1d: NDArray) -> None:
        """Preset AMG preconditioner must be SPD when applied to an SPD matrix.

        Theory: V/W-cycle with symmetric Jacobi smoother (equal pre/post steps)
        and Galerkin coarsening produces an SPD preconditioner M when A is SPD
        (Vanek, Mandel & Brezina 1996, Theorem 4.1).

        Verification: assemble M by applying the preconditioner to each canonical
        basis vector, then check symmetry and positive-definiteness via eigenvalues.
        """
        n = poisson_1d.shape[0]
        precond = amg_preset_class(poisson_1d, n_levels=2)
        M = np.column_stack([precond.apply(np.eye(n)[:, i]) for i in range(n)])

        # Symmetry
        np.testing.assert_allclose(
            M, M.T, atol=1e-12, err_msg="Preconditioner matrix is not symmetric"
        )

        # Positive definiteness
        eigenvalues = np.linalg.eigvalsh(M)
        assert np.all(eigenvalues > 0), (
            f"Non-positive eigenvalue(s): {eigenvalues[eigenvalues <= 0]}"
        )

    def test_requires_flexible_cg_false(self, amg_preset_class, poisson_1d: NDArray) -> None:
        """Both presets are linear preconditioners and must not request FCG."""
        precond = amg_preset_class(poisson_1d, n_levels=2)
        assert precond.requires_flexible_cg is False

    def test_wcycle_iters_not_worse_than_vcycle(
        self, poisson_1d: NDArray, poisson_rhs: NDArray
    ) -> None:
        """WCycleAMG must need ≤ iterations than VCycleAMG on SPD Poisson.

        Theory: W-cycle's extra coarse-grid correction never increases the
        iteration count for SPD problems (Briggs et al. 2000, §3.3).
        """
        from neuralls.domain.solver.factories import flexible_cg
        from neuralls.domain.solver.preconditioners.implementations.amg import (
            VCycleAMG,
            WCycleAMG,
        )

        _, info_v = flexible_cg(
            poisson_1d,
            poisson_rhs,
            preconditioner=VCycleAMG(poisson_1d, n_levels=2),
            rtol=1e-8,
        )
        _, info_w = flexible_cg(
            poisson_1d,
            poisson_rhs,
            preconditioner=WCycleAMG(poisson_1d, n_levels=2),
            rtol=1e-8,
        )
        assert info_w.iterations <= info_v.iterations
