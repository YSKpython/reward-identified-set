"""Tests for the Fisher LLF (Neural Manipulation Cost) module."""

import numpy as np
import pytest

from src.fisher.builder import build_fisher_matrix
from src.fisher.llf import (
    CGResult,
    ManipulationCostResult,
    compute_manipulation_cost,
    kl_budget_from_mc,
    llf_conservative_bound_check,
    solve_fisher_system,
)


class TestCGSolverCorrectness:
    """Tests for Conjugate Gradient solver correctness."""

    def test_cg_identity_system(self) -> None:
        """Test that for F = I and known g, s = g exactly (within 1e-10)."""
        rng = np.random.RandomState(42)
        n = 10
        F = np.eye(n)
        g = rng.randn(n)

        result = solve_fisher_system(F, g, damping=0.0, tol=1e-12, max_iter=n)

        assert np.allclose(result.solution, g, rtol=1e-10, atol=1e-10)

    def test_cg_diagonal_system(self) -> None:
        """Test that for F = diag([1, 2, 4, 8]) and known g, s = F^{-1} g."""
        rng = np.random.RandomState(42)
        diag_vals = np.array([1.0, 2.0, 4.0, 8.0])
        F = np.diag(diag_vals)
        g = rng.randn(len(diag_vals))

        result = solve_fisher_system(F, g, damping=0.0, tol=1e-12, max_iter=len(diag_vals))

        expected = g / diag_vals
        assert np.allclose(result.solution, expected, rtol=1e-10, atol=1e-10)

    def test_cg_convergence_flag(self) -> None:
        """Verify converged == True for well-conditioned systems."""
        rng = np.random.RandomState(42)
        n = 5
        F = np.eye(n) * 2.0  # Well-conditioned
        g = rng.randn(n)

        result = solve_fisher_system(F, g, damping=1e-6, tol=1e-8, max_iter=100)

        assert result.converged

    def test_cg_iteration_count(self) -> None:
        """For an n-dimensional diagonal system, CG should converge in at most n iterations."""
        rng = np.random.RandomState(42)
        n = 10
        diag_vals = np.arange(1, n + 1, dtype=float)
        F = np.diag(diag_vals)
        g = rng.randn(n)

        result = solve_fisher_system(F, g, damping=0.0, tol=1e-12, max_iter=100)

        assert result.n_iterations <= n

    def test_cg_damping_effect(self) -> None:
        """Verify that damping > 0 shifts the solution: (F + εI)^{-1} g differs from F^{-1} g."""
        rng = np.random.RandomState(42)
        n = 5
        F = np.eye(n) * 2.0
        g = rng.randn(n)

        result_no_damping = solve_fisher_system(F, g, damping=0.0, tol=1e-12, max_iter=n)
        result_with_damping = solve_fisher_system(F, g, damping=0.1, tol=1e-12, max_iter=n)

        # The solutions should be different
        assert not np.allclose(
            result_no_damping.solution, result_with_damping.solution, rtol=1e-6
        )

    def test_cg_non_square_raises(self) -> None:
        """Test that non-square F raises ValueError."""
        rng = np.random.RandomState(42)
        F = rng.randn(3, 4)
        g = rng.randn(3)

        with pytest.raises(ValueError, match="square"):
            solve_fisher_system(F, g)

    def test_cg_non_symmetric_raises(self) -> None:
        """Test that non-symmetric F raises ValueError."""
        rng = np.random.RandomState(42)
        F = rng.randn(4, 4)
        # Make it non-symmetric
        F[0, 1] = 1.0
        F[1, 0] = 2.0
        g = rng.randn(4)

        with pytest.raises(ValueError, match="symmetric"):
            solve_fisher_system(F, g)

    def test_cg_negative_damping_raises(self) -> None:
        """Test that damping < 0 raises ValueError."""
        rng = np.random.RandomState(42)
        n = 3
        F = np.eye(n)
        g = rng.randn(n)

        with pytest.raises(ValueError, match="damping"):
            solve_fisher_system(F, g, damping=-0.1)


class TestManipulationCostCorrectness:
    """Tests for manipulation cost computation correctness."""

    def test_mc_known_vulnerability_score(self) -> None:
        """Construct a system where V = g^T F^{-1} g is known analytically.

        For F = I, we have s = g, so V = g^T g = ||g||^2.
        Then MC = margin / sqrt(V) = margin / ||g||.
        """
        rng = np.random.RandomState(42)
        n = 5
        F = np.eye(n)
        g = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        margin = 0.1

        # V = g^T g = 1 + 4 + 9 + 16 + 25 = 55
        expected_V = np.sum(g**2)
        expected_mc = margin / np.sqrt(expected_V)

        result = compute_manipulation_cost(F, g, margin, damping=0.0, tol=1e-12)

        assert np.isclose(result.vulnerability_score, expected_V, rtol=1e-10)
        assert np.isclose(result.mc_value, expected_mc, rtol=1e-10)

    def test_mc_e17_reproduction(self) -> None:
        """Reproduce E17 finding: MC_LLF = 0.0129 for V = 59.68, m = 0.1.

        We construct a synthetic system where V = g^T F^{-1} g = 59.68.
        For simplicity, use F = I, then V = ||g||^2 = 59.68, so ||g|| = sqrt(59.68).
        Then MC = 0.1 / sqrt(59.68) ≈ 0.0129.
        """
        rng = np.random.RandomState(42)
        n = 10
        F = np.eye(n)

        # We want V = g^T F^{-1} g = g^T g = 59.68
        # So ||g|| = sqrt(59.68)
        target_V = 59.68
        g_norm = np.sqrt(target_V)

        # Create a unit vector and scale it
        g = rng.randn(n)
        g = g / np.linalg.norm(g) * g_norm

        margin = 0.1

        result = compute_manipulation_cost(F, g, margin, damping=0.0, tol=1e-12)

        # Expected MC = 0.1 / sqrt(59.68) ≈ 0.0129
        expected_mc = margin / np.sqrt(target_V)

        assert np.isclose(result.mc_value, expected_mc, rtol=1e-4)
        # Verify the specific value from the paper (with absolute tolerance)
        assert np.isclose(result.mc_value, 0.0129, atol=1e-4)

    def test_kl_budget_formula(self) -> None:
        """Verify kl_budget_from_mc(0.0129) ≈ 8.32e-5 (within 1e-6)."""
        mc_value = 0.0129
        expected_kl = (mc_value ** 2) / 2.0  # ≈ 8.3205e-5

        result = kl_budget_from_mc(mc_value)

        assert np.isclose(result, expected_kl, rtol=1e-6)
        assert np.isclose(result, 8.32e-5, atol=1e-6)

    def test_mc_zero_margin_raises(self) -> None:
        """Test that margin = 0 raises ValueError."""
        rng = np.random.RandomState(42)
        n = 3
        F = np.eye(n)
        g = rng.randn(n)

        with pytest.raises(ValueError, match="margin"):
            compute_manipulation_cost(F, g, margin=0.0)

    def test_mc_negative_margin_raises(self) -> None:
        """Test that margin < 0 raises ValueError."""
        rng = np.random.RandomState(42)
        n = 3
        F = np.eye(n)
        g = rng.randn(n)

        with pytest.raises(ValueError, match="margin"):
            compute_manipulation_cost(F, g, margin=-0.1)

    def test_mc_zero_gradient_raises(self) -> None:
        """Test that g = 0 raises ValueError (V would be 0)."""
        rng = np.random.RandomState(42)
        n = 3
        F = np.eye(n)
        g = np.zeros(n)

        with pytest.raises(ValueError, match="Vulnerability score"):
            compute_manipulation_cost(F, g, margin=0.1)


class TestConservativeBoundVerification:
    """Tests for conservative bound verification (Proposition 7.10)."""

    def test_conservative_bound_holds(self) -> None:
        """Construct a synthetic full-parameter Fisher F_full and its LLF submatrix.

        By zeroing out backbone rows/columns, we get F_llf. Then verify
        MC_full <= MC_llf.
        """
        rng = np.random.RandomState(42)
        n_full = 10
        n_llf = 5  # Last layer only

        # Create a full Fisher matrix (symmetric positive definite)
        A = rng.randn(n_full, n_full)
        F_full = A @ A.T + 0.1 * np.eye(n_full)

        # Create gradient vector
        g_full = rng.randn(n_full)

        # Create LLF version by zeroing out first (n_full - n_llf) rows/cols
        # Actually, for LLF we take the bottom-right block
        F_llf = F_full[n_full - n_llf :, n_full - n_llf :].copy()
        g_llf = g_full[n_full - n_llf :].copy()

        margin = 0.1

        result_full = compute_manipulation_cost(F_full, g_full, margin, damping=1e-6, tol=1e-10)
        result_llf = compute_manipulation_cost(F_llf, g_llf, margin, damping=1e-6, tol=1e-10)

        # Proposition 7.10: MC_full <= MC_llf
        assert result_full.mc_value <= result_llf.mc_value + 1e-10

    def test_conservative_bound_check_function(self) -> None:
        """Verify llf_conservative_bound_check function behavior."""
        # MC_full <= MC_llf should return True
        assert llf_conservative_bound_check(0.0129, 0.010) is True

        # MC_full > MC_llf should return False
        assert llf_conservative_bound_check(0.010, 0.0129) is False


class TestIntegrationWithFisherBuilder:
    """Tests for integration with Fisher builder module."""

    def test_end_to_end_with_builder(self) -> None:
        """Use build_fisher_matrix to construct a Fisher matrix from synthetic features.

        Then compute the manipulation cost and verify all fields of
        ManipulationCostResult are populated and consistent.
        """
        rng = np.random.RandomState(42)
        n_samples = 20
        feature_dim = 5

        # Generate synthetic features
        features = rng.randn(n_samples, feature_dim)

        # Generate comparison pairs
        pairs = [(i, i + 1) for i in range(n_samples - 1)]

        # Generate reward differences
        reward_diffs = rng.randn(len(pairs))

        # Build Fisher matrix
        fisher_result = build_fisher_matrix(features, pairs, reward_diffs)
        F = fisher_result.fisher_matrix

        # Create gradient vector (same dimension as features)
        g = rng.randn(feature_dim)

        margin = 0.1

        result = compute_manipulation_cost(F, g, margin, damping=1e-6, tol=1e-10)

        # Verify all fields are populated
        assert isinstance(result.mc_value, float)
        assert result.mc_value > 0
        assert isinstance(result.vulnerability_score, float)
        assert result.vulnerability_score > 0
        assert isinstance(result.kl_budget, float)
        assert result.kl_budget > 0
        assert isinstance(result.cg_result, CGResult)
        assert isinstance(result.is_vulnerable, (bool, np.bool_))

        # Verify consistency: KL budget should equal MC^2 / 2
        expected_kl = (result.mc_value ** 2) / 2.0
        assert np.isclose(result.kl_budget, expected_kl, rtol=1e-10)

    def test_vulnerability_flag(self) -> None:
        """Construct systems with low and high MC to verify is_vulnerable flag."""
        rng = np.random.RandomState(42)
        n = 5
        F = np.eye(n)
        margin = 0.1
        vulnerability_threshold = 0.05

        # Low MC (high vulnerability): large V means small MC
        # V = g^T g, so make g large
        g_large = rng.randn(n) * 100  # Large gradient
        result_vulnerable = compute_manipulation_cost(
            F, g_large, margin, damping=0.0, vulnerability_threshold=vulnerability_threshold
        )
        assert bool(result_vulnerable.is_vulnerable) is True

        # High MC (low vulnerability): small V means large MC
        # Make g small
        g_small = rng.randn(n) * 0.01  # Small gradient
        result_not_vulnerable = compute_manipulation_cost(
            F, g_small, margin, damping=0.0, vulnerability_threshold=vulnerability_threshold
        )
        assert bool(result_not_vulnerable.is_vulnerable) is False


class TestDeterminism:
    """Tests for determinism and reproducibility."""

    def test_cg_determinism(self) -> None:
        """Solving the same system twice produces identical results."""
        rng = np.random.RandomState(42)
        n = 10
        F = rng.randn(n, n)
        F = F @ F.T  # Make symmetric positive definite
        g = rng.randn(n)

        result1 = solve_fisher_system(F, g, damping=1e-6, tol=1e-10, max_iter=100)
        result2 = solve_fisher_system(F, g, damping=1e-6, tol=1e-10, max_iter=100)

        # Check bitwise or near-bitwise equality
        assert np.allclose(result1.solution, result2.solution, rtol=1e-14, atol=1e-14)
        assert result1.converged == result2.converged
        assert result1.n_iterations == result2.n_iterations
        assert np.isclose(result1.residual_norm, result2.residual_norm, rtol=1e-14)

    def test_mc_determinism(self) -> None:
        """Computing MC twice from the same inputs produces identical results."""
        rng = np.random.RandomState(42)
        n = 10
        F = rng.randn(n, n)
        F = F @ F.T  # Make symmetric positive definite
        g = rng.randn(n)
        margin = 0.1

        result1 = compute_manipulation_cost(F, g, margin, damping=1e-6, tol=1e-10)
        result2 = compute_manipulation_cost(F, g, margin, damping=1e-6, tol=1e-10)

        assert np.isclose(result1.mc_value, result2.mc_value, rtol=1e-14)
        assert np.isclose(result1.vulnerability_score, result2.vulnerability_score, rtol=1e-14)
        assert np.isclose(result1.kl_budget, result2.kl_budget, rtol=1e-14)
        assert result1.is_vulnerable == result2.is_vulnerable


class TestEdgeCasesAndValidation:
    """Additional tests for edge cases and input validation."""

    def test_cg_g_shape_mismatch_raises(self) -> None:
        """Test that g shape not matching F raises ValueError."""
        rng = np.random.RandomState(42)
        n = 4
        F = np.eye(n)
        g_wrong = rng.randn(n + 1)  # Wrong size

        with pytest.raises(ValueError, match="shape"):
            solve_fisher_system(F, g_wrong)

    def test_cg_invalid_tol_raises(self) -> None:
        """Test that tol <= 0 raises ValueError."""
        rng = np.random.RandomState(42)
        n = 3
        F = np.eye(n)
        g = rng.randn(n)

        with pytest.raises(ValueError, match="tol"):
            solve_fisher_system(F, g, tol=0.0)

        with pytest.raises(ValueError, match="tol"):
            solve_fisher_system(F, g, tol=-1e-8)

    def test_cg_invalid_max_iter_raises(self) -> None:
        """Test that max_iter < 1 raises ValueError."""
        rng = np.random.RandomState(42)
        n = 3
        F = np.eye(n)
        g = rng.randn(n)

        with pytest.raises(ValueError, match="max_iter"):
            solve_fisher_system(F, g, max_iter=0)

    def test_mc_cg_not_converged_raises(self) -> None:
        """Test that non-convergent CG raises ValueError."""
        rng = np.random.RandomState(42)
        n = 100
        # Create a challenging system where convergence is difficult
        # Use a very ill-conditioned matrix with large condition number
        diag_vals = np.concatenate([np.ones(50) * 1e-10, np.ones(50) * 1e10])
        F = np.diag(diag_vals)
        g = rng.randn(n)
        margin = 0.1

        # Use very few iterations to force non-convergence
        with pytest.raises(ValueError, match="did not converge"):
            compute_manipulation_cost(F, g, margin, damping=0.0, tol=1e-15, max_iter=2)


class TestCGResultDataclass:
    """Tests for CGResult dataclass."""

    def test_cg_result_fields(self) -> None:
        """Test that CGResult has all required fields."""
        rng = np.random.RandomState(42)
        n = 3
        F = np.eye(n)
        g = rng.randn(n)

        result = solve_fisher_system(F, g)

        assert isinstance(result.solution, np.ndarray)
        assert result.solution.shape == (n,)
        assert isinstance(result.converged, bool)
        assert isinstance(result.n_iterations, int)
        assert isinstance(result.residual_norm, float)


class TestManipulationCostResultDataclass:
    """Tests for ManipulationCostResult dataclass."""

    def test_mc_result_fields(self) -> None:
        """Test that ManipulationCostResult has all required fields."""
        rng = np.random.RandomState(42)
        n = 5
        F = np.eye(n)
        g = rng.randn(n)
        margin = 0.1

        result = compute_manipulation_cost(F, g, margin)

        assert isinstance(result.mc_value, float)
        assert isinstance(result.vulnerability_score, float)
        assert isinstance(result.kl_budget, float)
        assert isinstance(result.cg_result, CGResult)
        assert isinstance(result.is_vulnerable, (bool, np.bool_))


class TestKlBudgetFunction:
    """Tests for kl_budget_from_mc function."""

    def test_kl_budget_positive(self) -> None:
        """Test that KL budget is always positive for positive MC."""
        mc_values = [0.001, 0.01, 0.1, 1.0, 10.0]

        for mc in mc_values:
            kl = kl_budget_from_mc(mc)
            assert kl > 0

    def test_kl_budget_zero(self) -> None:
        """Test that KL budget is zero for zero MC."""
        kl = kl_budget_from_mc(0.0)
        assert kl == 0.0

    def test_kl_budget_quadratic(self) -> None:
        """Test that KL budget scales quadratically with MC."""
        mc1 = 0.01
        mc2 = 0.02  # 2x mc1

        kl1 = kl_budget_from_mc(mc1)
        kl2 = kl_budget_from_mc(mc2)

        # kl2 should be 4x kl1 (quadratic scaling)
        assert np.isclose(kl2, 4 * kl1, rtol=1e-10)
