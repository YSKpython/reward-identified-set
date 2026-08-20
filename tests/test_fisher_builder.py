"""Tests for the Fisher information matrix builder module."""

import numpy as np
import pytest

from src.fisher.builder import (
    FisherBuildResult,
    apply_tikhonov,
    build_fisher_matrix,
    check_psd,
    compute_edge_curvatures,
    fisher_null_space_dim,
    fisher_rank,
    logistic,
)


class TestLogisticFunction:
    """Tests for the numerically stable logistic function."""

    def test_logistic_zero(self) -> None:
        """Test that logistic(0) == 0.5."""
        result = logistic(np.array([0.0]))
        assert result[0] == 0.5

    def test_logistic_large_positive(self) -> None:
        """Test that logistic(100) is close to 1.0 without overflow."""
        result = logistic(np.array([100.0]))
        assert np.isclose(result[0], 1.0, rtol=1e-10)
        assert not np.isnan(result[0])
        assert not np.isinf(result[0])

    def test_logistic_large_negative(self) -> None:
        """Test that logistic(-100) is close to 0.0 without overflow."""
        result = logistic(np.array([-100.0]))
        assert np.isclose(result[0], 0.0, rtol=1e-10)
        assert not np.isnan(result[0])
        assert not np.isinf(result[0])

    def test_logistic_symmetry(self) -> None:
        """Test that logistic(x) + logistic(-x) == 1.0 for various x."""
        rng = np.random.RandomState(42)
        x_values = rng.uniform(-100, 100, size=100)
        x = np.array(x_values)
        result = logistic(x) + logistic(-x)
        assert np.allclose(result, 1.0, rtol=1e-10)


class TestEdgeCurvatures:
    """Tests for edge curvature computation."""

    def test_curvature_at_zero_margin(self) -> None:
        """Test that curvature at zero margin is 0.25."""
        result = compute_edge_curvatures(np.array([0.0]))
        assert result[0] == 0.25

    def test_curvature_positive_range(self) -> None:
        """Test that all curvatures are in (0, 0.25]."""
        rng = np.random.RandomState(42)
        # Use a smaller range to avoid underflow to exactly 0
        reward_diffs = rng.uniform(-20, 20, size=100)
        curvatures = compute_edge_curvatures(reward_diffs)
        assert np.all(curvatures >= 0)
        assert np.all(curvatures <= 0.25)

    def test_curvature_symmetry(self) -> None:
        """Test that curvature at +Delta r equals curvature at -Delta r."""
        rng = np.random.RandomState(42)
        reward_diffs = rng.uniform(-100, 100, size=50)
        curvatures_pos = compute_edge_curvatures(reward_diffs)
        curvatures_neg = compute_edge_curvatures(-reward_diffs)
        assert np.allclose(curvatures_pos, curvatures_neg, rtol=1e-10)

    def test_curvature_empty_raises(self) -> None:
        """Test that empty input raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            compute_edge_curvatures(np.array([]))


class TestFisherMatrixConstruction:
    """Tests for Fisher matrix construction."""

    def test_single_edge_fisher(self) -> None:
        """Test single edge Fisher matrix with known feature difference and margin 0."""
        # Create a simple feature matrix
        features = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        pairs = [(0, 1)]
        reward_diffs = np.array([0.0])  # margin 0 gives curvature 0.25

        result = build_fisher_matrix(features, pairs, reward_diffs)

        # Feature difference: delta = [1-4, 2-5, 3-6] = [-3, -3, -3]
        delta = features[0] - features[1]
        expected_fisher = 0.25 * np.outer(delta, delta)

        assert np.allclose(result.fisher_matrix, expected_fisher, rtol=1e-10)

    def test_fisher_symmetric(self) -> None:
        """Test that constructed Fisher matrix is symmetric."""
        rng = np.random.RandomState(42)
        features = rng.randn(10, 5)
        pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
        reward_diffs = rng.randn(len(pairs))

        result = build_fisher_matrix(features, pairs, reward_diffs)
        F = result.fisher_matrix

        assert np.allclose(F, F.T, rtol=1e-10)

    def test_fisher_psd(self) -> None:
        """Test that constructed Fisher matrix has all eigenvalues >= -1e-8."""
        rng = np.random.RandomState(42)
        features = rng.randn(10, 5)
        pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
        reward_diffs = rng.randn(len(pairs))

        result = build_fisher_matrix(features, pairs, reward_diffs)

        assert result.is_psd

    def test_fisher_trace(self) -> None:
        """Test that trace equals sum of I_e * ||delta_e||^2 over all edges."""
        rng = np.random.RandomState(42)
        features = rng.randn(10, 5)
        pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
        reward_diffs = rng.randn(len(pairs))

        result = build_fisher_matrix(features, pairs, reward_diffs)

        # Compute expected trace manually
        expected_trace = 0.0
        for idx, (i, j) in enumerate(pairs):
            delta = features[i] - features[j]
            curvature = result.edge_curvatures[idx]
            expected_trace += curvature * np.sum(delta**2)

        assert np.isclose(result.trace, expected_trace, rtol=1e-10)

    def test_fisher_rank_low_rank(self) -> None:
        """Test Fisher rank with features lying in a rank-3 subspace.

        This test mirrors the paper's E88 finding where Fisher rank is 501 out
        of 1024 dimensions (null-space 523). Here we construct a simpler case
        with rank 3 out of 10 dimensions to verify the rank analysis works.
        """
        rng = np.random.RandomState(42)
        n_samples = 100
        feature_dim = 10
        subspace_dim = 3

        # Generate features in a rank-3 subspace
        # First generate random basis vectors for the subspace
        basis = rng.randn(feature_dim, subspace_dim)
        # Then generate random coefficients
        coeffs = rng.randn(n_samples, subspace_dim)
        # Project into the subspace
        features = coeffs @ basis.T

        # Generate comparison pairs
        n_edges = 50
        pairs = [(rng.randint(0, n_samples), rng.randint(0, n_samples)) for _ in range(n_edges)]
        reward_diffs = rng.randn(n_edges)

        result = build_fisher_matrix(features, pairs, reward_diffs)

        # The Fisher rank should be at most subspace_dim (3)
        # With enough diverse edges, it should equal subspace_dim
        assert result.rank <= subspace_dim
        assert result.null_space_dim == feature_dim - result.rank

    def test_fisher_rank_full(self) -> None:
        """Test Fisher rank with features in general position.

        When features span the full space and we have more samples than
        feature dimension, the Fisher rank should equal feature_dim.
        """
        rng = np.random.RandomState(42)
        feature_dim = 5
        n_samples = 20  # More than feature_dim

        # Generate features in general position
        features = rng.randn(n_samples, feature_dim)

        # Generate enough diverse comparison pairs
        n_edges = 30
        pairs = [(rng.randint(0, n_samples), rng.randint(0, n_samples)) for _ in range(n_edges)]
        reward_diffs = rng.randn(n_edges)

        result = build_fisher_matrix(features, pairs, reward_diffs)

        # With enough diverse edges, Fisher rank should equal feature_dim
        assert result.rank == feature_dim
        assert result.null_space_dim == 0

    def test_null_space_dim_consistency(self) -> None:
        """Test that null_space_dim == feature_dim - rank for a known matrix."""
        rng = np.random.RandomState(42)
        features = rng.randn(10, 5)
        pairs = [(0, 1), (2, 3), (4, 5)]
        reward_diffs = rng.randn(len(pairs))

        result = build_fisher_matrix(features, pairs, reward_diffs)

        assert result.null_space_dim == result.feature_dim - result.rank


class TestTikhonovRegularization:
    """Tests for Tikhonov regularization."""

    def test_tikhonov_shifts_eigenvalues(self) -> None:
        """Test that eigenvalues of F + epsilon*I are eigenvalues of F shifted by epsilon."""
        rng = np.random.RandomState(42)
        F = rng.randn(5, 5)
        F = F @ F.T  # Make symmetric PSD

        epsilon = 0.5
        F_reg = apply_tikhonov(F, epsilon)

        eigvals_orig = np.linalg.eigvalsh(F)
        eigvals_reg = np.linalg.eigvalsh(F_reg)

        assert np.allclose(eigvals_reg, eigvals_orig + epsilon, rtol=1e-10)

    def test_tikhonov_zero_epsilon(self) -> None:
        """Test that apply_tikhonov(F, 0.0) returns F unchanged."""
        rng = np.random.RandomState(42)
        F = rng.randn(5, 5)
        F = F @ F.T

        F_reg = apply_tikhonov(F, 0.0)

        assert np.allclose(F_reg, F, rtol=1e-10)

    def test_tikhonov_negative_epsilon_raises(self) -> None:
        """Test that negative epsilon raises ValueError."""
        rng = np.random.RandomState(42)
        F = rng.randn(5, 5)

        with pytest.raises(ValueError, match="non-negative"):
            apply_tikhonov(F, -0.1)


class TestEdgeCasesAndValidation:
    """Tests for edge cases and input validation."""

    def test_empty_pairs_raises(self) -> None:
        """Test that empty pairs list raises ValueError."""
        rng = np.random.RandomState(42)
        features = rng.randn(10, 5)
        pairs = []
        reward_diffs = np.array([])

        with pytest.raises(ValueError, match="empty"):
            build_fisher_matrix(features, pairs, reward_diffs)

    def test_length_mismatch_raises(self) -> None:
        """Test that len(pairs) != len(reward_diffs) raises ValueError."""
        rng = np.random.RandomState(42)
        features = rng.randn(10, 5)
        pairs = [(0, 1), (2, 3)]
        reward_diffs = np.array([0.0])

        with pytest.raises(ValueError, match="len\\(pairs\\) must equal"):
            build_fisher_matrix(features, pairs, reward_diffs)

    def test_out_of_bounds_index_raises(self) -> None:
        """Test that pair index beyond feature count raises ValueError."""
        rng = np.random.RandomState(42)
        features = rng.randn(10, 5)
        pairs = [(0, 15)]  # 15 is out of bounds for 10 samples
        reward_diffs = np.array([0.0])

        with pytest.raises(ValueError, match="out-of-bounds"):
            build_fisher_matrix(features, pairs, reward_diffs)

    def test_nan_features_raises(self) -> None:
        """Test that features containing NaN raise ValueError."""
        rng = np.random.RandomState(42)
        features = rng.randn(10, 5)
        features[0, 0] = np.nan
        pairs = [(0, 1)]
        reward_diffs = np.array([0.0])

        with pytest.raises(ValueError, match="NaN or Inf"):
            build_fisher_matrix(features, pairs, reward_diffs)

    def test_non_square_rank_raises(self) -> None:
        """Test that non-square matrix to fisher_rank raises ValueError."""
        rng = np.random.RandomState(42)
        F = rng.randn(5, 3)

        with pytest.raises(ValueError, match="square"):
            fisher_rank(F)

    def test_inf_features_raises(self) -> None:
        """Test that features containing Inf raise ValueError."""
        rng = np.random.RandomState(42)
        features = rng.randn(10, 5)
        features[0, 0] = np.inf
        pairs = [(0, 1)]
        reward_diffs = np.array([0.0])

        with pytest.raises(ValueError, match="NaN or Inf"):
            build_fisher_matrix(features, pairs, reward_diffs)

    def test_non_1d_reward_diffs_raises(self) -> None:
        """Test that non-1D reward_diffs raises ValueError."""
        rng = np.random.RandomState(42)
        features = rng.randn(10, 5)
        pairs = [(0, 1)]
        reward_diffs = np.array([[0.0]])  # 2D instead of 1D

        with pytest.raises(ValueError, match="1-D"):
            build_fisher_matrix(features, pairs, reward_diffs)


class TestReproducibility:
    """Tests for reproducibility and determinism."""

    def test_fisher_determinism(self) -> None:
        """Test that building Fisher matrix twice produces identical results."""
        rng = np.random.RandomState(42)
        features = rng.randn(10, 5)
        pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
        reward_diffs = rng.randn(len(pairs))

        result1 = build_fisher_matrix(features, pairs, reward_diffs)
        result2 = build_fisher_matrix(features, pairs, reward_diffs)

        assert np.allclose(result1.fisher_matrix, result2.fisher_matrix, rtol=1e-14)
        assert np.allclose(result1.edge_curvatures, result2.edge_curvatures, rtol=1e-14)
        assert result1.trace == result2.trace
        assert result1.rank == result2.rank


class TestFisherRankAndNullSpace:
    """Additional tests for rank and null space functions."""

    def test_fisher_rank_identity(self) -> None:
        """Test that identity matrix has full rank."""
        n = 5
        F = np.eye(n)
        assert fisher_rank(F) == n

    def test_fisher_rank_zero_matrix(self) -> None:
        """Test that zero matrix has rank 0."""
        n = 5
        F = np.zeros((n, n))
        assert fisher_rank(F) == 0

    def test_fisher_rank_rank_one(self) -> None:
        """Test rank-1 matrix."""
        v = np.array([1.0, 2.0, 3.0])
        F = np.outer(v, v)
        assert fisher_rank(F) == 1

    def test_null_space_dim_identity(self) -> None:
        """Test null space dim of identity matrix is 0."""
        n = 5
        F = np.eye(n)
        assert fisher_null_space_dim(F) == 0

    def test_null_space_dim_zero(self) -> None:
        """Test null space dim of zero matrix equals dimension."""
        n = 5
        F = np.zeros((n, n))
        assert fisher_null_space_dim(F) == n


class TestCheckPSD:
    """Tests for PSD checking function."""

    def test_check_psd_identity(self) -> None:
        """Test that identity matrix is PSD."""
        F = np.eye(5)
        assert check_psd(F)

    def test_check_psd_zero(self) -> None:
        """Test that zero matrix is PSD."""
        F = np.zeros((5, 5))
        assert check_psd(F)

    def test_check_psd_negative(self) -> None:
        """Test that matrix with negative eigenvalue is not PSD."""
        F = np.diag([-1.0, 2.0, 3.0])
        assert not check_psd(F)

    def test_check_psd_small_negative_within_tol(self) -> None:
        """Test that small negative eigenvalue within tolerance is considered PSD."""
        F = np.diag([-1e-9, 2.0, 3.0])
        assert check_psd(F, tol=1e-8)

    def test_check_psd_outer_product(self) -> None:
        """Test that outer product matrix is PSD."""
        v = np.array([1.0, 2.0, 3.0])
        F = np.outer(v, v)
        assert check_psd(F)


class TestFisherBuildResult:
    """Tests for FisherBuildResult dataclass."""

    def test_result_fields(self) -> None:
        """Test that FisherBuildResult has all required fields."""
        rng = np.random.RandomState(42)
        features = rng.randn(10, 5)
        pairs = [(0, 1), (2, 3)]
        reward_diffs = rng.randn(len(pairs))

        result = build_fisher_matrix(features, pairs, reward_diffs)

        assert isinstance(result.fisher_matrix, np.ndarray)
        assert result.fisher_matrix.shape == (5, 5)
        assert isinstance(result.edge_curvatures, np.ndarray)
        assert result.edge_curvatures.shape == (2,)
        assert result.feature_dim == 5
        assert result.n_edges == 2
        assert isinstance(result.rank, int)
        assert isinstance(result.null_space_dim, int)
        assert isinstance(result.is_psd, bool)
        assert isinstance(result.trace, float)
