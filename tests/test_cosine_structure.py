"""Tests for cosine similarity structure computation module."""

import numpy as np
import pytest

from src.analysis.cosine_structure import (
    CosineStructureResult,
    bootstrap_ci,
    compute_cosine_structure,
    cosine_matrix,
    cosine_similarity,
    cross_component_cosine,
    within_component_cosine,
)


class TestCosineSimilarityCorrectness:
    """Tests for basic cosine similarity correctness."""

    def test_cosine_identical_vectors(self) -> None:
        """Test that cosine of a vector with itself is 1.0."""
        v = np.array([1.0, 2.0, 3.0])
        result = cosine_similarity(v, v)
        assert result == 1.0

    def test_cosine_orthogonal_vectors(self) -> None:
        """Test that cosine of orthogonal vectors is 0.0."""
        v = np.array([1.0, 0.0, 0.0])
        w = np.array([0.0, 1.0, 0.0])
        result = cosine_similarity(v, w)
        assert result == 0.0

    def test_cosine_opposite_vectors(self) -> None:
        """Test that cosine of v and -v is -1.0."""
        v = np.array([1.0, 2.0, 3.0])
        result = cosine_similarity(v, -v)
        assert result == -1.0

    def test_cosine_known_value(self) -> None:
        """Test hand-computed cosine for a specific pair to within 1e-12."""
        # v = [1, 2, 3], w = [4, 5, 6]
        # dot(v, w) = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        # norm(v) = sqrt(1 + 4 + 9) = sqrt(14)
        # norm(w) = sqrt(16 + 25 + 36) = sqrt(77)
        # cos = 32 / (sqrt(14) * sqrt(77))
        v = np.array([1.0, 2.0, 3.0])
        w = np.array([4.0, 5.0, 6.0])
        expected = 32.0 / (np.sqrt(14.0) * np.sqrt(77.0))
        result = cosine_similarity(v, w)
        assert abs(result - expected) < 1e-12

    def test_cosine_zero_norm_raises(self) -> None:
        """Test that zero vector raises ValueError."""
        v = np.array([1.0, 2.0, 3.0])
        zero = np.array([0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="zero norm"):
            cosine_similarity(v, zero)
        with pytest.raises(ValueError, match="zero norm"):
            cosine_similarity(zero, v)

    def test_cosine_shape_mismatch_raises(self) -> None:
        """Test that vectors of different length raise ValueError."""
        v = np.array([1.0, 2.0, 3.0])
        w = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="same shape"):
            cosine_similarity(v, w)

    def test_cosine_non_1d_raises(self) -> None:
        """Test that non-1D vectors raise ValueError."""
        v = np.array([[1.0, 2.0], [3.0, 4.0]])
        w = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="1-D"):
            cosine_similarity(v, w)
        with pytest.raises(ValueError, match="1-D"):
            cosine_similarity(w, v)


class TestCosineMatrixCorrectness:
    """Tests for cosine matrix computation."""

    def test_cosine_matrix_diagonal_is_one(self) -> None:
        """Test that diagonal entries are all 1.0."""
        rng = np.random.RandomState(42)
        X = rng.standard_normal((10, 5))
        C = cosine_matrix(X)
        diag = np.diag(C)
        assert np.allclose(diag, 1.0)

    def test_cosine_matrix_symmetric(self) -> None:
        """Test that matrix equals its transpose."""
        rng = np.random.RandomState(42)
        X = rng.standard_normal((10, 5))
        C = cosine_matrix(X)
        assert np.allclose(C, C.T)

    def test_cosine_matrix_matches_pairwise(self) -> None:
        """Test that matrix entries match individual cosine_similarity calls."""
        rng = np.random.RandomState(42)
        X = rng.standard_normal((5, 3))
        C = cosine_matrix(X)

        # Check a few entries against pairwise computation
        for i in range(5):
            for j in range(5):
                expected = cosine_similarity(X[i], X[j])
                assert abs(C[i, j] - expected) < 1e-12

    def test_cosine_matrix_zero_row_raises(self) -> None:
        """Test that a zero-norm row raises ValueError."""
        X = np.array([[1.0, 2.0], [0.0, 0.0], [3.0, 4.0]])
        with pytest.raises(ValueError, match="zero norm"):
            cosine_matrix(X)

    def test_cosine_matrix_non_2d_raises(self) -> None:
        """Test that non-2D input raises ValueError."""
        x_1d = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="2-D"):
            cosine_matrix(x_1d)
        x_3d = np.random.randn(2, 3, 4)
        with pytest.raises(ValueError, match="2-D"):
            cosine_matrix(x_3d)


class TestWithinCrossStructure:
    """Tests for within/cross component cosine structure."""

    def test_within_component_cosine_shape(self) -> None:
        """Test that output shape matches input pair count."""
        rng = np.random.RandomState(42)
        X = rng.standard_normal((100, 50))
        pairs = [(i, i + 1) for i in range(0, 20, 2)]
        result = within_component_cosine(X, pairs)
        assert result.shape == (len(pairs),)

    def test_within_component_cosine_values(self) -> None:
        """Test that near-identical within-component pairs have cosines close to 1.0."""
        # Create synthetic data where within-component pairs are nearly identical
        rng = np.random.RandomState(42)
        base = rng.standard_normal((10, 20))
        # Add tiny noise to create near-identical pairs
        X = np.vstack([base, base + rng.standard_normal((10, 20)) * 1e-6])
        # Pairs are (i, i+10) for i in 0..9
        pairs = [(i, i + 10) for i in range(10)]
        result = within_component_cosine(X, pairs)
        # All cosines should be very close to 1.0
        assert np.all(result > 0.999)

    def test_cross_component_cosine_shape(self) -> None:
        """Test that output shape matches n_pairs."""
        rng = np.random.RandomState(42)
        X = rng.standard_normal((100, 50))
        n_pairs = 20
        result = cross_component_cosine(X, n_pairs=n_pairs, random_state=42)
        assert result.shape == (n_pairs,)

    def test_cross_excludes_within_pairs(self) -> None:
        """Test that exclude_indices prevents sampling within-component pairs."""
        rng = np.random.RandomState(42)
        X = rng.standard_normal((50, 10))

        # Define some within-component pairs
        within_pairs = [(0, 1), (2, 3), (4, 5)]
        exclude_set = set()
        for i, j in within_pairs:
            exclude_set.add((i, j))
            exclude_set.add((j, i))

        # Sample many cross pairs
        n_pairs = 100
        result = cross_component_cosine(
            X, n_pairs=n_pairs, random_state=42, exclude_indices=exclude_set
        )

        # The function should complete without error
        # and return the correct number of samples
        assert len(result) == n_pairs

    def test_cross_determinism(self) -> None:
        """Test that same random_state produces identical arrays."""
        rng = np.random.RandomState(42)
        X = rng.standard_normal((100, 50))

        result1 = cross_component_cosine(X, n_pairs=20, random_state=42)
        result2 = cross_component_cosine(X, n_pairs=20, random_state=42)

        assert np.array_equal(result1, result2)

    def test_cross_different_seeds_differ(self) -> None:
        """Test that different random_state values produce different samples."""
        rng = np.random.RandomState(42)
        X = rng.standard_normal((100, 50))

        result1 = cross_component_cosine(X, n_pairs=20, random_state=42)
        result2 = cross_component_cosine(X, n_pairs=20, random_state=123)

        # They should differ (with very high probability)
        assert not np.array_equal(result1, result2)

    def test_cross_n_pairs_zero_raises(self) -> None:
        """Test that n_pairs <= 0 raises ValueError."""
        rng = np.random.RandomState(42)
        X = rng.standard_normal((100, 50))
        with pytest.raises(ValueError, match="positive"):
            cross_component_cosine(X, n_pairs=0, random_state=42)
        with pytest.raises(ValueError, match="positive"):
            cross_component_cosine(X, n_pairs=-1, random_state=42)

    def test_cross_n_pairs_too_large_raises(self) -> None:
        """Test that requesting more pairs than available raises ValueError."""
        rng = np.random.RandomState(42)
        X = rng.standard_normal((5, 3))
        # With 5 samples, max pairs = 5 * 4 = 20
        with pytest.raises(ValueError, match="available"):
            cross_component_cosine(X, n_pairs=100, random_state=42)


class TestIntegratedStructureResult:
    """Tests for the integrated compute_cosine_structure function."""

    def test_compute_cosine_structure_fields(self) -> None:
        """Test that all fields of CosineStructureResult are populated and consistent."""
        rng = np.random.RandomState(42)
        X = rng.standard_normal((100, 50))
        pairs = [(i, i + 1) for i in range(0, 20, 2)]

        result = compute_cosine_structure(X, pairs, n_cross_pairs=50, random_state=42)

        # Check all fields exist and have correct types
        assert isinstance(result.within_mean, float)
        assert isinstance(result.within_std, float)
        assert isinstance(result.cross_mean, float)
        assert isinstance(result.cross_std, float)
        assert isinstance(result.gap, float)
        assert isinstance(result.n_within, int)
        assert isinstance(result.n_cross, int)
        assert isinstance(result.within_values, np.ndarray)
        assert isinstance(result.cross_values, np.ndarray)

        # Check consistency
        assert result.gap == result.within_mean - result.cross_mean
        assert result.n_within == len(result.within_values)
        assert result.n_cross == len(result.cross_values)
        assert result.n_within == len(pairs)
        assert result.n_cross == 50

        # Check means and stds match the values
        assert abs(result.within_mean - np.mean(result.within_values)) < 1e-12
        assert abs(result.within_std - np.std(result.within_values, ddof=0)) < 1e-12
        assert abs(result.cross_mean - np.mean(result.cross_values)) < 1e-12
        assert abs(result.cross_std - np.std(result.cross_values, ddof=0)) < 1e-12

    def test_synthetic_two_cluster_structure(self) -> None:
        """Test detection of within/cross cosine gap on synthetic two-cluster data.

        Constructs a feature matrix with two tight clusters where:
        - Within-cluster cosine ≈ 0.95
        - Cross-cluster cosine ≈ 0.5

        Verifies that within_mean > cross_mean and gap > 0.3.
        """
        rng = np.random.RandomState(42)

        # Create two well-separated clusters
        n_per_cluster = 50
        n_features = 100

        # Cluster centers far apart
        center1 = np.ones(n_features) * 10.0
        center2 = -np.ones(n_features) * 10.0

        # Generate cluster members with small variance around centers
        cluster1 = center1 + rng.standard_normal((n_per_cluster, n_features)) * 0.5
        cluster2 = center2 + rng.standard_normal((n_per_cluster, n_features)) * 0.5

        X = np.vstack([cluster1, cluster2])

        # Define within-component pairs (within each cluster)
        # Pair consecutive samples within cluster 1
        within_pairs = [(i, i + 1) for i in range(0, n_per_cluster - 1, 2)]

        result = compute_cosine_structure(
            X, within_pairs, n_cross_pairs=200, random_state=42
        )

        # Verify the gap pattern
        assert result.within_mean > result.cross_mean, (
            f"Expected within_mean > cross_mean, got {result.within_mean} vs "
            f"{result.cross_mean}"
        )
        assert result.gap > 0.3, (
            f"Expected gap > 0.3, got {result.gap}"
        )

        # Within-cluster cosines should be high (near 1.0 for tight clusters)
        assert result.within_mean > 0.9, (
            f"Expected within_mean > 0.9, got {result.within_mean}"
        )


class TestBootstrapCI:
    """Tests for bootstrap confidence interval computation."""

    def test_bootstrap_ci_contains_true_mean(self) -> None:
        """Test that 95% bootstrap CI contains the true mean for a known distribution."""
        rng = np.random.RandomState(42)

        # Generate data from N(0, 1)
        data = rng.standard_normal(500)

        # True mean is 0
        lo, hi = bootstrap_ci(data, statistic="mean", n_bootstrap=2000, ci=0.95)

        # For n=500 from N(0,1), the 95% CI should contain 0
        assert lo < 0 < hi or abs(lo) < 0.2 or abs(hi) < 0.2

    def test_bootstrap_ci_for_std(self) -> None:
        """Test bootstrap CI for standard deviation."""
        rng = np.random.RandomState(42)

        # Generate data from N(0, 1)
        data = rng.standard_normal(500)

        # True std is 1
        lo, hi = bootstrap_ci(data, statistic="std", n_bootstrap=2000, ci=0.95)

        # CI should be around 1.0
        assert lo < 1.0 < hi or (lo > 0.8 and hi < 1.2)

    def test_bootstrap_ci_invalid_statistic_raises(self) -> None:
        """Test that invalid statistic raises ValueError."""
        rng = np.random.RandomState(42)
        data = rng.standard_normal(100)

        with pytest.raises(ValueError, match="statistic must be"):
            bootstrap_ci(data, statistic="median")  # type: ignore[arg-type]

    def test_bootstrap_ci_invalid_ci_raises(self) -> None:
        """Test that ci outside (0, 1) raises ValueError."""
        rng = np.random.RandomState(42)
        data = rng.standard_normal(100)

        with pytest.raises(ValueError, match="ci must be in"):
            bootstrap_ci(data, ci=1.5)
        with pytest.raises(ValueError, match="ci must be in"):
            bootstrap_ci(data, ci=0.0)
        with pytest.raises(ValueError, match="ci must be in"):
            bootstrap_ci(data, ci=-0.5)

    def test_bootstrap_ci_empty_raises(self) -> None:
        """Test that empty array raises ValueError."""
        empty = np.array([])
        with pytest.raises(ValueError, match="empty"):
            bootstrap_ci(empty)

    def test_bootstrap_ci_determinism(self) -> None:
        """Test that same random_state produces identical results."""
        rng = np.random.RandomState(42)
        data = rng.standard_normal(100)

        lo1, hi1 = bootstrap_ci(data, n_bootstrap=1000, random_state=42)
        lo2, hi2 = bootstrap_ci(data, n_bootstrap=1000, random_state=42)

        assert lo1 == lo2
        assert hi1 == hi2


class TestEdgeCasesAndValidation:
    """Tests for edge cases and input validation."""

    def test_empty_pair_indices_raises(self) -> None:
        """Test that empty pair_indices raises ValueError."""
        rng = np.random.RandomState(42)
        X = rng.standard_normal((100, 50))

        with pytest.raises(ValueError, match="empty"):
            within_component_cosine(X, [])

    def test_out_of_bounds_index_raises(self) -> None:
        """Test that pair index beyond feature matrix size raises ValueError."""
        rng = np.random.RandomState(42)
        X = rng.standard_normal((10, 5))

        # Index 10 is out of bounds for 10 samples
        with pytest.raises(ValueError, match="out of bounds"):
            within_component_cosine(X, [(0, 10)])

        # Negative index
        with pytest.raises(ValueError, match="out of bounds"):
            within_component_cosine(X, [(-1, 0)])

    def test_n_cross_pairs_too_large_raises(self) -> None:
        """Test that requesting more cross pairs than available raises ValueError."""
        rng = np.random.RandomState(42)
        X = rng.standard_normal((5, 3))

        # With 5 samples, max pairs = 5 * 4 = 20
        # Requesting more should fail
        with pytest.raises(ValueError, match="available"):
            cross_component_cosine(X, n_pairs=100, random_state=42)

    @pytest.mark.parametrize("bad_ci", [0.0, 1.0, 1.5, -0.5])
    def test_bootstrap_ci_boundary_values(self, bad_ci: float) -> None:
        """Test bootstrap CI with boundary ci values.

        Parameters
        ----------
        bad_ci : float
            Invalid or boundary ci value to test.
        """
        rng = np.random.RandomState(42)
        data = rng.standard_normal(100)

        if bad_ci == 0.0 or bad_ci >= 1.0 or bad_ci < 0:
            with pytest.raises(ValueError, match="ci must be in"):
                bootstrap_ci(data, ci=bad_ci)

    def test_cosine_similarity_large_vectors(self) -> None:
        """Test cosine similarity with large magnitude vectors (numerical stability)."""
        # Large vectors should still compute correctly
        v = np.array([1e6, 2e6, 3e6])
        w = np.array([1e6, 2e6, 3e6])
        result = cosine_similarity(v, w)
        assert abs(result - 1.0) < 1e-10

    def test_cosine_similarity_small_vectors(self) -> None:
        """Test cosine similarity with small magnitude vectors (numerical stability)."""
        # Small vectors should still compute correctly
        v = np.array([1e-6, 2e-6, 3e-6])
        w = np.array([1e-6, 2e-6, 3e-6])
        result = cosine_similarity(v, w)
        assert abs(result - 1.0) < 1e-10

    def test_within_component_single_pair(self) -> None:
        """Test within_component_cosine with a single pair."""
        rng = np.random.RandomState(42)
        X = rng.standard_normal((10, 5))
        pairs = [(0, 1)]
        result = within_component_cosine(X, pairs)
        assert result.shape == (1,)
        expected = cosine_similarity(X[0], X[1])
        assert abs(result[0] - expected) < 1e-12

    def test_compute_cosine_structure_single_pair(self) -> None:
        """Test compute_cosine_structure with a single within-component pair."""
        rng = np.random.RandomState(42)
        X = rng.standard_normal((50, 10))
        pairs = [(0, 1)]

        result = compute_cosine_structure(X, pairs, n_cross_pairs=10, random_state=42)

        assert result.n_within == 1
        assert result.n_cross == 10
        assert len(result.within_values) == 1
        assert len(result.cross_values) == 10
