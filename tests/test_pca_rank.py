"""Tests for PCA effective rank computation module."""

import numpy as np
import pytest

from src.analysis.pca_rank import (
    PcaResult,
    compute_pca,
    effective_rank,
    explain_variance_at_rank,
)
from src.utils.seed import seed_everything


class TestSyntheticLowRank:
    """Tests using synthetic low-rank matrices with known structure."""

    def test_rank_36_recovery(self) -> None:
        """Test recovery of a synthetic rank-36 matrix.

        Constructs a (500, 1024) matrix with true rank 36 by generating
        A = U @ V.T where U has shape (500, 36) and V has shape (1024, 36),
        both drawn from a fixed-seed Gaussian. Adds tiny noise (1e-8 scale).
        Verifies that effective_rank <= 36 at threshold 0.95 (since all
        significant variance is captured within the true rank).
        """
        rng = np.random.default_rng(42)

        # Generate random matrices for low-rank construction
        U = rng.standard_normal((500, 36))
        V = rng.standard_normal((1024, 36))

        # Construct low-rank matrix: A = U @ V.T has rank at most 36
        A = U @ V.T

        # Add tiny noise to simulate numerical precision
        noise = rng.standard_normal((500, 1024)) * 1e-8
        A_noisy = A + noise

        result = compute_pca(A_noisy, threshold=0.95)

        # The effective rank should be at most 36 (the true rank)
        # In practice with random matrices it may be slightly less due to
        # uneven variance distribution, but should never exceed the true rank
        assert result.effective_rank <= 36, (
            f"Expected effective_rank <= 36, got {result.effective_rank}"
        )
        # Also verify we capture significant structure (not just 1 component)
        assert result.effective_rank >= 10, (
            f"Expected effective_rank >= 10, got {result.effective_rank}"
        )

    def test_full_rank_matrix(self) -> None:
        """Test effective rank on a full-rank square matrix.

        Constructs a (50, 50) full-rank Gaussian matrix. At threshold 0.99,
        the effective rank should be close to min(n_samples, n_features) = 50,
        though for random Gaussian matrices the variance is spread across many
        components so fewer may reach 99%.
        """
        rng = np.random.default_rng(42)

        X = rng.standard_normal((50, 50))

        result = compute_pca(X, threshold=0.99)

        # For a full-rank random Gaussian matrix, variance is spread relatively
        # evenly, so we expect many components but not necessarily all 50 at 99%
        assert result.effective_rank >= 30, (
            f"Expected effective_rank >= 30, got {result.effective_rank}"
        )
        assert result.effective_rank <= 50, (
            f"Expected effective_rank <= 50, got {result.effective_rank}"
        )

    def test_rank_1_matrix(self) -> None:
        """Test effective rank on a rank-1 matrix.

        Constructs a rank-1 matrix via outer product of two random vectors.
        Verifies effective_rank == 1 at threshold 0.95.
        """
        rng = np.random.default_rng(42)

        u = rng.standard_normal(100)
        v = rng.standard_normal(50)

        # Outer product gives rank-1 matrix
        X = np.outer(u, v)

        result = compute_pca(X, threshold=0.95)

        assert result.effective_rank == 1, (
            f"Expected effective_rank=1, got {result.effective_rank}"
        )

    def test_cumulative_variance_monotonic(self) -> None:
        """Verify cumulative variance is non-decreasing and ends near 1.0."""
        rng = np.random.default_rng(42)

        X = rng.standard_normal((100, 50))

        result = compute_pca(X, threshold=0.95)

        cum_var = result.cumulative_variance

        # Check monotonicity
        assert np.all(np.diff(cum_var) >= 0), (
            "Cumulative variance must be non-decreasing"
        )

        # Check final value is approximately 1.0
        assert abs(cum_var[-1] - 1.0) < 1e-6, (
            f"Final cumulative variance should be ~1.0, got {cum_var[-1]}"
        )

    def test_explained_variance_ratio_sums_to_one(self) -> None:
        """Verify explained_variance_ratio sums to approximately 1.0."""
        rng = np.random.default_rng(42)

        X = rng.standard_normal((100, 50))

        result = compute_pca(X, threshold=0.95)

        total = np.sum(result.explained_variance_ratio)

        assert abs(total - 1.0) < 1e-6, (
            f"Explained variance ratio should sum to ~1.0, got {total}"
        )


class TestEdgeCasesAndValidation:
    """Tests for edge cases and input validation."""

    def test_threshold_boundary(self) -> None:
        """Test effective_rank at threshold exactly matching a cumulative value."""
        cum_var = np.array([0.5, 0.75, 0.90, 0.95, 0.98, 1.0])

        # Threshold exactly at 0.95 should return index 4 (1-indexed)
        rank = effective_rank(cum_var, threshold=0.95)
        assert rank == 4, f"Expected rank=4 at threshold=0.95, got {rank}"

        # Threshold exactly at 0.90 should return index 3
        rank = effective_rank(cum_var, threshold=0.90)
        assert rank == 3, f"Expected rank=3 at threshold=0.90, got {rank}"

    def test_threshold_never_reached(self) -> None:
        """Test that threshold=1.0 returns total components when not reached."""
        # Due to numerical precision, cumulative variance may never reach exactly 1.0
        cum_var = np.array([0.3, 0.55, 0.75, 0.90, 0.95, 0.99])

        # With threshold=1.0, should return total number of components
        rank = effective_rank(cum_var, threshold=1.0)
        assert rank == 6, (
            f"Expected rank=6 when threshold not reached, got {rank}"
        )

    def test_invalid_1d_input(self) -> None:
        """Test that passing a 1-D array raises ValueError."""
        rng = np.random.default_rng(42)
        x_1d = rng.standard_normal(100)

        with pytest.raises(ValueError, match="Expected 2-D array"):
            compute_pca(x_1d, threshold=0.95)

    def test_invalid_single_sample(self) -> None:
        """Test that passing a single sample raises ValueError."""
        rng = np.random.default_rng(42)
        x_single = rng.standard_normal((1, 10))

        with pytest.raises(ValueError, match="at least 2 samples"):
            compute_pca(x_single, threshold=0.95)

    @pytest.mark.parametrize("bad_threshold", [0.0, -0.5, 1.5])
    def test_invalid_threshold(self, bad_threshold: float) -> None:
        """Test that invalid thresholds raise ValueError.

        Parameters
        ----------
        bad_threshold : float
            Invalid threshold value to test.
        """
        rng = np.random.default_rng(42)
        X = rng.standard_normal((10, 5))

        with pytest.raises(ValueError, match="Threshold must be in"):
            compute_pca(X, threshold=bad_threshold)

    def test_empty_cumulative_variance(self) -> None:
        """Test that empty cumulative_variance array raises ValueError."""
        empty_array = np.array([])

        with pytest.raises(ValueError, match="empty"):
            effective_rank(empty_array, threshold=0.95)

    @pytest.mark.parametrize("bad_rank", [0, 10])
    def test_explain_variance_at_rank_bounds(self, bad_rank: int) -> None:
        """Test explain_variance_at_rank raises for out-of-bounds rank.

        Parameters
        ----------
        bad_rank : int
            Invalid rank value to test.
        """
        cum_var = np.array([0.3, 0.55, 0.75, 0.90, 0.95])

        with pytest.raises(ValueError):
            explain_variance_at_rank(cum_var, bad_rank)


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_determinism(self) -> None:
        """Test that compute_pca produces identical results with same seed.

        Calls compute_pca twice on the same input with seed_everything(42)
        before each call. Verifies the components arrays are identical
        within numerical tolerance (1e-12).
        """
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 50))

        seed_everything(42)
        result1 = compute_pca(X, threshold=0.95, random_state=42)

        seed_everything(42)
        result2 = compute_pca(X, threshold=0.95, random_state=42)

        # Components should be identical (bitwise or within 1e-12)
        assert np.allclose(result1.components, result2.components, atol=1e-12), (
            "Components should be deterministic across runs"
        )

        # Also verify other fields match
        assert result1.effective_rank == result2.effective_rank
        assert np.allclose(
            result1.explained_variance_ratio,
            result2.explained_variance_ratio,
            atol=1e-12,
        )
        assert np.allclose(
            result1.cumulative_variance,
            result2.cumulative_variance,
            atol=1e-12,
        )


class TestPcaResultStructure:
    """Tests verifying PcaResult dataclass structure."""

    def test_pca_result_fields(self) -> None:
        """Verify PcaResult contains all required fields with correct types."""
        rng = np.random.default_rng(42)

        X = rng.standard_normal((100, 50))

        result = compute_pca(X, threshold=0.95)

        # Verify type and presence of all fields
        assert isinstance(result.effective_rank, int)
        assert isinstance(result.explained_variance_ratio, np.ndarray)
        assert isinstance(result.cumulative_variance, np.ndarray)
        assert isinstance(result.components, np.ndarray)
        assert isinstance(result.threshold, float)
        assert isinstance(result.n_samples, int)
        assert isinstance(result.n_features, int)
        assert isinstance(result.total_variance_explained, float)

        # Verify shapes
        assert result.explained_variance_ratio.shape == (50,)
        assert result.cumulative_variance.shape == (50,)
        assert result.components.shape == (50, 50)

        # Verify dimensions are recorded correctly
        assert result.n_samples == 100
        assert result.n_features == 50

        # Verify threshold is stored
        assert result.threshold == 0.95

        # Verify total_variance_explained matches cumulative at effective_rank
        expected_total = result.cumulative_variance[result.effective_rank - 1]
        assert abs(result.total_variance_explained - expected_total) < 1e-12
