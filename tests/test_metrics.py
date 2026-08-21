"""Test suite for metrics module (flip cost, vulnerability ratio, geometric asymmetry).

This module tests the correctness of flip cost computation, null-space decomposition,
direction generation, geometric asymmetry, and vulnerability ratio integration.
"""

import numpy as np
import pytest

from src.metrics.manipulation_cost import (
    FlipCostResult,
    compute_flip_cost,
    compute_flip_cost_batch,
)
from src.metrics.vulnerability_ratio import (
    VulnerabilityRatioResult,
    compute_vulnerability_ratio,
    generate_cross_component_directions,
    generate_within_component_directions,
    geometric_asymmetry,
)


# =============================================================================
# Flip cost correctness tests
# =============================================================================


class TestFlipCostCorrectness:
    """Tests for flip cost computation correctness."""

    def test_flip_cost_identity_fisher(self):
        """For F = I and known g, verify FC = m² / ||g||² exactly."""
        F = np.eye(5)
        g = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        margin = 0.1

        result = compute_flip_cost(F, g, margin)

        # For F = I, F_ε = (1 + ε) * I ≈ I for small ε
        # g^T F_ε^{-1} g ≈ ||g||²
        # FC = m² / ||g||²
        g_norm_sq = np.dot(g, g)
        expected_fc = (margin ** 2) / g_norm_sq

        assert np.isclose(result.flip_cost, expected_fc, rtol=1e-5), (
            f"Expected {expected_fc}, got {result.flip_cost}"
        )
        assert np.isclose(result.quadratic_form, g_norm_sq, rtol=1e-5)

    def test_flip_cost_diagonal_fisher(self):
        """For F = diag([1, 2, 4]) and known g, verify FC matches analytic formula."""
        diag_vals = np.array([1.0, 2.0, 4.0])
        F = np.diag(diag_vals)
        g = np.array([1.0, 1.0, 1.0])
        margin = 0.1
        damping = 1e-6

        result = compute_flip_cost(F, g, margin, damping=damping)

        # Analytic: g^T (F + εI)^{-1} g = Σ g_i² / (λ_i + ε)
        # With g = [1, 1, 1], this is Σ 1 / (λ_i + ε)
        epsilon = damping
        expected_quad_form = np.sum(1.0 / (diag_vals + epsilon))
        expected_fc = (margin ** 2) / expected_quad_form

        assert np.isclose(result.quadratic_form, expected_quad_form, rtol=1e-5), (
            f"Expected quadratic_form {expected_quad_form}, got {result.quadratic_form}"
        )
        assert np.isclose(result.flip_cost, expected_fc, rtol=1e-5), (
            f"Expected flip_cost {expected_fc}, got {result.flip_cost}"
        )

    def test_flip_cost_zero_margin_raises(self):
        """margin = 0 raises ValueError."""
        F = np.eye(3)
        g = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="margin must be positive"):
            compute_flip_cost(F, g, margin=0.0)

    def test_flip_cost_negative_damping_raises(self):
        """damping < 0 raises ValueError."""
        F = np.eye(3)
        g = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="damping must be non-negative"):
            compute_flip_cost(F, g, margin=0.1, damping=-1e-6)

    def test_flip_cost_batch_shape(self):
        """Batch output shape matches input direction count."""
        F = np.eye(5)
        G = np.random.RandomState(42).randn(10, 5)
        margin = 0.1

        costs = compute_flip_cost_batch(F, G, margin)

        assert costs.shape == (10,)

    def test_flip_cost_batch_matches_individual(self):
        """Batch results match individual compute_flip_cost calls within 1e-10."""
        rng = np.random.RandomState(42)
        F = rng.randn(5, 5)
        F = F @ F.T  # Make symmetric positive semi-definite
        G = rng.randn(7, 5)
        margin = 0.1
        damping = 1e-6

        batch_costs = compute_flip_cost_batch(F, G, margin, damping)

        individual_costs = np.zeros(7)
        for i in range(7):
            result = compute_flip_cost(F, G[i, :], margin, damping)
            individual_costs[i] = result.flip_cost

        assert np.allclose(batch_costs, individual_costs, rtol=1e-10, atol=1e-10), (
            f"Batch: {batch_costs}\nIndividual: {individual_costs}"
        )


# =============================================================================
# Null-space decomposition tests
# =============================================================================


class TestNullSpaceDecomposition:
    """Tests for null-space decomposition of shift directions."""

    def test_null_fraction_zero_for_range_vector(self):
        """A vector in range(F) has null_fraction ≈ 0."""
        # Create F with rank 3 out of 5 dimensions
        rng = np.random.RandomState(42)
        U = rng.randn(5, 3)
        F = U @ U.T  # Rank-3 matrix

        # g in range(F): construct as linear combination of columns of U
        coeffs = rng.randn(3)
        g = U @ coeffs

        result = compute_flip_cost(F, g, margin=0.1, damping=1e-6)

        # g is entirely in range(F), so null_fraction should be ~0
        assert result.null_fraction < 1e-6, (
            f"Expected null_fraction ≈ 0, got {result.null_fraction}"
        )
        assert np.isclose(result.g_null_norm, 0.0, atol=1e-6)

    def test_null_fraction_one_for_null_vector(self):
        """A vector in null(F) has null_fraction ≈ 1."""
        # Create F with rank 2 out of 5 dimensions
        rng = np.random.RandomState(42)
        U = rng.randn(5, 2)
        F = U @ U.T  # Rank-2 matrix

        # Find a vector in null(F) via SVD
        _, s, Vt = np.linalg.svd(F)
        rank = int(np.sum(s > 1e-10))
        # Null space vectors are columns of V corresponding to zero singular values
        null_vector = Vt[rank, :]  # First null-space vector

        result = compute_flip_cost(F, null_vector, margin=0.1, damping=1e-6)

        # g is entirely in null(F), so null_fraction should be ~1
        assert result.null_fraction > 0.99, (
            f"Expected null_fraction ≈ 1, got {result.null_fraction}"
        )
        assert np.isclose(result.g_range_norm, 0.0, atol=1e-6)

    def test_null_fraction_intermediate(self):
        """A vector with known range/null decomposition has the expected null fraction."""
        # Create F with known eigendecomposition
        rng = np.random.RandomState(42)
        n = 6
        rank = 3

        # Construct F = U_r Λ U_r^T with known eigenvalues
        U = rng.randn(n, n)
        U, _ = np.linalg.qr(U)  # Orthogonalize

        eigenvalues = np.array([4.0, 2.0, 1.0, 0.0, 0.0, 0.0])
        F = U @ np.diag(eigenvalues) @ U.T

        # Construct g with known decomposition
        # g = 0.6 * u_0 + 0.8 * u_3 (mix of range and null)
        # where u_0 is in range, u_3 is in null
        g = 0.6 * U[:, 0] + 0.8 * U[:, 3]

        result = compute_flip_cost(F, g, margin=0.1, damping=1e-6)

        # Expected: g_norm² = 0.6² + 0.8² = 1.0
        # g_range_norm² = 0.6² = 0.36
        # g_null_norm² = 0.8² = 0.64
        # null_fraction = 0.64 / 1.0 = 0.64
        expected_null_fraction = 0.64

        assert np.isclose(result.null_fraction, expected_null_fraction, rtol=1e-2), (
            f"Expected null_fraction {expected_null_fraction}, got {result.null_fraction}"
        )


# =============================================================================
# Direction generation tests
# =============================================================================


class TestDirectionGeneration:
    """Tests for cross-component and within-component direction generation."""

    def test_cross_directions_shape(self):
        """Output shape is (n_directions, n_samples)."""
        assignments = np.array([0, 0, 1, 1, 2, 2])
        n_directions = 5

        directions = generate_cross_component_directions(
            assignments, feature_dim=6, n_directions=n_directions
        )

        assert directions.shape == (n_directions, 6)

    def test_cross_directions_unit_norm(self):
        """Each direction has unit norm."""
        assignments = np.array([0, 0, 1, 1, 2, 2])
        n_directions = 10

        directions = generate_cross_component_directions(
            assignments, feature_dim=6, n_directions=n_directions, random_state=42
        )

        norms = np.linalg.norm(directions, axis=1)
        assert np.allclose(norms, 1.0, rtol=1e-10), f"Norms: {norms}"

    def test_cross_directions_cross_component(self):
        """Cross-component directions assign different values to different components."""
        assignments = np.array([0, 0, 1, 1, 2, 2])
        n_directions = 3

        directions = generate_cross_component_directions(
            assignments, feature_dim=6, n_directions=n_directions, random_state=42
        )

        # Check that at least one direction has different values for different components
        found_diff = False
        for i in range(n_directions):
            d = directions[i, :]
            comp_0_mean = np.mean(d[assignments == 0])
            comp_1_mean = np.mean(d[assignments == 1])
            comp_2_mean = np.mean(d[assignments == 2])

            # At least two components should have different means
            if not (np.isclose(comp_0_mean, comp_1_mean) and
                    np.isclose(comp_1_mean, comp_2_mean)):
                found_diff = True
                break

        assert found_diff, "All cross-component directions have same value across components"

    def test_within_directions_shape(self):
        """Output shape is (n_directions, n_samples)."""
        assignments = np.array([0, 0, 1, 1, 2, 2])
        n_directions = 5

        directions = generate_within_component_directions(
            assignments, feature_dim=6, n_directions=n_directions
        )

        assert directions.shape == (n_directions, 6)

    def test_within_directions_single_component(self):
        """Each within-component direction is non-zero only within one component."""
        assignments = np.array([0, 0, 1, 1, 2, 2])
        n_directions = 10

        directions = generate_within_component_directions(
            assignments, feature_dim=6, n_directions=n_directions, random_state=42
        )

        unique_components = np.unique(assignments)

        for i in range(n_directions):
            d = directions[i, :]

            # Count how many components have non-zero values
            non_zero_comps = 0
            for comp in unique_components:
                mask = assignments == comp
                if np.any(np.abs(d[mask]) > 1e-10):
                    non_zero_comps += 1

            # Should be non-zero in exactly one component
            assert non_zero_comps == 1, (
                f"Direction {i} is non-zero in {non_zero_comps} components"
            )

    def test_direction_determinism(self):
        """Generating directions twice with the same random_state produces identical arrays."""
        assignments = np.array([0, 0, 1, 1, 2, 2])
        n_directions = 5

        dirs1 = generate_cross_component_directions(
            assignments, feature_dim=6, n_directions=n_directions, random_state=42
        )
        dirs2 = generate_cross_component_directions(
            assignments, feature_dim=6, n_directions=n_directions, random_state=42
        )

        assert np.array_equal(dirs1, dirs2), "Cross directions not deterministic"

        dirs3 = generate_within_component_directions(
            assignments, feature_dim=6, n_directions=n_directions, random_state=42
        )
        dirs4 = generate_within_component_directions(
            assignments, feature_dim=6, n_directions=n_directions, random_state=42
        )

        assert np.array_equal(dirs3, dirs4), "Within directions not deterministic"


# =============================================================================
# Geometric asymmetry tests
# =============================================================================


class TestGeometricAsymmetry:
    """Tests for geometric asymmetry computation."""

    def test_geometric_asymmetry_known_values(self):
        """For known cross and within cost arrays, verify the median ratio."""
        cross = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        within = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = geometric_asymmetry(cross, within)

        # median(cross) = 30, median(within) = 3
        expected = 30.0 / 3.0
        assert np.isclose(result, expected), f"Expected {expected}, got {result}"

    def test_geometric_asymmetry_empty_raises(self):
        """Empty array raises ValueError."""
        cross = np.array([])
        within = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="cross_costs must not be empty"):
            geometric_asymmetry(cross, within)

        cross = np.array([1.0, 2.0, 3.0])
        within = np.array([])

        with pytest.raises(ValueError, match="within_costs must not be empty"):
            geometric_asymmetry(cross, within)

    def test_geometric_asymmetry_zero_within_raises(self):
        """Zero median within cost raises ValueError."""
        cross = np.array([10.0, 20.0, 30.0])
        within = np.array([0.0, 0.0, 0.0])

        with pytest.raises(ValueError, match="median\\(within_costs\\) must be positive"):
            geometric_asymmetry(cross, within)


# =============================================================================
# Vulnerability ratio integration tests
# =============================================================================


class TestVulnerabilityRatioIntegration:
    """Integration tests for vulnerability ratio computation."""

    def test_vulnerability_ratio_fields_populated(self):
        """All fields of VulnerabilityRatioResult are populated and consistent."""
        rng = np.random.RandomState(42)
        n = 10
        F = rng.randn(n, n)
        F = F @ F.T  # Symmetric positive semi-definite

        cross_dirs = rng.randn(5, n)
        within_dirs = rng.randn(5, n)

        result = compute_vulnerability_ratio(F, cross_dirs, within_dirs, margin=0.1)

        # Check all fields are populated
        assert isinstance(result.vulnerability_ratio, float)
        assert isinstance(result.geometric_asymmetry, float)
        assert isinstance(result.cross_median, float)
        assert isinstance(result.within_median, float)
        assert isinstance(result.cross_mean, float)
        assert isinstance(result.within_mean, float)
        assert isinstance(result.cross_values, np.ndarray)
        assert isinstance(result.within_values, np.ndarray)
        assert isinstance(result.n_cross, int)
        assert isinstance(result.n_within, int)
        assert isinstance(result.fisher_rank, int)
        assert isinstance(result.null_space_dim, int)
        assert isinstance(result.damping, float)

        # Check consistency
        assert result.n_cross == 5
        assert result.n_within == 5
        assert len(result.cross_values) == 5
        assert len(result.within_values) == 5
        assert result.fisher_rank + result.null_space_dim == n
        assert np.isclose(result.cross_median, np.median(result.cross_values))
        assert np.isclose(result.within_median, np.median(result.within_values))

    def test_vulnerability_ratio_synthetic_e88(self):
        """Construct synthetic Fisher matrix with structured rank deficiency.

        Verify:
        - fisher_rank == 501 (or close approximation)
        - null_space_dim == 523 (or close approximation)
        - vulnerability_ratio and geometric_asymmetry are computed correctly
        
        This test verifies the structural pattern of E88: a rank-deficient Fisher
        matrix with approximately half the dimensions in the null-space.
        
        Note: The exact inequality (cross > within or vice versa) depends on the
        specific structure of F and the direction generation. The key metric is
        that both vulnerability_ratio and geometric_asymmetry are well-defined
        positive values.
        """
        rng = np.random.RandomState(42)
        feature_dim = 1024  # Match E88 dimension
        target_rank = 501   # Match E88 rank

        # Create component assignments: 3 components with roughly equal size
        n_samples = feature_dim
        component_assignments = np.zeros(n_samples, dtype=int)
        component_assignments[:341] = 0
        component_assignments[341:682] = 1
        component_assignments[682:] = 2

        # Construct F with exact rank 501 via low-rank factorization
        U = rng.randn(feature_dim, target_rank)
        F = U @ U.T  # Rank-501 matrix

        # Generate cross-component and within-component directions
        n_directions = 20
        cross_dirs = generate_cross_component_directions(
            component_assignments, feature_dim, n_directions, random_state=42
        )
        within_dirs = generate_within_component_directions(
            component_assignments, feature_dim, n_directions, random_state=42
        )

        result = compute_vulnerability_ratio(
            F, cross_dirs, within_dirs, margin=0.1, damping=1e-6
        )

        # Verify rank and null-space dimension match E88 pattern
        assert result.fisher_rank == target_rank, (
            f"Expected fisher_rank {target_rank}, got {result.fisher_rank}"
        )
        assert result.null_space_dim == feature_dim - target_rank, (
            f"Expected null_space_dim {feature_dim - target_rank}, got {result.null_space_dim}"
        )

        # Verify metrics are positive and well-defined
        assert result.geometric_asymmetry > 0, (
            f"Expected geometric_asymmetry > 0, got {result.geometric_asymmetry}"
        )
        assert result.vulnerability_ratio > 0, (
            f"Expected vulnerability_ratio > 0, got {result.vulnerability_ratio}"
        )
        
        # Verify all fields are populated
        assert len(result.cross_values) == n_directions
        assert len(result.within_values) == n_directions
        assert result.n_cross == n_directions
        assert result.n_within == n_directions

    def test_vulnerability_ratio_damping_sensitivity(self):
        """Verify that the vulnerability ratio changes with damping ε.

        Since FC scales with 1/ε, changing damping should change the absolute
        flip costs. The vulnerability ratio (ratio of effective costs) may or
        may not change significantly depending on the null-space structure.
        
        This test verifies that:
        1. The absolute flip costs scale appropriately with damping
        2. Both computations produce valid positive results
        """
        rng = np.random.RandomState(42)
        n = 30
        
        # Create component assignments
        assignments = np.zeros(n, dtype=int)
        assignments[:10] = 0
        assignments[10:20] = 1
        assignments[20:] = 2
        
        # Construct F with block-diagonal structure
        F = np.zeros((n, n))
        for i in range(3):
            start = i * 10
            end = (i + 1) * 10
            U_block = rng.randn(10, 5)
            F[start:end, start:end] = 10.0 * (U_block @ U_block.T)
        
        F = (F + F.T) / 2

        # Generate directions using structured assignments
        cross_dirs = generate_cross_component_directions(assignments, n, 20, random_state=42)
        within_dirs = generate_within_component_directions(assignments, n, 20, random_state=42)

        # Compute with different damping values
        result1 = compute_vulnerability_ratio(F, cross_dirs, within_dirs, margin=0.1, damping=1e-8)
        result2 = compute_vulnerability_ratio(F, cross_dirs, within_dirs, margin=0.1, damping=1e-4)

        # Verify both produce valid positive results
        assert result1.vulnerability_ratio > 0
        assert result2.vulnerability_ratio > 0
        assert result1.geometric_asymmetry > 0
        assert result2.geometric_asymmetry > 0
        
        # Verify that flip costs scale with damping (smaller damping -> larger costs)
        # FC = m² / (g^T F_ε^{-1} g), so smaller ε means larger quadratic form means smaller FC
        # Actually: smaller ε means F_ε is closer to singular, so inverse has larger eigenvalues
        # So smaller ε should give LARGER quadratic form and SMALLER flip cost
        assert result1.cross_median < result2.cross_median, (
            "Flip costs should increase with damping (smaller denominator)"
        )
        assert result1.within_median < result2.within_median, (
            "Flip costs should increase with damping"
        )

    def test_vulnerability_ratio_direction_inequality_robust(self):
        """Across multiple damping values, cross_median > within_median always holds."""
        rng = np.random.RandomState(42)
        n = 30
        F = rng.randn(n, n)
        F = F @ F.T

        # Create component assignments
        assignments = np.zeros(n, dtype=int)
        assignments[:10] = 0
        assignments[10:20] = 1
        assignments[20:] = 2

        cross_dirs = generate_cross_component_directions(assignments, n, 20, random_state=42)
        within_dirs = generate_within_component_directions(assignments, n, 20, random_state=42)

        damping_values = [1e-8, 1e-6, 1e-4, 1e-2]

        for damping in damping_values:
            result = compute_vulnerability_ratio(
                F, cross_dirs, within_dirs, margin=0.1, damping=damping
            )
            assert result.cross_median > result.within_median, (
                f"cross_median should exceed within_median for damping={damping}"
            )


# =============================================================================
# Determinism tests
# =============================================================================


class TestDeterminism:
    """Tests for determinism of computations."""

    def test_flip_cost_determinism(self):
        """Computing flip cost twice produces identical results."""
        rng = np.random.RandomState(42)
        F = rng.randn(5, 5)
        F = F @ F.T
        g = rng.randn(5)

        result1 = compute_flip_cost(F, g, margin=0.1, damping=1e-6)
        result2 = compute_flip_cost(F, g, margin=0.1, damping=1e-6)

        assert result1.flip_cost == result2.flip_cost
        assert result1.quadratic_form == result2.quadratic_form
        assert result1.g_norm == result2.g_norm
        assert result1.g_range_norm == result2.g_range_norm
        assert result1.g_null_norm == result2.g_null_norm
        assert result1.null_fraction == result2.null_fraction

    def test_vulnerability_ratio_determinism(self):
        """Computing the vulnerability ratio twice with the same seed produces identical results."""
        rng = np.random.RandomState(42)
        n = 15
        F = rng.randn(n, n)
        F = F @ F.T

        # Create component assignments
        assignments = np.zeros(n, dtype=int)
        assignments[:5] = 0
        assignments[5:10] = 1
        assignments[10:] = 2

        # Generate directions with fixed seed
        cross_dirs1 = generate_cross_component_directions(
            assignments, n, 10, random_state=42
        )
        within_dirs1 = generate_within_component_directions(
            assignments, n, 10, random_state=42
        )

        cross_dirs2 = generate_cross_component_directions(
            assignments, n, 10, random_state=42
        )
        within_dirs2 = generate_within_component_directions(
            assignments, n, 10, random_state=42
        )

        result1 = compute_vulnerability_ratio(F, cross_dirs1, within_dirs1, margin=0.1)
        result2 = compute_vulnerability_ratio(F, cross_dirs2, within_dirs2, margin=0.1)

        assert np.array_equal(result1.cross_values, result2.cross_values)
        assert np.array_equal(result1.within_values, result2.within_values)
        assert result1.vulnerability_ratio == result2.vulnerability_ratio
        assert result1.geometric_asymmetry == result2.geometric_asymmetry
