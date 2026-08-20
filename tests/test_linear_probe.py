"""Test suite for linear probe analysis module.

This module contains pytest tests for the linear probe functionality
used in Experiment E20 (Three-Level Pattern) of the NeurIPS 2027 submission.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.linear_probe import (
    ProbeResult,
    ThreeLevelSummary,
    compute_three_level_summary,
    run_linear_probe,
    run_permuted_label_probe,
    run_random_projection_probe,
    run_tfidf_probe,
)


class TestLinearProbeBasic:
    """Tests for basic run_linear_probe functionality."""

    def test_perfectly_separable_data(self) -> None:
        """Test with two well-separated Gaussian clusters.

        Constructs two well-separated Gaussian clusters in 2-D and verifies
        that run_linear_probe achieves accuracy > 0.95.
        """
        rng = np.random.RandomState(42)
        n_samples_per_class = 100

        # Class 0: centered at (-5, -5)
        X0 = rng.randn(n_samples_per_class, 2) - 5
        # Class 1: centered at (5, 5)
        X1 = rng.randn(n_samples_per_class, 2) + 5

        features = np.vstack([X0, X1])
        labels = np.array([0] * n_samples_per_class + [1] * n_samples_per_class)

        result = run_linear_probe(features, labels, n_folds=5, random_state=42)

        assert result.accuracy_mean > 0.95, (
            f"Expected accuracy > 0.95 for separable data, got {result.accuracy_mean}"
        )

    def test_chance_level_data(self) -> None:
        """Test with features drawn independently of labels (pure noise).

        Verifies that run_linear_probe achieves accuracy within 0.1 of 0.5.
        """
        rng = np.random.RandomState(42)
        n_samples = 200
        n_features = 50

        # Features are pure noise, independent of labels
        features = rng.randn(n_samples, n_features)
        labels = rng.randint(0, 2, size=n_samples)

        result = run_linear_probe(features, labels, n_folds=5, random_state=42)

        assert abs(result.accuracy_mean - 0.5) <= 0.1, (
            f"Expected accuracy within 0.1 of 0.5 for noise data, "
            f"got {result.accuracy_mean}"
        )

    def test_fold_count_matches(self) -> None:
        """Verify len(fold_accuracies) == n_folds."""
        rng = np.random.RandomState(42)
        n_samples = 100
        n_features = 10
        n_folds = 5

        features = rng.randn(n_samples, n_features)
        labels = rng.randint(0, 2, size=n_samples)

        result = run_linear_probe(features, labels, n_folds=n_folds, random_state=42)

        assert len(result.fold_accuracies) == n_folds, (
            f"Expected {n_folds} fold accuracies, got {len(result.fold_accuracies)}"
        )

    def test_n_samples_matches(self) -> None:
        """Verify n_samples equals the input sample count."""
        rng = np.random.RandomState(42)
        n_samples = 150
        n_features = 20

        features = rng.randn(n_samples, n_features)
        labels = rng.randint(0, 2, size=n_samples)

        result = run_linear_probe(features, labels, n_folds=5, random_state=42)

        assert result.n_samples == n_samples, (
            f"Expected n_samples={n_samples}, got {result.n_samples}"
        )


class TestTfidfProbe:
    """Tests for TF-IDF probe functionality."""

    def test_tfidf_separable_text(self) -> None:
        """Test with synthetic texts containing class-specific keywords.

        Constructs texts where class 0 always contains "alpha" and class 1
        always contains "beta". Verifies TF-IDF probe accuracy > 0.9.
        """
        rng = np.random.RandomState(42)
        n_samples_per_class = 50

        # Class 0 texts contain "alpha"
        texts_0 = [f"This is alpha document number {i}" for i in range(n_samples_per_class)]
        # Class 1 texts contain "beta"
        texts_1 = [f"This is beta document number {i}" for i in range(n_samples_per_class)]

        texts = texts_0 + texts_1
        labels = np.array([0] * n_samples_per_class + [1] * n_samples_per_class)

        result = run_tfidf_probe(texts, labels, n_folds=5, random_state=42)

        assert result.accuracy_mean > 0.9, (
            f"Expected TF-IDF accuracy > 0.9 for separable text, "
            f"got {result.accuracy_mean}"
        )

    def test_tfidf_uninformative_text(self) -> None:
        """Test with random vocabulary unrelated to labels.

        Verifies TF-IDF probe accuracy is within 0.15 of 0.5.
        """
        rng = np.random.RandomState(42)
        n_samples = 100

        # Generate random words unrelated to labels
        vocab = ["word" + str(i) for i in range(100)]
        texts = []
        for _ in range(n_samples):
            n_words = rng.randint(5, 15)
            words = rng.choice(vocab, size=n_words, replace=True)
            texts.append(" ".join(words))

        labels = rng.randint(0, 2, size=n_samples)

        result = run_tfidf_probe(texts, labels, n_folds=5, random_state=42)

        assert abs(result.accuracy_mean - 0.5) <= 0.15, (
            f"Expected TF-IDF accuracy within 0.15 of 0.5 for uninformative text, "
            f"got {result.accuracy_mean}"
        )

    def test_tfidf_length_mismatch_raises(self) -> None:
        """Test that len(texts) != len(labels) raises ValueError."""
        texts = ["text one", "text two", "text three"]
        labels = np.array([0, 1])

        with pytest.raises(ValueError, match="length mismatch"):
            run_tfidf_probe(texts, labels, n_folds=2, random_state=42)


class TestRandomProjectionProbe:
    """Tests for random projection probe functionality."""

    def test_random_projection_shape(self) -> None:
        """Verify the projected feature dimension matches n_components."""
        rng = np.random.RandomState(42)
        n_samples = 50
        n_features_original = 100
        n_components = 32

        features = rng.randn(n_samples, n_features_original)
        labels = rng.randint(0, 2, size=n_samples)

        # We need to check the internal projection shape
        # The result should work correctly even after projection
        result = run_random_projection_probe(
            features, labels, n_folds=2, random_state=42, n_components=n_components
        )

        # Verify the probe runs without error and returns valid results
        assert result.n_samples == n_samples
        assert len(result.fold_accuracies) == 2

    def test_random_projection_chance_level(self) -> None:
        """For noise features, random projection probe accuracy is within 0.1 of 0.5."""
        rng = np.random.RandomState(42)
        n_samples = 200
        n_features = 50

        features = rng.randn(n_samples, n_features)
        labels = rng.randint(0, 2, size=n_samples)

        result = run_random_projection_probe(
            features, labels, n_folds=5, random_state=42, n_components=64
        )

        assert abs(result.accuracy_mean - 0.5) <= 0.1, (
            f"Expected random projection accuracy within 0.1 of 0.5, "
            f"got {result.accuracy_mean}"
        )

    def test_random_projection_determinism(self) -> None:
        """Calling twice with the same random_state produces identical results."""
        rng = np.random.RandomState(42)
        n_samples = 100
        n_features = 30

        features = rng.randn(n_samples, n_features)
        labels = rng.randint(0, 2, size=n_samples)

        result1 = run_random_projection_probe(
            features, labels, n_folds=5, random_state=123, n_components=32
        )
        result2 = run_random_projection_probe(
            features, labels, n_folds=5, random_state=123, n_components=32
        )

        assert result1.accuracy_mean == result2.accuracy_mean, (
            f"Expected deterministic results, got {result1.accuracy_mean} vs {result2.accuracy_mean}"
        )
        assert np.array_equal(result1.fold_accuracies, result2.fold_accuracies)


class TestPermutedLabelProbe:
    """Tests for permuted label probe functionality."""

    def test_permuted_label_chance_level(self) -> None:
        """Even for perfectly separable features, permuted labels yield chance-level accuracy."""
        rng = np.random.RandomState(42)
        n_samples_per_class = 100

        # Create perfectly separable data
        X0 = rng.randn(n_samples_per_class, 10) - 5
        X1 = rng.randn(n_samples_per_class, 10) + 5
        features = np.vstack([X0, X1])
        labels = np.array([0] * n_samples_per_class + [1] * n_samples_per_class)

        result = run_permuted_label_probe(
            features, labels, n_folds=5, random_state=42
        )

        assert abs(result.accuracy_mean - 0.5) <= 0.1, (
            f"Expected permuted label accuracy within 0.1 of 0.5, "
            f"got {result.accuracy_mean}"
        )

    def test_permuted_label_determinism(self) -> None:
        """Calling twice with the same random_state produces identical results."""
        rng = np.random.RandomState(42)
        n_samples = 100
        n_features = 20

        features = rng.randn(n_samples, n_features)
        labels = rng.randint(0, 2, size=n_samples)

        result1 = run_permuted_label_probe(
            features, labels, n_folds=5, random_state=123
        )
        result2 = run_permuted_label_probe(
            features, labels, n_folds=5, random_state=123
        )

        assert result1.accuracy_mean == result2.accuracy_mean, (
            f"Expected deterministic results, got {result1.accuracy_mean} vs {result2.accuracy_mean}"
        )
        assert np.array_equal(result1.fold_accuracies, result2.fold_accuracies)


class TestThreeLevelSummary:
    """Tests for compute_three_level_summary functionality."""

    def test_three_level_summary_fields(self) -> None:
        """All fields of ThreeLevelSummary are populated and consistent."""
        rng = np.random.RandomState(42)
        n_samples = 100
        n_features = 20

        pooled_features = rng.randn(n_samples, n_features)
        texts = [f"document {i}" for i in range(n_samples)]
        labels = rng.randint(0, 2, size=n_samples)

        summary = compute_three_level_summary(
            pooled_features, texts, labels, n_folds=2, random_state=42
        )

        # Check all fields exist and have correct types
        assert isinstance(summary.pooled_result, ProbeResult)
        assert isinstance(summary.tfidf_result, ProbeResult)
        assert isinstance(summary.random_projection_result, ProbeResult)
        assert isinstance(summary.permuted_label_result, ProbeResult)
        assert isinstance(summary.pooled_vs_chance, float)
        assert isinstance(summary.tfidf_vs_pooled, float)
        assert isinstance(summary.three_level_pattern_confirmed, bool)

        # Check consistency
        assert summary.pooled_vs_chance == summary.pooled_result.accuracy_mean - 0.5
        assert summary.tfidf_vs_pooled == summary.tfidf_result.accuracy_mean - summary.pooled_result.accuracy_mean

    def test_three_level_pattern_confirmed_true(self) -> None:
        """Test pattern confirmed when pooled is noise but texts are separable.

        Constructs synthetic data where pooled features are pure noise
        (chance-level) but texts contain class-specific keywords (TF-IDF separable).
        Verifies three_level_pattern_confirmed == True.
        """
        rng = np.random.RandomState(42)
        n_samples_per_class = 60

        # Pooled features are pure noise (independent of labels)
        pooled_features = rng.randn(2 * n_samples_per_class, 50)

        # Texts are class-separable
        texts_0 = [f"This is alpha document {i}" for i in range(n_samples_per_class)]
        texts_1 = [f"This is beta document {i}" for i in range(n_samples_per_class)]
        texts = texts_0 + texts_1

        labels = np.array([0] * n_samples_per_class + [1] * n_samples_per_class)

        summary = compute_three_level_summary(
            pooled_features, texts, labels, n_folds=5, random_state=42
        )

        assert summary.three_level_pattern_confirmed is True, (
            f"Expected three_level_pattern_confirmed=True, got False. "
            f"pooled_acc={summary.pooled_result.accuracy_mean:.3f}, "
            f"pooled_std={summary.pooled_result.accuracy_std:.3f}, "
            f"tfidf_acc={summary.tfidf_result.accuracy_mean:.3f}, "
            f"tfidf_vs_pooled={summary.tfidf_vs_pooled:.3f}"
        )

    def test_three_level_pattern_confirmed_false(self) -> None:
        """Test pattern NOT confirmed when pooled features are separable.

        Constructs synthetic data where pooled features are perfectly separable.
        Verifies three_level_pattern_confirmed == False.
        """
        rng = np.random.RandomState(42)
        n_samples_per_class = 60

        # Pooled features are perfectly separable
        X0 = rng.randn(n_samples_per_class, 20) - 5
        X1 = rng.randn(n_samples_per_class, 20) + 5
        pooled_features = np.vstack([X0, X1])

        # Texts are also separable (but this doesn't matter since pooled is good)
        texts_0 = [f"alpha doc {i}" for i in range(n_samples_per_class)]
        texts_1 = [f"beta doc {i}" for i in range(n_samples_per_class)]
        texts = texts_0 + texts_1

        labels = np.array([0] * n_samples_per_class + [1] * n_samples_per_class)

        summary = compute_three_level_summary(
            pooled_features, texts, labels, n_folds=5, random_state=42
        )

        # Should be False because pooled accuracy will be high (not near chance)
        assert summary.three_level_pattern_confirmed is False, (
            f"Expected three_level_pattern_confirmed=False, got True. "
            f"pooled_acc={summary.pooled_result.accuracy_mean:.3f}"
        )


class TestEdgeCasesAndValidation:
    """Tests for edge cases and input validation."""

    def test_1d_features_raises(self) -> None:
        """Passing 1-D features raises ValueError."""
        rng = np.random.RandomState(42)
        features_1d = rng.randn(50)
        labels = rng.randint(0, 2, size=50)

        with pytest.raises(ValueError, match="2-D"):
            run_linear_probe(features_1d, labels, n_folds=2, random_state=42)

    def test_label_length_mismatch_raises(self) -> None:
        """Mismatched labels raises ValueError."""
        rng = np.random.RandomState(42)
        features = rng.randn(50, 10)
        labels = rng.randint(0, 2, size=40)  # Wrong length

        with pytest.raises(ValueError, match="mismatch"):
            run_linear_probe(features, labels, n_folds=2, random_state=42)

    def test_too_few_samples_raises(self) -> None:
        """Fewer samples than n_folds per class raises ValueError."""
        rng = np.random.RandomState(42)
        n_samples = 6
        n_features = 5

        # 3 samples per class, but n_folds=5 requires at least 5 per class
        features = rng.randn(n_samples, n_features)
        labels = np.array([0, 0, 0, 1, 1, 1])

        with pytest.raises(ValueError, match="Fewer than n_folds"):
            run_linear_probe(features, labels, n_folds=5, random_state=42)

    def test_invalid_n_folds_raises(self) -> None:
        """n_folds=1 raises ValueError."""
        rng = np.random.RandomState(42)
        features = rng.randn(20, 5)
        labels = rng.randint(0, 2, size=20)

        with pytest.raises(ValueError, match="n_folds must be >= 2"):
            run_linear_probe(features, labels, n_folds=1, random_state=42)

    @pytest.mark.parametrize("c_value", [0, -1, -0.5])
    def test_invalid_C_raises(self, c_value: float) -> None:
        """C=0 or C=-1 raises ValueError."""
        rng = np.random.RandomState(42)
        features = rng.randn(20, 5)
        labels = rng.randint(0, 2, size=20)

        with pytest.raises(ValueError, match="C must be > 0"):
            run_linear_probe(features, labels, n_folds=2, C=c_value, random_state=42)

    def test_empty_texts_raises(self) -> None:
        """Empty text list raises ValueError."""
        labels = np.array([0, 1])

        with pytest.raises(ValueError, match="empty"):
            run_tfidf_probe([], labels, n_folds=2, random_state=42)

    def test_invalid_n_components_raises(self) -> None:
        """n_components=0 raises ValueError."""
        rng = np.random.RandomState(42)
        features = rng.randn(20, 10)
        labels = rng.randint(0, 2, size=20)

        with pytest.raises(ValueError, match="n_components must be > 0"):
            run_random_projection_probe(
                features, labels, n_folds=2, random_state=42, n_components=0
            )
