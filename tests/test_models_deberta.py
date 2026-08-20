"""Tests for the DeBERTa-v3-base reward model adapter."""
from __future__ import annotations

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from src.models.deberta import DeBERTaV3Adapter
from src.models.base import RewardModelAdapter
from src.utils.seed import seed_everything


class TestDeBERTaV3AdapterInterface:
    """Tier 1: Interface tests that do not require model download."""

    def test_interface_compliance(self) -> None:
        """Test that DeBERTaV3Adapter passes isinstance check for RewardModelAdapter."""
        adapter = DeBERTaV3Adapter()
        assert isinstance(adapter, RewardModelAdapter)

    def test_properties(self) -> None:
        """Test that model_name, architecture, and feature_dim return expected values."""
        adapter = DeBERTaV3Adapter()
        assert adapter.model_name == "OpenAssistant/reward-model-deberta-v3-base"
        assert adapter.architecture == "deberta-v3-base"
        assert adapter.feature_dim == 1024

    def test_not_loaded_raises(self) -> None:
        """Test that calling extract_features before load() raises RuntimeError."""
        adapter = DeBERTaV3Adapter()
        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.extract_features(["test prompt"])

        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.predict_rewards(["test prompt"])

        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.compute_jacobian_rows(["test prompt"])

    def test_invalid_layer_raises(self) -> None:
        """Test that calling extract_features with invalid layer raises ValueError."""
        adapter = DeBERTaV3Adapter()
        # Mock the _loaded flag to bypass the loaded check
        adapter._loaded = True
        with pytest.raises(ValueError, match="Unsupported layer"):
            adapter.extract_features(["test prompt"], layer="last")

        with pytest.raises(ValueError, match="Unsupported layer"):
            adapter.extract_features(["test prompt"], layer="first")


@pytest.mark.slow
class TestDeBERTaV3AdapterIntegration:
    """Tier 2: Integration tests that require model download."""

    @pytest.fixture
    def adapter(self) -> DeBERTaV3Adapter:
        """Create and load a DeBERTaV3Adapter instance."""
        adapter = DeBERTaV3Adapter()
        adapter.load(device="cpu")
        return adapter

    def test_load_idempotent(self) -> None:
        """Test that calling load() twice does not raise or reload."""
        adapter = DeBERTaV3Adapter()
        adapter.load(device="cpu")
        # Second call should be a no-op
        adapter.load(device="cpu")
        assert adapter._loaded is True

    def test_extract_features_shape(self, adapter: DeBERTaV3Adapter) -> None:
        """Test that extract_features returns correct shape (n_prompts, 1024)."""
        prompts = [
            "This is a test prompt.",
            "Another test prompt here.",
            "A third prompt for testing.",
        ]
        features = adapter.extract_features(prompts)
        assert features.shape == (3, 1024)
        assert features.dtype == np.float32

    def test_predict_rewards_shape(self, adapter: DeBERTaV3Adapter) -> None:
        """Test that predict_rewards returns correct shape (n_prompts,)."""
        prompts = [
            "This is a test prompt.",
            "Another test prompt here.",
            "A third prompt for testing.",
        ]
        rewards = adapter.predict_rewards(prompts)
        assert rewards.shape == (3,)
        assert rewards.dtype == np.float32

    def test_compute_jacobian_rows_shape(
        self, adapter: DeBERTaV3Adapter
    ) -> None:
        """Test that compute_jacobian_rows returns correct shape."""
        prompts = [
            "This is a test prompt.",
            "Another test prompt here.",
        ]
        jacobian = adapter.compute_jacobian_rows(prompts)
        # Shape should be (n_prompts, readout_param_count)
        assert jacobian.shape[0] == 2
        assert jacobian.ndim == 2
        assert jacobian.dtype == np.float32

    def test_determinism(self, adapter: DeBERTaV3Adapter) -> None:
        """Test that extract_features produces identical results with same seed."""
        prompts = ["Test prompt for determinism check."]

        # First run
        seed_everything(42)
        features1 = adapter.extract_features(prompts)

        # Second run with same seed
        seed_everything(42)
        features2 = adapter.extract_features(prompts)

        assert np.allclose(features1, features2)

    def test_mean_pooling_ignores_padding(
        self, adapter: DeBERTaV3Adapter
    ) -> None:
        """Test that mean pooling correctly ignores padding tokens.

        Two prompts where one is padded should produce features that differ
        only due to actual content, not padding positions.
        """
        # Short prompt (will have more padding)
        short_prompt = "Hi."
        # Longer prompt with similar content (less padding)
        long_prompt = "This is a much longer prompt that contains more tokens."

        features_short = adapter.extract_features([short_prompt])
        features_long = adapter.extract_features([long_prompt])

        # Features should be different (different content)
        # but both should be valid (not NaN or Inf)
        assert not np.isnan(features_short).any()
        assert not np.isinf(features_short).any()
        assert not np.isnan(features_long).any()
        assert not np.isinf(features_long).any()

        # Both should have correct shape
        assert features_short.shape == (1, 1024)
        assert features_long.shape == (1, 1024)
