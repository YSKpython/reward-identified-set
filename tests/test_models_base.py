"""Tests for the RewardModelAdapter abstract base class."""
from __future__ import annotations

import pytest
import numpy as np
from collections.abc import Sequence

from src.models.base import RewardModelAdapter


class MockRewardModel(RewardModelAdapter):
    """Mock implementation of RewardModelAdapter for testing.

    This class provides deterministic outputs:
    - features: array of ones
    - rewards: arange based on prompt index
    - jacobian: array of twos
    """

    def __init__(
        self,
        model_name: str = "mock/reward-model",
        architecture: str = "mock-base",
        feature_dim: int = 512,
    ) -> None:
        """Initialize the mock reward model.

        Args:
            model_name: HuggingFace model identifier.
            architecture: Architecture family string.
            feature_dim: Dimensionality of the feature space.
        """
        self._model_name = model_name
        self._architecture = architecture
        self._feature_dim = feature_dim
        self._loaded = False

    @property
    def model_name(self) -> str:
        """Return the HuggingFace model identifier."""
        return self._model_name

    @property
    def architecture(self) -> str:
        """Return the architecture family string."""
        return self._architecture

    @property
    def feature_dim(self) -> int:
        """Return the dimensionality of the feature space."""
        return self._feature_dim

    def load(self, device: str = "auto") -> None:
        """Load model weights (mock implementation).

        This method is idempotent.

        Args:
            device: Target device (ignored in mock).
        """
        if not self._loaded:
            self._loaded = True

    def extract_features(
        self, prompts: Sequence[str], layer: str = "penultimate"
    ) -> np.ndarray:
        """Extract features (mock implementation).

        Returns an array of ones with shape (len(prompts), feature_dim).

        Args:
            prompts: A sequence of prompt strings.
            layer: Layer specification (ignored in mock).

        Returns:
            Numpy array of shape (len(prompts), feature_dim).
        """
        n_prompts = len(prompts)
        return np.ones((n_prompts, self._feature_dim), dtype=np.float32)

    def predict_rewards(self, prompts: Sequence[str]) -> np.ndarray:
        """Predict rewards (mock implementation).

        Returns arange based on prompt indices.

        Args:
            prompts: A sequence of prompt strings.

        Returns:
            Numpy array of shape (len(prompts),).
        """
        n_prompts = len(prompts)
        return np.arange(n_prompts, dtype=np.float32)

    def compute_jacobian_rows(
        self, prompts: Sequence[str]
    ) -> np.ndarray:
        """Compute Jacobian rows (mock implementation).

        Returns an array of twos with shape (len(prompts), readout_param_count).

        Args:
            prompts: A sequence of prompt strings.

        Returns:
            Numpy array of shape (len(prompts), readout_param_count).
        """
        n_prompts = len(prompts)
        # Mock readout_param_count as feature_dim + 1 (bias term)
        readout_param_count = self._feature_dim + 1
        return 2.0 * np.ones(
            (n_prompts, readout_param_count), dtype=np.float32
        )


class TestRewardModelAdapter:
    """Test suite for RewardModelAdapter and its implementations."""

    def test_mock_is_instance(self) -> None:
        """Test that MockRewardModel passes isinstance check."""
        model = MockRewardModel()
        assert isinstance(model, RewardModelAdapter)

    def test_cannot_instantiate_abc(self) -> None:
        """Test that instantiating RewardModelAdapter directly raises TypeError."""
        with pytest.raises(TypeError):
            RewardModelAdapter()

    @pytest.mark.parametrize("n_prompts", [1, 5, 10])
    def test_extract_features_shape(self, n_prompts: int) -> None:
        """Test that extract_features returns the correct shape."""
        model = MockRewardModel(feature_dim=512)
        prompts = [f"prompt_{i}" for i in range(n_prompts)]
        features = model.extract_features(prompts)

        assert features.shape == (n_prompts, 512)
        assert features.dtype == np.float32

    @pytest.mark.parametrize("n_prompts", [1, 5, 10])
    def test_predict_rewards_shape(self, n_prompts: int) -> None:
        """Test that predict_rewards returns the correct shape."""
        model = MockRewardModel()
        prompts = [f"prompt_{i}" for i in range(n_prompts)]
        rewards = model.predict_rewards(prompts)

        assert rewards.shape == (n_prompts,)
        assert rewards.dtype == np.float32

    @pytest.mark.parametrize("n_prompts", [1, 5, 10])
    def test_compute_jacobian_rows_shape(self, n_prompts: int) -> None:
        """Test that compute_jacobian_rows returns the correct shape."""
        model = MockRewardModel(feature_dim=512)
        prompts = [f"prompt_{i}" for i in range(n_prompts)]
        jacobian = model.compute_jacobian_rows(prompts)

        # readout_param_count = feature_dim + 1 (bias term)
        expected_params = 512 + 1
        assert jacobian.shape == (n_prompts, expected_params)
        assert jacobian.dtype == np.float32

    def test_repr_contains_model_name(self) -> None:
        """Test that __repr__ contains the model name."""
        model = MockRewardModel(model_name="test/model-name")
        repr_str = repr(model)

        assert "test/model-name" in repr_str
        assert "MockRewardModel" in repr_str

    def test_load_idempotent(self) -> None:
        """Test that load() is idempotent."""
        model = MockRewardModel()
        assert not model._loaded

        model.load()
        assert model._loaded

        # Calling again should not change state
        model.load()
        assert model._loaded

    @pytest.mark.parametrize("device", ["auto", "cpu", "cuda", "tpu"])
    def test_load_accepts_device_strings(self, device: str) -> None:
        """Test that load() accepts various device strings."""
        model = MockRewardModel()
        # Should not raise
        model.load(device=device)
        assert model._loaded

    def test_extract_features_deterministic(self) -> None:
        """Test that extract_features produces deterministic output."""
        model = MockRewardModel(feature_dim=256)
        prompts = ["test prompt 1", "test prompt 2"]

        features1 = model.extract_features(prompts)
        features2 = model.extract_features(prompts)

        np.testing.assert_array_equal(features1, features2)

    def test_predict_rewards_deterministic(self) -> None:
        """Test that predict_rewards produces deterministic output."""
        model = MockRewardModel()
        prompts = ["test prompt 1", "test prompt 2", "test prompt 3"]

        rewards1 = model.predict_rewards(prompts)
        rewards2 = model.predict_rewards(prompts)

        np.testing.assert_array_equal(rewards1, rewards2)

    def test_compute_jacobian_rows_deterministic(self) -> None:
        """Test that compute_jacobian_rows produces deterministic output."""
        model = MockRewardModel(feature_dim=128)
        prompts = ["test prompt"]

        jacobian1 = model.compute_jacobian_rows(prompts)
        jacobian2 = model.compute_jacobian_rows(prompts)

        np.testing.assert_array_equal(jacobian1, jacobian2)

    @pytest.mark.parametrize(
        "feature_dim,expected_jacobian_cols",
        [(64, 65), (128, 129), (512, 513), (1024, 1025)],
    )
    def test_jacobian_param_count(
        self, feature_dim: int, expected_jacobian_cols: int
    ) -> None:
        """Test that Jacobian has correct number of columns."""
        model = MockRewardModel(feature_dim=feature_dim)
        prompts = ["single prompt"]
        jacobian = model.compute_jacobian_rows(prompts)

        assert jacobian.shape == (1, expected_jacobian_cols)
