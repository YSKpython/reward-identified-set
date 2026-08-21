"""Abstract base class for reward model adapters."""

from __future__ import annotations

import abc
from collections.abc import Sequence

import numpy as np


class RewardModelAdapter(abc.ABC):
    """Abstract base class for RLHF reward model adapters.

    This protocol provides a uniform interface for feature extraction,
    reward prediction, and Jacobian computation across different reward
    model architectures (e.g., DeBERTa-v3-base, Llama-based RMs).

    The adapter pattern enables the same audit code (E16 PCA rank collapse,
    E20 linear probe, E88 Fisher audit) to run on any reward model that
    implements this interface.

    Attributes:
        model_name: HuggingFace model identifier.
        architecture: Architecture family string.
        feature_dim: Dimensionality of the feature space.
    """

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        """Return the HuggingFace model identifier.

        Returns:
            A string identifying the model, e.g.,
            "OpenAssistant/reward-model-deberta-v3-base".
        """
        pass

    @property
    @abc.abstractmethod
    def architecture(self) -> str:
        """Return the architecture family string.

        Returns:
            A string describing the architecture, e.g.,
            "deberta-v3-base" or "llama-3.1-8b-rm".
        """
        pass

    @property
    @abc.abstractmethod
    def feature_dim(self) -> int:
        """Return the dimensionality of the feature space.

        Returns:
            An integer representing the feature dimension, e.g., 1024.
        """
        pass

    @abc.abstractmethod
    def load(self, device: str = "auto") -> None:
        """Load model weights into memory.

        This method must be idempotent: calling it multiple times should
        have no effect after the first successful load.

        Args:
            device: Target device for loading the model. Options are:
                - "auto": Automatically select the best available device.
                - "cpu": Force CPU-only execution.
                - "cuda": Force GPU execution.
                - "tpu": Force TPU execution.
        """
        pass

    @abc.abstractmethod
    def extract_features(
        self, prompts: Sequence[str], layer: str = "penultimate"
    ) -> np.ndarray:
        """Extract hidden-state features from the reward model.

        Args:
            prompts: A sequence of prompt strings to process.
            layer: Which hidden layer to extract features from.
                Common options: "penultimate", "last", or layer index.

        Returns:
            A numpy array of shape (len(prompts), feature_dim) containing
            the extracted features for each prompt.
        """
        pass

    @abc.abstractmethod
    def predict_rewards(self, prompts: Sequence[str]) -> np.ndarray:
        """Predict scalar rewards for a sequence of prompts.

        Args:
            prompts: A sequence of prompt strings to evaluate.

        Returns:
            A numpy array of shape (len(prompts),) containing the
            predicted reward values for each prompt.
        """
        pass

    @abc.abstractmethod
    def compute_jacobian_rows(self, prompts: Sequence[str]) -> np.ndarray:
        """Compute Jacobian rows d(reward)/d(readout_params) for each prompt.

        This method is used in E21 (VJP Range Finder) and E22 (Regime Hunt)
        for range-based audits of the reward manipulation surface.

        Args:
            prompts: A sequence of prompt strings to process.

        Returns:
            A numpy array of shape (len(prompts), readout_param_count)
            containing the Jacobian row for each prompt.
        """
        pass

    def __repr__(self) -> str:
        """Return a string representation of the adapter.

        Returns:
            A string showing the model name and feature dimension.
        """
        return f"{self.__class__.__name__}(model_name={self.model_name!r}, feature_dim={self.feature_dim})"
