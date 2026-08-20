"""DeBERTa-v3-base reward model adapter implementation."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, List, Optional

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.models.base import RewardModelAdapter


class DeBERTaV3Adapter(RewardModelAdapter):
    """Adapter for OpenAssistant/reward-model-deberta-v3-base.

    This adapter wraps the DeBERTa-v3-base reward model used in the paper's
    primary experiments (E16, E17, E20, E21, E22, E88). The model consists of
    a DeBERTa-v3-base backbone with a scalar reward head. Feature extraction
    uses mean pooling over the sequence dimension of the penultimate layer.

    Attributes:
        _model: The loaded HuggingFace model (None until load() is called).
        _tokenizer: The loaded tokenizer (None until load() is called).
        _device: The device where the model resides (None until load() is called).
        _loaded: Flag indicating whether the model has been loaded.
    """

    def __init__(self) -> None:
        """Initialize the DeBERTa-v3 adapter.

        The adapter is not loaded until load() is called.
        """
        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
        self._device: Optional[str] = None
        self._loaded: bool = False

    @property
    def model_name(self) -> str:
        """Return the HuggingFace model identifier.

        Returns:
            The model identifier string.
        """
        return "OpenAssistant/reward-model-deberta-v3-base"

    @property
    def architecture(self) -> str:
        """Return the architecture family string.

        Returns:
            The architecture family identifier.
        """
        return "deberta-v3-base"

    @property
    def feature_dim(self) -> int:
        """Return the dimensionality of the feature space.

        Returns:
            The feature dimension (1024 for DeBERTa-v3-base).
        """
        return 1024

    def _ensure_loaded(self) -> None:
        """Raise RuntimeError if the model is not loaded.

        Raises:
            RuntimeError: If load() has not been called.
        """
        if not self._loaded:
            raise RuntimeError(
                f"Model not loaded. Call load() before using {self.__class__.__name__}."
            )

    def load(self, device: str = "auto") -> None:
        """Load the model and tokenizer into memory.

        This method is idempotent: calling it multiple times has no effect
        after the first successful load.

        Args:
            device: Target device for loading the model. Options are:
                - "auto": Automatically select CUDA if available, else CPU.
                - "cpu": Force CPU-only execution.
                - "cuda": Force GPU execution.

        Raises:
            RuntimeError: If an invalid device string is provided.
        """
        if self._loaded:
            return

        # Resolve "auto" to CUDA if available, else CPU
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        elif device not in ("cpu", "cuda"):
            raise RuntimeError(
                f"Invalid device '{device}'. Must be 'auto', 'cpu', or 'cuda'."
            )

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)  # type: ignore[no-untyped-call]
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        )
        self._model = self._model.to(device)
        self._model.eval()
        self._device = device
        self._loaded = True

    def extract_features(
        self, prompts: Sequence[str], layer: str = "penultimate"
    ) -> np.ndarray:
        """Extract hidden-state features from the reward model.

        Tokenizes prompts and performs a forward pass with hidden states enabled.
        Extracts the penultimate layer hidden states and applies mean pooling
        over non-padding tokens using the attention mask.

        Args:
            prompts: A sequence of prompt strings to process.
            layer: Which hidden layer to extract features from. Only
                "penultimate" is supported currently.

        Returns:
            A float32 numpy array of shape (len(prompts), 1024) containing
            the mean-pooled features for each prompt.

        Raises:
            RuntimeError: If the model has not been loaded.
            ValueError: If layer is not "penultimate".
        """
        self._ensure_loaded()

        if layer != "penultimate":
            raise ValueError(
                f"Unsupported layer '{layer}'. Only 'penultimate' is supported."
            )

        assert self._model is not None
        assert self._tokenizer is not None

        with torch.no_grad():
            inputs = self._tokenizer(
                list(prompts),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self._device)

            outputs = self._model(
                **inputs, output_hidden_states=True, return_dict=True
            )

            hidden_states = outputs.hidden_states
            # Extract penultimate layer: hidden_states[-2]
            # Shape: (batch_size, seq_len, hidden_dim)
            penultimate_hidden = hidden_states[-2]

            # Get attention mask and expand for broadcasting
            # Shape: (batch_size, seq_len)
            attention_mask = inputs["attention_mask"]

            # Expand mask to match hidden state dimensions
            # Shape: (batch_size, seq_len, 1)
            mask_expanded = attention_mask.unsqueeze(-1).float()

            # Apply mean pooling over non-padding tokens
            # Sum over sequence dimension, then divide by number of valid tokens
            sum_hidden = (penultimate_hidden * mask_expanded).sum(dim=1)
            count_tokens = mask_expanded.sum(dim=1)

            # Avoid division by zero (shouldn't happen with valid input)
            count_tokens = count_tokens.clamp(min=1e-9)
            mean_pooled = sum_hidden / count_tokens

            # Convert to numpy float32
            features = mean_pooled.cpu().numpy().astype(np.float32)

        return features  # type: ignore[no-any-return]

    def predict_rewards(self, prompts: Sequence[str]) -> np.ndarray:
        """Predict scalar rewards for a sequence of prompts.

        Tokenizes prompts and performs a forward pass without hidden states.
        The model returns a scalar logit per prompt.

        Args:
            prompts: A sequence of prompt strings to evaluate.

        Returns:
            A float32 numpy array of shape (len(prompts),) containing
            the predicted reward values for each prompt.

        Raises:
            RuntimeError: If the model has not been loaded.
        """
        self._ensure_loaded()

        assert self._model is not None
        assert self._tokenizer is not None

        with torch.no_grad():
            inputs = self._tokenizer(
                list(prompts),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self._device)

            outputs = self._model(**inputs, return_dict=True)
            logits = outputs.logits

            # Extract scalar rewards: shape (batch_size,)
            rewards = logits.squeeze(-1).cpu().numpy().astype(np.float32)

        return rewards  # type: ignore[no-any-return]

    def compute_jacobian_rows(
        self, prompts: Sequence[str]
    ) -> np.ndarray:
        """Compute Jacobian rows d(reward)/d(readout_params) for each prompt.

        For each prompt, computes the gradient of the scalar reward output
        with respect to the readout layer parameters (final linear layer
        weights and bias). Uses torch.autograd.grad with create_graph=False
        and retain_graph=False.

        Processes prompts one at a time to avoid memory issues with large
        batch gradients. This method is used in E21 (VJP Range Finder) and
        E22 (Regime Hunt) for range-based audits.

        Args:
            prompts: A sequence of prompt strings to process.

        Returns:
            A float32 numpy array of shape (len(prompts), readout_param_count)
            containing the flattened Jacobian row for each prompt.

        Raises:
            RuntimeError: If the model has not been loaded.
        """
        self._ensure_loaded()

        assert self._model is not None
        assert self._tokenizer is not None

        # Get readout layer parameters
        # The classifier is typically model.classifier or similar
        # For DeBERTa classification models, it's usually model.classifier
        readout_params: List[torch.Tensor] = []

        # Find the classification head parameters
        # In HF AutoModelForSequenceClassification, this is typically in
        # model.classifier or model.score
        if hasattr(self._model, "classifier"):
            for param in self._model.classifier.parameters():
                readout_params.append(param)
        elif hasattr(self._model, "score"):
            for param in self._model.score.parameters():
                readout_params.append(param)
        else:
            # Fallback: look for any linear layer at the end
            for name, param in self._model.named_parameters():
                if "classifier" in name.lower() or "head" in name.lower():
                    readout_params.append(param)

        if not readout_params:
            raise RuntimeError("Could not find readout layer parameters.")

        # Calculate total parameter count
        param_count = sum(p.numel() for p in readout_params)

        jacobian_rows = []

        # Process one prompt at a time to save memory
        for prompt in prompts:
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self._device)

            # Forward pass
            outputs = self._model(**inputs, return_dict=True)
            reward = outputs.logits.squeeze(-1)  # Scalar

            # Compute gradients with respect to readout parameters
            grads = torch.autograd.grad(
                reward,
                readout_params,
                create_graph=False,
                retain_graph=False,
                allow_unused=True,
            )

            # Flatten and concatenate all gradients
            flat_grads = []
            for g in grads:
                if g is not None:
                    flat_grads.append(g.view(-1))

            jacobian_row = torch.cat(flat_grads).cpu().numpy().astype(np.float32)
            jacobian_rows.append(jacobian_row)

        return np.stack(jacobian_rows, axis=0)
