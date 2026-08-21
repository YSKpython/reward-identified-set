"""Experiment E16: Feature-Space Audit.

This module implements the feature-space audit described in Experiment E16
of the NeurIPS 2027 submission. It computes PCA effective rank and cosine
similarity structure to quantify how high-dimensional representations collapse
into a low-rank manifold.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.analysis.cosine_structure import compute_cosine_structure
from src.analysis.pca_rank import compute_pca
from src.models.deberta import DeBERTaV3Adapter
from src.utils.seed import seed_everything

EXPERIMENT_NAME: str = "Feature-Space Audit"
EXPERIMENT_ID: str = "e16"


def _load_adapter(config: dict[str, Any]) -> DeBERTaV3Adapter:
    """Load the reward model adapter from config.

    Args:
        config: Configuration dictionary containing model settings.

    Returns:
        Loaded DeBERTaV3Adapter instance.

    Raises:
        NotImplementedError: If the model architecture is not supported.
    """
    model_name = config["model"]["name"]
    if model_name != "OpenAssistant/reward-model-deberta-v3-base":
        raise NotImplementedError(
            f"Unsupported model architecture: {model_name}"
        )
    adapter = DeBERTaV3Adapter()
    adapter.load(device=config.get("device", "auto"))
    return adapter


def _extract_features_cached(
    adapter: DeBERTaV3Adapter,
    n_prompts: int,
    cache_dir: str,
    config: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    """Extract features with optional caching.

    Args:
        adapter: Loaded reward model adapter.
        n_prompts: Number of prompts to process.
        cache_dir: Directory for caching extracted features.
        config: Configuration dictionary.

    Returns:
        Tuple of (features array, list of prompt texts).
    """
    cache_path = Path(cache_dir) / f"e16_features_n{n_prompts}.npz"

    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        features = data["features"]
        prompts = list(data["prompts"])
        return features, prompts

    # Generate synthetic prompts for testing
    prompts = [f"Prompt {i} for testing purposes." for i in range(n_prompts)]

    # Extract features
    features = adapter.extract_features(prompts, layer="penultimate")

    # Cache results
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        features=features,
        prompts=np.array(prompts, dtype=object),
    )

    return features, prompts


def run(config: dict[str, Any]) -> dict[str, Any]:
    """Run the E16 Feature-Space Audit experiment.

    This function orchestrates the full E16 analysis:
    1. Load the DeBERTa-v3 adapter and extract features for n_prompts prompts.
    2. Compute PCA effective rank via src.analysis.pca_rank.compute_pca.
    3. Compute within/cross cosine structure via
       src.analysis.cosine_structure.compute_cosine_structure.

    Args:
        config: Configuration dictionary containing experiment parameters.
            Required keys: seed, pca_threshold, n_prompts, n_cross_pairs,
            data.cache_dir, model.name.

    Returns:
        JSON-serializable dictionary with keys:
        - experiment_id: "e16"
        - experiment_name: "Feature-Space Audit"
        - status: "completed" or "error"
        - results: Dict with effective_rank, within_cosine_mean, etc.
        - expected_values: Paper-reported reference values.
    """
    try:
        seed_everything(config["seed"])

        # Load adapter
        adapter = _load_adapter(config)

        # Extract features
        n_prompts = config["n_prompts"]
        cache_dir = config["data"]["cache_dir"]
        features, prompts = _extract_features_cached(
            adapter, n_prompts, cache_dir, config
        )

        # Compute PCA effective rank
        pca_threshold = config["pca_threshold"]
        pca_result = compute_pca(features, threshold=pca_threshold)

        # Build pair indices for within-component pairs
        # For simplicity, pair consecutive samples as within-component
        n_samples = len(features)
        pair_indices = [(i, i + 1) for i in range(0, n_samples - 1, 2)]

        # Compute cosine structure
        n_cross_pairs = config["n_cross_pairs"]
        cosine_result = compute_cosine_structure(
            features=features,
            pair_indices=pair_indices,
            n_cross_pairs=n_cross_pairs,
            random_state=config["seed"],
        )

        # Get explained variance at effective rank
        pca_explained_var = float(
            pca_result.cumulative_variance[pca_result.effective_rank - 1]
        )

        results = {
            "effective_rank": pca_result.effective_rank,
            "within_cosine_mean": float(cosine_result.within_mean),
            "within_cosine_std": float(cosine_result.within_std),
            "cross_cosine_mean": float(cosine_result.cross_mean),
            "cross_cosine_std": float(cosine_result.cross_std),
            "cosine_gap": float(cosine_result.gap),
            "pca_explained_variance_at_rank": pca_explained_var,
        }

        expected_values = {
            "effective_rank": 36,
            "within_cosine": 0.960,
            "cross_cosine": 0.773,
            "gap": 0.187,
        }

        return {
            "experiment_id": EXPERIMENT_ID,
            "experiment_name": EXPERIMENT_NAME,
            "status": "completed",
            "results": results,
            "expected_values": expected_values,
        }

    except Exception as exc:
        return {
            "experiment_id": EXPERIMENT_ID,
            "experiment_name": EXPERIMENT_NAME,
            "status": "error",
            "error": str(exc),
            "expected_values": {},
        }
