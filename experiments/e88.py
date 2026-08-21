"""Experiment E88: Real-World Feature-Space Audit.

This module implements the comprehensive feature-space audit described in
Experiment E88 of the NeurIPS 2027 submission. It computes vulnerability
ratios and geometric asymmetry between cross-component and within-component
manipulation directions.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.fisher.builder import build_fisher_matrix, fisher_rank
from src.metrics.vulnerability_ratio import (
    compute_vulnerability_ratio,
    generate_cross_component_directions,
    generate_within_component_directions,
)
from src.models.deberta import DeBERTaV3Adapter
from src.utils.seed import seed_everything

EXPERIMENT_NAME: str = "Real-World Feature-Space Audit"
EXPERIMENT_ID: str = "e88"


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
        raise NotImplementedError(f"Unsupported model architecture: {model_name}")
    adapter = DeBERTaV3Adapter()
    adapter.load(device=config.get("device", "auto"))
    return adapter


def run(config: dict[str, Any]) -> dict[str, Any]:
    """Run the E88 Real-World Feature-Space Audit experiment.

    This function orchestrates the full E88 analysis:
    1. Load adapter, extract features for n_prompts prompts.
    2. Build the within-component Fisher matrix via
       src.fisher.builder.build_fisher_matrix.
    3. Generate cross-component and within-component directions via
       src.metrics.vulnerability_ratio.generate_cross_component_directions
       and generate_within_component_directions.
    4. Compute the vulnerability ratio via
       src.metrics.vulnerability_ratio.compute_vulnerability_ratio.

    Args:
        config: Configuration dictionary containing experiment parameters.
            Required keys: seed, n_prompts, margin, damping, n_cross_directions,
            n_within_directions, data.cache_dir, model.name.

    Returns:
        JSON-serializable dictionary with keys:
        - experiment_id: "e88"
        - experiment_name: "Real-World Feature-Space Audit"
        - status: "completed" or "error"
        - results: Dict with vulnerability_ratio, geometric_asymmetry, etc.
        - expected_values: Paper-reported reference values.
    """
    try:
        seed_everything(config["seed"])

        # Load adapter
        adapter = _load_adapter(config)

        # Generate synthetic prompts and extract features
        n_prompts = config["n_prompts"]
        prompts = [f"E88 audit prompt {i}." for i in range(n_prompts)]
        features = adapter.extract_features(prompts, layer="penultimate")

        # Predict rewards
        rewards = adapter.predict_rewards(prompts)

        # Create component assignments for generating directions
        # Assign each prompt to a component (e.g., 10 components)
        n_components = 10
        component_assignments = np.array([i % n_components for i in range(n_prompts)])

        # Build comparison pairs for Fisher construction
        n_samples = len(features)
        pairs_list = [(i, i + 1) for i in range(0, n_samples - 1, 2)]
        reward_diffs = np.array([rewards[i] - rewards[j] for i, j in pairs_list])

        # Build Fisher matrix
        fisher_result = build_fisher_matrix(features, pairs_list, reward_diffs)
        F = fisher_result.fisher_matrix  # noqa: N806

        # Generate cross-component and within-component directions
        n_cross = config["n_cross_directions"]
        n_within = config["n_within_directions"]
        feature_dim = features.shape[1]

        cross_directions = generate_cross_component_directions(
            component_assignments=component_assignments,
            feature_dim=feature_dim,
            n_directions=n_cross,
            random_state=config["seed"],
        )

        within_directions = generate_within_component_directions(
            component_assignments=component_assignments,
            feature_dim=feature_dim,
            n_directions=n_within,
            random_state=config["seed"],
        )

        # Compute vulnerability ratio
        margin = config["margin"]
        damping = config["damping"]

        vr_result = compute_vulnerability_ratio(
            F=F,
            cross_directions=cross_directions,
            within_directions=within_directions,
            margin=margin,
            damping=damping,
        )

        # Compute Fisher rank
        fisher_rank_val = fisher_rank(F)

        results = {
            "vulnerability_ratio": float(vr_result.vulnerability_ratio),
            "geometric_asymmetry": float(vr_result.geometric_asymmetry),
            "cross_median": float(vr_result.cross_median),
            "within_median": float(vr_result.within_median),
            "fisher_rank": int(fisher_result.rank),
            "null_space_dim": int(fisher_result.null_space_dim),
            "feature_dim": int(feature_dim),
            "n_cross": int(n_cross),
            "n_within": int(n_within),
        }

        expected_values = {
            "fisher_rank": 501,
            "null_space": 523,
            "cross": 247.33,
            "within": 3.63,
            "asymmetry": 68,
            "vuln_ratio": 14.6,
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
