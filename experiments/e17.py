"""Experiment E17: Neural Manipulation Cost Audit.

This module implements the neural manipulation cost audit described in
Experiment E17 of the NeurIPS 2027 submission. It computes the Fisher
information matrix and uses conjugate gradient to solve for the minimum
manipulation cost.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.fisher.builder import build_fisher_matrix
from src.fisher.llf import compute_manipulation_cost
from src.models.deberta import DeBERTaV3Adapter
from src.utils.seed import seed_everything

EXPERIMENT_NAME: str = "Neural Manipulation Cost Audit"
EXPERIMENT_ID: str = "e17"


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


def run(config: dict[str, Any]) -> dict[str, Any]:
    """Run the E17 Neural Manipulation Cost Audit experiment.

    This function orchestrates the full E17 analysis:
    1. Load adapter, extract features for n_audit_prompts prompts.
    2. Build the within-component Fisher matrix via
       src.fisher.builder.build_fisher_matrix.
    3. Construct a constraint gradient g (use the mean feature difference
       direction).
    4. Compute the manipulation cost via
       src.fisher.llf.compute_manipulation_cost.

    Args:
        config: Configuration dictionary containing experiment parameters.
            Required keys: seed, n_audit_prompts, margin, damping,
            cg_tolerance, cg_max_iter, data.cache_dir, model.name.

    Returns:
        JSON-serializable dictionary with keys:
        - experiment_id: "e17"
        - experiment_name: "Neural Manipulation Cost Audit"
        - status: "completed" or "error"
        - results: Dict with mc_llf, vulnerability_score, etc.
        - expected_values: Paper-reported reference values.
    """
    try:
        seed_everything(config["seed"])

        # Load adapter
        adapter = _load_adapter(config)

        # Generate synthetic prompts and extract features
        n_prompts = config["n_audit_prompts"]
        prompts = [f"Audit prompt {i} for testing." for i in range(n_prompts)]
        features = adapter.extract_features(prompts, layer="penultimate")

        # Predict rewards for computing reward differences
        rewards = adapter.predict_rewards(prompts)

        # Build comparison pairs (consecutive pairs)
        n_samples = len(features)
        pairs = [(i, i + 1) for i in range(0, n_samples - 1, 2)]
        reward_diffs = np.array([rewards[i] - rewards[j] for i, j in pairs])

        # Build Fisher matrix
        fisher_result = build_fisher_matrix(features, pairs, reward_diffs)
        F = fisher_result.fisher_matrix

        # Construct constraint gradient g as mean feature difference direction
        # g = mean(phi(x_i) - phi(x_j)) over all pairs
        feature_diffs = features[pairs[:, 0]] - features[pairs[:, 1]]
        g = np.mean(feature_diffs, axis=0)

        # Normalize g to unit norm
        g_norm = np.linalg.norm(g)
        if g_norm > 0:
            g = g / g_norm

        # Compute manipulation cost
        margin = config["margin"]
        damping = config["damping"]
        cg_tol = config["cg_tolerance"]
        cg_max_iter = config["cg_max_iter"]

        mc_result = compute_manipulation_cost(
            F=F,
            g=g,
            margin=margin,
            damping=damping,
            tol=cg_tol,
            max_iter=cg_max_iter,
        )

        results = {
            "mc_llf": float(mc_result.mc_value),
            "vulnerability_score": float(mc_result.vulnerability_score),
            "kl_budget": float(mc_result.kl_budget),
            "fisher_rank": int(fisher_result.rank),
            "null_space_dim": int(fisher_result.null_space_dim),
            "cg_converged": bool(mc_result.cg_result.converged),
            "cg_iterations": int(mc_result.cg_result.n_iterations),
        }

        expected_values = {
            "V": 59.68,
            "MC_LLF": 0.0129,
            "KL_budget": 8.3e-5,
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
