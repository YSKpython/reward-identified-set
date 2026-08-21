"""Metrics for feature-space vulnerability audit.

This module provides metrics for computing flip costs, vulnerability ratios,
and geometric asymmetry in RLHF reward model audits.
"""

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

__all__ = [
    "FlipCostResult",
    "compute_flip_cost",
    "compute_flip_cost_batch",
    "VulnerabilityRatioResult",
    "compute_vulnerability_ratio",
    "generate_cross_component_directions",
    "generate_within_component_directions",
    "geometric_asymmetry",
]
