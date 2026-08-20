"""Within-component Fisher Information Matrix builder for RLHF reward models.

This module provides functions to construct the Fisher information matrix
from pre-extracted features, comparison pair indices, and predicted reward
differences under the Bradley-Terry model.
"""

from src.fisher.builder import (
    FisherBuildResult,
    logistic,
    compute_edge_curvatures,
    build_fisher_matrix,
    fisher_rank,
    fisher_null_space_dim,
    apply_tikhonov,
    check_psd,
)

__all__ = [
    "FisherBuildResult",
    "logistic",
    "compute_edge_curvatures",
    "build_fisher_matrix",
    "fisher_rank",
    "fisher_null_space_dim",
    "apply_tikhonov",
    "check_psd",
]
