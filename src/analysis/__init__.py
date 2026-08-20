"""Analysis module for feature-space vulnerability audit pipeline."""

from src.analysis.pca_rank import (
    PcaResult,
    compute_pca,
    effective_rank,
    explain_variance_at_rank,
)

__all__ = [
    "PcaResult",
    "compute_pca",
    "effective_rank",
    "explain_variance_at_rank",
]
