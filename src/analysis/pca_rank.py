"""PCA effective rank computation for feature-space vulnerability audit.

This module implements the PCA-based effective rank analysis described in
Experiment E16 of the NeurIPS 2027 submission. It computes the minimum number
of principal components required to explain a given threshold of variance in
a pre-extracted feature matrix.

The effective rank metric quantifies how high-dimensional representations
collapse into a low-rank manifold, which is critical for understanding
feature-space vulnerabilities in RLHF reward models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np
from sklearn.decomposition import PCA


@dataclass
class PcaResult:
    """Container for PCA analysis results.

    Attributes
    ----------
    effective_rank : int
        Minimum number of components reaching the variance threshold.
    explained_variance_ratio : np.ndarray
        Per-component explained variance ratio. Shape: (n_components,).
    cumulative_variance : np.ndarray
        Cumulative sum of explained variance ratio. Shape: (n_components,).
    components : np.ndarray
        Principal component axes. Shape: (n_components, n_features).
    threshold : float
        The variance threshold used (e.g., 0.95).
    n_samples : int
        Number of samples in the input.
    n_features : int
        Number of features in the input.
    total_variance_explained : float
        Cumulative variance at the effective rank.
    """

    effective_rank: int
    explained_variance_ratio: np.ndarray
    cumulative_variance: np.ndarray
    components: np.ndarray
    threshold: float
    n_samples: int
    n_features: int
    total_variance_explained: float


def compute_pca(
    features: np.ndarray,
    threshold: float = 0.95,
    random_state: int = 42,
) -> PcaResult:
    """Run PCA on the input feature matrix and compute effective rank.

    This function centers the data by subtracting the mean, then performs
    exact PCA using scikit-learn's `svd_solver="full"`. The effective rank
    is defined as the minimum number of principal components required to
    explain the specified threshold of total variance.

    Parameters
    ----------
    features : np.ndarray
        Input feature matrix of shape (n_samples, n_features).
    threshold : float, optional
        Variance threshold for computing effective rank. Must be in (0, 1].
        Default is 0.95.
    random_state : int, optional
        Random seed for reproducibility. Note that with svd_solver="full",
        the result is deterministic regardless of this parameter. Default is 42.

    Returns
    -------
    PcaResult
        Populated dataclass containing PCA results including effective rank,
        explained variance ratios, cumulative variance, and component axes.

    Raises
    ------
    ValueError
        If `features` is not 2-D.
    ValueError
        If `features` has fewer than 2 samples.
    ValueError
        If `threshold` is not in the open interval (0, 1].

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> X = rng.standard_normal((100, 50))
    >>> result = compute_pca(X, threshold=0.95)
    >>> result.effective_rank  # doctest: +SKIP
    50
    """
    # Validate input dimensions
    if features.ndim != 2:
        raise ValueError(f"Expected 2-D array, got {features.ndim}-D")

    n_samples, n_features = features.shape

    if n_samples < 2:
        raise ValueError(
            f"Expected at least 2 samples, got {n_samples}"
        )

    # Validate threshold
    if not (0 < threshold <= 1):
        raise ValueError(
            f"Threshold must be in (0, 1], got {threshold}"
        )

    # Run PCA with full SVD for exact computation
    pca = PCA(n_components=None, svd_solver="full", random_state=random_state)
    pca.fit(features)

    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    components = pca.components_

    # Compute effective rank
    eff_rank = effective_rank(cumulative_variance, threshold)

    # Get total variance explained at the effective rank
    total_variance_explained = float(cumulative_variance[eff_rank - 1])

    return PcaResult(
        effective_rank=eff_rank,
        explained_variance_ratio=explained_variance_ratio,
        cumulative_variance=cumulative_variance,
        components=components,
        threshold=threshold,
        n_samples=n_samples,
        n_features=n_features,
        total_variance_explained=total_variance_explained,
    )


def effective_rank(
    cumulative_variance: np.ndarray,
    threshold: float = 0.95,
) -> int:
    """Compute the effective rank from cumulative explained variance.

    Given a cumulative explained-variance array, return the smallest index k
    (1-indexed count) such that cumulative_variance[k-1] >= threshold. If the
    threshold is never reached (e.g., threshold=1.0 with numerical error),
    return the total number of components.

    Parameters
    ----------
    cumulative_variance : np.ndarray
        Cumulative explained variance array. Must be non-empty and
        non-decreasing.
    threshold : float, optional
        Variance threshold for determining effective rank. Default is 0.95.

    Returns
    -------
    int
        The effective rank (minimum number of components to reach threshold).

    Raises
    ------
    ValueError
        If `cumulative_variance` is empty.

    Examples
    --------
    >>> import numpy as np
    >>> cum_var = np.array([0.5, 0.75, 0.90, 0.95, 0.98, 1.0])
    >>> effective_rank(cum_var, threshold=0.95)
    4
    >>> effective_rank(cum_var, threshold=0.99)
    6
    """
    if cumulative_variance.size == 0:
        raise ValueError("cumulative_variance array is empty")

    n_components = len(cumulative_variance)

    # Find first index where cumulative variance meets or exceeds threshold
    indices = np.where(cumulative_variance >= threshold)[0]

    if len(indices) == 0:
        # Threshold never reached; return total number of components
        return n_components

    # Return 1-indexed count
    return int(indices[0] + 1)


def explain_variance_at_rank(
    cumulative_variance: np.ndarray,
    rank: int,
) -> float:
    """Return the cumulative explained variance at a given rank.

    This utility function retrieves the cumulative variance captured by
    the first `rank` principal components. Useful for reporting how much
    variance a chosen rank captures.

    Parameters
    ----------
    cumulative_variance : np.ndarray
        Cumulative explained variance array.
    rank : int
        The rank (number of components) at which to query variance.
        Must satisfy 1 <= rank <= len(cumulative_variance).

    Returns
    -------
    float
        The cumulative explained variance at the specified rank.

    Raises
    ------
    ValueError
        If `rank < 1` or `rank > len(cumulative_variance)`.

    Examples
    --------
    >>> import numpy as np
    >>> cum_var = np.array([0.3, 0.55, 0.75, 0.90, 0.95])
    >>> explain_variance_at_rank(cum_var, 3)
    0.75
    >>> explain_variance_at_rank(cum_var, 5)
    0.95
    """
    n_components = len(cumulative_variance)

    if rank < 1:
        raise ValueError(f"Rank must be >= 1, got {rank}")

    if rank > n_components:
        raise ValueError(
            f"Rank must be <= {n_components}, got {rank}"
        )

    return float(cumulative_variance[rank - 1])
