"""Cosine similarity structure computation for feature-space vulnerability audit.

This module implements the cosine similarity analysis described in Experiment
E16 of the NeurIPS 2027 submission. It measures how a trained reward model's
hidden-state representations implicitly couple structurally disconnected
components of the tabular comparison graph.

The key insight is that within-component pairs (chosen/rejected responses
sharing the same prompt) should have high cosine similarity if the model
captures semantic relatedness, while cross-component pairs (responses from
different prompts) provide a baseline. The gap between these confirms that
the transformer's inductive bias implicitly connects disconnected components.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class CosineStructureResult:
    """Container for cosine similarity structure analysis results.

    Attributes
    ----------
    within_mean : float
        Mean cosine similarity of within-component pairs.
    within_std : float
        Standard deviation of within-component cosines.
    cross_mean : float
        Mean cosine similarity of cross-component pairs.
    cross_std : float
        Standard deviation of cross-component cosines.
    gap : float
        Difference between within_mean and cross_mean.
    n_within : int
        Number of within-component pairs evaluated.
    n_cross : int
        Number of cross-component pairs evaluated.
    within_values : np.ndarray
        Full array of within-component cosine values. Shape: (n_within,).
    cross_values : np.ndarray
        Full array of cross-component cosine values. Shape: (n_cross,).
    """

    within_mean: float
    within_std: float
    cross_mean: float
    cross_std: float
    gap: float
    n_within: int
    n_cross: int
    within_values: np.ndarray
    cross_values: np.ndarray


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the cosine similarity between two 1-D vectors.

    The cosine similarity is defined as the dot product of the two vectors
    divided by the product of their norms: dot(a, b) / (norm(a) * norm(b)).

    Parameters
    ----------
    a : np.ndarray
        First input vector. Must be 1-D.
    b : np.ndarray
        Second input vector. Must be 1-D and have the same shape as a.

    Returns
    -------
    float
        The cosine similarity between a and b. Value in [-1, 1].

    Raises
    ------
    ValueError
        If either vector has zero norm.
    ValueError
        If the vectors have different shapes or are not 1-D.

    Examples
    --------
    >>> import numpy as np
    >>> v = np.array([1.0, 0.0, 0.0])
    >>> cosine_similarity(v, v)
    1.0
    >>> w = np.array([0.0, 1.0, 0.0])
    >>> cosine_similarity(v, w)
    0.0
    >>> cosine_similarity(v, -v)
    -1.0
    """
    # Validate dimensions
    if a.ndim != 1:
        raise ValueError(f"Expected 1-D array for 'a', got {a.ndim}-D")
    if b.ndim != 1:
        raise ValueError(f"Expected 1-D array for 'b', got {b.ndim}-D")

    # Validate shapes match
    if a.shape != b.shape:
        raise ValueError(
            f"Vectors must have the same shape, got {a.shape} and {b.shape}"
        )

    # Compute norms
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    # Check for zero norms
    if norm_a == 0.0:
        raise ValueError("Vector 'a' has zero norm")
    if norm_b == 0.0:
        raise ValueError("Vector 'b' has zero norm")

    # Compute cosine similarity
    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_matrix(features: np.ndarray) -> np.ndarray:
    """Compute the full pairwise cosine similarity matrix for a feature matrix.

    Given a feature matrix of shape (n_samples, n_features), this function
    returns an (n_samples, n_samples) symmetric matrix where entry (i, j)
    is the cosine similarity between row i and row j. The diagonal entries
    are all 1.0 (each sample has cosine similarity 1 with itself).

    The implementation normalizes each row to unit norm before computing
    the Gram matrix for numerical stability.

    Parameters
    ----------
    features : np.ndarray
        Input feature matrix of shape (n_samples, n_features).

    Returns
    -------
    np.ndarray
        Pairwise cosine similarity matrix of shape (n_samples, n_samples).
        Symmetric with ones on the diagonal.

    Raises
    ------
    ValueError
        If `features` is not 2-D.
    ValueError
        If any row has zero norm.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    >>> C = cosine_matrix(X)
    >>> C.shape
    (3, 3)
    >>> np.allclose(np.diag(C), 1.0)
    True
    """
    # Validate dimensions
    if features.ndim != 2:
        raise ValueError(f"Expected 2-D array, got {features.ndim}-D")

    n_samples, _ = features.shape

    # Compute row norms
    norms = np.linalg.norm(features, axis=1)

    # Check for zero norms
    zero_mask = norms == 0.0
    if np.any(zero_mask):
        raise ValueError("Feature matrix contains rows with zero norm")

    # Normalize rows to unit norm
    normalized = features / norms[:, np.newaxis]

    # Compute Gram matrix (pairwise dot products of normalized rows)
    result: np.ndarray = normalized @ normalized.T
    return result


def within_component_cosine(
    features: np.ndarray,
    pair_indices: Sequence[tuple[int, int]],
) -> np.ndarray:
    """Compute cosine similarities for within-component pairs.

    Given a feature matrix and a list of (i, j) index pairs representing
    within-component comparisons (e.g., chosen/rejected responses for the
    same prompt), compute the cosine similarity for each pair.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    pair_indices : Sequence[tuple[int, int]]
        List of (i, j) index pairs specifying which pairs to compare.
        Each index must be in range [0, n_samples).

    Returns
    -------
    np.ndarray
        1-D array of shape (len(pair_indices),) containing cosine
        similarities for each specified pair.

    Raises
    ------
    ValueError
        If any index is out of bounds.
    ValueError
        If `pair_indices` is empty.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    >>> pairs = [(0, 1)]  # Compare first two samples
    >>> result = within_component_cosine(X, pairs)
    >>> result.shape
    (1,)
    >>> result[0] > 0.9  # These vectors are similar
    True
    """
    n_samples, _ = features.shape

    # Validate non-empty
    if len(pair_indices) == 0:
        raise ValueError("pair_indices cannot be empty")

    # Validate indices
    for idx, (i, j) in enumerate(pair_indices):
        if not (0 <= i < n_samples and 0 <= j < n_samples):
            raise ValueError(
                f"Index out of bounds at pair {idx}: ({i}, {j}) "
                f"with n_samples={n_samples}"
            )

    # Compute cosine similarities for each pair
    cosines: np.ndarray = np.empty(len(pair_indices), dtype=np.float64)
    for k, (i, j) in enumerate(pair_indices):
        cosines[k] = cosine_similarity(features[i], features[j])

    return cosines


def cross_component_cosine(
    features: np.ndarray,
    n_pairs: int,
    random_state: int = 42,
    exclude_indices: set[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Sample random cross-component pairs and compute their cosine similarities.

    This function samples `n_pairs` random pairs of indices from the feature
    matrix, excluding any pairs listed in `exclude_indices`. The exclusion
    mechanism is used to avoid accidentally sampling a within-component pair
    when constructing cross-component baselines.

    Uses a dedicated `np.random.RandomState` seeded with `random_state` for
    reproducibility, independent of global RNG state.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    n_pairs : int
        Number of random pairs to sample.
    random_state : int, optional
        Random seed for reproducibility. Default is 42.
    exclude_indices : set[tuple[int, int]] | None, optional
        Set of (i, j) pairs to exclude from sampling. Both (i, j) and
        (j, i) should be included for symmetric exclusion. Default is None.

    Returns
    -------
    np.ndarray
        1-D array of shape (n_pairs,) containing cosine similarities
        for each sampled pair.

    Raises
    ------
    ValueError
        If `n_pairs` exceeds the number of available pairs.
    ValueError
        If `n_pairs <= 0`.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.default_rng(42).standard_normal((100, 50))
    >>> cosines = cross_component_cosine(X, n_pairs=10, random_state=42)
    >>> cosines.shape
    (10,)
    """
    n_samples, _ = features.shape

    # Validate n_pairs
    if n_pairs <= 0:
        raise ValueError(f"n_pairs must be positive, got {n_pairs}")

    # Total number of possible pairs (excluding self-pairs)
    total_possible_pairs = n_samples * (n_samples - 1)

    # Account for excluded pairs
    if exclude_indices is not None:
        available_pairs = total_possible_pairs - len(exclude_indices)
    else:
        available_pairs = total_possible_pairs

    if n_pairs > available_pairs:
        raise ValueError(
            f"Requested {n_pairs} pairs but only {available_pairs} available"
        )

    # Initialize RNG with seed
    rng = np.random.RandomState(random_state)

    # Build exclusion set for fast lookup
    if exclude_indices is None:
        exclude_set = set()
    else:
        exclude_set = set(exclude_indices)

    # Sample pairs using rejection sampling
    cosines_list: list[float] = []
    attempts = 0
    max_attempts = n_pairs * 100  # Prevent infinite loops

    while len(cosines_list) < n_pairs and attempts < max_attempts:
        i = rng.randint(0, n_samples)
        j = rng.randint(0, n_samples)

        # Skip self-pairs
        if i == j:
            attempts += 1
            continue

        # Skip excluded pairs
        if (i, j) in exclude_set:
            attempts += 1
            continue

        # Compute cosine similarity
        cos_sim = cosine_similarity(features[i], features[j])
        cosines_list.append(cos_sim)
        attempts += 1

    if len(cosines_list) < n_pairs:
        raise ValueError(
            f"Could not sample {n_pairs} pairs after {max_attempts} attempts. "
            f"Too many exclusions or insufficient samples."
        )

    return np.array(cosines_list, dtype=np.float64)


def compute_cosine_structure(
    features: np.ndarray,
    pair_indices: Sequence[tuple[int, int]],
    n_cross_pairs: int,
    random_state: int = 42,
) -> CosineStructureResult:
    """Orchestrate the full E16 cosine similarity structure analysis.

    This function computes both within-component and cross-component cosine
    similarities, then aggregates them into summary statistics including
    means, standard deviations, and the gap between them.

    The gap (within_mean - cross_mean) quantifies how much more similar
    within-component pairs are compared to random pairs. A significant gap
    indicates that the model's representations capture semantic relatedness
    even when the tabular comparison graph is nearly disconnected.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    pair_indices : Sequence[tuple[int, int]]
        List of (i, j) index pairs representing within-component comparisons.
    n_cross_pairs : int
        Number of cross-component pairs to sample for baseline.
    random_state : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    CosineStructureResult
        Populated dataclass containing all cosine structure metrics.

    Examples
    --------
    >>> import numpy as np
    >>> # Create synthetic data with two clusters
    >>> rng = np.random.RandomState(42)
    >>> cluster1 = rng.randn(50, 10) + np.array([5.0] * 10)
    >>> cluster2 = rng.randn(50, 10) - np.array([5.0] * 10)
    >>> X = np.vstack([cluster1, cluster2])
    >>> # Within-cluster pairs (first 50 samples are cluster 1)
    >>> pairs = [(i, i+1) for i in range(0, 48, 2)]
    >>> result = compute_cosine_structure(X, pairs, n_cross_pairs=100)
    >>> result.n_within == len(pairs)
    True
    >>> result.n_cross == 100
    True
    """
    # Compute within-component cosines
    within_values = within_component_cosine(features, pair_indices)

    # Build exclusion set (both directions for symmetry)
    exclude_set: set[tuple[int, int]] = set()
    for i, j in pair_indices:
        exclude_set.add((i, j))
        exclude_set.add((j, i))

    # Sample cross-component cosines
    cross_values = cross_component_cosine(
        features,
        n_cross_pairs,
        random_state=random_state,
        exclude_indices=exclude_set,
    )

    # Compute summary statistics
    within_mean = float(np.mean(within_values))
    within_std = float(np.std(within_values, ddof=0))
    cross_mean = float(np.mean(cross_values))
    cross_std = float(np.std(cross_values, ddof=0))
    gap = within_mean - cross_mean

    return CosineStructureResult(
        within_mean=within_mean,
        within_std=within_std,
        cross_mean=cross_mean,
        cross_std=cross_std,
        gap=gap,
        n_within=len(within_values),
        n_cross=len(cross_values),
        within_values=within_values,
        cross_values=cross_values,
    )


def bootstrap_ci(
    values: np.ndarray,
    statistic: Literal["mean", "std"] = "mean",
    n_bootstrap: int = 4000,
    random_state: int = 42,
    ci: float = 0.95,
) -> tuple[float, float]:
    """Compute a bootstrap confidence interval for a given statistic.

    Uses non-parametric bootstrap resampling to estimate the confidence
    interval of either the mean or standard deviation of a 1-D array.

    Parameters
    ----------
    values : np.ndarray
        1-D array of values to analyze.
    statistic : Literal["mean", "std"], optional
        Which statistic to compute CI for. Must be "mean" or "std".
        Default is "mean".
    n_bootstrap : int, optional
        Number of bootstrap resamples. Default is 4000.
    random_state : int, optional
        Random seed for reproducibility. Default is 42.
    ci : float, optional
        Confidence level (e.g., 0.95 for 95% CI). Must be in (0, 1).
        Default is 0.95.

    Returns
    -------
    tuple[float, float]
        Tuple of (lower, upper) confidence interval bounds.

    Raises
    ------
    ValueError
        If `values` is empty.
    ValueError
        If `statistic` is not "mean" or "std".
    ValueError
        If `ci` is not in the open interval (0, 1).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> data = rng.normal(0, 1, 100)
    >>> lo, hi = bootstrap_ci(data, statistic="mean", n_bootstrap=1000)
    >>> lo < 0 < hi  # True mean (0) should be in CI most of the time
    True
    """
    # Validate inputs
    if values.size == 0:
        raise ValueError("values array cannot be empty")

    if values.ndim != 1:
        raise ValueError(f"Expected 1-D array, got {values.ndim}-D")

    if statistic not in ("mean", "std"):
        raise ValueError(
            f"statistic must be 'mean' or 'std', got '{statistic}'"
        )

    if not (0 < ci < 1):
        raise ValueError(f"ci must be in (0, 1), got {ci}")

    # Initialize RNG
    rng = np.random.RandomState(random_state)

    n = len(values)

    # Perform bootstrap resampling
    bootstrap_stats = np.empty(n_bootstrap, dtype=np.float64)

    for b in range(n_bootstrap):
        # Resample with replacement
        indices = rng.randint(0, n, size=n)
        resample = values[indices]

        # Compute statistic
        if statistic == "mean":
            bootstrap_stats[b] = np.mean(resample)
        else:  # statistic == "std"
            bootstrap_stats[b] = np.std(resample, ddof=0)

    # Compute percentile-based confidence interval
    alpha = 1 - ci
    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)

    lower = float(np.percentile(bootstrap_stats, lower_percentile))
    upper = float(np.percentile(bootstrap_stats, upper_percentile))

    return lower, upper
