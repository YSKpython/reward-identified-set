"""Within-component Fisher Information Matrix builder for RLHF reward models.

This module implements the Fisher information matrix construction from pre-extracted
features, comparison pair indices, and predicted reward differences under the
Bradley-Terry model. The Fisher matrix is used for feature-space vulnerability
audit pipelines in RLHF reward models.

Under the Bradley-Terry model, the probability of preferring response x_i over
x_j is P(x_i > x_j) = sigma(r(x_i) - r(x_j)), where sigma is the logistic function.
For a linear readout r(x) = W phi(x) with feature vector phi(x) in R^d, the Fisher
information contribution of a single comparison edge e = (i, j) is:

    F_e = I_e * (phi(x_i) - phi(x_j))(phi(x_i) - phi(x_j))^T

where I_e = sigma(Delta r_e)(1 - sigma(Delta r_e)) is the edge curvature and
Delta r_e = r(x_i) - r(x_j) is the predicted reward margin.

The full within-component Fisher matrix is the sum over all comparison edges:

    F_within = sum_{e in E} F_e
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class FisherBuildResult:
    """Result of building the Fisher information matrix.

    Attributes:
        fisher_matrix: The full Fisher matrix, shape (feature_dim, feature_dim).
        edge_curvatures: Per-edge curvature values, shape (n_edges,).
        feature_dim: Dimensionality of the feature space.
        n_edges: Number of comparison edges used.
        rank: Numerical rank of the Fisher matrix.
        null_space_dim: feature_dim - rank.
        is_psd: Whether the matrix is positive semidefinite (all eigenvalues >= -tol).
        trace: Trace of the Fisher matrix (sum of edge curvatures times squared feature diffs).
    """

    fisher_matrix: np.ndarray
    edge_curvatures: np.ndarray
    feature_dim: int
    n_edges: int
    rank: int
    null_space_dim: int
    is_psd: bool
    trace: float


def logistic(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function.

    Computes sigma(x) = 1 / (1 + exp(-x)) using a numerically stable formulation
    that avoids overflow for large positive or negative values.

    For x >= 0: sigma(x) = 1 / (1 + exp(-x))
    For x < 0: sigma(x) = exp(x) / (1 + exp(x))

    Args:
        x: Input array or scalar. Can be any shape.

    Returns:
        Logistic sigmoid of x, same shape as input. Values are in (0, 1).

    Examples:
        >>> logistic(np.array([0.0]))
        array([0.5])
        >>> logistic(np.array([100.0]))
        array([1.])
        >>> logistic(np.array([-100.0]))
        array([0.])
    """
    x = np.asarray(x)
    result = np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )
    return result


def compute_edge_curvatures(reward_diffs: np.ndarray) -> np.ndarray:
    """Compute edge curvature values for given reward differences.

    Under the Bradley-Terry model, the edge curvature is:

        I_e = sigma(Delta r_e) * (1 - sigma(Delta r_e))

    This is maximized at Delta r_e = 0 where I_e = 0.25, and approaches 0 as
    |Delta r_e| -> infinity.

    Args:
        reward_diffs: Array of predicted reward margins, shape (n_edges,).

    Returns:
        Array of curvature values, shape (n_edges,). Values are in (0, 0.25].

    Raises:
        ValueError: If reward_diffs is empty or not 1-D.

    Examples:
        >>> compute_edge_curvatures(np.array([0.0]))
        array([0.25])
    """
    reward_diffs = np.asarray(reward_diffs)

    if reward_diffs.ndim != 1:
        raise ValueError("reward_diffs must be 1-D")

    if reward_diffs.size == 0:
        raise ValueError("reward_diffs cannot be empty")

    sigma_vals = logistic(reward_diffs)
    curvatures = sigma_vals * (1.0 - sigma_vals)
    return curvatures


def build_fisher_matrix(
    features: np.ndarray,
    pairs: Sequence[Tuple[int, int]],
    reward_diffs: np.ndarray,
) -> FisherBuildResult:
    """Construct the within-component Fisher information matrix.

    For each edge e = (i, j), computes the feature difference delta_e = phi(x_i) - phi(x_j)
    and accumulates F += I_e * delta_e * delta_e^T, where I_e is the edge curvature.

    Args:
        features: Pre-extracted feature vectors, shape (n_samples, feature_dim).
        pairs: List of (i, j) index pairs, one per comparison edge.
        reward_diffs: Predicted reward margin for each pair, shape (n_edges,).

    Returns:
        FisherBuildResult containing the constructed Fisher matrix and metadata.

    Raises:
        ValueError: If features is not 2-D.
        ValueError: If len(pairs) != len(reward_diffs).
        ValueError: If pairs is empty.
        ValueError: If any pair index is out of bounds.
        ValueError: If features contains any NaN or Inf.

    Examples:
        >>> rng = np.random.RandomState(42)
        >>> features = rng.randn(10, 5)
        >>> pairs = [(0, 1), (2, 3)]
        >>> reward_diffs = np.array([0.0, 1.0])
        >>> result = build_fisher_matrix(features, pairs, reward_diffs)
        >>> result.fisher_matrix.shape
        (5, 5)
    """
    features = np.asarray(features)
    reward_diffs = np.asarray(reward_diffs)

    # Validate features
    if features.ndim != 2:
        raise ValueError("features must be 2-D")

    if not np.isfinite(features).all():
        raise ValueError("features contains NaN or Inf")

    n_samples, feature_dim = features.shape

    # Validate pairs
    if len(pairs) == 0:
        raise ValueError("pairs cannot be empty")

    if len(pairs) != len(reward_diffs):
        raise ValueError("len(pairs) must equal len(reward_diffs)")

    # Validate pair indices
    for idx, (i, j) in enumerate(pairs):
        if i < 0 or i >= n_samples or j < 0 or j >= n_samples:
            raise ValueError(f"Pair {idx} has out-of-bounds index: ({i}, {j})")

    n_edges = len(pairs)

    # Compute edge curvatures
    edge_curvatures = compute_edge_curvatures(reward_diffs)

    # Compute feature differences for all edges: shape (n_edges, feature_dim)
    pairs_array = np.array(pairs)
    i_indices = pairs_array[:, 0]
    j_indices = pairs_array[:, 1]

    feature_diffs = features[i_indices] - features[j_indices]

    # Vectorized Fisher construction:
    # F = sum_e I_e * delta_e * delta_e^T
    # This can be written as F = (sqrt(I) * delta)^T @ (sqrt(I) * delta)
    # where sqrt(I) is diagonal matrix of sqrt(curvatures)
    # More efficiently: F = delta^T @ diag(I) @ delta
    # Using broadcasting: weighted_diffs = feature_diffs * sqrt(edge_curvatures[:, None])
    # Then F = weighted_diffs.T @ weighted_diffs

    sqrt_curvatures = np.sqrt(edge_curvatures)
    weighted_diffs = feature_diffs * sqrt_curvatures[:, np.newaxis]

    fisher_matrix = weighted_diffs.T @ weighted_diffs

    # Compute trace: sum of I_e * ||delta_e||^2
    squared_norms = np.sum(feature_diffs**2, axis=1)
    trace_val = float(np.sum(edge_curvatures * squared_norms))

    # Compute rank via SVD
    rank_val = fisher_rank(fisher_matrix)

    # Compute null space dimension
    null_space_dim_val = fisher_null_space_dim(fisher_matrix)

    # Check PSD
    is_psd_val = check_psd(fisher_matrix)

    return FisherBuildResult(
        fisher_matrix=fisher_matrix,
        edge_curvatures=edge_curvatures,
        feature_dim=feature_dim,
        n_edges=n_edges,
        rank=rank_val,
        null_space_dim=null_space_dim_val,
        is_psd=is_psd_val,
        trace=trace_val,
    )


def fisher_rank(F: np.ndarray, tol: float = 1e-10) -> int:
    """Compute the numerical rank of a symmetric matrix via SVD.

    Counts the number of singular values greater than the tolerance.

    Args:
        F: Square matrix to analyze.
        tol: Tolerance for singular value threshold. Singular values <= tol
            are considered zero.

    Returns:
        Numerical rank of the matrix.

    Raises:
        ValueError: If F is not square.

    Examples:
        >>> F = np.array([[1.0, 0.0], [0.0, 0.0]])
        >>> fisher_rank(F)
        1
    """
    F = np.asarray(F)

    if F.ndim != 2 or F.shape[0] != F.shape[1]:
        raise ValueError("F must be a square matrix")

    singular_values = np.linalg.svd(F, compute_uv=False)
    rank_val = int(np.sum(singular_values > tol))
    return rank_val


def fisher_null_space_dim(F: np.ndarray, tol: float = 1e-10) -> int:
    """Compute the null space dimension of a symmetric matrix.

    Returns feature_dim - rank(F).

    Args:
        F: Square matrix to analyze.
        tol: Tolerance for singular value threshold.

    Returns:
        Dimension of the null space.

    Raises:
        ValueError: If F is not square.

    Examples:
        >>> F = np.array([[1.0, 0.0], [0.0, 0.0]])
        >>> fisher_null_space_dim(F)
        1
    """
    F = np.asarray(F)

    if F.ndim != 2 or F.shape[0] != F.shape[1]:
        raise ValueError("F must be a square matrix")

    rank_val = fisher_rank(F, tol)
    null_dim = int(F.shape[0]) - rank_val
    return null_dim


def apply_tikhonov(F: np.ndarray, epsilon: float) -> np.ndarray:
    """Apply Tikhonov regularization to the Fisher matrix.

    Returns F + epsilon * I, where I is the identity matrix. This regularized
    Fisher is used in flip-cost computation: FC = m^2 / (g^T F_epsilon^{-1} g).

    Args:
        F: Square Fisher matrix.
        epsilon: Regularization strength. Must be non-negative.

    Returns:
        Regularized matrix F + epsilon * I.

    Raises:
        ValueError: If epsilon < 0.
        ValueError: If F is not square.

    Examples:
        >>> F = np.array([[1.0, 0.0], [0.0, 2.0]])
        >>> apply_tikhonov(F, 0.5)
        array([[1.5, 0. ],
               [0. , 2.5]])
    """
    F = np.asarray(F)

    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")

    if F.ndim != 2 or F.shape[0] != F.shape[1]:
        raise ValueError("F must be a square matrix")

    n = F.shape[0]
    return F + epsilon * np.eye(n)


def check_psd(F: np.ndarray, tol: float = 1e-8) -> bool:
    """Check if a matrix is positive semidefinite.

    Uses eigendecomposition to verify all eigenvalues are >= -tol.

    Args:
        F: Square matrix to check.
        tol: Tolerance for eigenvalue check. Eigenvalues >= -tol are considered
            non-negative.

    Returns:
        True if all eigenvalues >= -tol, False otherwise.

    Examples:
        >>> F = np.array([[1.0, 0.0], [0.0, 2.0]])
        >>> check_psd(F)
        True
        >>> F = np.array([[-1.0, 0.0], [0.0, 2.0]])
        >>> check_psd(F)
        False
    """
    F = np.asarray(F)

    if F.ndim != 2 or F.shape[0] != F.shape[1]:
        raise ValueError("F must be a square matrix")

    eigenvalues = np.linalg.eigvalsh(F)
    return bool(np.all(eigenvalues >= -tol))
