"""Flip cost computation for reward shift directions under the Fisher metric.

This module implements the flip cost metric from Experiment E88, which computes
the squared minimum parameter-space norm required to achieve a reward shift g
with margin m under the Fisher metric:

    FC(g) = m² / (g^T F_ε^{-1} g),   where F_ε = F + ε I

The flip cost represents the deterministic optimal-direction manipulation cost,
using the analytic optimum rather than sampling random directions.
"""

from dataclasses import dataclass

import numpy as np

from src.fisher.llf import solve_fisher_system


@dataclass
class FlipCostResult:
    """Result of computing the flip cost for a reward shift direction.

    Attributes:
        flip_cost: The flip cost m² / (g^T F_ε^{-1} g).
        quadratic_form: The value g^T F_ε^{-1} g.
        margin: The target margin m used.
        damping: The Tikhonov damping ε used.
        g_norm: ||g||_2, the norm of the shift direction.
        g_range_norm: Norm of the component of g in range(F).
        g_null_norm: Norm of the component of g in null(F).
        null_fraction: g_null_norm² / g_norm², fraction of g in null-space.
    """

    flip_cost: float
    quadratic_form: float
    margin: float
    damping: float
    g_norm: float
    g_range_norm: float
    g_null_norm: float
    null_fraction: float


def _decompose_g_null_range(  # noqa: N803
    F: np.ndarray,  # noqa: N803
    g: np.ndarray,
    tol: float = 1e-8,
) -> tuple[float, float, float]:
    """Decompose g into range and null components of F via eigendecomposition.

    Uses eigendecomposition of F to compute the orthogonal projection onto
    range(F) and null(F). For a symmetric matrix F = U Λ U^T, the projection
    onto range(F) is P = U_r U_r^T where U_r contains eigenvectors with
    eigenvalues > tol.

    Args:
        F: Square Fisher matrix, shape (n, n). Must be symmetric.
        g: Shift direction vector, shape (n,).
        tol: Tolerance for eigenvalue threshold. Eigenvalues <= tol are
            considered zero (null-space).

    Returns:
        Tuple of (g_norm, g_range_norm, g_null_norm).
    """
    # Compute g norm
    g_norm = float(np.linalg.norm(g))

    # Eigendecomposition of F
    eigenvalues, eigenvectors = np.linalg.eigh(F)

    # Identify range space: eigenvectors with eigenvalues > tol
    rank_mask = eigenvalues > tol
    if not np.any(rank_mask):
        # All eigenvalues are zero or negative; g is entirely in null-space
        return g_norm, 0.0, g_norm

    U_r = eigenvectors[:, rank_mask]  # noqa: N806

    # Project g onto range(F): g_range = U_r U_r^T g
    g_range = U_r @ (U_r.T @ g)
    g_range_norm = float(np.linalg.norm(g_range))

    # Null component: g_null = g - g_range
    g_null = g - g_range
    g_null_norm = float(np.linalg.norm(g_null))

    return g_norm, g_range_norm, g_null_norm


def compute_flip_cost(  # noqa: N803
    F: np.ndarray,  # noqa: N803
    g: np.ndarray,
    margin: float,
    damping: float = 1e-6,
    tol: float = 1e-8,
    max_iter: int = 1000,
) -> FlipCostResult:
    """Compute the deterministic optimal-direction flip cost.

    The flip cost is defined as:

        FC(g) = m² / (g^T F_ε^{-1} g),   where F_ε = F + ε I

    This represents the squared minimum parameter-space norm required to
    achieve the reward shift g with margin m under the Fisher metric.

    Args:
        F: Square Fisher information matrix, shape (n, n). Must be symmetric.
        g: Reward shift direction vector, shape (n,).
        margin: Target margin m. Must be positive.
        damping: Tikhonov regularization parameter ε. Default is 1e-6.
        tol: Convergence tolerance for the CG solver. Default is 1e-8.
        max_iter: Maximum number of CG iterations. Default is 1000.

    Returns:
        FlipCostResult containing the flip cost, quadratic form, margin,
        damping, and null-space decomposition statistics.

    Raises:
        ValueError: If F is not square or not symmetric.
        ValueError: If g shape does not match F.
        ValueError: If margin <= 0.
        ValueError: If damping < 0.
        ValueError: If the quadratic form is non-positive (numerical failure).

    Examples:
        >>> F = np.eye(3)
        >>> g = np.array([1.0, 2.0, 3.0])
        >>> result = compute_flip_cost(F, g, margin=0.1)
        >>> result.flip_cost > 0
        True
    """
    F = np.asarray(F, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)

    # Validate F is square
    if F.ndim != 2 or F.shape[0] != F.shape[1]:
        raise ValueError("F must be a square matrix")

    n = F.shape[0]

    # Validate F is symmetric
    if not np.allclose(F, F.T, rtol=1e-10, atol=1e-10):
        raise ValueError("F must be symmetric")

    # Validate g shape matches F
    if g.ndim != 1 or g.shape[0] != n:
        raise ValueError("g shape does not match F")

    # Validate margin
    if margin <= 0:
        raise ValueError("margin must be positive")

    # Validate damping
    if damping < 0:
        raise ValueError("damping must be non-negative")

    # Apply Tikhonov damping: F_ε = F + damping * I
    F_epsilon = F + damping * np.eye(n)  # noqa: N806

    # Solve F_ε s = g using CG solver
    cg_result = solve_fisher_system(
        F=F_epsilon,
        g=g,
        damping=0.0,  # Already applied damping
        tol=tol,
        max_iter=max_iter,
    )

    s = cg_result.solution

    # Compute quadratic form V = g^T s
    quadratic_form = float(np.dot(g, s))

    # Validate quadratic form is positive
    if quadratic_form <= 0:
        raise ValueError("Quadratic form is non-positive; indicates numerical failure")

    # Compute flip cost FC = m² / V
    flip_cost = (margin**2) / quadratic_form

    # Decompose g into range and null components of F
    g_norm, g_range_norm, g_null_norm = _decompose_g_null_range(F, g, tol=tol)

    # Compute null fraction
    null_fraction = g_null_norm**2 / g_norm**2 if g_norm > 0 else 0.0

    return FlipCostResult(
        flip_cost=flip_cost,
        quadratic_form=quadratic_form,
        margin=margin,
        damping=damping,
        g_norm=g_norm,
        g_range_norm=g_range_norm,
        g_null_norm=g_null_norm,
        null_fraction=null_fraction,
    )


def compute_flip_cost_batch(  # noqa: N803
    F: np.ndarray,  # noqa: N803
    G: np.ndarray,  # noqa: N803
    margin: float,
    damping: float = 1e-6,
    tol: float = 1e-8,
    max_iter: int = 1000,
) -> np.ndarray:
    """Compute flip costs for a batch of shift directions.

    Vectorized computation of flip costs for multiple directions. Falls back
    to a loop over compute_flip_cost for clarity.

    Args:
        F: Square Fisher information matrix, shape (n, n). Must be symmetric.
        G: Batch of shift direction vectors, shape (n_directions, n).
        margin: Target margin m. Must be positive.
        damping: Tikhonov regularization parameter ε. Default is 1e-6.
        tol: Convergence tolerance for the CG solver. Default is 1e-8.
        max_iter: Maximum number of CG iterations. Default is 1000.

    Returns:
        Array of shape (n_directions,) containing flip costs for each direction.

    Raises:
        ValueError: If G is not 2-D.
        ValueError: If G.shape[1] != F.shape[0].
        ValueError: If margin <= 0.
        ValueError: If damping < 0.

    Examples:
        >>> F = np.eye(3)
        >>> G = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        >>> costs = compute_flip_cost_batch(F, G, margin=0.1)
        >>> costs.shape
        (2,)
    """
    F = np.asarray(F, dtype=np.float64)  # noqa: N806
    G = np.asarray(G, dtype=np.float64)  # noqa: N806

    # Validate G is 2-D
    if G.ndim != 2:
        raise ValueError("G must be 2-D")

    n_directions, feature_dim = G.shape

    # Validate G.shape[1] matches F
    if feature_dim != F.shape[0]:
        raise ValueError("G.shape[1] must match F.shape[0]")

    # Validate margin
    if margin <= 0:
        raise ValueError("margin must be positive")

    # Validate damping
    if damping < 0:
        raise ValueError("damping must be non-negative")

    # Compute flip costs for each direction
    flip_costs = np.zeros(n_directions, dtype=np.float64)

    for i in range(n_directions):
        g_i = G[i, :]
        result = compute_flip_cost(
            F=F,
            g=g_i,
            margin=margin,
            damping=damping,
            tol=tol,
            max_iter=max_iter,
        )
        flip_costs[i] = result.flip_cost

    return flip_costs
