"""Neural Manipulation Cost computation via Conjugate Gradient solver.

This module implements the Neural Manipulation Cost (MC) metric from Definition 7.9,
which quantifies the vulnerability of a reward model to manipulation under the Fisher
information geometry. The MC is computed by solving a linear system using the
Conjugate Gradient (CG) method.

The CG solver finds s in (F + damping*I) s = g, where F is the Fisher information
matrix, g is the gradient vector, and damping is a regularization parameter.

The manipulation cost is then computed as:
    MC = margin / sqrt(V)
where V = g^T s is the vulnerability score.

The KL divergence budget is approximated as:
    KL ≈ MC^2 / 2

This follows from the second-order expansion of KL divergence under the Fisher metric:
    D_KL ≈ (1/2) Δθ^T F Δθ
and MC = sqrt(Δθ^T F Δθ), so KL ≈ MC^2 / 2.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class CGResult:
    """Result of the Conjugate Gradient solver.

    Attributes:
        solution: The solution vector s, shape (n,).
        converged: Whether the solver converged within max_iter iterations.
        n_iterations: Number of iterations performed.
        residual_norm: Final residual norm ||b - A @ s||.
    """

    solution: np.ndarray
    converged: bool
    n_iterations: int
    residual_norm: float


@dataclass
class ManipulationCostResult:
    """Result of the Neural Manipulation Cost computation.

    Attributes:
        mc_value: The computed manipulation cost value.
        vulnerability_score: The vulnerability score V = g^T s.
        kl_budget: The approximate KL divergence budget ≈ MC^2 / 2.
        cg_result: The CG solver result containing convergence info.
        is_vulnerable: Whether the system is vulnerable (MC < threshold).
    """

    mc_value: float
    vulnerability_score: float
    kl_budget: float
    cg_result: CGResult
    is_vulnerable: bool


def solve_fisher_system(
    F: np.ndarray,
    g: np.ndarray,
    damping: float = 1e-6,
    tol: float = 1e-8,
    max_iter: int = 1000,
) -> CGResult:
    """Solve the linear system (F + damping*I) s = g using Conjugate Gradient.

    Implements the Conjugate Gradient algorithm for solving symmetric positive
    definite linear systems. The system is regularized with Tikhonov damping:
    (F + damping*I) s = g.

    Args:
        F: Square Fisher information matrix, shape (n, n). Must be symmetric.
        g: Right-hand side vector, shape (n,).
        damping: Tikhonov regularization parameter. Must be non-negative.
            Default is 1e-6.
        tol: Convergence tolerance for residual norm. Iteration stops when
            ||r|| < tol * ||g||. Must be positive. Default is 1e-8.
        max_iter: Maximum number of iterations. Must be at least 1.
            Default is 1000.

    Returns:
        CGResult containing the solution, convergence flag, iteration count,
        and final residual norm.

    Raises:
        ValueError: If F is not square.
        ValueError: If F is not symmetric (within 1e-10).
        ValueError: If g shape does not match F.
        ValueError: If damping < 0.
        ValueError: If tol <= 0 or max_iter < 1.

    Examples:
        >>> F = np.eye(3)
        >>> g = np.array([1.0, 2.0, 3.0])
        >>> result = solve_fisher_system(F, g)
        >>> np.allclose(result.solution, g)
        True
    """
    F = np.asarray(F, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)

    # Validate F is square
    if F.ndim != 2 or F.shape[0] != F.shape[1]:
        raise ValueError("F must be a square matrix")

    n = F.shape[0]

    # Validate g shape matches F
    if g.ndim != 1 or g.shape[0] != n:
        raise ValueError("g shape does not match F")

    # Validate F is symmetric
    if not np.allclose(F, F.T, rtol=1e-10, atol=1e-10):
        raise ValueError("F must be symmetric")

    # Validate damping
    if damping < 0:
        raise ValueError("damping must be non-negative")

    # Validate tol and max_iter
    if tol <= 0:
        raise ValueError("tol must be positive")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")

    # Initialize: A = F + damping * I
    # We don't explicitly form A; we compute A @ v as F @ v + damping * v
    x = np.zeros(n, dtype=np.float64)  # Initial guess s = 0
    r = g.copy()  # Initial residual r = g - A @ 0 = g
    p = r.copy()  # Initial search direction

    rs_old = np.dot(r, r)  # ||r||^2
    b_norm = np.linalg.norm(g)
    tol_abs = tol * b_norm if b_norm > 0 else tol

    # Check if initial guess is already good enough
    if np.sqrt(rs_old) < tol_abs:
        return CGResult(
            solution=x,
            converged=True,
            n_iterations=0,
            residual_norm=np.sqrt(rs_old),
        )

    converged = False
    n_iterations = 0

    for k in range(max_iter):
        n_iterations = k + 1

        # Compute A @ p = F @ p + damping * p
        Ap = F @ p + damping * p

        # Check for numerical stability: p @ Ap must be positive
        pAp = np.dot(p, Ap)
        if pAp <= 0:
            # Numerical failure; return current best estimate
            residual_norm = float(np.linalg.norm(r))
            return CGResult(
                solution=x,
                converged=False,
                n_iterations=n_iterations,
                residual_norm=residual_norm,
            )

        # Step size: alpha = r^T r / (p^T A p)
        alpha = rs_old / pAp

        # Update solution: x = x + alpha * p
        x = x + alpha * p

        # Update residual: r = r - alpha * A @ p
        r = r - alpha * Ap

        # Check convergence
        rs_new = np.dot(r, r)
        residual_norm = float(np.sqrt(rs_new))

        if residual_norm < tol_abs:
            converged = True
            break

        # Compute beta for next search direction
        beta = rs_new / rs_old
        rs_old = rs_new

        # Update search direction: p = r + beta * p
        p = r + beta * p

    return CGResult(
        solution=x,
        converged=converged,
        n_iterations=n_iterations,
        residual_norm=residual_norm,
    )


def compute_manipulation_cost(
    F: np.ndarray,
    g: np.ndarray,
    margin: float,
    damping: float = 1e-6,
    tol: float = 1e-8,
    max_iter: int = 1000,
    vulnerability_threshold: float = 0.05,
) -> ManipulationCostResult:
    """Compute the Neural Manipulation Cost (Definition 7.9).

    The Neural Manipulation Cost quantifies how susceptible a reward model is to
    manipulation. It is defined as:

        MC = margin / sqrt(V)

    where V = g^T s is the vulnerability score, s = (F + damping*I)^{-1} g is
    the solution to the regularized Fisher system, and margin is the decision
    margin.

    The KL divergence budget is approximately MC^2 / 2, following from the
    second-order expansion under the Fisher metric.

    Args:
        F: Square Fisher information matrix, shape (n, n). Must be symmetric.
        g: Gradient vector, shape (n,).
        margin: Decision margin. Must be positive.
        damping: Tikhonov regularization parameter. Default is 1e-6.
        tol: CG convergence tolerance. Default is 1e-8.
        max_iter: Maximum CG iterations. Default is 1000.
        vulnerability_threshold: Threshold below which the system is considered
            vulnerable. Default is 0.05.

    Returns:
        ManipulationCostResult containing the MC value, vulnerability score,
        KL budget, CG result, and vulnerability flag.

    Raises:
        ValueError: If F is not square or not symmetric.
        ValueError: If g shape does not match F.
        ValueError: If damping < 0.
        ValueError: If tol <= 0 or max_iter < 1.
        ValueError: If margin <= 0.
        ValueError: If V <= 0 (numerical failure or zero gradient).
        ValueError: If CG did not converge.

    Examples:
        >>> F = np.eye(3)
        >>> g = np.array([1.0, 2.0, 3.0])
        >>> result = compute_manipulation_cost(F, g, margin=0.1)
        >>> result.mc_value > 0
        True
    """
    # Validate margin
    if margin <= 0:
        raise ValueError("margin must be positive")

    # Solve the Fisher system
    cg_result = solve_fisher_system(
        F=F,
        g=g,
        damping=damping,
        tol=tol,
        max_iter=max_iter,
    )

    # Check CG convergence
    if not cg_result.converged:
        raise ValueError("CG did not converge; MC estimate is unreliable")

    s = cg_result.solution

    # Compute vulnerability score V = g^T s
    V = np.dot(g, s)

    # Validate V > 0
    if V <= 0:
        raise ValueError("Vulnerability score V <= 0; indicates numerical failure or zero gradient")

    # Compute MC = margin / sqrt(V)
    mc_value = margin / np.sqrt(V)

    # Compute KL budget ≈ MC^2 / 2
    kl_budget = kl_budget_from_mc(mc_value)

    # Determine vulnerability flag
    is_vulnerable = bool(mc_value < vulnerability_threshold)

    return ManipulationCostResult(
        mc_value=float(mc_value),
        vulnerability_score=float(V),
        kl_budget=float(kl_budget),
        cg_result=cg_result,
        is_vulnerable=is_vulnerable,
    )


def llf_conservative_bound_check(mc_llf: float, mc_full: float) -> bool:
    """Verify Proposition 7.10: MC_full <= MC_LLF (within numerical tolerance).

    This function validates that the full-parameter manipulation cost is bounded
    above by the LLF (Last Layer Fine-tuning) manipulation cost, as stated in
    Proposition 7.10.

    Args:
        mc_llf: Manipulation cost computed using only LLF parameters.
        mc_full: Manipulation cost computed using all parameters.

    Returns:
        True if mc_full <= mc_llf (within tolerance 1e-10), False otherwise.

    Examples:
        >>> llf_conservative_bound_check(0.0129, 0.010)
        True
        >>> llf_conservative_bound_check(0.010, 0.0129)
        False
    """
    return mc_full <= mc_llf + 1e-10


def kl_budget_from_mc(mc_value: float) -> float:
    """Return the approximate KL divergence budget MC^2 / 2.

    This follows from the second-order expansion of KL divergence under the
    Fisher metric:

        D_KL ≈ (1/2) Δθ^T F Δθ

    and MC = sqrt(Δθ^T F Δθ), so KL ≈ MC^2 / 2.

    Args:
        mc_value: The manipulation cost value.

    Returns:
        The approximate KL divergence budget.

    Examples:
        >>> kl_budget_from_mc(0.0129)
        8.3205e-05
    """
    return (mc_value ** 2) / 2.0
