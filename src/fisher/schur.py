"""Corrected block-inverse decomposition and LLF exactness verification.

This module implements the mathematical core for resolving the "Schur gap"
miscalculation and establishing when the LLF bound is exactly tight.

The key results are:
- Lemma 7.12: Corrected Block-Inverse Decomposition with shifted residual
- Theorem 7.14: LLF Exactness under Range Alignment
- Theorem 7.15: Range-Based Cost Formula

The corrected decomposition accounts for the cross-block coupling between
backbone and readout parameters, showing that the naive unshifted residual
produces a spurious Schur gap.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class BlockFisherPartition:
    """Partition of the full Fisher matrix into backbone and readout blocks.

    Attributes:
        F_theta: Backbone Fisher block, shape (p_theta, p_theta).
        F_w: Readout Fisher block, shape (p_w, p_w).
        F_theta_w: Cross-block from backbone to readout, shape (p_theta, p_w).
        F_w_theta: Transpose cross-block, shape (p_w, p_theta).
        p_theta: Backbone parameter dimension.
        p_w: Readout parameter dimension.
    """

    F_theta: np.ndarray
    F_w: np.ndarray
    F_theta_w: np.ndarray
    F_w_theta: np.ndarray
    p_theta: int
    p_w: int


@dataclass
class QuadraticFormDecomposition:
    """Decomposition of the quadratic form g^T F^dagger g.

    Attributes:
        total: Full quadratic form g^T F^dagger g.
        backbone_term: g_theta^T S^{-1} g_theta where S is the Schur complement.
        shifted_residual: The vector r = g_w - F_w_theta F_theta^{-1} g_theta.
        readout_term_corrected: r^T F_w^{-1} r (the corrected readout contribution).
        readout_term_naive: g_w^T F_w^{-1} g_w (the unshifted, incorrect form).
        schur_gap_corrected: total - backbone_term (should equal readout_term_corrected).
        schur_gap_naive: readout_term_naive (the spurious gap from unshifted residual).
        correction_magnitude: abs(readout_term_corrected - readout_term_naive).
    """

    total: float
    backbone_term: float
    shifted_residual: np.ndarray
    readout_term_corrected: float
    readout_term_naive: float
    schur_gap_corrected: float
    schur_gap_naive: float
    correction_magnitude: float


@dataclass
class RangeProjectionResult:
    """Result of computing orthogonal projection onto range(J_w).

    Attributes:
        P: Orthogonal projection onto range(J_w), shape (n, n).
        Q: Orthogonal projection onto range(J_w)^perp, shape (n, n).
        basis_U: Left singular vectors of J_w, shape (n, r).
        rank: Numerical rank of J_w.
        singular_values: Singular values of J_w, shape (r,).
    """

    P: np.ndarray
    Q: np.ndarray
    basis_U: np.ndarray  # noqa: N815
    rank: int
    singular_values: np.ndarray


@dataclass
class RangeInclusionResult:
    """Result of checking range inclusion condition.

    Attributes:
        is_included: True if range(J_theta) ⊆ range(J_w).
        max_projection_residual: Maximum over columns of ||Q_{J_w} j_theta_i||.
        rank_theta: Numerical rank of J_theta.
        rank_w: Numerical rank of J_w.
        rank_joint: Numerical rank of [J_theta | J_w].
        tolerance: The tolerance used for rank determination.
    """

    is_included: bool
    max_projection_residual: float
    rank_theta: int
    rank_w: int
    rank_joint: int
    tolerance: float


@dataclass
class LLFExactnessResult:
    """Result of verifying Theorem 7.14 numerically.

    Attributes:
        is_exact: True if Q_{J_w} a = 0 (within tolerance).
        residual_norm: ||Q_{J_w} a||_2.
        Q_a: The residual vector Q_{J_w} a.
        mc_full: Manipulation cost under full Fisher.
        mc_llf: Manipulation cost under LLF.
        relative_gap: |MC_full - MC_LLF| / MC_LLF.
    """

    is_exact: bool
    residual_norm: float
    Q_a: np.ndarray
    mc_full: float
    mc_llf: float
    relative_gap: float


def partition_fisher(F: np.ndarray, p_theta: int) -> BlockFisherPartition:  # noqa: N803
    """Partition a full Fisher matrix into backbone and readout blocks.

    The first p_theta rows/columns are the backbone; the remaining are the readout.

    Args:
        F: Full Fisher matrix, shape (p, p) where p = p_theta + p_w.
        p_theta: Dimension of backbone parameters. Must satisfy 1 <= p_theta < p.

    Returns:
        BlockFisherPartition containing the four blocks and dimensions.

    Raises:
        ValueError: If F is not square.
        ValueError: If F is not symmetric (within 1e-10).
        ValueError: If p_theta < 1 or p_theta >= F.shape[0].
    """
    F = np.asarray(F)  # noqa: N806

    # Validate F is square
    if F.ndim != 2 or F.shape[0] != F.shape[1]:
        raise ValueError("F must be a square matrix")

    n = F.shape[0]

    # Validate p_theta
    if p_theta < 1 or p_theta >= n:
        raise ValueError(f"p_theta must satisfy 1 <= p_theta < {n}, got {p_theta}")

    # Validate symmetry
    if not np.allclose(F, F.T, rtol=1e-10, atol=1e-10):
        raise ValueError("F must be symmetric")

    p_w = n - p_theta

    # Extract blocks
    F_theta = F[:p_theta, :p_theta]  # noqa: N806
    F_w = F[p_theta:, p_theta:]  # noqa: N806
    F_theta_w = F[:p_theta, p_theta:]  # noqa: N806
    F_w_theta = F[p_theta:, :p_theta]  # noqa: N806

    return BlockFisherPartition(
        F_theta=F_theta,
        F_w=F_w,
        F_theta_w=F_theta_w,
        F_w_theta=F_w_theta,
        p_theta=p_theta,
        p_w=p_w,
    )


def _solve_symmetric(A: np.ndarray, b: np.ndarray, damping: float) -> np.ndarray:  # noqa: N803
    """Solve (A + damping*I) x = b using Cholesky or LU fallback.

    Args:
        A: Symmetric matrix, shape (n, n).
        b: Right-hand side, shape (n,) or (n, m).
        damping: Tikhonov regularization parameter.

    Returns:
        Solution x, same shape as b.
    """
    n = A.shape[0]
    A_reg = A + damping * np.eye(n)  # noqa: N806

    # Try Cholesky first (faster for SPD matrices)
    try:
        L = np.linalg.cholesky(A_reg)  # noqa: N806
        # Solve L y = b, then L^T x = y
        if b.ndim == 1:
            y = np.linalg.solve(L, b)
            x = np.linalg.solve(L.T, y)
        else:
            y = np.linalg.solve(L, b)
            x = np.linalg.solve(L.T, y)
        return x
    except np.linalg.LinAlgError:
        # Fall back to LU
        return np.linalg.solve(A_reg, b)


def corrected_quadratic_form(  # noqa: N803
    F: np.ndarray,  # noqa: N803
    g: np.ndarray,
    p_theta: int,
    damping: float = 1e-6,
) -> QuadraticFormDecomposition:
    """Compute the corrected Pythagorean decomposition of g^T F^dagger g.

    Uses Tikhonov damping for numerical stability. The decomposition is:

        g^T F^dagger g = g_theta^T S^{-1} g_theta
                       + (g_w - F_w_theta F_theta^{-1} g_theta)^T F_w^{-1}
                         (g_w - F_w_theta F_theta^{-1} g_theta)

    where S = F_theta - F_theta_w F_w^{-1} F_w_theta is the Schur complement.

    Args:
        F: Full Fisher matrix, shape (p, p).
        g: Gradient vector, shape (p,).
        p_theta: Dimension of backbone parameters.
        damping: Tikhonov regularization parameter. Must be non-negative.

    Returns:
        QuadraticFormDecomposition containing all terms of the decomposition.

    Raises:
        ValueError: If shapes are inconsistent.
        ValueError: If damping < 0.
    """
    F = np.asarray(F)  # noqa: N806
    g = np.asarray(g)

    # Validate damping
    if damping < 0:
        raise ValueError("damping must be non-negative")

    # Validate shapes
    if F.ndim != 2 or F.shape[0] != F.shape[1]:
        raise ValueError("F must be a square matrix")

    if g.ndim != 1 or g.shape[0] != F.shape[0]:
        raise ValueError("g shape does not match F")

    # Partition F
    partition = partition_fisher(F, p_theta)

    F_theta = partition.F_theta  # noqa: N806
    F_w = partition.F_w  # noqa: N806
    F_theta_w = partition.F_theta_w  # noqa: N806
    F_w_theta = partition.F_w_theta  # noqa: N806

    # Split g
    g_theta = g[:p_theta]
    g_w = g[p_theta:]

    # Compute inverses with damping
    eps = damping

    # Step 2: Compute Schur complement S = F_theta - F_theta_w (F_w + eps I)^{-1} F_w_theta
    F_w_inv_F_w_theta = _solve_symmetric(F_w, F_w_theta, eps)  # noqa: N806
    S = F_theta - F_theta_w @ F_w_inv_F_w_theta  # noqa: N806

    # Step 3: Compute backbone term g_theta^T (S + eps I)^{-1} g_theta
    S_inv_g_theta = _solve_symmetric(S, g_theta, eps)  # noqa: N806
    backbone_term = float(g_theta @ S_inv_g_theta)

    # Step 4: Compute shifted residual r = g_w - F_w_theta (F_theta + eps I)^{-1} g_theta
    F_theta_inv_g_theta = _solve_symmetric(F_theta, g_theta, eps)  # noqa: N806
    shifted_residual = g_w - F_w_theta @ F_theta_inv_g_theta

    # Step 5: Compute corrected readout term r^T (F_w + eps I)^{-1} r
    F_w_inv_r = _solve_symmetric(F_w, shifted_residual, eps)  # noqa: N806
    readout_term_corrected = float(shifted_residual @ F_w_inv_r)

    # Step 6: Compute naive readout term g_w^T (F_w + eps I)^{-1} g_w
    F_w_inv_g_w = _solve_symmetric(F_w, g_w, eps)  # noqa: N806
    readout_term_naive = float(g_w @ F_w_inv_g_w)

    # Step 7: Compute total via direct pseudoinverse g^T (F + eps I)^{-1} g
    F_inv_g = _solve_symmetric(F, g, eps)  # noqa: N806
    total = float(g @ F_inv_g)

    # Step 8: Compute gaps
    schur_gap_corrected = total - backbone_term
    schur_gap_naive = readout_term_naive
    correction_magnitude = abs(readout_term_corrected - readout_term_naive)

    return QuadraticFormDecomposition(
        total=total,
        backbone_term=backbone_term,
        shifted_residual=shifted_residual,
        readout_term_corrected=readout_term_corrected,
        readout_term_naive=readout_term_naive,
        schur_gap_corrected=schur_gap_corrected,
        schur_gap_naive=schur_gap_naive,
        correction_magnitude=correction_magnitude,
    )


def range_projection(J_w: np.ndarray, tol: float = 1e-10) -> RangeProjectionResult:  # noqa: N803
    """Compute the orthogonal projection onto range(J_w) via SVD.

    Uses numpy.linalg.svd with full_matrices=False. The rank is determined
    as the number of singular values > tol. Constructs P = U U^T and Q = I - P.

    Args:
        J_w: Jacobian matrix with respect to readout parameters, shape (n, m).
        tol: Tolerance for singular value threshold.

    Returns:
        RangeProjectionResult containing P, Q, basis_U, rank, and singular_values.

    Raises:
        ValueError: If J_w is not 2-D.
        ValueError: If J_w has zero rows or columns.
    """
    J_w = np.asarray(J_w)  # noqa: N806

    # Validate J_w is 2-D
    if J_w.ndim != 2:
        raise ValueError("J_w must be 2-D")

    n_rows, n_cols = J_w.shape

    # Validate J_w has non-zero dimensions
    if n_rows == 0 or n_cols == 0:
        raise ValueError("J_w must have non-zero rows and columns")

    # Check for all-zero matrix (rank will be 0)
    if np.allclose(J_w, 0):
        raise ValueError("J_w must have non-zero rows and columns")

    # Compute thin SVD
    U, s, Vt = np.linalg.svd(J_w, full_matrices=False)  # noqa: N806

    # Determine rank
    rank = int(np.sum(s > tol))

    # Truncate U to significant singular vectors
    if rank > 0:
        U_r = U[:, :rank]  # noqa: N806
        s_r = s[:rank]
    else:
        U_r = np.zeros((n_rows, 0))  # noqa: N806
        s_r = np.array([])

    # Construct P = U U^T
    P = U_r @ U_r.T  # noqa: N806

    # Construct Q = I - P
    Q = np.eye(n_rows) - P  # noqa: N806

    return RangeProjectionResult(
        P=P,
        Q=Q,
        basis_U=U_r,
        rank=rank,
        singular_values=s_r,
    )


def _matrix_rank(A: np.ndarray, tol: float = 1e-10) -> int:  # noqa: N803
    """Compute numerical rank of a matrix via SVD.

    Args:
        A: Matrix, shape (m, n).
        tol: Tolerance for singular value threshold.

    Returns:
        Numerical rank of A.
    """
    s = np.linalg.svd(A, compute_uv=False)
    return int(np.sum(s > tol))


def check_range_inclusion(  # noqa: N803
    J_theta: np.ndarray,  # noqa: N803
    J_w: np.ndarray,  # noqa: N803
    tol: float = 1e-10,
) -> RangeInclusionResult:
    """Verify whether range(J_theta) ⊆ range(J_w).

    For each column j_theta_i of J_theta, computes ||Q_{J_w} j_theta_i||.
    range inclusion holds iff the maximum projection residual < tol.

    Args:
        J_theta: Jacobian matrix with respect to backbone parameters, shape (n, m_theta).
        J_w: Jacobian matrix with respect to readout parameters, shape (n, m_w).
        tol: Tolerance for projection residual and rank determination.

    Returns:
        RangeInclusionResult containing is_included, max_projection_residual,
        ranks, and tolerance.

    Raises:
        ValueError: If row counts differ.
        ValueError: If either matrix is not 2-D.
    """
    J_theta = np.asarray(J_theta)  # noqa: N806
    J_w = np.asarray(J_w)  # noqa: N806

    # Validate both are 2-D
    if J_theta.ndim != 2:
        raise ValueError("J_theta must be 2-D")
    if J_w.ndim != 2:
        raise ValueError("J_w must be 2-D")

    # Validate row counts match
    if J_theta.shape[0] != J_w.shape[0]:
        raise ValueError("J_theta and J_w must have the same number of rows")

    n = J_w.shape[0]  # noqa: F841

    # Compute Q_{J_w}
    proj_result = range_projection(J_w, tol)
    Q = proj_result.Q  # noqa: N806

    # Compute projection residuals for each column of J_theta
    max_residual = 0.0
    for i in range(J_theta.shape[1]):
        j_theta_i = J_theta[:, i]
        residual = Q @ j_theta_i
        residual_norm = float(np.linalg.norm(residual))
        max_residual = max(max_residual, residual_norm)

    is_included = max_residual < tol

    # Compute ranks
    rank_theta = _matrix_rank(J_theta, tol)
    rank_w = proj_result.rank

    # Compute joint rank
    J_joint = np.hstack([J_theta, J_w])  # noqa: N806
    rank_joint = _matrix_rank(J_joint, tol)

    return RangeInclusionResult(
        is_included=is_included,
        max_projection_residual=max_residual,
        rank_theta=rank_theta,
        rank_w=rank_w,
        rank_joint=rank_joint,
        tolerance=tol,
    )


def _compute_mc_from_jacobian(  # noqa: N803
    a: np.ndarray,
    J: np.ndarray,  # noqa: N803
    margin: float,
    damping: float = 1e-6,
) -> float:
    """Compute manipulation cost MC = margin / sqrt(a^T (J J^T)^+ a).

    Args:
        a: Target reward-shift vector, shape (n,).
        J: Jacobian matrix, shape (n, m).
        margin: Decision margin. Must be positive.
        damping: Tikhonov regularization for (J J^T).

    Returns:
        Manipulation cost value.
    """
    # Compute J J^T
    JJT = J @ J.T  # noqa: N806

    # Solve (J J^T + damping*I) x = a
    n = JJT.shape[0]
    JJT_reg = JJT + damping * np.eye(n)  # noqa: N806
    x = np.linalg.solve(JJT_reg, a)

    # Compute quadratic form a^T x
    quad_form = float(a @ x)

    if quad_form <= 0:
        raise ValueError("Quadratic form must be positive")

    return margin / np.sqrt(quad_form)


def llf_exactness_check(  # noqa: N803
    a: np.ndarray,
    J_w: np.ndarray,  # noqa: N803
    J_theta: np.ndarray,  # noqa: N803
    F_full: np.ndarray,  # noqa: N803
    F_llf: np.ndarray,  # noqa: N803
    margin: float,
    tol: float = 1e-10,
    damping: float = 1e-6,
) -> LLFExactnessResult:
    """Verify Theorem 7.14 numerically.

    The theorem states that MC_full = MC_LLF if and only if Q_{J_w} a = 0,
    where Q_{J_w} is the orthogonal projection onto range(J_w)^perp.

    When range(J_theta) ⊆ range(J_w), we have range([J_theta | J_w]) = range(J_w),
    so MC_full = MC_LLF if a ∈ range(J_w).

    Args:
        a: Target reward-shift vector, shape (n,).
        J_w: Jacobian with respect to readout parameters, shape (n, m_w).
        J_theta: Jacobian with respect to backbone parameters, shape (n, m_theta).
        F_full: Full Fisher matrix (not directly used, kept for API consistency).
        F_llf: LLF Fisher matrix (not directly used, kept for API consistency).
        margin: Decision margin. Must be positive.
        tol: Tolerance for residual norm check.
        damping: Tikhonov regularization for matrix inversion.

    Returns:
        LLFExactnessResult containing is_exact, residual_norm, Q_a, mc_full,
        mc_llf, and relative_gap.

    Raises:
        ValueError: If shapes are inconsistent.
        ValueError: If margin <= 0.
    """
    a = np.asarray(a)
    J_w = np.asarray(J_w)  # noqa: N806
    J_theta = np.asarray(J_theta)  # noqa: N806

    # Validate margin
    if margin <= 0:
        raise ValueError("margin must be positive")

    # Validate shapes
    if a.ndim != 1:
        raise ValueError("a must be 1-D")
    if J_w.ndim != 2:
        raise ValueError("J_w must be 2-D")
    if J_theta.ndim != 2:
        raise ValueError("J_theta must be 2-D")
    if J_w.shape[0] != a.shape[0]:
        raise ValueError("J_w row count must match a length")
    if J_theta.shape[0] != a.shape[0]:
        raise ValueError("J_theta row count must match a length")

    # Step 1: Compute Q_{J_w} a and its norm
    proj_result = range_projection(J_w, tol)
    Q = proj_result.Q  # noqa: N806
    Q_a = Q @ a  # noqa: N806
    residual_norm = float(np.linalg.norm(Q_a))

    # Step 2: Compute MC_full using the full Jacobian J = [J_theta | J_w]
    # When range(J_theta) ⊆ range(J_w), range(J_full) = range(J_w)
    J_full = np.hstack([J_theta, J_w])  # noqa: N806
    mc_full = _compute_mc_from_jacobian(a, J_full, margin, damping)

    # Step 3: Compute MC_LLF using J_w only
    mc_llf = _compute_mc_from_jacobian(a, J_w, margin, damping)

    # Step 4: Determine is_exact
    is_exact = residual_norm < tol

    # Step 5: Compute relative gap
    relative_gap = abs(mc_full - mc_llf) / mc_llf if mc_llf > 0 else float("inf")

    return LLFExactnessResult(
        is_exact=is_exact,
        residual_norm=residual_norm,
        Q_a=Q_a,
        mc_full=mc_full,
        mc_llf=mc_llf,
        relative_gap=relative_gap,
    )


def range_based_cost(  # noqa: N803
    a: np.ndarray,
    J_w: np.ndarray,  # noqa: N803
    W_C: np.ndarray,  # noqa: N803
    margin: float,
    tol: float = 1e-10,
) -> float:
    """Compute the manipulation cost using Theorem 7.15's range-based formula.

    Under range inclusion, the cost is:
        MC = margin / sqrt(a^T U (U^T W_C U)^{-1} U^T a)

    where U is the left singular vector basis of J_w. This expression is
    independent of the singular values Sigma and right singular vectors V.

    Args:
        a: Target reward-shift vector, shape (n,).
        J_w: Jacobian with respect to readout parameters, shape (n, m).
        W_C: Comparison-space Fisher weight matrix, shape (n, n).
        margin: Decision margin. Must be positive.
        tol: Tolerance for singular value threshold.

    Returns:
        Manipulation cost computed via the range-based formula.

    Raises:
        ValueError: If W_C is not square or shape-mismatched with a.
        ValueError: If margin <= 0.
        ValueError: If the quadratic form is non-positive.
    """
    a = np.asarray(a)
    J_w = np.asarray(J_w)
    W_C = np.asarray(W_C)

    # Validate margin
    if margin <= 0:
        raise ValueError("margin must be positive")

    # Validate shapes
    if a.ndim != 1:
        raise ValueError("a must be 1-D")
    if J_w.ndim != 2:
        raise ValueError("J_w must be 2-D")
    if W_C.ndim != 2 or W_C.shape[0] != W_C.shape[1]:
        raise ValueError("W_C must be square")
    if W_C.shape[0] != a.shape[0]:
        raise ValueError("W_C shape does not match a")
    if J_w.shape[0] != a.shape[0]:
        raise ValueError("J_w row count must match a length")

    # Compute SVD of J_w
    proj_result = range_projection(J_w, tol)
    U = proj_result.basis_U  # noqa: N806
    rank = proj_result.rank

    if rank == 0:
        raise ValueError("J_w has zero rank; cannot compute range-based cost")

    # Compute U^T W_C U
    UT_WC = U.T @ W_C  # noqa: N806
    UT_WC_U = UT_WC @ U  # noqa: N806

    # Invert with damping for stability
    eps = 1e-12
    UT_WC_U_inv = np.linalg.inv(UT_WC_U + eps * np.eye(rank))  # noqa: N806

    # Compute a^T U (U^T W_C U)^{-1} U^T a
    U_T_a = U.T @ a  # noqa: N806
    intermediate = UT_WC_U_inv @ U_T_a
    quad_form = float(U_T_a @ intermediate)

    if quad_form <= 0:
        raise ValueError("Quadratic form must be positive")

    return margin / np.sqrt(quad_form)


def schur_gap_diagnostic(  # noqa: N803
    F: np.ndarray,  # noqa: N803
    g: np.ndarray,
    p_theta: int,
    damping: float = 1e-6,
) -> dict[str, object]:
    """Return a diagnostic dictionary comparing corrected and naive Schur gaps.

    This replicates the diagnostic from the paper's Experiment 18, which
    diagnosed the spurious gap produced by using the unshifted residual.

    Args:
        F: Full Fisher matrix, shape (p, p).
        g: Gradient vector, shape (p,).
        p_theta: Dimension of backbone parameters.
        damping: Tikhonov regularization parameter.

    Returns:
        Dictionary with keys:
            "corrected_gap": The corrected Schur gap (readout_term_corrected).
            "naive_gap": The naive Schur gap (readout_term_naive).
            "correction_magnitude": |corrected - naive|.
            "relative_correction": correction_magnitude / |corrected_gap|.
            "diagnosis": English string explaining the finding.
    """
    decomp = corrected_quadratic_form(F, g, p_theta, damping)

    corrected_gap = decomp.readout_term_corrected
    naive_gap = decomp.readout_term_naive
    correction_magnitude = decomp.correction_magnitude

    if abs(corrected_gap) > 1e-15:
        relative_correction = correction_magnitude / abs(corrected_gap)
    else:
        relative_correction = float("inf") if correction_magnitude > 0 else 0.0

    # Generate diagnosis
    if correction_magnitude < 1e-10:
        diagnosis = "No significant correction needed; naive and corrected forms agree."
    elif relative_correction > 0.1:
        diagnosis = (
            "Substantial correction detected. The naive unshifted residual would "
            "have produced a spurious Schur gap that does not reflect the true "
            "cost difference. This confirms the necessity of the shift term "
            "F_w_theta F_theta^{-1} g_theta."
        )
    else:
        diagnosis = (
            "Minor correction detected. The naive form slightly overestimates "
            "the readout contribution due to ignoring the cross-block coupling."
        )

    return {
        "corrected_gap": corrected_gap,
        "naive_gap": naive_gap,
        "correction_magnitude": correction_magnitude,
        "relative_correction": relative_correction,
        "diagnosis": diagnosis,
    }
